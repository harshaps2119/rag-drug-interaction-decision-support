"""
services/pair_retrieval_service.py
=====================================
Phase 6: given two drugs, retrieve and classify evidence about their
RELATIONSHIP — not just evidence about each drug individually.

WHY THIS FILE EXISTS SEPARATELY FROM services/retrieval_service.py
------------------------------------------------------------------------
Phase 5's retrieve() answers "what stored text is semantically similar to
this query?" — a general-purpose primitive. It has no concept of "drug
pairs" at all. This file is where that concept is built: given Drug A and
Drug B, it decides WHAT to search for (several queries, not one), scopes
each search appropriately, merges and deduplicates the results, and —
critically — classifies each result by how directly it actually connects
the two drugs, rather than just how similar it is to a query.

THE CORE SAFETY REQUIREMENT THIS FILE IMPLEMENTS
------------------------------------------------------
Retrieving relevant evidence about Warfarin, and separately retrieving
relevant evidence about Ibuprofen, does NOT mean evidence of a
Warfarin-Ibuprofen interaction was found. This file never conflates
those. A pair only gets `evidence_status="pair_specific_evidence_found"`
when a retrieved chunk's OWN source drug is one of the pair AND its text
explicitly names the other drug (or a known synonym) — see
`classify_chunk()` below and `docs/retrieval.md` for the full reasoning.
"no interaction" is not a possible outcome this file can produce, ever —
only "we did/didn't find evidence connecting them", which is a
fundamentally weaker and more honest claim.

RETRIEVAL STRATEGY (hybrid: multiple queries, then merge)
------------------------------------------------------------
For a pair, `generate_pair_queries()` builds a small, bounded set of
queries of three kinds:
  1. pair_specific  — e.g. "warfarin and ibuprofen interaction",
     including combinations with each drug's brand name.
  2. drug_a_specific — e.g. "warfarin drug interactions"
  3. drug_b_specific — e.g. "ibuprofen drug interactions"
Each is executed via Phase 5's retrieve(), scoped (via ChromaDB's `$in`
filter — see rag/vector_store.build_where) to only the evidence already
indexed for drug A and/or drug B, as appropriate to the query type. This
is the "hybrid" part: no single query type alone reliably surfaces
pair-relevant text, so several angles are tried and the results merged.

EFFICIENCY FOR MULTI-DRUG LISTS
------------------------------------
`PairRetrievalEngine` caches drug-specific query results (keyed by
rxcui + query text) across calls to `retrieve_pair()`. For a list of N
drugs, `retrieve_multi()` generates C(N,2) pairs, and every pair sharing
a drug reuses that drug's cached "drug X interactions"-type query
results instead of re-querying ChromaDB — see
tests/test_pair_retrieval_service.py::test_retrieve_multi_reuses_cached_drug_specific_queries
for a concrete, executed test proving the cache is actually hit, not
just present in the code.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_pair_retrieval_service.py -v

Uses controlled SQLite + ChromaDB fixtures built directly in the test
file (not live RxNorm/DailyMed/embedding calls) — proves the retrieval
and classification LOGIC, independent of real API availability.
"""

from __future__ import annotations

import itertools

from sqlalchemy.orm import Session

from app.rag.entity_matching import find_class_mention_containing, find_mentions
from app.schemas.evidence_assessment import (
    DrugMentionResult,
    DrugRef,
    EvidenceClassification,
    PairEvidenceItem,
    PairRetrievalResult,
)
from app.schemas.retrieval import RetrievedChunk
from app.services.knowledge_base_service import get_drug_by_rxcui, search_drugs_by_name
from app.services.retrieval_service import retrieve

# Always attached to every PairRetrievalResult — see module docstring.
STANDARD_LIMITATIONS: list[str] = [
    "Drug mention detection uses simple case-insensitive whole-word string matching on known names/synonyms "
    "(app/rag/entity_matching.py) — it is not clinical natural language entity resolution and does not "
    "understand negation (text saying an interaction was NOT observed still counts as a 'mention').",
    "Drug-class matching uses a small, illustrative, non-exhaustive internal vocabulary — its absence of a "
    "match does not mean no class-level relationship exists.",
    "Evidence is drawn only from each drug's own previously-ingested FDA label sections (Phases 3-5) — if "
    "neither drug's own label happens to mention the other, no pair-specific evidence will be found here, "
    "even if a real interaction is documented elsewhere in the medical literature.",
    "A low distance (high similarity) score reflects textual/topical similarity only — it is never "
    "interpreted as interaction probability, severity, or clinical risk.",
    "This retrieval layer does not itself determine whether an interaction exists — it only surfaces and "
    "classifies evidence for the downstream explanation layer, which must remain grounded in the retrieved evidence.",
]

_MAX_NAME_VARIANTS = 2  # bounds query-combination growth — see generate_pair_queries()
_SUPPORT_RANK = {
    EvidenceClassification.DRUG_SPECIFIC: 0,
    EvidenceClassification.CLASS_LEVEL: 1,
    EvidenceClassification.GENERAL_LABEL: 2,
}


# ---------------------------------------------------------------------------
# Pair generation (for multi-drug lists)
# ---------------------------------------------------------------------------

def generate_unique_pairs(drug_names: list[str]) -> list[tuple[str, str]]:
    """
    Generates unique, order-independent drug pairs from a list of names.
    Repeated names (case-insensitive) are deduplicated first, so
    [A, B, A] never produces (A, A). For N distinct names this produces
    C(N, 2) = N*(N-1)/2 pairs — e.g. 4 drugs -> 6 pairs, 10 drugs -> 45
    pairs. That growth is inherent to "every unique pair" and isn't
    reduced algorithmically here; see docs/retrieval.md for this as a
    documented scaling limitation. What IS avoided is generating both
    (A, B) and (B, A) as separate, redundant pairs, and re-processing an
    exact duplicate name entered twice.
    """
    seen_lower: set[str] = set()
    unique_names: list[str] = []
    for name in drug_names:
        clean = name.strip()
        key = clean.lower()
        if clean and key not in seen_lower:
            seen_lower.add(key)
            unique_names.append(clean)

    return list(itertools.combinations(unique_names, 2))


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def _bounded_variants(drug: DrugRef, max_variants: int = _MAX_NAME_VARIANTS) -> list[str]:
    """Returns up to max_variants distinct name strings for a drug (normalized name first, then synonyms)."""
    candidates = [v for v in ([drug.normalized_name] + list(drug.search_terms)) if v]
    seen: set[str] = set()
    unique: list[str] = []
    for v in candidates:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique[:max_variants] or [drug.input_name]


def generate_pair_queries(drug_a: DrugRef, drug_b: DrugRef) -> list[tuple[str, str]]:
    """
    Builds a small, bounded, deduplicated set of (query_text, query_type)
    tuples for a drug pair. query_type is one of "pair_specific",
    "drug_a_specific", "drug_b_specific".
    """
    a_variants = _bounded_variants(drug_a)
    b_variants = _bounded_variants(drug_b)

    queries: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(text: str, qtype: str) -> None:
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            queries.append((text.strip(), qtype))

    for a_term in a_variants:
        for b_term in b_variants:
            add(f"{a_term} and {b_term} interaction", "pair_specific")
            add(f"{a_term} {b_term}", "pair_specific")

    for a_term in a_variants[:1]:
        add(f"{a_term} drug interactions", "drug_a_specific")
        add(f"{a_term} interaction warnings", "drug_a_specific")

    for b_term in b_variants[:1]:
        add(f"{b_term} drug interactions", "drug_b_specific")
        add(f"{b_term} interaction warnings", "drug_b_specific")

    return queries


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_chunk(
    chunk: RetrievedChunk, drug_a: DrugRef, drug_b: DrugRef
) -> tuple[EvidenceClassification, DrugMentionResult]:
    """
    Classifies one retrieved chunk in the context of a specific drug
    pair. See app/schemas/evidence_assessment.py's module docstring for
    what each classification means.

    A chunk's OWN source drug (chunk.rxcui, from its Phase 4/5
    provenance) counts as an implicit "mention" of that drug even if the
    text doesn't literally repeat the drug's name (FDA labels often say
    "this drug" rather than naming themselves) — but PAIR_SPECIFIC still
    requires the OTHER drug's name (or synonym) to appear in the text
    itself, which is what actually makes it pair-specific rather than
    just self-referential.
    """
    matched_a = find_mentions(chunk.text, drug_a.search_terms or [drug_a.normalized_name or ""])
    matched_b = find_mentions(chunk.text, drug_b.search_terms or [drug_b.normalized_name or ""])

    mentions_a = bool(matched_a) or (chunk.rxcui is not None and chunk.rxcui == drug_a.rxcui)
    mentions_b = bool(matched_b) or (chunk.rxcui is not None and chunk.rxcui == drug_b.rxcui)

    class_match_a = None if mentions_a else find_class_mention_containing(chunk.text, drug_a.normalized_name or "")
    class_match_b = None if mentions_b else find_class_mention_containing(chunk.text, drug_b.normalized_name or "")

    mention_result = DrugMentionResult(
        mentions_drug_a=mentions_a,
        mentions_drug_b=mentions_b,
        matched_terms_a=matched_a,
        matched_terms_b=matched_b,
        class_match_a=class_match_a,
        class_match_b=class_match_b,
    )

    if mentions_a and mentions_b:
        classification = EvidenceClassification.PAIR_SPECIFIC
    elif (mentions_a and class_match_b) or (mentions_b and class_match_a):
        classification = EvidenceClassification.CLASS_LEVEL
    elif mentions_a or mentions_b:
        classification = EvidenceClassification.DRUG_SPECIFIC
    else:
        classification = EvidenceClassification.GENERAL_LABEL

    return classification, mention_result


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PairRetrievalEngine:
    """
    Stateful wrapper (holds a session, embedding model, ChromaDB
    collection, and a query-result cache) so a batch of related pair
    retrievals — e.g. all pairs from one multi-drug list — can share
    cached drug-specific query results. See module docstring's
    "efficiency" section.
    """

    def __init__(self, session: Session, embedding_model, collection, top_k_per_query: int = 5):
        self.session = session
        self.embedding_model = embedding_model
        self.collection = collection
        self.top_k_per_query = top_k_per_query
        self._query_cache: dict[str, list[RetrievedChunk]] = {}
        self.cache_hits = 0  # exposed for tests to prove the cache is actually used
        self.cache_misses = 0

    def resolve_drug(self, name_or_rxcui: str) -> DrugRef:
        """
        Resolves a drug name (or a bare RxCUI string) against the local
        SQLite knowledge base (Phase 4). Does NOT call RxNorm live —
        Phase 6 operates on top of whatever has already been ingested;
        see docs/retrieval.md for why that's the right boundary here.
        """
        clean = name_or_rxcui.strip()
        drug = None

        if clean.isdigit():
            drug = get_drug_by_rxcui(self.session, clean)

        if drug is None:
            matches = search_drugs_by_name(self.session, clean)
            if matches:
                exact = [d for d in matches if (d.normalized_name or "").lower() == clean.lower()]
                drug = exact[0] if exact else matches[0]

        if drug is None:
            return DrugRef(input_name=clean, resolved=False)

        search_terms = [t for t in ([drug.normalized_name] + [s.synonym for s in drug.synonyms]) if t]
        return DrugRef(
            input_name=clean,
            rxcui=drug.rxcui,
            normalized_name=drug.normalized_name,
            resolved=True,
            search_terms=search_terms,
        )

    def _cached_retrieve(self, query_text: str, rxcui_in: list[str]) -> list[RetrievedChunk]:
        cache_key = f"{sorted(rxcui_in)}::{query_text.lower()}::{self.top_k_per_query}"
        if cache_key in self._query_cache:
            self.cache_hits += 1
            return self._query_cache[cache_key]

        self.cache_misses += 1
        results = retrieve(
            self.embedding_model,
            self.collection,
            query_text,
            top_k=self.top_k_per_query,
            rxcui=rxcui_in,
        )
        self._query_cache[cache_key] = results
        return results

    def retrieve_pair(self, drug_a_input: str, drug_b_input: str) -> PairRetrievalResult:
        """
        Retrieves and classifies evidence for one drug pair. Never
        raises for "drug not found" or "nothing retrieved" — reported
        via evidence_status, consistent with every other service in
        this project.
        """
        drug_a = self.resolve_drug(drug_a_input)
        drug_b = self.resolve_drug(drug_b_input)

        if not drug_a.resolved or not drug_b.resolved:
            unresolved = [n for n, d in ((drug_a_input, drug_a), (drug_b_input, drug_b)) if not d.resolved]
            return PairRetrievalResult(
                drug_a=drug_a,
                drug_b=drug_b,
                evidence_status="drug_not_found",
                warnings=[
                    f"Could not resolve in the local knowledge base: {', '.join(unresolved)}. "
                    "Has this drug been ingested yet (scripts/ingest_labels.py + build_vector_index.py)?"
                ],
                limitations=list(STANDARD_LIMITATIONS),
            )

        # A pairwise DDI check requires two distinct medications.
        # Compare resolved RxCUIs rather than raw input strings so aliases
        # such as "Coumadin" and "warfarin" are also rejected.
        if drug_a.rxcui == drug_b.rxcui:
            return PairRetrievalResult(
                drug_a=drug_a,
                drug_b=drug_b,
                evidence_status="invalid_input",
                warnings=[
                    "Drug A and Drug B resolve to the same RxCUI; a pairwise interaction check "
                    "requires two distinct medications."
                ],
                limitations=list(STANDARD_LIMITATIONS),
            )

        queries = generate_pair_queries(drug_a, drug_b)
        queries_used: list[str] = []
        merged: dict[str, PairEvidenceItem] = {}

        try:
            for query_text, qtype in queries:
                queries_used.append(query_text)
                if qtype == "drug_a_specific":
                    rxcui_in = [drug_a.rxcui] if drug_a.rxcui else []
                elif qtype == "drug_b_specific":
                    rxcui_in = [drug_b.rxcui] if drug_b.rxcui else []
                else:
                    rxcui_in = [r for r in (drug_a.rxcui, drug_b.rxcui) if r]

                if not rxcui_in:
                    continue

                chunks = self._cached_retrieve(query_text, rxcui_in)

                for chunk in chunks:
                    classification, mentions = classify_chunk(chunk, drug_a, drug_b)

                    existing = merged.get(chunk.chunk_id)
                    if existing is not None:
                        existing.distance = min(existing.distance, chunk.distance)
                        if query_text not in existing.retrieval_queries:
                            existing.retrieval_queries.append(query_text)
                            existing.retrieval_query_types.append(qtype)
                        continue

                    merged[chunk.chunk_id] = PairEvidenceItem(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        distance=chunk.distance,
                        drug_name=chunk.drug_name,
                        rxcui=chunk.rxcui,
                        label_setid=chunk.label_setid,
                        spl_version=chunk.spl_version,
                        manufacturer=chunk.manufacturer,
                        section_key=chunk.section_key,
                        section_name=chunk.section_name,
                        section_code=chunk.section_code,
                        source_url=chunk.source_url,
                        api_url=chunk.api_url,
                        db_record_id=chunk.db_record_id,
                        evidence_status_of_label=chunk.evidence_status,
                        classification=classification,
                        drug_mentions=mentions,
                        retrieval_queries=[query_text],
                        retrieval_query_types=[qtype],
                    )
        except Exception as exc:
            return PairRetrievalResult(
                drug_a=drug_a,
                drug_b=drug_b,
                evidence_status="retrieval_error",
                error=str(exc),
                queries_used=queries_used,
                limitations=list(STANDARD_LIMITATIONS),
            )

        pair_evidence = sorted(
            (item for item in merged.values() if item.classification == EvidenceClassification.PAIR_SPECIFIC),
            key=lambda i: i.distance,
        )
        supporting_evidence = sorted(
            (item for item in merged.values() if item.classification != EvidenceClassification.PAIR_SPECIFIC),
            key=lambda i: (_SUPPORT_RANK[i.classification], i.distance),
        )

        if pair_evidence:
            status = "pair_specific_evidence_found"
        elif supporting_evidence:
            status = "supporting_evidence_found"
        else:
            status = "insufficient_evidence"

        return PairRetrievalResult(
            drug_a=drug_a,
            drug_b=drug_b,
            evidence_status=status,
            pair_evidence=pair_evidence,
            supporting_evidence=supporting_evidence,
            queries_used=queries_used,
            limitations=list(STANDARD_LIMITATIONS),
        )

    def retrieve_multi(self, drug_names: list[str]) -> dict[tuple[str, str], PairRetrievalResult]:
        """
        Generates all unique pairs from `drug_names` (see
        generate_unique_pairs()) and retrieves evidence for each,
        reusing cached drug-specific query results across pairs that
        share a drug.
        """
        pairs = generate_unique_pairs(drug_names)
        return {pair: self.retrieve_pair(pair[0], pair[1]) for pair in pairs}
