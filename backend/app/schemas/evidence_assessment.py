"""
schemas/evidence_assessment.py
=================================
Schemas for Phase 6's pair-evidence retrieval: classifying individual
retrieved chunks by how directly they bear on a specific drug PAIR
(rather than just "how similar is this text to my query", which is all
Phase 5 alone can tell you).

THE CENTRAL DISTINCTION THIS FILE ENCODES
----------------------------------------------
Retrieving relevant evidence about Drug A, and separately retrieving
relevant evidence about Drug B, is NOT the same as finding evidence that
A and B actually interact. `EvidenceClassification` exists specifically
to keep these apart:

  - PAIR_SPECIFIC: the chunk's own source drug is one of the pair, AND
    the chunk's text also names the OTHER drug (or a known synonym of
    it) directly. This is the strongest signal this phase can produce —
    still just "these two are mentioned together in FDA label text", not
    proof of a clinically real interaction.
  - CLASS_LEVEL: the chunk names one target drug, and separately mentions
    a drug CLASS the other target drug belongs to (e.g. "NSAIDs"),
    without naming that other drug specifically.
  - DRUG_SPECIFIC: the chunk is relevant evidence about ONE of the two
    target drugs, but has no textual connection (direct or class-level)
    to the other one at all.
  - GENERAL_LABEL: the chunk was retrieved (semantically similar to a
    query) but doesn't clearly mention either target drug by name or
    class — e.g. boilerplate label language.

None of these four classifications, on their own or combined, are ever
read as "an interaction was found". They describe the TEXTUAL RELEVANCE
of a piece of retrieved evidence to the pair being asked about — nothing
about clinical truth. See docs/retrieval.md for the full explanation.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceClassification(str, Enum):
    PAIR_SPECIFIC = "pair_specific_evidence"
    DRUG_SPECIFIC = "drug_specific_evidence"
    CLASS_LEVEL = "class_level_evidence"
    GENERAL_LABEL = "general_label_evidence"


class DrugMentionResult(BaseModel):
    """The result of running entity_matching against one chunk of text for one drug pair."""

    mentions_drug_a: bool = False
    mentions_drug_b: bool = False
    matched_terms_a: list[str] = Field(
        default_factory=list, description="Which of drug A's name/synonym variants matched, if any."
    )
    matched_terms_b: list[str] = Field(default_factory=list)
    class_match_a: str | None = Field(
        default=None, description="A drug-class term found in the text that drug A belongs to, if drug A wasn't named directly."
    )
    class_match_b: str | None = None


class PairEvidenceItem(BaseModel):
    """
    One retrieved chunk, classified in the context of a specific drug
    pair. Extends Phase 5's RetrievedChunk with pair-specific
    classification, drug-mention detail, and which retrieval quer(y/ies)
    surfaced it.
    """

    chunk_id: str
    text: str
    distance: float = Field(
        description="Vector-space retrieval-relevance signal ONLY — see services/retrieval_service.retrieve()'s "
        "docstring. NOT a clinical severity/probability/interaction-proof value, here or anywhere."
    )

    drug_name: str | None = None
    rxcui: str | None = None
    label_setid: str | None = None
    spl_version: str | None = None
    manufacturer: str | None = None
    section_key: str | None = None
    section_name: str | None = None
    section_code: str | None = None
    source_url: str | None = None
    api_url: str | None = None
    db_record_id: int | None = None
    evidence_status_of_label: str | None = Field(
        default=None,
        description="The Phase 3/4 evidence_status of the LABEL this chunk came from (e.g. 'interaction_evidence_found'). "
        "Renamed from Phase 5's 'evidence_status' field to avoid confusion with PairRetrievalResult.evidence_status below, "
        "which is a different, pair-level concept introduced in this phase.",
    )

    classification: EvidenceClassification
    drug_mentions: DrugMentionResult

    retrieval_queries: list[str] = Field(
        default_factory=list, description="Every distinct query text that retrieved this chunk (deduplicated)."
    )
    retrieval_query_types: list[str] = Field(
        default_factory=list, description="Query type(s) ('pair_specific' / 'drug_a_specific' / 'drug_b_specific') that retrieved this chunk."
    )


class DrugRef(BaseModel):
    """A drug as resolved (or not) against the local SQLite knowledge base for pair retrieval."""

    input_name: str = Field(description="Exactly what was passed in for this drug.")
    rxcui: str | None = None
    normalized_name: str | None = None
    resolved: bool = Field(default=False, description="Whether this name/rxcui was found in the local knowledge base.")
    search_terms: list[str] = Field(
        default_factory=list,
        description="Name variants (normalized name + known synonyms/brand names) used for mention detection.",
    )


PairEvidenceStatus = Literal[
    "pair_specific_evidence_found",
    "supporting_evidence_found",
    "insufficient_evidence",
    "drug_not_found",
    "invalid_input",
    "retrieval_error",
]


class PairRetrievalResult(BaseModel):
    """
    The result of retrieving and classifying evidence for one drug pair.
    "no interaction" is deliberately NOT a possible evidence_status value
    — this layer never asserts that, one way or the other; see
    docs/retrieval.md.
    """

    drug_a: DrugRef
    drug_b: DrugRef

    evidence_status: PairEvidenceStatus

    pair_evidence: list[PairEvidenceItem] = Field(
        default_factory=list, description="Chunks classified PAIR_SPECIFIC, ranked by distance ascending."
    )
    supporting_evidence: list[PairEvidenceItem] = Field(
        default_factory=list,
        description="Chunks classified DRUG_SPECIFIC / CLASS_LEVEL / GENERAL_LABEL, ranked by classification strength then distance.",
    )

    queries_used: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list, description="Always-attached, standard caveats about this retrieval approach — see pair_retrieval_service.STANDARD_LIMITATIONS."
    )
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
