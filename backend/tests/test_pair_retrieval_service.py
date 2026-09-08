"""
tests/test_pair_retrieval_service.py
=======================================
Comprehensive tests for app/services/pair_retrieval_service.py.

Uses controlled fixtures: a real in-memory SQLite database (Phase 4
models, populated directly), a real local ChromaDB instance (pytest
tmp_path), and DeterministicFakeEmbeddingModel (Phase 5) — proving the
retrieval STRATEGY and CLASSIFICATION LOGIC, independent of any live
RxNorm/DailyMed/embedding-model API. Distances from the fake embedding
model carry no semantic meaning (see app/rag/embeddings.py) — tests here
never assert anything about WHICH result ranks higher by fake-embedding
distance, only about classification, filtering, dedup, and provenance,
which do not depend on real semantic quality.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_pair_retrieval_service.py -v
"""

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import init_db, make_engine
from app.models.db_models import Drug, DrugLabelLink, DrugSynonym, Label, LabelSectionRecord
from app.rag.embeddings import DeterministicFakeEmbeddingModel
from app.rag.ingest import ingest_rag_index
from app.rag.vector_store import get_client, get_collection
from app.schemas.evidence_assessment import EvidenceClassification
from app.services.pair_retrieval_service import (
    PairRetrievalEngine,
    classify_chunk,
    generate_pair_queries,
    generate_unique_pairs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


@pytest.fixture
def collection(tmp_path):
    client = get_client(persist_dir=str(tmp_path / "chroma"))
    return get_collection(client=client)


@pytest.fixture
def embedding_model():
    return DeterministicFakeEmbeddingModel(dimension=16)


def _add_drug(session, *, rxcui, name, synonyms=None):
    drug = Drug(rxcui=rxcui, normalized_name=name, input_name=name, match_type="exact")
    session.add(drug)
    session.commit()
    for syn, syn_type in (synonyms or []):
        session.add(DrugSynonym(drug_id=drug.id, synonym=syn, synonym_type=syn_type))
    session.commit()
    return drug


def _add_label_with_sections(session, drug, *, setid, sections):
    """sections: dict[section_key -> (section_code, section_name, text)]"""
    label = Label(setid=setid, spl_version="1", title=f"{drug.normalized_name.upper()} LABEL")
    session.add(label)
    session.commit()
    for key, (code, name, text) in sections.items():
        session.add(LabelSectionRecord(
            label_id=label.id, section_key=key, section_code=code, section_name=name, text=text, found=True,
        ))
    session.commit()
    link = DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()
    return label, link


def _seed_warfarin_ibuprofen(session, collection, embedding_model):
    """Seeds a realistic pair: Warfarin's label MENTIONS ibuprofen/NSAIDs; Ibuprofen's label doesn't mention warfarin."""
    warfarin = _add_drug(session, rxcui="11289", name="warfarin", synonyms=[("Coumadin", "brand")])
    ibuprofen = _add_drug(session, rxcui="5640", name="ibuprofen", synonyms=[("Advil", "brand")])

    _add_label_with_sections(session, warfarin, setid="warfarin-setid", sections={
        "drug_interactions": (
            "34073-7", "Drug Interactions",
            "Concomitant use of warfarin with NSAIDs such as ibuprofen may increase the risk of bleeding.",
        ),
        "contraindications": ("34070-3", "Contraindications", "Active pathological bleeding."),
    })
    _add_label_with_sections(session, ibuprofen, setid="ibuprofen-setid", sections={
        "drug_interactions": (
            "34073-7", "Drug Interactions",
            "Ibuprofen may reduce the antihypertensive effect of ACE inhibitors and diuretics.",
        ),
        "adverse_reactions": ("34084-4", "Adverse Reactions", "Gastrointestinal upset, nausea, dyspepsia."),
    })

    ingest_rag_index(session, embedding_model, collection)
    return warfarin, ibuprofen


# ---------------------------------------------------------------------------
# Tests: pair query generation
# ---------------------------------------------------------------------------

def test_generate_pair_queries_produces_all_three_query_types(session, collection, embedding_model):
    engine = PairRetrievalEngine(session, embedding_model, collection)
    _seed_warfarin_ibuprofen(session, collection, embedding_model)
    a = engine.resolve_drug("warfarin")
    b = engine.resolve_drug("ibuprofen")

    queries = generate_pair_queries(a, b)
    types = {qtype for _, qtype in queries}

    assert types == {"pair_specific", "drug_a_specific", "drug_b_specific"}


def test_generate_pair_queries_deduplicates():
    from app.schemas.evidence_assessment import DrugRef

    a = DrugRef(input_name="x", normalized_name="drugx", resolved=True, search_terms=["drugx", "drugx"])
    b = DrugRef(input_name="y", normalized_name="drugy", resolved=True, search_terms=["drugy"])

    queries = generate_pair_queries(a, b)
    texts = [q for q, _ in queries]
    assert len(texts) == len(set(t.lower() for t in texts))  # no duplicates


def test_generate_pair_queries_is_bounded_not_combinatorially_explosive():
    from app.schemas.evidence_assessment import DrugRef

    a = DrugRef(input_name="x", normalized_name="drugx", resolved=True, search_terms=["drugx", "syn1", "syn2", "syn3"])
    b = DrugRef(input_name="y", normalized_name="drugy", resolved=True, search_terms=["drugy", "syn1", "syn2"])

    queries = generate_pair_queries(a, b)
    assert len(queries) < 20  # bounded despite many synonyms available


# ---------------------------------------------------------------------------
# Tests: synonym handling
# ---------------------------------------------------------------------------

def test_resolve_drug_populates_search_terms_from_synonyms(session, collection, embedding_model):
    _add_drug(session, rxcui="11289", name="warfarin", synonyms=[("Coumadin", "brand"), ("Jantoven", "brand")])
    engine = PairRetrievalEngine(session, embedding_model, collection)

    drug = engine.resolve_drug("warfarin")

    assert drug.resolved is True
    assert "warfarin" in drug.search_terms
    assert "Coumadin" in drug.search_terms
    assert "Jantoven" in drug.search_terms


def test_classify_chunk_matches_via_brand_synonym():
    from app.schemas.evidence_assessment import DrugRef
    from app.schemas.retrieval import RetrievedChunk

    drug_a = DrugRef(input_name="warfarin", rxcui="11289", normalized_name="warfarin", resolved=True, search_terms=["warfarin", "Coumadin"])
    drug_b = DrugRef(input_name="ibuprofen", rxcui="5640", normalized_name="ibuprofen", resolved=True, search_terms=["ibuprofen"])

    chunk = RetrievedChunk(
        chunk_id="c1", text="Coumadin should be used cautiously with ibuprofen.", distance=0.1, rxcui="5640",
    )
    classification, mentions = classify_chunk(chunk, drug_a, drug_b)

    assert classification == EvidenceClassification.PAIR_SPECIFIC
    assert "Coumadin" in mentions.matched_terms_a
    assert "ibuprofen" in mentions.matched_terms_b


# ---------------------------------------------------------------------------
# Tests: exact pair-specific retrieval (end-to-end)
# ---------------------------------------------------------------------------

def test_retrieve_pair_finds_pair_specific_evidence(session, collection, embedding_model):
    _seed_warfarin_ibuprofen(session, collection, embedding_model)
    engine = PairRetrievalEngine(session, embedding_model, collection)

    result = engine.retrieve_pair("warfarin", "ibuprofen")

    assert result.evidence_status == "pair_specific_evidence_found"
    assert len(result.pair_evidence) >= 1
    assert "ibuprofen" in result.pair_evidence[0].text.lower()
    assert result.pair_evidence[0].classification == EvidenceClassification.PAIR_SPECIFIC


def test_retrieve_pair_rejects_same_rxcui_for_both_drugs(session, collection, embedding_model):
    """Aliases resolving to the same RxCUI must not be treated as a drug pair."""
    _add_drug(
        session,
        rxcui="11289",
        name="warfarin",
        synonyms=[("Coumadin", "brand")],
    )

    engine = PairRetrievalEngine(session, embedding_model, collection)

    result = engine.retrieve_pair("warfarin", "Coumadin")

    assert result.drug_a.resolved is True
    assert result.drug_b.resolved is True
    assert result.drug_a.rxcui == "11289"
    assert result.drug_b.rxcui == "11289"

    assert result.evidence_status == "invalid_input"
    assert result.pair_evidence == []
    assert result.supporting_evidence == []
    assert result.queries_used == []
    assert any("same RxCUI" in warning for warning in result.warnings)


def test_retrieve_pair_does_not_conclude_interaction_from_two_unrelated_labels(session, collection, embedding_model):
    """CORE SAFETY TEST: two drugs that each have evidence, but whose
    labels never mention each other, must NOT be reported as
    pair_specific_evidence_found."""
    warfarin = _add_drug(session, rxcui="11289", name="warfarin")
    metformin = _add_drug(session, rxcui="6809", name="metformin")

    _add_label_with_sections(session, warfarin, setid="w-setid", sections={
        "drug_interactions": ("34073-7", "Drug Interactions", "Warfarin interacts with vitamin K antagonists and alcohol."),
    })
    _add_label_with_sections(session, metformin, setid="m-setid", sections={
        "drug_interactions": ("34073-7", "Drug Interactions", "Metformin should be used cautiously with iodinated contrast media."),
    })
    ingest_rag_index(session, embedding_model, collection)

    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("warfarin", "metformin")

    assert result.evidence_status != "pair_specific_evidence_found"
    assert result.pair_evidence == []
    # Both drugs DO have their own evidence -- that must land in supporting_evidence, not be mistaken for a pair match.
    assert result.evidence_status == "supporting_evidence_found"
    assert len(result.supporting_evidence) >= 1


# ---------------------------------------------------------------------------
# Tests: drug-specific retrieval / supporting evidence
# ---------------------------------------------------------------------------

def test_retrieve_pair_classifies_unrelated_evidence_as_drug_specific(session, collection, embedding_model):
    warfarin = _add_drug(session, rxcui="11289", name="warfarin")
    metformin = _add_drug(session, rxcui="6809", name="metformin")
    _add_label_with_sections(session, warfarin, setid="w-setid", sections={
        "drug_interactions": ("34073-7", "Drug Interactions", "Warfarin interacts with alcohol and vitamin K."),
    })
    _add_label_with_sections(session, metformin, setid="m-setid", sections={
        "drug_interactions": ("34073-7", "Drug Interactions", "Metformin should be used cautiously with contrast media."),
    })
    ingest_rag_index(session, embedding_model, collection)

    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("warfarin", "metformin")

    classifications = {item.classification for item in result.supporting_evidence}
    assert EvidenceClassification.DRUG_SPECIFIC in classifications


# ---------------------------------------------------------------------------
# Tests: class-level evidence
# ---------------------------------------------------------------------------

def test_retrieve_pair_detects_class_level_evidence(session, collection, embedding_model):
    """Warfarin's label mentions 'NSAIDs' (a class) without naming ibuprofen specifically."""
    warfarin = _add_drug(session, rxcui="11289", name="warfarin")
    ibuprofen = _add_drug(session, rxcui="5640", name="ibuprofen")

    _add_label_with_sections(session, warfarin, setid="w-setid", sections={
        "warnings_and_precautions": (
            "43685-7", "Warnings and Precautions",
            "Increased bleeding risk when combined with NSAIDs.",
        ),
    })
    _add_label_with_sections(session, ibuprofen, setid="i-setid", sections={
        "adverse_reactions": ("34084-4", "Adverse Reactions", "Gastrointestinal upset."),
    })
    ingest_rag_index(session, embedding_model, collection)

    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("warfarin", "ibuprofen")

    all_items = result.pair_evidence + result.supporting_evidence
    classifications = {item.classification for item in all_items}
    assert EvidenceClassification.CLASS_LEVEL in classifications

    class_item = next(i for i in all_items if i.classification == EvidenceClassification.CLASS_LEVEL)
    assert class_item.drug_mentions.class_match_b == "nsaids" or class_item.drug_mentions.class_match_a == "nsaids"


# ---------------------------------------------------------------------------
# Tests: duplicate evidence removal
# ---------------------------------------------------------------------------

def test_retrieve_pair_deduplicates_chunk_retrieved_by_multiple_queries(session, collection, embedding_model):
    _seed_warfarin_ibuprofen(session, collection, embedding_model)
    engine = PairRetrievalEngine(session, embedding_model, collection, top_k_per_query=10)

    result = engine.retrieve_pair("warfarin", "ibuprofen")

    all_ids = [item.chunk_id for item in result.pair_evidence + result.supporting_evidence]
    assert len(all_ids) == len(set(all_ids))  # no chunk appears twice

    if result.pair_evidence:
        assert len(result.pair_evidence[0].retrieval_queries) >= 1


# ---------------------------------------------------------------------------
# Tests: ranking
# ---------------------------------------------------------------------------

def test_pair_evidence_ranked_by_distance_ascending(session, collection, embedding_model):
    _seed_warfarin_ibuprofen(session, collection, embedding_model)
    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("warfarin", "ibuprofen")

    distances = [item.distance for item in result.pair_evidence]
    assert distances == sorted(distances)


_SUPPORT_RANK_FOR_TEST = {
    EvidenceClassification.DRUG_SPECIFIC: 0,
    EvidenceClassification.CLASS_LEVEL: 1,
    EvidenceClassification.GENERAL_LABEL: 2,
}


def test_supporting_evidence_ranked_by_classification_strength_then_distance(session, collection, embedding_model):
    warfarin = _add_drug(session, rxcui="11289", name="warfarin")
    metformin = _add_drug(session, rxcui="6809", name="metformin")
    _add_label_with_sections(session, warfarin, setid="w-setid", sections={
        "drug_interactions": ("34073-7", "Drug Interactions", "Warfarin interacts with alcohol."),
        "warnings_and_precautions": ("43685-7", "Warnings and Precautions", "Caution with NSAIDs."),
    })
    _add_label_with_sections(session, metformin, setid="m-setid", sections={
        "adverse_reactions": ("34084-4", "Adverse Reactions", "Nausea and diarrhea are common."),
    })
    ingest_rag_index(session, embedding_model, collection)

    engine = PairRetrievalEngine(session, embedding_model, collection, top_k_per_query=10)
    result = engine.retrieve_pair("warfarin", "metformin")

    ranks = [_SUPPORT_RANK_FOR_TEST[item.classification] for item in result.supporting_evidence]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# Tests: provenance preservation
# ---------------------------------------------------------------------------

def test_pair_evidence_preserves_full_provenance(session, collection, embedding_model):
    _seed_warfarin_ibuprofen(session, collection, embedding_model)
    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("warfarin", "ibuprofen")

    item = result.pair_evidence[0]
    assert item.rxcui == "11289"
    assert item.label_setid == "warfarin-setid"
    assert item.spl_version == "1"
    assert item.section_code == "34073-7"
    assert item.section_name == "Drug Interactions"
    assert item.db_record_id is not None
    assert item.chunk_id.startswith("link-")
    assert len(item.retrieval_queries) >= 1
    assert item.evidence_status_of_label == "interaction_evidence_found"


# ---------------------------------------------------------------------------
# Tests: unknown drugs
# ---------------------------------------------------------------------------

def test_retrieve_pair_unknown_drug_a(session, collection, embedding_model):
    _add_drug(session, rxcui="5640", name="ibuprofen")
    engine = PairRetrievalEngine(session, embedding_model, collection)

    result = engine.retrieve_pair("notarealdrugxyz", "ibuprofen")

    assert result.evidence_status == "drug_not_found"
    assert result.drug_a.resolved is False
    assert result.drug_b.resolved is True


def test_retrieve_pair_both_drugs_unknown(session, collection, embedding_model):
    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("notarealdrug1", "notarealdrug2")

    assert result.evidence_status == "drug_not_found"
    assert result.drug_a.resolved is False
    assert result.drug_b.resolved is False


# ---------------------------------------------------------------------------
# Tests: empty retrieval (drugs known, but no evidence indexed)
# ---------------------------------------------------------------------------

def test_retrieve_pair_insufficient_evidence_when_nothing_indexed(session, collection, embedding_model):
    _add_drug(session, rxcui="11289", name="warfarin")
    _add_drug(session, rxcui="5640", name="ibuprofen")
    # Note: no labels/sections ingested into ChromaDB at all.

    engine = PairRetrievalEngine(session, embedding_model, collection)
    result = engine.retrieve_pair("warfarin", "ibuprofen")

    assert result.evidence_status == "insufficient_evidence"
    assert result.pair_evidence == []
    assert result.supporting_evidence == []


def test_evidence_status_never_uses_no_interaction_as_a_value():
    """Static assertion on the schema itself: 'no interaction' must never be a possible evidence_status."""
    import typing

    from app.schemas.evidence_assessment import PairEvidenceStatus

    allowed_values = typing.get_args(PairEvidenceStatus)
    assert "no_interaction" not in allowed_values
    assert all("no_interaction" not in v for v in allowed_values)


# ---------------------------------------------------------------------------
# Tests: retrieval failure
# ---------------------------------------------------------------------------

def test_retrieve_pair_handles_retrieval_exception_gracefully(session, collection, embedding_model, monkeypatch):
    _seed_warfarin_ibuprofen(session, collection, embedding_model)
    engine = PairRetrievalEngine(session, embedding_model, collection)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated ChromaDB failure")

    monkeypatch.setattr("app.services.pair_retrieval_service.retrieve", _boom)

    result = engine.retrieve_pair("warfarin", "ibuprofen")

    assert result.evidence_status == "retrieval_error"
    assert result.error is not None
    assert "simulated ChromaDB failure" in result.error


# ---------------------------------------------------------------------------
# Tests: multi-drug pair generation
# ---------------------------------------------------------------------------

def test_generate_unique_pairs_count_matches_combinations_formula():
    pairs = generate_unique_pairs(["A", "B", "C", "D"])
    assert len(pairs) == 6  # C(4,2)


def test_generate_unique_pairs_no_reversed_duplicates():
    pairs = generate_unique_pairs(["Warfarin", "Ibuprofen", "Aspirin"])
    pair_set = {frozenset(p) for p in pairs}
    assert len(pair_set) == len(pairs)  # each unordered pair appears once


def test_generate_unique_pairs_deduplicates_repeated_names_case_insensitive():
    pairs = generate_unique_pairs(["Warfarin", "warfarin", "Ibuprofen"])
    assert len(pairs) == 1
    assert pairs[0] == ("Warfarin", "Ibuprofen")


def test_generate_unique_pairs_single_drug_produces_no_pairs():
    assert generate_unique_pairs(["Warfarin"]) == []


def test_generate_unique_pairs_empty_list():
    assert generate_unique_pairs([]) == []


def test_retrieve_multi_generates_correct_number_of_results(session, collection, embedding_model):
    _add_drug(session, rxcui="11289", name="warfarin")
    _add_drug(session, rxcui="5640", name="ibuprofen")
    _add_drug(session, rxcui="1191", name="aspirin")
    engine = PairRetrievalEngine(session, embedding_model, collection)

    results = engine.retrieve_multi(["warfarin", "ibuprofen", "aspirin"])

    assert len(results) == 3  # C(3,2)
    assert ("warfarin", "ibuprofen") in results
    assert ("warfarin", "aspirin") in results
    assert ("ibuprofen", "aspirin") in results


def test_retrieve_multi_reuses_cached_drug_specific_queries(session, collection, embedding_model):
    """EFFICIENCY TEST: drug-specific queries for a drug shared across
    multiple pairs must be served from cache, not re-queried."""
    warfarin = _add_drug(session, rxcui="11289", name="warfarin")
    ibuprofen = _add_drug(session, rxcui="5640", name="ibuprofen")
    aspirin = _add_drug(session, rxcui="1191", name="aspirin")

    for drug, setid in ((warfarin, "w"), (ibuprofen, "i"), (aspirin, "a")):
        _add_label_with_sections(session, drug, setid=f"{setid}-setid", sections={
            "drug_interactions": ("34073-7", "Drug Interactions", f"{drug.normalized_name} general interaction text."),
        })
    ingest_rag_index(session, embedding_model, collection)

    engine = PairRetrievalEngine(session, embedding_model, collection)
    engine.retrieve_multi(["warfarin", "ibuprofen", "aspirin"])

    # Warfarin appears in 2 of the 3 pairs (warfarin-ibuprofen, warfarin-aspirin) —
    # its drug-specific queries should be cached and reused on the second occurrence.
    assert engine.cache_hits > 0
