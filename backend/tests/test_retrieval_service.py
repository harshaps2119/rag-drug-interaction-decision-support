"""
tests/test_retrieval_service.py
==================================
Tests for app/services/retrieval_service.py.

Uses DeterministicFakeEmbeddingModel and a real ChromaDB instance in a
pytest tmp_path — tests retrieval PLUMBING (filtering, top_k, empty
results, metadata pass-through) fully offline. Does NOT and cannot test
retrieval QUALITY/relevance — see rag/embeddings.py's docstring.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_retrieval_service.py -v
"""

import pytest

from app.rag.embeddings import DeterministicFakeEmbeddingModel
from app.rag.vector_store import get_client, get_collection, upsert_chunks
from app.services.retrieval_service import retrieve


@pytest.fixture
def embedding_model():
    return DeterministicFakeEmbeddingModel(dimension=16)


@pytest.fixture
def collection(tmp_path):
    client = get_client(persist_dir=str(tmp_path / "chroma"))
    return get_collection(client=client)


def _seed(collection, embedding_model):
    texts = [
        "Warfarin may increase bleeding risk when combined with NSAIDs.",
        "Ibuprofen can reduce the effectiveness of certain blood pressure medications.",
        "Amiodarone should not be combined with certain antiarrhythmic drugs.",
    ]
    metadatas = [
        {"drug_name": "warfarin", "rxcui": "11289", "section_key": "drug_interactions", "section_name": "Drug Interactions", "section_code": "34073-7", "label_setid": "s1", "spl_version": "1", "manufacturer": "", "source_url": "https://dailymed.example/s1", "api_url": "", "db_record_id": 1, "evidence_status": "interaction_evidence_found"},
        {"drug_name": "ibuprofen", "rxcui": "5640", "section_key": "drug_interactions", "section_name": "Drug Interactions", "section_code": "34073-7", "label_setid": "s2", "spl_version": "1", "manufacturer": "", "source_url": "https://dailymed.example/s2", "api_url": "", "db_record_id": 2, "evidence_status": "interaction_evidence_found"},
        {"drug_name": "amiodarone", "rxcui": "703", "section_key": "warnings_and_precautions", "section_name": "Warnings and Precautions", "section_code": "43685-7", "label_setid": "s3", "spl_version": "1", "manufacturer": "", "source_url": "https://dailymed.example/s3", "api_url": "", "db_record_id": 3, "evidence_status": "interaction_evidence_found"},
    ]
    ids = ["c1", "c2", "c3"]
    embeddings = embedding_model.embed(texts)
    upsert_chunks(collection, ids, texts, embeddings, metadatas)


def test_retrieve_returns_results_with_full_metadata(collection, embedding_model):
    _seed(collection, embedding_model)

    results = retrieve(embedding_model, collection, "warfarin bleeding risk", top_k=3)

    assert len(results) == 3
    top = results[0]
    assert top.text
    assert isinstance(top.distance, float)
    assert top.drug_name is not None
    assert top.rxcui is not None
    assert top.section_name is not None
    assert top.source_url is not None


def test_retrieve_respects_top_k(collection, embedding_model):
    _seed(collection, embedding_model)
    results = retrieve(embedding_model, collection, "warfarin", top_k=1)
    assert len(results) == 1

    results2 = retrieve(embedding_model, collection, "warfarin", top_k=2)
    assert len(results2) == 2


def test_retrieve_uses_configured_default_top_k_when_none_given(collection, embedding_model, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "RAG_DEFAULT_TOP_K", 2)
    _seed(collection, embedding_model)

    results = retrieve(embedding_model, collection, "warfarin")
    assert len(results) == 2


def test_retrieve_filters_by_rxcui(collection, embedding_model):
    _seed(collection, embedding_model)

    results = retrieve(embedding_model, collection, "drug interaction", top_k=5, rxcui="11289")

    assert len(results) == 1
    assert results[0].rxcui == "11289"


def test_retrieve_filters_by_section_key(collection, embedding_model):
    _seed(collection, embedding_model)

    results = retrieve(embedding_model, collection, "drug interaction", top_k=5, section_key="warnings_and_precautions")

    assert len(results) == 1
    assert results[0].section_key == "warnings_and_precautions"


def test_retrieve_combined_filters(collection, embedding_model):
    _seed(collection, embedding_model)

    results = retrieve(
        embedding_model, collection, "drug interaction", top_k=5,
        rxcui="5640", section_key="drug_interactions",
    )
    assert len(results) == 1
    assert results[0].rxcui == "5640"


def test_retrieve_empty_query_returns_empty_list(collection, embedding_model):
    _seed(collection, embedding_model)
    assert retrieve(embedding_model, collection, "") == []
    assert retrieve(embedding_model, collection, "   ") == []


def test_retrieve_against_empty_collection_returns_empty_list(collection, embedding_model):
    results = retrieve(embedding_model, collection, "any query at all")
    assert results == []


def test_retrieve_filter_matching_nothing_returns_empty_list(collection, embedding_model):
    _seed(collection, embedding_model)
    results = retrieve(embedding_model, collection, "warfarin", rxcui="00000000")
    assert results == []


def test_distance_field_is_present_but_carries_no_clinical_meaning(collection, embedding_model):
    """Regression test for the project's core safety requirement: the
    distance field must be a plain float retrieval signal — this test
    only asserts its TYPE and presence, never attaches any clinical
    interpretation to its value, which is the point."""
    _seed(collection, embedding_model)
    results = retrieve(embedding_model, collection, "warfarin", top_k=1)
    assert isinstance(results[0].distance, float)
