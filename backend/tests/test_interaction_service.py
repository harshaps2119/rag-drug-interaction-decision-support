"""
tests/test_interaction_service.py
====================================
Tests for app/services/interaction_service.py — the orchestration layer
tying Phases 2-7 together for the API. Mocks ingest_drug/ingest_rag_index
where a live network call would otherwise be needed; uses real in-memory
SQLite + real ChromaDB + fake embeddings/LLM for everything else.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_interaction_service.py -v
"""

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import init_db, make_engine
from app.rag.embeddings import DeterministicFakeEmbeddingModel
from app.rag.vector_store import get_client, get_collection
from app.schemas.ingestion import IngestionResult
from app.services import interaction_service as svc
from tests._fake_llm import FakeLLMClient, make_llm_json
from tests.conftest import seed_drug_pair


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


# ---------------------------------------------------------------------------
# ensure_drug_available: already-known drug is a no-op
# ---------------------------------------------------------------------------

def test_ensure_drug_available_noop_when_already_known(session, collection, embedding_model, monkeypatch):
    seed_drug_pair(session, embedding_model, collection)

    called = {"ingest": False}
    monkeypatch.setattr(svc, "ingest_drug", lambda *a, **k: called.update(ingest=True))

    svc.ensure_drug_available(session, embedding_model, collection, "warfarin")

    assert called["ingest"] is False  # never called -- drug was already known locally


# ---------------------------------------------------------------------------
# ensure_drug_available: unknown drug triggers on-demand ingestion
# ---------------------------------------------------------------------------

def test_ensure_drug_available_ingests_unknown_drug(session, collection, embedding_model, monkeypatch):
    def fake_ingest_drug(name, session):
        from app.models.db_models import Drug

        drug = Drug(rxcui="99999", normalized_name=name, input_name=name, match_type="exact")
        session.add(drug)
        session.commit()
        return IngestionResult(drug_name=name, rxcui="99999", drug_status="added", evidence_status="no_label_found")

    monkeypatch.setattr(svc, "ingest_drug", fake_ingest_drug)

    called = {"rag_ingest": False}
    monkeypatch.setattr(svc, "ingest_rag_index", lambda *a, **k: called.update(rag_ingest=True))

    svc.ensure_drug_available(session, embedding_model, collection, "newdrug")

    from app.services.knowledge_base_service import get_drug_by_rxcui

    assert get_drug_by_rxcui(session, "99999") is not None
    assert called["rag_ingest"] is True


def test_ensure_drug_available_handles_failed_normalization_gracefully(session, collection, embedding_model, monkeypatch):
    monkeypatch.setattr(
        svc, "ingest_drug",
        lambda name, session: IngestionResult(drug_name=name, rxcui=None, drug_status="failed"),
    )
    rag_called = {"called": False}
    monkeypatch.setattr(svc, "ingest_rag_index", lambda *a, **k: rag_called.update(called=True))

    svc.ensure_drug_available(session, embedding_model, collection, "notarealdrug")  # should not raise

    assert rag_called["called"] is False  # never indexed -- nothing to index


# ---------------------------------------------------------------------------
# check_pair: end-to-end with already-known drugs
# ---------------------------------------------------------------------------

def test_check_pair_with_known_drugs_returns_llm_grounded_response(session, collection, embedding_model):
    seed_drug_pair(session, embedding_model, collection)
    llm_client = FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=["EVIDENCE-001"]))

    response = svc.check_pair(session, embedding_model, collection, "warfarin", "ibuprofen", llm_client=llm_client)

    assert response.mode == "llm_grounded"
    assert response.drug_a.rxcui == "11289"
    assert response.drug_b.rxcui == "5640"


def test_check_pair_unknown_drug_reports_drug_not_found(session, collection, embedding_model, monkeypatch):
    monkeypatch.setattr(
        svc, "ingest_drug",
        lambda name, session: IngestionResult(drug_name=name, rxcui=None, drug_status="failed"),
    )
    llm_client = FakeLLMClient(response_text=make_llm_json())

    response = svc.check_pair(session, embedding_model, collection, "notarealdrug1", "notarealdrug2", llm_client=llm_client)

    assert response.interaction_assessment == "drug_not_found"
    assert response.mode == "evidence_only"
    assert llm_client.call_count == 0  # never reaches the LLM


# ---------------------------------------------------------------------------
# check_multiple: multi-drug pair generation + caching reuse
# ---------------------------------------------------------------------------

def test_check_multiple_returns_all_pairs(session, collection, embedding_model):
    seed_drug_pair(session, embedding_model, collection)
    from tests.conftest import seed_single_drug

    seed_single_drug(session, embedding_model, collection, "aspirin", "1191", "Aspirin has various interactions.")

    llm_client = FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=[]))
    results = svc.check_multiple(session, embedding_model, collection, ["warfarin", "ibuprofen", "aspirin"], llm_client=llm_client)

    assert len(results) == 3  # C(3,2)
    assert ("warfarin", "ibuprofen") in results
    assert ("warfarin", "aspirin") in results
    assert ("ibuprofen", "aspirin") in results
