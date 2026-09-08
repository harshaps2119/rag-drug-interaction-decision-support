"""
tests/conftest.py
====================
Shared fixtures for Phase 8 API tests. Wires up the real FastAPI app
with overridden dependencies -- in-memory SQLite (shared across sessions
via the StaticPool fix in app/db.py), a tmp_path ChromaDB collection,
DeterministicFakeEmbeddingModel, and a FakeLLMClient. No live
RxNorm/DailyMed/Gemini calls anywhere in the test suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from app.api.deps import (
    get_chroma_collection,
    get_db_session,
    get_embedding_model,
    get_llm_client,
    get_rate_limiter,
)
from app.core.rate_limiter import InMemoryRateLimiter
from app.db import init_db, make_engine
from app.main import app
from app.rag.embeddings import DeterministicFakeEmbeddingModel
from app.rag.vector_store import get_client, get_collection
from tests._fake_llm import FakeLLMClient, make_llm_json


@pytest.fixture
def db_session_factory():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def db_session(db_session_factory):
    s = db_session_factory()
    yield s
    s.close()


@pytest.fixture
def chroma_collection(tmp_path):
    client = get_client(persist_dir=str(tmp_path / "chroma"))
    return get_collection(client=client)


@pytest.fixture
def fake_embedding_model():
    return DeterministicFakeEmbeddingModel(dimension=16)


@pytest.fixture
def fake_llm_client():
    return FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=["EVIDENCE-001"]))


@pytest.fixture
def api_client(db_session_factory, chroma_collection, fake_embedding_model, fake_llm_client):
    """
    A TestClient wired to the real app, with all external/heavy
    dependencies overridden by fakes. Rate limiting is set generous by
    default (1000 req/min) so unrelated tests aren't accidentally
    throttled -- tests specifically about rate limiting override
    get_rate_limiter again themselves with a tight limit.
    """

    def override_session():
        s = db_session_factory()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db_session] = override_session
    app.dependency_overrides[get_embedding_model] = lambda: fake_embedding_model
    app.dependency_overrides[get_chroma_collection] = lambda: chroma_collection
    app.dependency_overrides[get_llm_client] = lambda: fake_llm_client
    app.dependency_overrides[get_rate_limiter] = lambda: InMemoryRateLimiter(max_requests=1000, window_seconds=60)

    # raise_server_exceptions=False: we want to observe the actual HTTP
    # response our exception handlers produce (a safe 500 envelope), not
    # have TestClient re-raise the raw exception for interactive debugging
    # -- discovered by testing test_error_handling.py, where the default
    # True caused a genuine server-side 500 to appear as a raised
    # exception instead of the response the real deployed app would give.
    client = TestClient(app, raise_server_exceptions=False)
    yield client

    app.dependency_overrides.clear()


def seed_drug_pair(session, embedding_model, collection):
    """Seeds warfarin+ibuprofen with pair-specific evidence, indexed into ChromaDB. Returns (warfarin, ibuprofen)."""
    from app.models.db_models import Drug, DrugLabelLink, Label, LabelSectionRecord
    from app.rag.ingest import ingest_rag_index

    warfarin = session.query(Drug).filter_by(rxcui="11289").one_or_none()
    ibuprofen = session.query(Drug).filter_by(rxcui="5640").one_or_none()
    if warfarin is None:
        warfarin = Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact")
        session.add(warfarin)
    if ibuprofen is None:
        ibuprofen = Drug(rxcui="5640", normalized_name="ibuprofen", input_name="ibuprofen", match_type="exact")
        session.add(ibuprofen)
    session.commit()

    label = Label(setid="w-setid", spl_version="1", title="WARFARIN LABEL")
    session.add(label)
    session.commit()
    sec = LabelSectionRecord(
        label_id=label.id,
        section_key="drug_interactions",
        section_code="34073-7",
        section_name="Drug Interactions",
        text="Warfarin combined with ibuprofen may increase bleeding risk.",
        found=True,
    )
    session.add(sec)
    session.commit()
    link = DrugLabelLink(drug_id=warfarin.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()

    ingest_rag_index(session, embedding_model, collection)
    return warfarin, ibuprofen


def seed_single_drug(session, embedding_model, collection, name: str, rxcui: str, text: str):
    """Seeds one drug with drug-specific (non-pair) evidence."""
    from app.models.db_models import Drug, DrugLabelLink, Label, LabelSectionRecord
    from app.rag.ingest import ingest_rag_index

    drug = session.query(Drug).filter_by(rxcui=rxcui).one_or_none()
    if drug is None:
        drug = Drug(rxcui=rxcui, normalized_name=name, input_name=name, match_type="exact")
        session.add(drug)
        session.commit()

    label = Label(setid=f"{rxcui}-setid", spl_version="1", title=f"{name.upper()} LABEL")
    session.add(label)
    session.commit()
    sec = LabelSectionRecord(
        label_id=label.id, section_key="drug_interactions", section_code="34073-7",
        section_name="Drug Interactions", text=text, found=True,
    )
    session.add(sec)
    session.commit()
    link = DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()

    ingest_rag_index(session, embedding_model, collection)
    return drug
