"""
tests/test_health_concurrency.py
==================================
Regression tests for GET /api/health non-blocking execution,
ChromaDB liveness checks via get(limit=1), absence of collection.count(),
and concurrency during retrieval.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from app.api.deps import get_chroma_collection
from app.config import settings
from app.main import app


def test_health_reports_ok_for_functioning_collection(api_client, monkeypatch):
    """Verifies that a normal functioning collection reports healthy status."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    r = api_client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["checks"]["chromadb"]["status"] == "ok"
    assert data["checks"]["sqlite"]["status"] == "ok"


def test_health_reports_ok_for_empty_functioning_collection(api_client, monkeypatch):
    """
    Verifies that an empty collection (where get(limit=1) returns empty lists)
    is correctly reported as healthy, not as an error.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    class EmptyFunctioningCollection:
        def get(self, limit: int = 1):
            return {
                "ids": [],
                "embeddings": None,
                "documents": [],
                "metadatas": [],
            }

    app.dependency_overrides[get_chroma_collection] = lambda: EmptyFunctioningCollection()
    try:
        r = api_client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["checks"]["chromadb"]["status"] == "ok"
    finally:
        app.dependency_overrides.pop(get_chroma_collection, None)


def test_health_reports_chromadb_error_when_get_raises(api_client, monkeypatch):
    """
    Verifies that if ChromaDB collection storage fails on get(limit=1),
    the endpoint catches the error and reports degraded/error status safely.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    class StorageFailureCollection:
        def get(self, limit: int = 1):
            raise RuntimeError("Database storage connection lost")

    app.dependency_overrides[get_chroma_collection] = lambda: StorageFailureCollection()
    try:
        r = api_client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["checks"]["chromadb"]["status"] == "error"
        assert "Database storage connection lost" in data["checks"]["chromadb"]["detail"]
        assert data["status"] == "degraded"
    finally:
        app.dependency_overrides.pop(get_chroma_collection, None)


def test_health_does_not_call_collection_count(api_client, chroma_collection, monkeypatch):
    """
    Verifies that the health endpoint does NOT call collection.count()
    (which would cause expensive table scans and SQLite locks under load).
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    count_called = False

    class SpyCollection:
        def get(self, limit: int = 1):
            return chroma_collection.get(limit=limit)

        def count(self):
            nonlocal count_called
            count_called = True
            raise AssertionError("collection.count() must never be called by the health check")

    app.dependency_overrides[get_chroma_collection] = lambda: SpyCollection()
    try:
        r = api_client.get("/api/health")
        assert r.status_code == 200
        assert not count_called, "collection.count() was invoked by /api/health"
        assert r.json()["checks"]["chromadb"]["status"] == "ok"
    finally:
        app.dependency_overrides.pop(get_chroma_collection, None)


def test_health_check_non_blocking_during_retrieval(api_client, monkeypatch):
    """
    Verifies that GET /api/health returns HTTP 200 and low latency even
    when the ChromaDB collection or embedding model is being queried.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    start = time.monotonic()
    r = api_client.get("/api/health")
    duration = time.monotonic() - start

    assert r.status_code == 200
    assert duration < 1.0  # Must be fast (< 1s), well under Render's 5s limit
    data = r.json()
    assert data["status"] == "ok"
    assert data["checks"]["chromadb"]["status"] == "ok"


def test_concurrent_health_requests_during_retrievals(
    api_client, db_session, fake_embedding_model, chroma_collection, monkeypatch
):
    """
    Simulates concurrent client traffic hitting GET /api/health and POST /api/interaction/check.
    Health checks should not fail or time out.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")
    from tests.conftest import seed_drug_pair

    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)

    def send_health():
        return api_client.get("/api/health")

    def send_check():
        return api_client.post(
            "/api/interaction/check",
            json={"drug_a": "warfarin", "drug_b": "ibuprofen"},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        f_check = executor.submit(send_check)
        f_health1 = executor.submit(send_health)
        f_health2 = executor.submit(send_health)

        r_check = f_check.result(timeout=10)
        r_h1 = f_health1.result(timeout=5)
        r_h2 = f_health2.result(timeout=5)

    assert r_check.status_code == 200
    assert r_h1.status_code == 200
    assert r_h2.status_code == 200
    assert r_h1.json()["status"] == "ok"
    assert r_h2.json()["status"] == "ok"


def test_batch_embedding_in_pair_retrieval(db_session, fake_embedding_model, chroma_collection):
    """
    Verifies that PairRetrievalEngine correctly precomputes embeddings in batch
    and successfully classifies and returns evidence chunks.
    """
    from app.services.pair_retrieval_service import PairRetrievalEngine
    from tests.conftest import seed_drug_pair

    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)

    engine = PairRetrievalEngine(db_session, fake_embedding_model, chroma_collection)
    result = engine.retrieve_pair("warfarin", "ibuprofen")

    assert result.evidence_status == "pair_specific_evidence_found"
    assert len(result.pair_evidence) > 0
    assert result.pair_evidence[0].classification.value == "pair_specific_evidence"


def test_batch_embedding_order_and_caching(db_session, chroma_collection):
    """
    Verifies that batch embeddings preserve query-to-vector ordering
    and that repeat queries hit the cache without redundant embeddings.
    """
    from app.rag.embeddings import DeterministicFakeEmbeddingModel
    from app.services.pair_retrieval_service import PairRetrievalEngine
    from tests.conftest import seed_drug_pair

    class TrackingModel(DeterministicFakeEmbeddingModel):
        def __init__(self):
            super().__init__(dimension=16)
            self.embed_calls: list[list[str]] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.embed_calls.append(list(texts))
            return super().embed(texts)

    tracker = TrackingModel()
    seed_drug_pair(db_session, tracker, chroma_collection)
    tracker.embed_calls.clear()

    engine = PairRetrievalEngine(db_session, tracker, chroma_collection)
    res1 = engine.retrieve_pair("warfarin", "ibuprofen")
    assert res1.evidence_status == "pair_specific_evidence_found"
    assert len(tracker.embed_calls) == 1  # Exactly 1 batched forward pass!
    first_call_texts = tracker.embed_calls[0]
    assert len(first_call_texts) > 1

    # Second call for the same pair should hit cache for all queries
    tracker.embed_calls.clear()
    res2 = engine.retrieve_pair("warfarin", "ibuprofen")
    assert res2.evidence_status == "pair_specific_evidence_found"
    assert len(tracker.embed_calls) == 0  # 0 calls, 100% cache hit!
    assert engine.cache_hits > 0


def test_health_cache_immutability_and_expiration(monkeypatch):
    """
    Verifies that:
    1. Health results are cached when no dependency overrides are present.
    2. Returned response objects cannot mutate the internal cache.
    3. The cache expires after _HEALTH_CACHE_TTL_SECONDS.
    4. Dependency overrides bypass the cache.
    """
    from unittest.mock import MagicMock
    import app.api.health as health_module
    from app.api.health import health as health_endpoint, _HEALTH_CACHE_TTL_SECONDS

    # Reset cache before test
    health_module._cached_health = None
    monkeypatch.setattr(settings, "LLM_PROVIDER", "none")

    mock_app = MagicMock()
    mock_app.dependency_overrides = {}
    mock_request = MagicMock()
    mock_request.app = mock_app
    mock_request.state.request_id = "test-req-123"
    mock_request.headers = {}

    mock_session = MagicMock()
    mock_collection = MagicMock()

    try:
        # Call 1: live check executes and populates cache
        resp1 = health_endpoint(request=mock_request, session=mock_session, collection=mock_collection)
        assert resp1.status == "ok"
        assert mock_session.execute.call_count == 1
        assert mock_collection.get.call_count == 1

        # Caller mutates returned response checks
        resp1.checks["sqlite"].status = "corrupted_by_caller"
        resp1.checks["sqlite"].detail = "malicious mutation"

        # Call 2: served from cache within TTL, unaffected by mutation
        resp2 = health_endpoint(request=mock_request, session=mock_session, collection=mock_collection)
        assert resp2.status == "ok"
        assert resp2.checks["sqlite"].status == "ok"  # Immune to caller mutation!
        assert resp2.checks["sqlite"].detail is None
        assert mock_session.execute.call_count == 1  # No extra db calls
        assert mock_collection.get.call_count == 1

        # Call 3: after TTL expires, live check executes
        current_time = time.monotonic()
        monkeypatch.setattr(time, "monotonic", lambda: current_time + _HEALTH_CACHE_TTL_SECONDS + 1.0)
        resp3 = health_endpoint(request=mock_request, session=mock_session, collection=mock_collection)
        assert resp3.status == "ok"
        assert mock_session.execute.call_count == 2
        assert mock_collection.get.call_count == 2

        # Call 4: with dependency_overrides active, cache is bypassed
        mock_app.dependency_overrides = {"some_dep": "override"}
        resp4 = health_endpoint(request=mock_request, session=mock_session, collection=mock_collection)
        assert mock_session.execute.call_count == 3
    finally:
        health_module._cached_health = None
