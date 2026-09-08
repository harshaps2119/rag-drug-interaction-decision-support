"""
tests/test_interaction_router.py
===================================
End-to-end tests for POST /api/interaction/check and
POST /api/interaction/check-multiple through the real FastAPI app, with
all external dependencies faked (see tests/conftest.py). No live
RxNorm/DailyMed/Gemini calls anywhere.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_interaction_router.py -v
"""

from app.api.deps import get_rate_limiter
from app.core.rate_limiter import InMemoryRateLimiter
from app.main import app
from app.models.db_models import DrugSynonym
from tests.conftest import seed_drug_pair, seed_single_drug


# ---------------------------------------------------------------------------
# Two-drug endpoint
# ---------------------------------------------------------------------------

def test_check_interaction_known_pair_returns_llm_grounded(api_client, db_session, fake_embedding_model, chroma_collection):
    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)

    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin", "drug_b": "ibuprofen"})

    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "llm_grounded"
    assert body["interaction_assessment"] == "pair_specific_evidence_found"
    assert body["drug_a"]["rxcui"] == "11289"
    assert body["drug_b"]["rxcui"] == "5640"
    assert len(body["cited_evidence"]) >= 1


def test_check_interaction_response_includes_request_id_header(api_client, db_session, fake_embedding_model, chroma_collection):
    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)
    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin", "drug_b": "ibuprofen"})
    assert "X-Request-ID" in r.headers


def test_check_interaction_same_rxcui_returns_invalid_input_without_llm(
    api_client, db_session, fake_embedding_model, chroma_collection, fake_llm_client
):
    warfarin = seed_single_drug(
        db_session,
        fake_embedding_model,
        chroma_collection,
        "warfarin",
        "11289",
        "Warfarin label evidence.",
    )
    db_session.add(DrugSynonym(drug_id=warfarin.id, synonym="Coumadin", synonym_type="brand"))
    db_session.commit()

    r = api_client.post("/api/interaction/check", json={"drug_a": "Warfarin", "drug_b": "Coumadin"})

    assert r.status_code == 200
    body = r.json()
    assert body["interaction_assessment"] == "invalid_input"
    assert body["mode"] == "evidence_only"
    assert body["fallback_reason"] == "Cannot generate an explanation: retrieval result was 'invalid_input'."
    assert fake_llm_client.call_count == 0


def test_check_interaction_unknown_drug_returns_drug_not_found(api_client, monkeypatch):
    from app.services import interaction_service as svc

    monkeypatch.setattr(
        svc, "ingest_drug",
        lambda name, session: __import__("app.schemas.ingestion", fromlist=["IngestionResult"]).IngestionResult(
            drug_name=name, rxcui=None, drug_status="failed"
        ),
    )

    r = api_client.post("/api/interaction/check", json={"drug_a": "notarealdrug1", "drug_b": "notarealdrug2"})

    assert r.status_code == 200  # a valid, structured result -- not an HTTP error
    body = r.json()
    assert body["interaction_assessment"] == "drug_not_found"
    assert body["mode"] == "evidence_only"
    # Explicit safety rule: NEVER "no interaction exists" anywhere in the response.
    assert "no interaction exists" not in str(body).lower()


# ---------------------------------------------------------------------------
# Gemini unavailable / evidence-only fallback through the real endpoint
# ---------------------------------------------------------------------------

def test_check_interaction_gemini_unavailable_falls_back_to_evidence_only(
    api_client, db_session, fake_embedding_model, chroma_collection,
):
    from app.exceptions import LLMServiceUnavailableError
    from app.api.deps import get_llm_client

    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)

    class BrokenClient:
        def generate(self, prompt):
            raise LLMServiceUnavailableError("simulated outage")

    app.dependency_overrides[get_llm_client] = lambda: BrokenClient()

    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin", "drug_b": "ibuprofen"})

    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "evidence_only"
    assert body["fallback_reason"] is not None
    assert len(body["cited_evidence"]) >= 1  # evidence still shown despite LLM outage


def test_check_interaction_retrieval_failure_handled_gracefully(api_client, db_session, fake_embedding_model, chroma_collection, monkeypatch):
    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)

    from app.services import pair_retrieval_service

    def _boom(*a, **k):
        raise RuntimeError("simulated ChromaDB outage")

    monkeypatch.setattr(pair_retrieval_service, "retrieve", _boom)

    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin", "drug_b": "ibuprofen"})

    assert r.status_code == 200
    body = r.json()
    assert body["interaction_assessment"] == "retrieval_error"
    assert "no interaction exists" not in str(body).lower()


# ---------------------------------------------------------------------------
# Multi-drug endpoint
# ---------------------------------------------------------------------------

def test_check_multiple_returns_all_unique_pairs(api_client, db_session, fake_embedding_model, chroma_collection):
    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)
    seed_single_drug(db_session, fake_embedding_model, chroma_collection, "aspirin", "1191", "Aspirin has interactions.")

    r = api_client.post("/api/interaction/check-multiple", json={"drugs": ["warfarin", "ibuprofen", "aspirin"]})

    assert r.status_code == 200
    body = r.json()
    assert body["total_pairs"] == 3
    pair_names = {frozenset((p["drug_a"], p["drug_b"])) for p in body["pairs"]}
    assert frozenset(("warfarin", "ibuprofen")) in pair_names
    assert frozenset(("warfarin", "aspirin")) in pair_names
    assert frozenset(("ibuprofen", "aspirin")) in pair_names


def test_check_multiple_rejects_too_few_drugs(api_client):
    r = api_client.post("/api/interaction/check-multiple", json={"drugs": ["warfarin"]})
    assert r.status_code == 422


def test_check_multiple_rejects_duplicates(api_client):
    r = api_client.post("/api/interaction/check-multiple", json={"drugs": ["warfarin", "Warfarin", "aspirin"]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Clinical safety rule: never "no interaction exists", across every failure mode
# ---------------------------------------------------------------------------

def test_no_scenario_ever_returns_no_interaction_exists_phrase(
    api_client, db_session, fake_embedding_model, chroma_collection,
):
    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)

    scenarios = [
        {"drug_a": "warfarin", "drug_b": "ibuprofen"},
        {"drug_a": "unknowndrug1", "drug_b": "unknowndrug2"},
    ]
    for payload in scenarios:
        r = api_client.post("/api/interaction/check", json=payload)
        assert "no interaction exists" not in r.text.lower()
        assert '"interaction_assessment":"no_interaction"' not in r.text.replace(" ", "").lower()


# ---------------------------------------------------------------------------
# Rate limiting, through the real endpoint
# ---------------------------------------------------------------------------

def test_rate_limit_blocks_after_threshold(api_client, db_session, fake_embedding_model, chroma_collection):
    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)
    # IMPORTANT: override with a lambda returning the SAME instance every
    # call, not a fresh one -- FastAPI calls the override callable on
    # every request, so `lambda: InMemoryRateLimiter(...)` would silently
    # reset state to zero hits each time and the limit would never
    # trigger. Discovered by actually running this test.
    shared_limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)
    app.dependency_overrides[get_rate_limiter] = lambda: shared_limiter

    payload = {"drug_a": "warfarin", "drug_b": "ibuprofen"}
    r1 = api_client.post("/api/interaction/check", json=payload)
    r2 = api_client.post("/api/interaction/check", json=payload)
    r3 = api_client.post("/api/interaction/check", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert r3.json()["error"]["code"] == "rate_limit_exceeded"


def test_rate_limit_disabled_via_config(api_client, db_session, fake_embedding_model, chroma_collection, monkeypatch):
    from app.config import settings

    seed_drug_pair(db_session, fake_embedding_model, chroma_collection)
    app.dependency_overrides[get_rate_limiter] = lambda: InMemoryRateLimiter(max_requests=1, window_seconds=60)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)

    payload = {"drug_a": "warfarin", "drug_b": "ibuprofen"}
    r1 = api_client.post("/api/interaction/check", json=payload)
    r2 = api_client.post("/api/interaction/check", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200  # not rate limited -- disabled entirely


def test_health_endpoint_is_not_rate_limited(api_client):
    """Health checks need to be pollable frequently -- only the expensive
    interaction endpoints carry the rate_limit_dependency."""
    app.dependency_overrides[get_rate_limiter] = lambda: InMemoryRateLimiter(max_requests=1, window_seconds=60)
    for _ in range(5):
        r = api_client.get("/api/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

def test_cors_headers_present_for_configured_origin(api_client):
    from app.config import settings

    r = api_client.options(
        "/api/health",
        headers={
            "Origin": settings.FRONTEND_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") == settings.FRONTEND_ORIGIN


def test_cors_rejects_unconfigured_origin(api_client):
    r = api_client.options(
        "/api/health",
        headers={
            "Origin": "https://evil-unrelated-site.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") != "https://evil-unrelated-site.example"
