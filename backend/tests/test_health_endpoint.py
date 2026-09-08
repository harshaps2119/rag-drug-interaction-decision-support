"""
tests/test_health_endpoint.py
================================
Tests for GET /api/health.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_health_endpoint.py -v
"""


def test_health_returns_200(api_client):
    r = api_client.get("/api/health")
    assert r.status_code == 200


def test_health_reports_ok_checks_when_everything_available(api_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fake-key-for-test")
    r = api_client.get("/api/health")
    body = r.json()
    assert body["checks"]["sqlite"]["status"] == "ok"
    assert body["checks"]["chromadb"]["status"] == "ok"
    assert body["checks"]["gemini_configured"]["status"] == "ok"
    assert body["status"] == "ok"


def test_health_reports_degraded_when_gemini_key_missing(api_client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "GEMINI_API_KEY", "")
    r = api_client.get("/api/health")
    body = r.json()
    assert body["checks"]["gemini_configured"]["status"] == "error"
    assert body["status"] == "degraded"
    # Overall request still succeeds -- a missing LLM key is not a service outage.
    assert r.status_code == 200


def test_health_includes_request_id(api_client):
    r = api_client.get("/api/health")
    assert "request_id" in r.json()
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


def test_health_never_calls_external_medical_apis_or_gemini(api_client, monkeypatch):
    """
    Patches every external-network-capable function this project has to
    raise if called -- proves the health endpoint genuinely never touches
    RxNorm, DailyMed, or Gemini, rather than just trusting the code
    review.
    """

    def _boom(*a, **k):
        raise AssertionError("Health check must never call external APIs.")

    monkeypatch.setattr("app.services.rxnorm_service.normalize_drug_name", _boom)
    monkeypatch.setattr("app.services.dailymed_service.get_drug_label_info", _boom)
    monkeypatch.setattr("app.services.llm_service.generate_grounded_response", _boom)

    r = api_client.get("/api/health")
    assert r.status_code == 200  # no exception was raised -- none of the patched functions were called
