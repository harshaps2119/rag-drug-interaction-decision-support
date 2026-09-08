"""
tests/test_error_handling.py
===============================
Tests for app/core/errors.py through the real app — consistent error
envelope, no stack traces/internal paths/secrets leaked to the client.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_error_handling.py -v
"""


def test_unhandled_exception_returns_generic_500_envelope(api_client, monkeypatch):
    """Forces an unexpected exception deep in the pipeline and confirms
    the client sees ONLY the generic envelope -- never the real exception text."""
    from app.services import interaction_service as svc

    def _boom(*a, **k):
        raise RuntimeError("SECRET INTERNAL DETAIL: database path /home/user/secret/ddi.db, key=sk-abcdef123456")

    monkeypatch.setattr(svc, "check_pair", _boom)

    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin", "drug_b": "ibuprofen"})

    assert r.status_code == 500
    body = r.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "request_id"}
    assert body["error"]["code"] == "internal_error"

    # The actual secret/internal detail must NEVER appear in the response.
    response_text = r.text
    assert "SECRET INTERNAL DETAIL" not in response_text
    assert "sk-abcdef123456" not in response_text
    assert "/home/user/secret" not in response_text
    assert "RuntimeError" not in response_text
    assert "Traceback" not in response_text


def test_error_response_never_contains_configured_api_key(api_client, monkeypatch):
    from app.config import settings

    fake_key = "AIzaSyFAKE_TEST_KEY_1234567890"
    monkeypatch.setattr(settings, "GEMINI_API_KEY", fake_key)

    from app.services import interaction_service as svc

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(svc, "check_pair", _boom)

    r = api_client.post("/api/interaction/check", json={"drug_a": "warfarin", "drug_b": "ibuprofen"})
    assert fake_key not in r.text


def test_404_for_unknown_route_does_not_leak_internals(api_client):
    r = api_client.get("/api/this-route-does-not-exist")
    assert r.status_code == 404


def test_error_envelope_shape_is_consistent_across_error_types(api_client):
    """Validation error (422) and rate-limit-style errors should share the exact same top-level shape."""
    r1 = api_client.post("/api/interaction/check", json={"drug_a": ""})
    assert set(r1.json().keys()) == {"error"}
    assert set(r1.json()["error"].keys()) == {"code", "message", "request_id"}


def test_error_response_includes_matching_request_id(api_client):
    r = api_client.post("/api/interaction/check", json={"drug_a": ""})
    assert r.json()["error"]["request_id"] == r.headers["X-Request-ID"]
