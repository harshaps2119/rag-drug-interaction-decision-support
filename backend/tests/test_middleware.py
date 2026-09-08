"""
tests/test_middleware.py
===========================
Tests for app/core/middleware.py and app/core/request_id.py, using
minimal standalone Starlette/FastAPI apps (not the full app) to isolate
middleware behavior and ordering.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_middleware.py -v
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.middleware import AuditMiddleware, BodySizeLimitMiddleware
from app.core.request_id import REQUEST_ID_HEADER, RequestIDMiddleware


def _build_app(max_body_bytes: int = 20_000) -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    @app.post("/echo")
    def echo(body: dict):
        return body

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=max_body_bytes)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestIDMiddleware)
    return app


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

def test_request_id_is_generated_when_not_supplied():
    client = TestClient(_build_app())
    r = client.get("/ping")
    assert REQUEST_ID_HEADER in r.headers
    assert len(r.headers[REQUEST_ID_HEADER]) > 0


def test_request_id_is_reused_when_client_supplies_one():
    client = TestClient(_build_app())
    r = client.get("/ping", headers={REQUEST_ID_HEADER: "my-custom-id"})
    assert r.headers[REQUEST_ID_HEADER] == "my-custom-id"


def test_request_id_differs_across_requests_when_not_supplied():
    client = TestClient(_build_app())
    r1 = client.get("/ping")
    r2 = client.get("/ping")
    assert r1.headers[REQUEST_ID_HEADER] != r2.headers[REQUEST_ID_HEADER]


# ---------------------------------------------------------------------------
# Middleware ordering: request_id must be available to everything downstream
# ---------------------------------------------------------------------------

def test_request_id_available_to_body_size_middleware():
    """
    If ordering were wrong, a 413 response from BodySizeLimitMiddleware
    would show request_id="unknown" instead of a real one, because
    RequestIDMiddleware wouldn't have run yet. This is verified directly,
    not just assumed from Starlette's documented middleware order.
    """
    client = TestClient(_build_app(max_body_bytes=10))
    r = client.post("/echo", json={"a_fairly_long_field_name": "some value that exceeds ten bytes easily"})
    assert r.status_code == 413
    body = r.json()
    assert body["error"]["request_id"] != "unknown"
    assert body["error"]["request_id"] == r.headers[REQUEST_ID_HEADER]


# ---------------------------------------------------------------------------
# Body size limit middleware
# ---------------------------------------------------------------------------

def test_body_size_limit_rejects_oversized_request():
    client = TestClient(_build_app(max_body_bytes=10))
    r = client.post("/echo", json={"key": "a value definitely longer than ten bytes"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"


def test_body_size_limit_allows_small_request():
    client = TestClient(_build_app(max_body_bytes=10_000))
    r = client.post("/echo", json={"key": "small"})
    assert r.status_code == 200


def test_body_size_limit_error_envelope_shape():
    client = TestClient(_build_app(max_body_bytes=5))
    r = client.post("/echo", json={"key": "value"})
    body = r.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "request_id"}
