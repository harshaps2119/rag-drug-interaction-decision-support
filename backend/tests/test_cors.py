"""
tests/test_cors.py
==================
Regression tests for CORS origin handling and middleware:
1. Normalization (whitespace, trailing slashes, comma separation).
2. Allowed origins (production Vercel app, local dev).
3. Rejected origins (arbitrary *.vercel.app, evil.com).
"""

from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.main import create_app, parse_allowed_origins


def test_parse_allowed_origins_single():
    origins = parse_allowed_origins("https://rag-drug-interaction-decision-suppo.vercel.app")
    assert "https://rag-drug-interaction-decision-suppo.vercel.app" in origins
    assert "http://localhost:5173" in origins
    assert "http://127.0.0.1:5173" in origins


def test_parse_allowed_origins_trailing_slash_stripped():
    origins = parse_allowed_origins("https://rag-drug-interaction-decision-suppo.vercel.app/")
    assert "https://rag-drug-interaction-decision-suppo.vercel.app" in origins
    assert "https://rag-drug-interaction-decision-suppo.vercel.app/" not in origins


def test_parse_allowed_origins_comma_separated_and_whitespace():
    raw = "  https://origin1.com/ ,  https://origin2.com  "
    origins = parse_allowed_origins(raw)
    assert "https://origin1.com" in origins
    assert "https://origin2.com" in origins
    assert "http://localhost:5173" in origins


def test_parse_allowed_origins_deduplication():
    raw = "http://localhost:5173, http://localhost:5173/"
    origins = parse_allowed_origins(raw)
    assert origins.count("http://localhost:5173") == 1


def test_cors_preflight_allowed_origin(monkeypatch):
    monkeypatch.setattr(
        settings,
        "FRONTEND_ORIGIN",
        "https://rag-drug-interaction-decision-suppo.vercel.app",
    )
    app = create_app()
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://rag-drug-interaction-decision-suppo.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers.get("access-control-allow-origin")
        == "https://rag-drug-interaction-decision-suppo.vercel.app"
    )


def test_cors_preflight_disallowed_origin(monkeypatch):
    monkeypatch.setattr(
        settings,
        "FRONTEND_ORIGIN",
        "https://rag-drug-interaction-decision-suppo.vercel.app",
    )
    app = create_app()
    client = TestClient(app)

    # Malicious or untrusted origin
    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://evil.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_cors_preflight_arbitrary_vercel_subdomain_rejected(monkeypatch):
    """
    Ensures wildcard *.vercel.app is NOT permitted by default,
    blocking arbitrary preview deployments from unauthorized access.
    """
    monkeypatch.setattr(
        settings,
        "FRONTEND_ORIGIN",
        "https://rag-drug-interaction-decision-suppo.vercel.app",
    )
    app = create_app()
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://unauthorized-branch.vercel.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
