"""
api/deps.py
=============
FastAPI dependency providers. Centralizing these here means:
  1. Route functions stay thin (see Phase 8 requirement: business logic
     lives in services, not route functions).
  2. Tests can override any of these via `app.dependency_overrides[...]`
     to inject fakes (an in-memory SQLite session, DeterministicFakeEmbeddingModel,
     a tmp_path ChromaDB collection, FakeLLMClient) -- see tests/conftest.py.

Heavy resources (the embedding model, the ChromaDB client) are lazily
created as module-level singletons on first use, NOT at import time --
importing this module (e.g. for tests that override everything anyway)
never triggers a model download or a filesystem/DB connection.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.config import settings
from app.core.audit_log import log_rate_limit_exceeded
from app.core.errors import RateLimitExceededError
from app.core.rate_limiter import InMemoryRateLimiter
from app.core.request_id import get_request_id
from app.db import get_session_factory
from app.rag.embeddings import SentenceTransformerEmbeddingModel
from app.rag.vector_store import get_collection
from app.services.llm_service import get_configured_llm_client

_embedding_model = None
_chroma_collection = None
_rate_limiter: InMemoryRateLimiter | None = None


def get_db_session():
    """Yields a SQLAlchemy session, closed after the request completes."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformerEmbeddingModel()
    return _embedding_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        _chroma_collection = get_collection()
    return _chroma_collection


def get_llm_client():
    try:
        return get_configured_llm_client()
    except Exception as exc:
        from app.exceptions import LLMServiceError, LLMServiceUnavailableError

        class FailingLLMClient:
            def __init__(self, err):
                self._err = err
                self.source = getattr(err, "source", "configuration")

            def generate(self, prompt: str) -> str:
                if isinstance(self._err, LLMServiceError):
                    raise self._err
                raise LLMServiceUnavailableError(f"LLM client initialization failed: {self._err}", source="configuration")

        return FailingLLMClient(exc)


def get_rate_limiter() -> InMemoryRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = InMemoryRateLimiter(
            max_requests=settings.RATE_LIMIT_MAX_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )
    return _rate_limiter


def rate_limit_dependency(request: Request, limiter: InMemoryRateLimiter = Depends(get_rate_limiter)) -> None:
    """
    Applied to expensive endpoints (interaction checking). Keyed by
    client IP. See core/rate_limiter.py for the in-memory/single-process
    trade-off, documented there and in docs/security.md.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    client_key = request.client.host if request.client else "unknown"
    allowed, retry_after = limiter.check(client_key)
    if not allowed:
        request_id = get_request_id(request)
        log_rate_limit_exceeded(request_id=request_id, endpoint=request.url.path, client_host=client_key)
        raise RateLimitExceededError(
            f"Rate limit exceeded. Try again in {retry_after:.0f} seconds.",
        )
