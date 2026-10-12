"""
main.py
=========
The FastAPI application entry point (Phase 8). Assembles middleware,
exception handlers, and routers. Business logic lives in app/services/;
this file only wires things together.

HOW TO RUN
-----------
    cd backend
    uvicorn app.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs
(FastAPI's automatic Swagger UI), or:
    curl http://localhost:8000/api/health

MIDDLEWARE ORDER
------------------
Starlette applies middleware in the REVERSE of the order they're added
with `add_middleware()` -- the LAST one added runs FIRST on the way in
(and last on the way out). RequestIDMiddleware is added last so it runs
FIRST, meaning `request.state.request_id` is guaranteed to exist before
BodySizeLimitMiddleware or AuditMiddleware (both of which read it) ever
execute. This ordering is verified directly by
tests/test_middleware.py::test_request_id_available_to_body_size_middleware,
not just assumed correct from documentation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.api import drugs, health, interaction
from app.config import settings
from app.core.audit_log import configure_audit_logging
from app.core.errors import AppError, app_error_handler, unhandled_exception_handler, validation_error_handler
from app.core.middleware import AuditMiddleware, BodySizeLimitMiddleware
from app.core.request_id import RequestIDMiddleware

API_DESCRIPTION = (
    "Evidence-grounded drug-drug interaction decision support. "
    "This is an educational/informational prototype, NOT a medical device. "
    "It is not for autonomous prescribing, and does not replace professional clinical judgment."
)


def parse_allowed_origins(origin_setting: str) -> list[str]:
    """
    Parses configured frontend origins from a single URL or comma-separated list,
    normalizing whitespace and removing trailing slashes. Preserves local dev origins.
    """
    origins: list[str] = []
    if origin_setting:
        for item in origin_setting.split(","):
            cleaned = item.strip().rstrip("/")
            if cleaned:
                origins.append(cleaned)
    origins.extend(["http://localhost:5173", "http://127.0.0.1:5173"])
    return list(dict.fromkeys(origins))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.api.deps import get_embedding_model

    # Warm up embedding model in background thread on startup so request time
    # doesn't pay the heavy model-loading latency on low-CPU containers.
    if get_embedding_model not in app.dependency_overrides:
        model = get_embedding_model()
        if hasattr(model, "warmup"):
            import logging
            import threading

            def _warmup_worker():
                try:
                    model.warmup()
                except Exception as exc:
                    logging.getLogger("ddi.embeddings").warning(
                        "Background model warmup failed: %s", exc
                    )

            threading.Thread(target=_warmup_worker, daemon=True, name="model-warmup").start()
    yield


def create_app() -> FastAPI:
    configure_audit_logging()

    app = FastAPI(
        title="Drug-Drug Interaction Decision Support API",
        description=API_DESCRIPTION,
        version="0.8.0",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # See module docstring for why this order matters.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_BODY_BYTES)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RequestIDMiddleware)

    allowed_origins = parse_allowed_origins(settings.FRONTEND_ORIGIN)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(drugs.router, prefix="/api", tags=["drugs"])
    app.include_router(interaction.router, prefix="/api", tags=["interaction"])

    return app


app = create_app()
