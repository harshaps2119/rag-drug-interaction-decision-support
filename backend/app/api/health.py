"""
api/health.py
================
GET /api/health -- reports whether this service's OWN local dependencies
(SQLite, ChromaDB, and whether the selected LLM key is configured) are working.

WHY THIS NEVER CALLS RxNorm, DailyMed, OR AN LLM PROVIDER
------------------------------------------------------
A health check exists to answer "is this service itself up and able to
serve requests" -- quickly, cheaply, and reliably enough to be polled
frequently (e.g. by a load balancer or container orchestrator every few
seconds). Calling out to THIRD-PARTY services would make health totally
dependent on THEIR uptime and latency, and would burn API quota /
LLM cost on every health check tick for no operational benefit -- if
RxNorm has a bad five minutes, that should not make THIS service report
unhealthy and get restarted by an orchestrator. "llm_configured" below
checks only whether the selected provider's key STRING is present locally,
never whether that provider itself is reachable.

OPERATIONAL CACHING TRADE-OFF (10-SECOND TTL)
---------------------------------------------
To prevent orchestrator probe timeouts (e.g. Render's 5-second HTTP health-check
timeout) during CPU-intensive retrieval and model operations, successful or
degraded health results are cached for 10.0 seconds. A genuine storage failure
(SQLite corruption or ChromaDB storage failure) can remain undetected by this
endpoint for up to this 10-second window before the cache expires and live checks
re-execute. The cache is automatically bypassed when test dependency overrides
are present.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_chroma_collection, get_db_session
from app.config import settings
from app.core.request_id import get_request_id
from app.schemas.api import HealthCheckDetail, HealthResponse

router = APIRouter()

_HEALTH_CACHE_TTL_SECONDS = 10.0
_cached_health: tuple[float, dict[str, HealthCheckDetail], str] | None = None


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    session: Session = Depends(get_db_session),
    collection=Depends(get_chroma_collection),
) -> HealthResponse:
    global _cached_health

    # If recent health results exist and no test dependency overrides are active,
    # serve immediately to prevent orchestrator 5s timeouts under heavy CPU contention.
    now = time.monotonic()
    has_overrides = bool(getattr(request.app, "dependency_overrides", None))
    if not has_overrides and _cached_health is not None:
        cached_time, cached_checks, cached_overall = _cached_health
        if now - cached_time < _HEALTH_CACHE_TTL_SECONDS:
            return HealthResponse(
                status=cached_overall,
                checks={
                    k: HealthCheckDetail(status=c.status, detail=c.detail)
                    for k, c in cached_checks.items()
                },
                request_id=get_request_id(request),
            )

    checks: dict[str, HealthCheckDetail] = {}

    try:
        session.execute(text("SELECT 1"))
        checks["sqlite"] = HealthCheckDetail(status="ok")
    except Exception as exc:
        checks["sqlite"] = HealthCheckDetail(status="error", detail=str(exc))

    try:
        collection.get(limit=1)
        checks["chromadb"] = HealthCheckDetail(status="ok")
    except Exception as exc:
        checks["chromadb"] = HealthCheckDetail(status="error", detail=str(exc))

    if settings.LLM_PROVIDER in ("none", "disabled", "off"):
        checks["llm_configured"] = HealthCheckDetail(
            status="ok",
            detail="LLM disabled -- running in evidence-only mode.",
        )
    else:
        api_key = settings.XAI_API_KEY if settings.LLM_PROVIDER == "xai" else settings.GEMINI_API_KEY
        if api_key:
            checks["llm_configured"] = HealthCheckDetail(status="ok")
        else:
            required_key = "XAI_API_KEY" if settings.LLM_PROVIDER == "xai" else "GEMINI_API_KEY"
            checks["llm_configured"] = HealthCheckDetail(
                status="error",
                detail=f"{required_key} is not set -- requests will use evidence-only fallback mode.",
            )

    overall = "ok" if all(c.status == "ok" for c in checks.values()) else "degraded"
    if not has_overrides:
        _cached_health = (
            now,
            {k: HealthCheckDetail(status=c.status, detail=c.detail) for k, c in checks.items()},
            overall,
        )
    return HealthResponse(status=overall, checks=checks, request_id=get_request_id(request))
