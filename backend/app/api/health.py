"""
api/health.py
================
GET /api/health -- reports whether this service's OWN local dependencies
(SQLite, ChromaDB, and whether an LLM key is configured) are working.

WHY THIS NEVER CALLS RxNorm, DailyMed, OR Gemini
------------------------------------------------------
A health check exists to answer "is this service itself up and able to
serve requests" -- quickly, cheaply, and reliably enough to be polled
frequently (e.g. by a load balancer or container orchestrator every few
seconds). Calling out to THIRD-PARTY services would make health totally
dependent on THEIR uptime and latency, and would burn API quota /
Gemini cost on every health check tick for no operational benefit -- if
RxNorm has a bad five minutes, that should not make THIS service report
unhealthy and get restarted by an orchestrator. "gemini_configured"
below checks only whether a key STRING is present locally, never
whether Gemini itself is reachable.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_chroma_collection, get_db_session
from app.config import settings
from app.core.request_id import get_request_id
from app.schemas.api import HealthCheckDetail, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    request: Request,
    session: Session = Depends(get_db_session),
    collection=Depends(get_chroma_collection),
) -> HealthResponse:
    checks: dict[str, HealthCheckDetail] = {}

    try:
        session.execute(text("SELECT 1"))
        checks["sqlite"] = HealthCheckDetail(status="ok")
    except Exception as exc:
        checks["sqlite"] = HealthCheckDetail(status="error", detail=str(exc))

    try:
        collection.count()
        checks["chromadb"] = HealthCheckDetail(status="ok")
    except Exception as exc:
        checks["chromadb"] = HealthCheckDetail(status="error", detail=str(exc))

    if settings.GEMINI_API_KEY:
        checks["gemini_configured"] = HealthCheckDetail(status="ok")
    else:
        checks["gemini_configured"] = HealthCheckDetail(
            status="error",
            detail="GEMINI_API_KEY is not set -- requests will use evidence-only fallback mode.",
        )

    overall = "ok" if all(c.status == "ok" for c in checks.values()) else "degraded"
    return HealthResponse(status=overall, checks=checks, request_id=get_request_id(request))
