"""
core/middleware.py
=====================
Two small pieces of application-wide middleware:

1. AuditMiddleware — logs one structured audit line for EVERY request
   (method, path, status code, latency, request id) via
   core/audit_log.log_http_request(). This is the generic, always-on
   audit trail; domain-specific detail (which drugs, what evidence
   status) is logged separately by the interaction router itself, which
   is the only place that actually has that context.

2. BodySizeLimitMiddleware — rejects requests with a body larger than
   settings.MAX_REQUEST_BODY_BYTES before the route handler (or Pydantic
   validation) ever runs, returning a 413 in this project's standard
   error envelope. This is a blunt, cheap guard against "unreasonable
   request sizes" (an explicit Phase 8 requirement) — this API has no
   legitimate reason to receive a multi-megabyte JSON body for two drug
   names.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_middleware.py -v
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.core.audit_log import log_http_request


class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000

        request_id = getattr(request.state, "request_id", "unknown")
        log_http_request(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            client_host=request.client.host if request.client else None,
        )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                size = int(content_length)
            except ValueError:
                size = None
            if size is not None and size > self.max_bytes:
                request_id = getattr(request.state, "request_id", "unknown")
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "request_too_large",
                            "message": f"Request body exceeds the maximum allowed size of {self.max_bytes} bytes.",
                            "request_id": request_id,
                        }
                    },
                )
        return await call_next(request)
