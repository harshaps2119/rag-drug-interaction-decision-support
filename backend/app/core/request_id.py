"""
core/request_id.py
=====================
Assigns a correlation/request ID to every incoming HTTP request, so a
single request can be traced across logs, audit records, and (when
useful) error responses — this is the standard "request ID" / "trace
ID" pattern used across most production APIs.

WHY THIS MATTERS FOR DEBUGGING
-----------------------------------
Without a request ID, correlating "the user reported this error at
14:32" with the right log lines is guesswork, especially once multiple
requests are in flight concurrently. With one, the response's
`X-Request-ID` header (and any error envelope's `request_id` field) is
the single key to grep every log line, audit record, and downstream
service call for that exact request.

HOW IT WORKS
--------------
If the client supplies an `X-Request-ID` header, it's reused (useful for
a frontend or gateway that already generates one, so tracing stays
consistent end-to-end). Otherwise a fresh UUID4 is generated. Either way
it's stored on `request.state.request_id` for route handlers to read,
and echoed back in the `X-Request-ID` response header.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_request_id_middleware.py -v
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def get_request_id(request: Request) -> str:
    """FastAPI dependency: reads the request ID assigned by RequestIDMiddleware.
    Falls back to generating one if the middleware somehow wasn't applied
    (e.g. a unit test hitting a route function directly) rather than raising."""
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())
