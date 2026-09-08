"""
core/errors.py
=================
A consistent API error envelope, and the exception handlers that
produce it. Every error response from this API — validation failures,
rate limiting, request-too-large, or an unexpected server error — has
exactly this shape:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

WHY A CONSISTENT ENVELOPE MATTERS
--------------------------------------
A frontend (or a viva panel reading the API) should never have to guess
what shape an error takes based on which endpoint or failure mode
produced it. One shape, always, makes client-side error handling simple
and makes this API's failure modes easy to document and reason about.

WHY NOTHING SENSITIVE EVER REACHES `message`
--------------------------------------------------
`message` is always a short, safe, human-readable string. Stack traces,
internal file paths, raw exception text from third-party libraries, and
any other implementation detail are logged server-side (via
`logger.exception(...)`, which includes the full traceback in the
server's own logs) and NEVER included in the JSON returned to the
client. This is enforced by construction: the generic
`unhandled_exception_handler` below builds its own fixed, generic
message string — it does not interpolate `str(exc)` into the response
body at all, so there's no risk of an unexpected exception message
accidentally leaking something sensitive (a file path, a partial SQL
query, etc.) to an API consumer.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_error_handling.py -v
"""

from __future__ import annotations

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("ddi.app")


class AppError(Exception):
    """Base class for this application's own deliberately-raised HTTP errors."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    code: str = "app_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None):
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(message)


class RateLimitExceededError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limit_exceeded"


class RequestTooLargeError(AppError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "request_too_large"


def _error_body(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "unknown"


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request_id = _get_request_id(request)
    return JSONResponse(status_code=exc.status_code, content=_error_body(exc.code, exc.message, request_id))


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Reformats FastAPI/Pydantic's default validation error (which includes
    internal field-path details useful for developers, not end users)
    into this project's consistent envelope with a single readable
    message, while the full structured detail still goes to server logs.
    """
    request_id = _get_request_id(request)
    logger.info("Validation error on %s: %s", request.url.path, exc.errors())

    first_error = exc.errors()[0] if exc.errors() else None
    if first_error:
        field = ".".join(str(loc) for loc in first_error.get("loc", []) if loc != "body")
        message = f"{field}: {first_error.get('msg')}" if field else str(first_error.get("msg"))
    else:
        message = "The request could not be validated."

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_error_body("validation_error", message, request_id),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort handler for anything not already handled. Logs the FULL
    exception (with traceback) server-side, but returns only a fixed,
    generic message to the client -- see module docstring for why this
    is a hard rule, not a judgment call made per-exception.
    """
    request_id = _get_request_id(request)
    logger.exception("Unhandled exception on %s [request_id=%s]", request.url.path, request_id)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body("internal_error", "An unexpected error occurred. Please try again.", request_id),
    )
