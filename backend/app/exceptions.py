"""
exceptions.py
=============
Custom exception types shared across backend services.

WHY THIS FILE EXISTS
---------------------
Generic exceptions (like a bare `Exception` or `httpx.HTTPError`) don't tell
the caller WHAT went wrong in domain terms. By defining specific exception
classes, the FastAPI layer (built in Phase 9) can catch them individually
and return the right HTTP status + a clear message to the frontend, instead
of a generic 500 error that hides what actually happened.

This matters extra for a health-adjacent tool: if RxNorm is down, the user
should see "drug lookup service unavailable, try again" — not a stack trace,
and not a silently wrong/blank result that looks like "no interaction."
"""


class ExternalAPIError(Exception):
    """Base class for errors talking to an external API (RxNorm, DailyMed, LLM)."""

    def __init__(self, message: str, *, source: str, status_code: int | None = None):
        self.source = source
        self.status_code = status_code
        super().__init__(message)


class ExternalAPITimeoutError(ExternalAPIError):
    """Raised when an external API call times out."""


class ExternalAPIUnavailableError(ExternalAPIError):
    """Raised when an external API is unreachable or returns a 5xx error."""


class ExternalAPIBadResponseError(ExternalAPIError):
    """Raised when an external API returns a 2xx response we can't parse as expected."""


class LLMServiceError(Exception):
    """Base class for all errors from the LLM layer (Phase 7)."""

    def __init__(self, message: str, *, source: str = "Gemini"):
        self.source = source
        super().__init__(message)


class LLMAPIKeyMissingError(LLMServiceError):
    """Raised when no Gemini API key is configured at all."""


class LLMAPIKeyInvalidError(LLMServiceError):
    """Raised when the Gemini API rejects the configured key."""


class LLMTimeoutError(LLMServiceError):
    """Raised when a Gemini request times out."""


class LLMRateLimitError(LLMServiceError):
    """Raised when Gemini's rate limit is hit."""


class LLMServiceUnavailableError(LLMServiceError):
    """Raised when Gemini is unreachable or returns a server-side error."""


class LLMOutputParsingError(LLMServiceError):
    """Raised when Gemini's response isn't valid JSON, or doesn't match the expected schema."""


class DrugNotFoundError(Exception):
    """Raised when a drug name cannot be normalized/found in RxNorm at all."""

    def __init__(self, drug_name: str):
        self.drug_name = drug_name
        super().__init__(f"Drug '{drug_name}' could not be found or normalized.")
