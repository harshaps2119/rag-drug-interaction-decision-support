"""
core/audit_log.py
====================
Structured, privacy-by-design audit logging.

PRIVACY BY DESIGN — ENFORCED BY FUNCTION SIGNATURE, NOT JUST POLICY
------------------------------------------------------------------------
This project collects NO patient information at all — no patient names,
IDs, phone numbers, addresses, or medical records ever enter the system
in the first place (the only user-supplied input anywhere is drug
names). So the real privacy question for audit logging is narrower:
what should be logged about a REQUEST, and what shouldn't?

The answer is enforced here structurally, not just by policy: every
audit-logging function below takes an explicit, fixed set of named
keyword arguments — request_id, endpoint, method, status_code,
latency_ms, rxcui values, evidence_status, llm_mode, error_category, and
so on. There is no `**kwargs` passthrough anywhere, and no code path
that logs an arbitrary free-text request body or a raw drug-name string
(only its resolved RxCUI, once normalized — the STRUCTURED identifier,
never the free-text the user typed). A future contributor extending this
file would have to deliberately add a new parameter to log something
new — accidentally logging an unsafe field is exactly the mistake this
signature-based approach is designed to make hard to do by accident.

Explicitly NEVER logged, anywhere in this codebase: patient names,
patient IDs, phone numbers, addresses, medical record numbers,
unnecessary free-text, API keys, or raw secrets. This project has no
legitimate use for any of them, so the simplest and safest rule is:
they never had a parameter to be passed through in the first place.

WHY RxCUI, NOT THE RAW DRUG NAME THE USER TYPED
------------------------------------------------------
Logging the normalized RxCUI (once resolved) gives full operational
value — "this identifier was queried this many times" is useful for
spotting popular drugs, catching abuse patterns, and debugging — while
being a step more sanitized than echoing back raw user input verbatim
into a log file. Free-text is not logged even for drug names; when a
name fails to resolve at all, `rxcui` is simply logged as `None`.

HOW LOGS ARE STRUCTURED
----------------------------
Every audit event is a single JSON line, so logs are easy to grep,
ship to a log aggregator, or parse programmatically. See `JsonFormatter`
below.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_audit_log.py -v
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

AUDIT_LOGGER_NAME = "ddi.audit"

audit_logger = logging.getLogger(AUDIT_LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    """Formats each log record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields:
            payload.update(extra_fields)
        else:
            payload["message"] = record.getMessage()
        return json.dumps(payload, default=str)


def configure_audit_logging(handler: logging.Handler | None = None) -> None:
    """
    Attaches a JSON-formatted handler to the audit logger. Safe to call
    more than once (clears existing handlers first) — used at app
    startup, and by tests that want to capture audit output.
    """
    audit_logger.handlers.clear()
    h = handler or logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    audit_logger.addHandler(h)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False


def log_http_request(
    *,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    client_host: str | None = None,
) -> None:
    """Generic per-request audit line, logged by the audit middleware for every request."""
    _emit("http_request", {
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "client_host": client_host,
    })


def log_interaction_check(
    *,
    request_id: str,
    endpoint: str,
    rxcui_a: str | None,
    rxcui_b: str | None,
    evidence_status: str,
    llm_mode: str | None,
    latency_ms: float,
    error_category: str | None = None,
) -> None:
    """Domain-specific audit line for a single drug-pair interaction check."""
    _emit("interaction_check", {
        "request_id": request_id,
        "endpoint": endpoint,
        "rxcui_a": rxcui_a,
        "rxcui_b": rxcui_b,
        "evidence_status": evidence_status,
        "llm_mode": llm_mode,
        "latency_ms": round(latency_ms, 2),
        "error_category": error_category,
    })


def log_rate_limit_exceeded(*, request_id: str, endpoint: str, client_host: str | None) -> None:
    _emit("rate_limit_exceeded", {
        "request_id": request_id,
        "endpoint": endpoint,
        "client_host": client_host,
    })


def _emit(event: str, fields: dict) -> None:
    audit_logger.info(event, extra={"extra_fields": {"event": event, **fields}})
