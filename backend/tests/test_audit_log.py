"""
tests/test_audit_log.py
==========================
Tests for app/core/audit_log.py — verifies structured JSON output and,
critically, that the audit logging FUNCTIONS have no code path capable
of accepting/emitting free-text or unsafe fields (the privacy-by-design
claim in that module's docstring, checked mechanically here via
signature inspection, not just by reading the code).

HOW TO RUN
-----------
    cd backend
    pytest tests/test_audit_log.py -v
"""

import inspect
import json
import logging

from app.core.audit_log import (
    JsonFormatter,
    audit_logger,
    configure_audit_logging,
    log_http_request,
    log_interaction_check,
    log_rate_limit_exceeded,
)


def test_json_formatter_produces_valid_json():
    record = logging.LogRecord(
        name="ddi.audit", level=logging.INFO, pathname="x", lineno=1, msg="test", args=None, exc_info=None,
    )
    record.extra_fields = {"event": "test_event", "request_id": "abc-123"}
    formatted = JsonFormatter().format(record)
    parsed = json.loads(formatted)
    assert parsed["event"] == "test_event"
    assert parsed["request_id"] == "abc-123"
    assert "timestamp" in parsed
    assert "level" in parsed


def test_configure_audit_logging_captures_output(caplog_json_handler=None):
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    configure_audit_logging(handler=handler)

    log_http_request(request_id="r1", method="GET", path="/api/health", status_code=200, latency_ms=5.5)

    output = stream.getvalue().strip()
    parsed = json.loads(output)
    assert parsed["event"] == "http_request"
    assert parsed["request_id"] == "r1"
    assert parsed["status_code"] == 200


def test_log_interaction_check_emits_expected_fields():
    import io

    stream = io.StringIO()
    configure_audit_logging(handler=logging.StreamHandler(stream))

    log_interaction_check(
        request_id="r2", endpoint="/api/interaction/check", rxcui_a="11289", rxcui_b="5640",
        evidence_status="pair_specific_evidence_found", llm_mode="llm_grounded", latency_ms=120.4,
    )

    parsed = json.loads(stream.getvalue().strip())
    assert parsed["event"] == "interaction_check"
    assert parsed["rxcui_a"] == "11289"
    assert parsed["rxcui_b"] == "5640"
    assert parsed["evidence_status"] == "pair_specific_evidence_found"
    assert parsed["llm_mode"] == "llm_grounded"


def test_log_rate_limit_exceeded_emits_expected_fields():
    import io

    stream = io.StringIO()
    configure_audit_logging(handler=logging.StreamHandler(stream))

    log_rate_limit_exceeded(request_id="r3", endpoint="/api/interaction/check", client_host="127.0.0.1")

    parsed = json.loads(stream.getvalue().strip())
    assert parsed["event"] == "rate_limit_exceeded"
    assert parsed["client_host"] == "127.0.0.1"


def test_audit_logger_does_not_propagate_to_root():
    """Ensures audit lines aren't duplicated into the root/app logger."""
    configure_audit_logging()
    assert audit_logger.propagate is False


# ---------------------------------------------------------------------------
# Privacy-by-design: mechanically verify no unsafe field can be passed through
# ---------------------------------------------------------------------------

_FORBIDDEN_PARAM_NAME_FRAGMENTS = [
    "patient", "name", "phone", "address", "medical_record", "ssn", "api_key", "secret", "password",
]
# 'name' fragment would false-positive on legitimate params -- allowlist the ones we know are safe.
_ALLOWED_EXCEPTIONS = set()


def test_audit_functions_have_no_kwargs_passthrough():
    """No audit-logging function accepts **kwargs -- every loggable field
    must be an explicit, named, reviewed parameter (see module docstring)."""
    for fn in (log_http_request, log_interaction_check, log_rate_limit_exceeded):
        sig = inspect.signature(fn)
        kinds = [p.kind for p in sig.parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD not in kinds, f"{fn.__name__} must not accept **kwargs"


def test_audit_functions_do_not_expose_forbidden_parameter_names():
    """Static check: none of the fixed, explicit parameters on any audit
    function are named after a category of data this project must never log."""
    for fn in (log_http_request, log_interaction_check, log_rate_limit_exceeded):
        sig = inspect.signature(fn)
        for param_name in sig.parameters:
            if param_name in _ALLOWED_EXCEPTIONS:
                continue
            for fragment in _FORBIDDEN_PARAM_NAME_FRAGMENTS:
                assert fragment not in param_name.lower(), (
                    f"{fn.__name__} has a parameter '{param_name}' matching forbidden fragment '{fragment}'"
                )
