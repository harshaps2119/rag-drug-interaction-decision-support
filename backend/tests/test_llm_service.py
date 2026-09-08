"""
tests/test_llm_service.py
============================
Tests for app/services/llm_service.py.

Uses FakeLLMClient (tests/_fake_llm.py) — no live Gemini call. Tests
prompt content, response parsing, and exception translation.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_llm_service.py -v
"""

import pytest

from app.exceptions import (
    LLMAPIKeyInvalidError,
    LLMAPIKeyMissingError,
    LLMOutputParsingError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.schemas.evidence_assessment import DrugMentionResult, EvidenceClassification, PairEvidenceItem
from app.services.llm_service import (
    GeminiLLMClient,
    _translate_gemini_exception,
    build_prompt,
    generate_grounded_response,
)
from tests._fake_llm import FakeLLMClient, make_llm_json


def _evidence_item(classification=EvidenceClassification.PAIR_SPECIFIC, text="Warfarin interacts with ibuprofen."):
    return PairEvidenceItem(
        chunk_id="c1", text=text, distance=0.1, drug_name="warfarin", rxcui="11289",
        section_name="Drug Interactions", section_code="34073-7",
        classification=classification, drug_mentions=DrugMentionResult(),
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def test_build_prompt_includes_both_drug_names():
    prompt = build_prompt("warfarin", "ibuprofen", {})
    assert "warfarin" in prompt
    assert "ibuprofen" in prompt


def test_build_prompt_includes_evidence_ids_and_text():
    evidence_map = {"EVIDENCE-001": _evidence_item()}
    prompt = build_prompt("warfarin", "ibuprofen", evidence_map)
    assert "EVIDENCE-001" in prompt
    assert "Warfarin interacts with ibuprofen." in prompt


def test_build_prompt_states_explicitly_when_no_evidence():
    prompt = build_prompt("warfarin", "ibuprofen", {})
    assert "none retrieved" in prompt.lower()


def test_build_prompt_contains_strict_safety_instructions():
    prompt = build_prompt("warfarin", "ibuprofen", {})
    assert "do not" in prompt.lower()
    assert "EVIDENCE-XXX" in prompt or "evidence" in prompt.lower()
    assert "severity" in prompt.lower()


def test_build_prompt_truncates_very_long_evidence_text():
    long_text = "x" * 5000
    evidence_map = {"EVIDENCE-001": _evidence_item(text=long_text)}
    prompt = build_prompt("warfarin", "ibuprofen", evidence_map)
    # The full 5000-char blob should not appear verbatim (defensive cap applied).
    assert long_text not in prompt


# ---------------------------------------------------------------------------
# Response parsing (via generate_grounded_response with a fake client)
# ---------------------------------------------------------------------------

def test_generate_grounded_response_parses_valid_json():
    client = FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=["EVIDENCE-001"]))
    result = generate_grounded_response("warfarin", "ibuprofen", {"EVIDENCE-001": _evidence_item()}, client=client)

    assert result.interaction_assessment == "pair_specific_evidence_found"
    assert result.cited_evidence_ids == ["EVIDENCE-001"]
    assert client.call_count == 1


def test_generate_grounded_response_strips_markdown_code_fences():
    fenced = "```json\n" + make_llm_json() + "\n```"
    client = FakeLLMClient(response_text=fenced)
    result = generate_grounded_response("warfarin", "ibuprofen", {}, client=client)
    assert result.interaction_assessment == "pair_specific_evidence_found"


def test_generate_grounded_response_malformed_json_raises_parsing_error():
    client = FakeLLMClient(response_text="this is not json at all {{{")
    with pytest.raises(LLMOutputParsingError):
        generate_grounded_response("warfarin", "ibuprofen", {}, client=client)


def test_generate_grounded_response_schema_mismatch_raises_parsing_error():
    import json

    bad_payload = json.dumps({"interaction_assessment": "no_interaction", "evidence_summary": "x", "safety_notice": "y"})
    client = FakeLLMClient(response_text=bad_payload)
    with pytest.raises(LLMOutputParsingError):
        generate_grounded_response("warfarin", "ibuprofen", {}, client=client)


def test_generate_grounded_response_valid_json_missing_required_field_raises():
    import json

    incomplete = json.dumps({"interaction_assessment": "insufficient_evidence"})  # missing evidence_summary, safety_notice
    client = FakeLLMClient(response_text=incomplete)
    with pytest.raises(LLMOutputParsingError):
        generate_grounded_response("warfarin", "ibuprofen", {}, client=client)


# ---------------------------------------------------------------------------
# Client-level failure propagation
# ---------------------------------------------------------------------------

def test_generate_grounded_response_propagates_client_timeout():
    client = FakeLLMClient(raises=LLMTimeoutError("simulated timeout"))
    with pytest.raises(LLMTimeoutError):
        generate_grounded_response("warfarin", "ibuprofen", {}, client=client)


def test_generate_grounded_response_propagates_rate_limit():
    client = FakeLLMClient(raises=LLMRateLimitError("simulated rate limit"))
    with pytest.raises(LLMRateLimitError):
        generate_grounded_response("warfarin", "ibuprofen", {}, client=client)


def test_generate_grounded_response_propagates_service_unavailable():
    client = FakeLLMClient(raises=LLMServiceUnavailableError("simulated 503"))
    with pytest.raises(LLMServiceUnavailableError):
        generate_grounded_response("warfarin", "ibuprofen", {}, client=client)


# ---------------------------------------------------------------------------
# GeminiLLMClient: missing/invalid key handling (no network call needed)
# ---------------------------------------------------------------------------

def test_gemini_client_missing_api_key_raises_before_network_call():
    client = GeminiLLMClient(api_key="")
    with pytest.raises(LLMAPIKeyMissingError):
        client.generate("some prompt")


def test_gemini_client_none_api_key_raises():
    client = GeminiLLMClient(api_key=None)
    # api_key=None falls back to settings.GEMINI_API_KEY, which is "" by default
    # in a test environment without a real .env — either way, missing key must raise.
    from app.config import settings

    if not settings.GEMINI_API_KEY:
        with pytest.raises(LLMAPIKeyMissingError):
            client.generate("some prompt")


# ---------------------------------------------------------------------------
# Exception translation
# ---------------------------------------------------------------------------

def test_translate_gemini_exception_unauthenticated_maps_to_invalid_key():
    exc = Exception("400 API_KEY_INVALID: the provided key is invalid")
    translated = _translate_gemini_exception(exc)
    assert isinstance(translated, LLMAPIKeyInvalidError)


def test_translate_gemini_exception_deadline_maps_to_timeout():
    class DeadlineExceeded(Exception):
        pass

    translated = _translate_gemini_exception(DeadlineExceeded("deadline exceeded"))
    assert isinstance(translated, LLMTimeoutError)


def test_translate_gemini_exception_resource_exhausted_maps_to_rate_limit():
    class ResourceExhausted(Exception):
        pass

    translated = _translate_gemini_exception(ResourceExhausted("429 quota exceeded"))
    assert isinstance(translated, LLMRateLimitError)


def test_translate_gemini_exception_unknown_error_maps_to_service_unavailable():
    translated = _translate_gemini_exception(Exception("something totally unexpected"))
    assert isinstance(translated, LLMServiceUnavailableError)
