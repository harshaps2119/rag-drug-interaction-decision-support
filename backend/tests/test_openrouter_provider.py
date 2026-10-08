"""
tests/test_openrouter_provider.py
===================================
Offline tests for the OpenRouter OpenAI-compatible LLM client.

All tests use a FakeOpenAIClient that mimics the openai.OpenAI
chat.completions.create surface — no live network call is ever made.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_openrouter_provider.py -v
"""

from types import SimpleNamespace

import pytest

from app.exceptions import (
    LLMAPIKeyInvalidError,
    LLMAPIKeyMissingError,
    LLMOutputParsingError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.schemas.evidence_assessment import (
    DrugMentionResult,
    DrugRef,
    EvidenceClassification,
    PairEvidenceItem,
    PairRetrievalResult,
)
from app.services.explanation_service import generate_explanation
from app.services.llm_service import (
    OpenRouterLLMClient,
    _translate_openrouter_exception,
    generate_grounded_response,
    get_configured_llm_client,
)
from tests._fake_llm import make_llm_json


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class FakeOpenAIClient:
    """Minimal mock of openai.OpenAI(…).chat.completions.create used by OpenRouterLLMClient."""

    def __init__(self, content: str | None = None, raises: Exception | None = None):
        self.content = content
        self.raises = raises
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


class FakeStatusError(Exception):
    def __init__(self, message: str, status_code: int):
        self.status_code = status_code
        super().__init__(message)


def _evidence_item(classification: EvidenceClassification = EvidenceClassification.PAIR_SPECIFIC) -> PairEvidenceItem:
    return PairEvidenceItem(
        chunk_id="or-evidence-1",
        text="Warfarin combined with ibuprofen may increase bleeding risk.",
        distance=0.12,
        drug_name="warfarin",
        rxcui="11289",
        section_name="Drug Interactions",
        section_code="34073-7",
        source_url="https://dailymed.example/warfarin",
        classification=classification,
        drug_mentions=DrugMentionResult(),
    )


def _pair_result(
    evidence_status: str = "pair_specific_evidence_found",
    classification: EvidenceClassification = EvidenceClassification.PAIR_SPECIFIC,
) -> PairRetrievalResult:
    item = _evidence_item(classification)
    return PairRetrievalResult(
        drug_a=DrugRef(input_name="warfarin", rxcui="11289", normalized_name="warfarin", resolved=True),
        drug_b=DrugRef(input_name="ibuprofen", rxcui="5640", normalized_name="ibuprofen", resolved=True),
        evidence_status=evidence_status,
        pair_evidence=[item] if classification == EvidenceClassification.PAIR_SPECIFIC else [],
        supporting_evidence=[] if classification == EvidenceClassification.PAIR_SPECIFIC else [item],
    )


def _client(fake: FakeOpenAIClient) -> OpenRouterLLMClient:
    return OpenRouterLLMClient(
        api_key="sk-or-test-key",
        model_name="meta-llama/llama-3.3-70b-instruct",
        sdk_client=fake,
    )


# ---------------------------------------------------------------------------
# Basic request / response
# ---------------------------------------------------------------------------


def test_openrouter_structured_response_is_parsed():
    fake = FakeOpenAIClient(make_llm_json(cited_evidence_ids=["EVIDENCE-001"]))
    result = generate_grounded_response(
        "warfarin", "ibuprofen", {"EVIDENCE-001": _evidence_item()}, client=_client(fake)
    )
    assert result.cited_evidence_ids == ["EVIDENCE-001"]
    assert len(fake.calls) == 1


def test_openrouter_uses_json_object_response_format():
    """OpenRouter uses json_object mode (not strict json_schema) for broad model compatibility."""
    fake = FakeOpenAIClient(make_llm_json())
    _client(fake).generate("prompt")
    call = fake.calls[0]
    assert call["response_format"]["type"] == "json_object"


def test_openrouter_request_uses_configured_model():
    fake = FakeOpenAIClient(make_llm_json())
    _client(fake).generate("prompt")
    assert fake.calls[0]["model"] == "meta-llama/llama-3.3-70b-instruct"


def test_openrouter_uses_zero_temperature():
    fake = FakeOpenAIClient(make_llm_json())
    _client(fake).generate("prompt")
    assert fake.calls[0]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Missing / invalid key (no network call)
# ---------------------------------------------------------------------------


def test_openrouter_missing_api_key_raises_before_network_call():
    with pytest.raises(LLMAPIKeyMissingError) as exc_info:
        OpenRouterLLMClient(api_key="", sdk_client=FakeOpenAIClient()).generate("prompt")
    assert exc_info.value.source == "OpenRouter"


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def test_openrouter_malformed_json_raises_parsing_error():
    fake = FakeOpenAIClient("not valid JSON {{{")
    with pytest.raises(LLMOutputParsingError) as exc_info:
        generate_grounded_response("warfarin", "ibuprofen", {}, client=_client(fake))
    assert exc_info.value.source == "OpenRouter"


def test_openrouter_401_translates_to_invalid_key_error():
    fake = FakeOpenAIClient(raises=FakeStatusError("No auth credentials found", 401))
    with pytest.raises(LLMAPIKeyInvalidError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "OpenRouter"


def test_openrouter_403_translates_to_invalid_key_error():
    fake = FakeOpenAIClient(raises=FakeStatusError("Unauthorized — invalid key", 403))
    with pytest.raises(LLMAPIKeyInvalidError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "OpenRouter"


def test_openrouter_429_translates_to_rate_limit_error():
    fake = FakeOpenAIClient(raises=FakeStatusError("rate limit exceeded", 429))
    with pytest.raises(LLMRateLimitError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "OpenRouter"


def test_openrouter_503_translates_to_service_unavailable():
    fake = FakeOpenAIClient(raises=FakeStatusError("service unavailable", 503))
    with pytest.raises(LLMServiceUnavailableError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "OpenRouter"


def test_openrouter_timeout_translates_to_timeout_error():
    fake = FakeOpenAIClient(raises=TimeoutError("request timed out"))
    with pytest.raises(LLMTimeoutError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "OpenRouter"


# ---------------------------------------------------------------------------
# _translate_openrouter_exception unit tests
# ---------------------------------------------------------------------------


def test_translate_openrouter_unauthenticated_maps_to_invalid_key():
    exc = FakeStatusError("invalid api key provided", 401)
    assert isinstance(_translate_openrouter_exception(exc), LLMAPIKeyInvalidError)


def test_translate_openrouter_timeout_maps_to_timeout():
    class DeadlineExceeded(Exception):
        pass
    assert isinstance(_translate_openrouter_exception(DeadlineExceeded("timed out")), LLMTimeoutError)


def test_translate_openrouter_quota_maps_to_rate_limit():
    assert isinstance(_translate_openrouter_exception(FakeStatusError("quota exceeded", 429)), LLMRateLimitError)


def test_translate_openrouter_unknown_maps_to_service_unavailable():
    assert isinstance(_translate_openrouter_exception(Exception("something unknown")), LLMServiceUnavailableError)


# ---------------------------------------------------------------------------
# Integration with explanation_service (evidence-only fallback)
# ---------------------------------------------------------------------------


def test_openrouter_failure_uses_evidence_only_fallback():
    fake = FakeOpenAIClient(raises=FakeStatusError("service unavailable", 503))
    response = generate_explanation(_pair_result(), llm_client=_client(fake))

    assert response.mode == "evidence_only"
    assert response.interaction_assessment == "pair_specific_evidence_found"
    assert response.cited_evidence[0].evidence_id == "EVIDENCE-001"
    assert "OpenRouter" in response.fallback_reason


def test_openrouter_hallucinated_citation_is_rejected_by_grounding_validation():
    fake = FakeOpenAIClient(make_llm_json(cited_evidence_ids=["EVIDENCE-999"]))
    response = generate_explanation(_pair_result(), llm_client=_client(fake))

    assert response.mode == "evidence_only"
    assert response.validation is not None
    assert response.validation.passed is False
    assert "EVIDENCE-999" in response.validation.fatal_issues[0]


def test_openrouter_cannot_upgrade_supporting_to_pair_specific_claims():
    fake = FakeOpenAIClient(
        make_llm_json(
            interaction_assessment="pair_specific_evidence_found",
            cited_evidence_ids=["EVIDENCE-001"],
            clinical_effect="The pair increases bleeding risk.",
        )
    )
    response = generate_explanation(
        _pair_result("supporting_evidence_found", EvidenceClassification.DRUG_SPECIFIC),
        llm_client=_client(fake),
    )
    assert response.mode == "evidence_only"
    assert response.validation is not None
    assert response.validation.passed is False


# ---------------------------------------------------------------------------
# get_configured_llm_client factory
# ---------------------------------------------------------------------------


def test_get_configured_llm_client_openrouter_returns_correct_type():
    client = get_configured_llm_client("openrouter")
    assert isinstance(client, OpenRouterLLMClient)


def test_get_configured_llm_client_openrouter_case_insensitive():
    client = get_configured_llm_client("OpenRouter")
    assert isinstance(client, OpenRouterLLMClient)
