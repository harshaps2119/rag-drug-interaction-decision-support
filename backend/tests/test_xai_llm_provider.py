"""Offline tests for the xAI OpenAI-compatible structured-output client."""

from types import SimpleNamespace

import pytest

from app.exceptions import (
    LLMAPIKeyInvalidError,
    LLMAPIKeyMissingError,
    LLMOutputParsingError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from app.schemas.evidence_assessment import DrugMentionResult, DrugRef, EvidenceClassification, PairEvidenceItem, PairRetrievalResult
from app.services.explanation_service import generate_explanation
from app.services.llm_service import XAILLMClient, generate_grounded_response
from tests._fake_llm import make_llm_json


class FakeXAIClient:
    """Minimal mock of OpenAI(...).chat.completions.create used by XAILLMClient."""

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


def _evidence_item(classification=EvidenceClassification.PAIR_SPECIFIC):
    return PairEvidenceItem(
        chunk_id="xai-evidence-1",
        text="Warfarin combined with ibuprofen may increase bleeding risk.",
        distance=0.1,
        drug_name="warfarin",
        rxcui="11289",
        section_name="Drug Interactions",
        section_code="34073-7",
        source_url="https://dailymed.example/warfarin",
        classification=classification,
        drug_mentions=DrugMentionResult(),
    )


def _pair_result(evidence_status="pair_specific_evidence_found", classification=EvidenceClassification.PAIR_SPECIFIC):
    item = _evidence_item(classification)
    return PairRetrievalResult(
        drug_a=DrugRef(input_name="warfarin", rxcui="11289", normalized_name="warfarin", resolved=True),
        drug_b=DrugRef(input_name="ibuprofen", rxcui="5640", normalized_name="ibuprofen", resolved=True),
        evidence_status=evidence_status,
        pair_evidence=[item] if classification == EvidenceClassification.PAIR_SPECIFIC else [],
        supporting_evidence=[] if classification == EvidenceClassification.PAIR_SPECIFIC else [item],
    )


def _client(fake: FakeXAIClient) -> XAILLMClient:
    return XAILLMClient(api_key="xai-test-key", model_name="grok-4.6", sdk_client=fake)


def test_xai_structured_response_is_parsed_and_requests_strict_schema():
    fake = FakeXAIClient(make_llm_json(cited_evidence_ids=["EVIDENCE-001"]))
    result = generate_grounded_response("warfarin", "ibuprofen", {"EVIDENCE-001": _evidence_item()}, client=_client(fake))

    assert result.cited_evidence_ids == ["EVIDENCE-001"]
    request = fake.calls[0]
    assert request["model"] == "grok-4.6"
    assert request["temperature"] == 0.0
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_xai_malformed_structured_response_is_rejected():
    fake = FakeXAIClient("not JSON")
    with pytest.raises(LLMOutputParsingError) as exc_info:
        generate_grounded_response("warfarin", "ibuprofen", {}, client=_client(fake))
    assert exc_info.value.source == "xAI"


def test_xai_api_failure_is_translated_without_a_network_call():
    fake = FakeXAIClient(raises=FakeStatusError("server error", 503))
    with pytest.raises(LLMServiceUnavailableError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "xAI"


def test_xai_incorrect_api_key_is_translated_to_key_error():
    fake = FakeXAIClient(raises=FakeStatusError("Incorrect API key provided", 400))
    with pytest.raises(LLMAPIKeyInvalidError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "xAI"


def test_xai_timeout_is_translated_without_a_network_call():
    fake = FakeXAIClient(raises=TimeoutError("request timed out"))
    with pytest.raises(LLMTimeoutError) as exc_info:
        _client(fake).generate("prompt")
    assert exc_info.value.source == "xAI"


def test_xai_missing_key_fails_before_provider_call():
    with pytest.raises(LLMAPIKeyMissingError) as exc_info:
        XAILLMClient(api_key="", sdk_client=FakeXAIClient()).generate("prompt")
    assert exc_info.value.source == "xAI"


def test_xai_failure_uses_existing_evidence_only_fallback():
    fake = FakeXAIClient(raises=FakeStatusError("service unavailable", 503))
    response = generate_explanation(_pair_result(), llm_client=_client(fake))

    assert response.mode == "evidence_only"
    assert response.interaction_assessment == "pair_specific_evidence_found"
    assert response.cited_evidence[0].evidence_id == "EVIDENCE-001"
    assert "xAI" in response.fallback_reason


def test_xai_hallucinated_citation_is_rejected_by_existing_grounding_validation():
    fake = FakeXAIClient(make_llm_json(cited_evidence_ids=["EVIDENCE-999"]))
    response = generate_explanation(_pair_result(), llm_client=_client(fake))

    assert response.mode == "evidence_only"
    assert response.validation is not None
    assert response.validation.passed is False
    assert "EVIDENCE-999" in response.validation.fatal_issues[0]


def test_xai_cannot_upgrade_supporting_evidence_to_pair_specific_claims():
    fake = FakeXAIClient(
        make_llm_json(
            interaction_assessment="pair_specific_evidence_found",
            cited_evidence_ids=["EVIDENCE-001"],
            clinical_effect="The pair increases bleeding risk.",
        )
    )
    response = generate_explanation(
        _pair_result("supporting_evidence_found", EvidenceClassification.DRUG_SPECIFIC), llm_client=_client(fake)
    )

    assert response.mode == "evidence_only"
    assert response.validation is not None
    assert response.validation.passed is False
