"""
tests/test_explanation_service.py
====================================
Tests for app/services/explanation_service.py — the orchestration and
fallback logic. Uses FakeLLMClient (tests/_fake_llm.py) and directly
constructed PairRetrievalResult fixtures. No live Gemini call, no live
retrieval pipeline.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_explanation_service.py -v
"""

from app.exceptions import LLMRateLimitError, LLMServiceUnavailableError, LLMTimeoutError
from app.schemas.evidence_assessment import (
    DrugMentionResult,
    DrugRef,
    EvidenceClassification,
    PairEvidenceItem,
    PairRetrievalResult,
)
from app.services.explanation_service import build_evidence_id_map, generate_explanation
from tests._fake_llm import FakeLLMClient, make_llm_json


def _drug_ref(name, rxcui, resolved=True):
    return DrugRef(input_name=name, rxcui=rxcui, normalized_name=name, resolved=resolved, search_terms=[name])


def _evidence_item(chunk_id="c1", classification=EvidenceClassification.PAIR_SPECIFIC, text="Warfarin interacts with ibuprofen, increasing bleeding risk."):
    return PairEvidenceItem(
        chunk_id=chunk_id, text=text, distance=0.1, drug_name="warfarin", rxcui="11289",
        label_setid="w-setid", spl_version="1", section_name="Drug Interactions", section_code="34073-7",
        source_url="https://dailymed.example/w-setid",
        classification=classification, drug_mentions=DrugMentionResult(),
        evidence_status_of_label="interaction_evidence_found",
    )


def _pair_result(evidence_status="pair_specific_evidence_found", pair_evidence=None, supporting_evidence=None):
    return PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"),
        drug_b=_drug_ref("ibuprofen", "5640"),
        evidence_status=evidence_status,
        pair_evidence=pair_evidence or [],
        supporting_evidence=supporting_evidence or [],
        queries_used=["warfarin and ibuprofen interaction"],
        limitations=["Standard limitation text."],
    )


# ---------------------------------------------------------------------------
# build_evidence_id_map
# ---------------------------------------------------------------------------

def test_build_evidence_id_map_orders_pair_before_supporting():
    pair_item = _evidence_item(chunk_id="pair1", classification=EvidenceClassification.PAIR_SPECIFIC)
    support_item = _evidence_item(chunk_id="support1", classification=EvidenceClassification.DRUG_SPECIFIC)
    result = _pair_result(pair_evidence=[pair_item], supporting_evidence=[support_item])

    evidence_map = build_evidence_id_map(result)

    assert list(evidence_map.keys()) == ["EVIDENCE-001", "EVIDENCE-002"]
    assert evidence_map["EVIDENCE-001"].chunk_id == "pair1"
    assert evidence_map["EVIDENCE-002"].chunk_id == "support1"


def test_build_evidence_id_map_respects_max_items_cap():
    items = [_evidence_item(chunk_id=f"c{i}") for i in range(15)]
    result = _pair_result(pair_evidence=items)

    evidence_map = build_evidence_id_map(result, max_items=5)

    assert len(evidence_map) == 5


def test_build_evidence_id_map_empty_when_no_evidence():
    result = _pair_result(evidence_status="insufficient_evidence")
    assert build_evidence_id_map(result) == {}


# ---------------------------------------------------------------------------
# Valid grounded response (LLM path succeeds)
# ---------------------------------------------------------------------------

def test_generate_explanation_valid_grounded_response():
    item = _evidence_item()
    pair_result = _pair_result(pair_evidence=[item])
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="pair_specific_evidence_found",
        cited_evidence_ids=["EVIDENCE-001"],
        severity=None,
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "llm_grounded"
    assert response.interaction_assessment == "pair_specific_evidence_found"
    assert len(response.cited_evidence) == 1
    assert response.cited_evidence[0].evidence_id == "EVIDENCE-001"
    assert response.cited_evidence[0].source_url == "https://dailymed.example/w-setid"
    assert response.fallback_reason is None


def test_generate_explanation_preserves_phase6_limitations():
    item = _evidence_item()
    pair_result = _pair_result(pair_evidence=[item])
    client = FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=["EVIDENCE-001"]))

    response = generate_explanation(pair_result, llm_client=client)

    assert "Standard limitation text." in response.limitations


# ---------------------------------------------------------------------------
# drug_not_found / invalid_input / retrieval_error never reach the LLM
# ---------------------------------------------------------------------------

def test_drug_not_found_never_calls_llm():
    pair_result = _pair_result(evidence_status="drug_not_found")
    client = FakeLLMClient(response_text=make_llm_json())

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert client.call_count == 0
    assert "drug_not_found" in response.fallback_reason


def test_retrieval_error_never_calls_llm():
    pair_result = _pair_result(evidence_status="retrieval_error")
    client = FakeLLMClient(response_text=make_llm_json())

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert client.call_count == 0


# ---------------------------------------------------------------------------
# LLM failure modes -> evidence-only fallback
# ---------------------------------------------------------------------------

def test_llm_timeout_falls_back_to_evidence_only():
    item = _evidence_item()
    pair_result = _pair_result(pair_evidence=[item])
    client = FakeLLMClient(raises=LLMTimeoutError("simulated timeout"))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert "timeout" in response.fallback_reason.lower() or "unavailable" in response.fallback_reason.lower()
    assert len(response.cited_evidence) >= 1  # evidence still shown


def test_llm_rate_limit_falls_back_to_evidence_only():
    pair_result = _pair_result(pair_evidence=[_evidence_item()])
    client = FakeLLMClient(raises=LLMRateLimitError("simulated rate limit"))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"


def test_llm_service_unavailable_falls_back_to_evidence_only():
    pair_result = _pair_result(pair_evidence=[_evidence_item()])
    client = FakeLLMClient(raises=LLMServiceUnavailableError("simulated 503"))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"


def test_malformed_json_falls_back_to_evidence_only():
    pair_result = _pair_result(pair_evidence=[_evidence_item()])
    client = FakeLLMClient(response_text="not valid json {{{")

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert "malformed" in response.fallback_reason.lower() or "json" in response.fallback_reason.lower() or "unavailable" in response.fallback_reason.lower()


# ---------------------------------------------------------------------------
# Validation failure -> evidence-only fallback
# ---------------------------------------------------------------------------

def test_hallucinated_citation_falls_back_to_evidence_only():
    pair_result = _pair_result(pair_evidence=[_evidence_item()])
    client = FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=["EVIDENCE-999"]))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert response.validation is not None
    assert response.validation.passed is False


def test_evidence_upgrade_falls_back_to_evidence_only():
    """LLM claims pair_specific but Phase 6 only found supporting evidence."""
    support_item = _evidence_item(classification=EvidenceClassification.DRUG_SPECIFIC)
    pair_result = _pair_result(evidence_status="supporting_evidence_found", supporting_evidence=[support_item])
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="pair_specific_evidence_found", cited_evidence_ids=["EVIDENCE-001"],
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert "upgrade" in response.fallback_reason.lower()


def test_no_interaction_language_falls_back_to_evidence_only():
    pair_result = _pair_result(pair_evidence=[_evidence_item()])
    client = FakeLLMClient(response_text=make_llm_json(
        evidence_summary="These two drugs do not interact.", cited_evidence_ids=["EVIDENCE-001"],
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"


# ---------------------------------------------------------------------------
# Unsupported severity -> non-fatal correction, LLM path still succeeds
# ---------------------------------------------------------------------------

def test_unsupported_severity_stripped_but_response_still_llm_grounded():
    item = _evidence_item(text="Warfarin interacts with ibuprofen, increasing bleeding risk.")
    pair_result = _pair_result(pair_evidence=[item])
    client = FakeLLMClient(response_text=make_llm_json(
        severity="High severity", cited_evidence_ids=["EVIDENCE-001"],
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "llm_grounded"  # not fatal, so LLM path still used
    assert response.severity is None  # but the unsupported severity is gone
    assert response.validation.corrections  # and it's recorded why


# ---------------------------------------------------------------------------
# Missing evidence (insufficient_evidence) never calls the LLM
# ---------------------------------------------------------------------------

def test_insufficient_evidence_bypasses_llm_with_empty_evidence():
    pair_result = _pair_result(evidence_status="insufficient_evidence")
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="insufficient_evidence", cited_evidence_ids=[],
        evidence_summary="No relevant evidence was retrieved for this pair.",
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert client.call_count == 0
    assert response.mode == "evidence_only"
    assert response.interaction_assessment == "insufficient_evidence"


# ---------------------------------------------------------------------------
# Evidence-only fallback structure
# ---------------------------------------------------------------------------

def test_evidence_only_response_still_has_full_citations():
    item = _evidence_item()
    pair_result = _pair_result(pair_evidence=[item])
    client = FakeLLMClient(raises=LLMServiceUnavailableError("down"))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert response.evidence_summary is None
    assert response.clinical_effect is None
    assert len(response.cited_evidence) == 1
    assert response.cited_evidence[0].text == item.text
    assert response.cited_evidence[0].source_url == item.source_url
    assert response.safety_notice  # always present, even in fallback
