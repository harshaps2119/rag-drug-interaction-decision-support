"""
tests/test_safety_regression.py
==================================
The six explicit safety regression tests specified for Phase 7, each
mapped directly to its numbered requirement. These are deliberately
kept separate from test_explanation_service.py / test_grounding_validator.py
(which cover the same mechanisms more exhaustively) so each one stands
as an unambiguous, individually-named regression check that a reviewer
can find and re-run in isolation.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_safety_regression.py -v
"""

from app.schemas.evidence_assessment import (
    DrugMentionResult,
    DrugRef,
    EvidenceClassification,
    PairEvidenceItem,
    PairRetrievalResult,
)
from app.services.explanation_service import generate_explanation
from app.services.grounding_validator import validate
from tests._fake_llm import FakeLLMClient, make_llm_json


def _drug_ref(name, rxcui):
    return DrugRef(input_name=name, rxcui=rxcui, normalized_name=name, resolved=True, search_terms=[name])


def _item(classification, text, drug_name="warfarin", rxcui="11289"):
    return PairEvidenceItem(
        chunk_id="c1", text=text, distance=0.1, drug_name=drug_name, rxcui=rxcui,
        section_name="Drug Interactions", section_code="34073-7",
        classification=classification, drug_mentions=DrugMentionResult(),
    )


# ---------------------------------------------------------------------------
# Test 1: Drug A label discusses Drug A only; Drug B unrelated -> NOT pair-specific
# ---------------------------------------------------------------------------

def test_1_drug_specific_only_evidence_is_not_reported_as_pair_specific():
    item = _item(EvidenceClassification.DRUG_SPECIFIC, "Warfarin requires regular INR monitoring.")
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("metformin", "6809"),
        evidence_status="supporting_evidence_found", supporting_evidence=[item],
    )
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="supporting_evidence_found", cited_evidence_ids=["EVIDENCE-001"],
        clinical_effect=None, mechanism=None,
        evidence_summary="Only drug-specific evidence about warfarin was found; no connection to metformin.",
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.interaction_assessment != "pair_specific_evidence_found"
    assert response.mode == "llm_grounded"  # a well-behaved response should still pass


# ---------------------------------------------------------------------------
# Test 2: Drug A label explicitly mentions Drug B -> pair-specific MAY be reported
# ---------------------------------------------------------------------------

def test_2_explicit_cross_mention_allows_pair_specific_report():
    item = _item(EvidenceClassification.PAIR_SPECIFIC, "Warfarin combined with ibuprofen may increase bleeding risk.")
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("ibuprofen", "5640"),
        evidence_status="pair_specific_evidence_found", pair_evidence=[item],
    )
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="pair_specific_evidence_found", cited_evidence_ids=["EVIDENCE-001"],
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "llm_grounded"
    assert response.interaction_assessment == "pair_specific_evidence_found"


# ---------------------------------------------------------------------------
# Test 3: Drug A label discusses a class containing Drug B -> class-level, NOT auto pair-specific
# ---------------------------------------------------------------------------

def test_3_class_level_evidence_is_not_automatically_pair_specific():
    item = _item(EvidenceClassification.CLASS_LEVEL, "Increased bleeding risk when combined with NSAIDs.")
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("ibuprofen", "5640"),
        evidence_status="supporting_evidence_found", supporting_evidence=[item],
    )
    # A misbehaving LLM tries to upgrade class-level to pair-specific -- must be rejected.
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="pair_specific_evidence_found", cited_evidence_ids=["EVIDENCE-001"],
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"  # rejected -- forced safe fallback
    assert response.interaction_assessment != "pair_specific_evidence_found"


def test_3b_well_behaved_llm_correctly_reports_class_level_as_supporting():
    item = _item(EvidenceClassification.CLASS_LEVEL, "Increased bleeding risk when combined with NSAIDs.")
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("ibuprofen", "5640"),
        evidence_status="supporting_evidence_found", supporting_evidence=[item],
    )
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="supporting_evidence_found", cited_evidence_ids=["EVIDENCE-001"],
        clinical_effect=None, mechanism=None,
        evidence_summary="Only class-level evidence (NSAIDs generally) was found -- not specific to ibuprofen by name.",
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "llm_grounded"
    assert response.interaction_assessment == "supporting_evidence_found"


def test_3c_supporting_only_pair_specific_prose_fails_validation():
    item = _item(EvidenceClassification.DRUG_SPECIFIC, "Warfarin requires regular INR monitoring.")
    from app.schemas.llm_output import LLMStructuredOutput

    result = validate(
        LLMStructuredOutput(
            interaction_assessment="supporting_evidence_found",
            evidence_summary="Warfarin evidence was retrieved.",
            clinical_effect="The pair increases bleeding risk.",
            mechanism="The combination inhibits platelet function.",
            cited_evidence_ids=["EVIDENCE-001"],
            safety_notice="Consult a physician.",
        ),
        {"EVIDENCE-001": item},
        "supporting_evidence_found",
    )

    assert result.passed is False
    assert any("pair-specific" in issue.lower() for issue in result.fatal_issues)


def test_3d_pair_specific_evidence_permits_effect_and_mechanism_claims():
    item = _item(EvidenceClassification.PAIR_SPECIFIC, "Warfarin combined with ibuprofen may increase bleeding risk.")
    from app.schemas.llm_output import LLMStructuredOutput

    result = validate(
        LLMStructuredOutput(
            interaction_assessment="pair_specific_evidence_found",
            evidence_summary="The pair is described in the evidence.",
            clinical_effect="Increased bleeding risk.",
            mechanism="Additive anticoagulant effects.",
            cited_evidence_ids=["EVIDENCE-001"],
            safety_notice="Consult a physician.",
        ),
        {"EVIDENCE-001": item},
        "pair_specific_evidence_found",
    )

    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 4: No relevant evidence -> insufficient_evidence, NEVER "no interaction exists"
# ---------------------------------------------------------------------------

def test_4_no_evidence_yields_insufficient_not_no_interaction():
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("metformin", "6809"),
        evidence_status="insufficient_evidence",
    )
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="insufficient_evidence", cited_evidence_ids=[],
        evidence_summary="No relevant evidence was retrieved for this drug pair.",
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.interaction_assessment == "insufficient_evidence"
    assert "no interaction exists" not in (response.evidence_summary or "").lower()
    assert "no interaction exists" not in (response.safety_notice or "").lower()


def test_4b_misbehaving_llm_claiming_no_interaction_is_rejected():
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("metformin", "6809"),
        evidence_status="insufficient_evidence",
    )
    client = FakeLLMClient(response_text=make_llm_json(
        interaction_assessment="insufficient_evidence", cited_evidence_ids=[],
        evidence_summary="No interaction exists between warfarin and metformin.",
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"  # forced safe fallback, the claim never reaches the user


# ---------------------------------------------------------------------------
# Test 5: Gemini says "High severity" but evidence doesn't state severity -> stripped
# ---------------------------------------------------------------------------

def test_5_unsupported_severity_claim_is_rejected_by_validator():
    item = _item(EvidenceClassification.PAIR_SPECIFIC, "Warfarin combined with ibuprofen may increase bleeding risk.")
    evidence_map = {"EVIDENCE-001": item}
    from app.schemas.llm_output import LLMStructuredOutput

    llm_output = LLMStructuredOutput(
        interaction_assessment="pair_specific_evidence_found",
        evidence_summary="Evidence indicates increased bleeding risk.",
        severity="High severity",  # NOT present anywhere in the evidence text
        cited_evidence_ids=["EVIDENCE-001"],
        safety_notice="Consult a physician.",
    )

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True  # non-fatal -- rest of response still usable
    assert result.corrected_severity is None  # but the unsupported severity is gone
    assert any("severity" in c.lower() and "stripped" in c.lower() for c in result.corrections)


def test_5b_end_to_end_unsupported_severity_never_reaches_user():
    item = _item(EvidenceClassification.PAIR_SPECIFIC, "Warfarin combined with ibuprofen may increase bleeding risk.")
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("ibuprofen", "5640"),
        evidence_status="pair_specific_evidence_found", pair_evidence=[item],
    )
    client = FakeLLMClient(response_text=make_llm_json(
        severity="High severity", cited_evidence_ids=["EVIDENCE-001"],
    ))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.severity is None
    assert response.mode == "llm_grounded"  # response itself still usable, just without the bad severity


# ---------------------------------------------------------------------------
# Test 6: Gemini cites EVIDENCE-999 but only EVIDENCE-001/002 exist -> validation failure
# ---------------------------------------------------------------------------

def test_6_citation_of_nonexistent_evidence_id_fails_validation():
    item1 = _item(EvidenceClassification.PAIR_SPECIFIC, "Warfarin combined with ibuprofen may increase bleeding risk.")
    item2 = _item(EvidenceClassification.DRUG_SPECIFIC, "Warfarin requires INR monitoring.")
    evidence_map = {"EVIDENCE-001": item1, "EVIDENCE-002": item2}
    from app.schemas.llm_output import LLMStructuredOutput

    llm_output = LLMStructuredOutput(
        interaction_assessment="pair_specific_evidence_found",
        evidence_summary="Evidence indicates increased bleeding risk.",
        cited_evidence_ids=["EVIDENCE-001", "EVIDENCE-999"],  # EVIDENCE-999 does not exist
        safety_notice="Consult a physician.",
    )

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False
    assert any("EVIDENCE-999" in issue for issue in result.fatal_issues)


def test_6b_end_to_end_hallucinated_citation_forces_evidence_only_fallback():
    item1 = _item(EvidenceClassification.PAIR_SPECIFIC, "Warfarin combined with ibuprofen may increase bleeding risk.")
    pair_result = PairRetrievalResult(
        drug_a=_drug_ref("warfarin", "11289"), drug_b=_drug_ref("ibuprofen", "5640"),
        evidence_status="pair_specific_evidence_found", pair_evidence=[item1],
    )
    client = FakeLLMClient(response_text=make_llm_json(cited_evidence_ids=["EVIDENCE-001", "EVIDENCE-999"]))

    response = generate_explanation(pair_result, llm_client=client)

    assert response.mode == "evidence_only"
    assert response.validation.passed is False
    assert any("EVIDENCE-999" in issue for issue in response.validation.fatal_issues)
