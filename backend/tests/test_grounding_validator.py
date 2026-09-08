"""
tests/test_grounding_validator.py
====================================
Tests for app/services/grounding_validator.py — pure function tests,
constructing LLMStructuredOutput and evidence maps directly. No LLM, no
network, runs instantly offline.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_grounding_validator.py -v
"""

from app.schemas.evidence_assessment import DrugMentionResult, EvidenceClassification, PairEvidenceItem
from app.schemas.llm_output import LLMStructuredOutput
from app.services.grounding_validator import validate


def _evidence(text="Warfarin may increase bleeding risk when combined with ibuprofen.", classification=EvidenceClassification.PAIR_SPECIFIC):
    return PairEvidenceItem(
        chunk_id="c1", text=text, distance=0.1, drug_name="warfarin", rxcui="11289",
        section_name="Drug Interactions", section_code="34073-7",
        classification=classification, drug_mentions=DrugMentionResult(),
    )


def _output(**overrides):
    defaults = dict(
        interaction_assessment="pair_specific_evidence_found",
        evidence_summary="Evidence indicates increased bleeding risk.",
        clinical_effect=None, mechanism=None, severity=None,
        cited_evidence_ids=["EVIDENCE-001"], limitations=[], safety_notice="Consult a physician.",
    )
    defaults.update(overrides)
    return LLMStructuredOutput(**defaults)


# ---------------------------------------------------------------------------
# Test 5 (project spec): unsupported severity is stripped, not fatal
# ---------------------------------------------------------------------------

def test_unsupported_severity_is_stripped_not_fatal():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(severity="High severity")  # not present anywhere in the evidence text

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True
    assert result.corrected_severity is None
    assert any("severity" in c.lower() for c in result.corrections)


def test_supported_severity_survives_validation():
    evidence_map = {"EVIDENCE-001": _evidence(text="Warfarin is contraindicated with ibuprofen due to bleeding risk.")}
    llm_output = _output(severity="contraindicated")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True
    assert result.corrected_severity == "contraindicated"
    assert result.corrections == []


def test_null_severity_passes_through_unchanged():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(severity=None)

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True
    assert result.corrected_severity is None
    assert result.corrections == []


def test_not_stated_severity_string_treated_as_null():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(severity="Not stated in retrieved source")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True
    assert result.corrected_severity is None
    assert result.corrections == []  # already-null-equivalent, not a "correction"


# ---------------------------------------------------------------------------
# Test 6 (project spec): hallucinated citation -> fatal
# ---------------------------------------------------------------------------

def test_hallucinated_citation_id_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence(), "EVIDENCE-002": _evidence()}
    llm_output = _output(cited_evidence_ids=["EVIDENCE-001", "EVIDENCE-999"])

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False
    assert any("EVIDENCE-999" in issue for issue in result.fatal_issues)


def test_all_valid_citations_pass():
    evidence_map = {"EVIDENCE-001": _evidence(), "EVIDENCE-002": _evidence()}
    llm_output = _output(cited_evidence_ids=["EVIDENCE-001", "EVIDENCE-002"])

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True


# ---------------------------------------------------------------------------
# Evidence-strength upgrade prevention
# ---------------------------------------------------------------------------

def test_llm_claiming_pair_specific_when_retrieval_only_found_supporting_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence(classification=EvidenceClassification.DRUG_SPECIFIC)}
    llm_output = _output(interaction_assessment="pair_specific_evidence_found")

    result = validate(llm_output, evidence_map, "supporting_evidence_found")

    assert result.passed is False
    assert any("upgrade" in issue.lower() for issue in result.fatal_issues)


def test_llm_claiming_supporting_when_retrieval_found_pair_specific_is_allowed():
    """The LLM is allowed to be MORE conservative than retrieval, just never less."""
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(interaction_assessment="supporting_evidence_found")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True


def test_llm_matching_phase6_status_exactly_is_allowed():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(interaction_assessment="insufficient_evidence")

    result = validate(llm_output, evidence_map, "insufficient_evidence")

    assert result.passed is True


# ---------------------------------------------------------------------------
# "No interaction" language detection
# ---------------------------------------------------------------------------

def test_no_interaction_phrase_in_evidence_summary_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(evidence_summary="These drugs do not interact and are safe to combine.")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False
    assert any("no interaction" in issue.lower() for issue in result.fatal_issues)


def test_no_interaction_phrase_in_safety_notice_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(safety_notice="No known interaction exists between these two drugs.")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False


def test_no_interaction_phrase_in_clinical_effect_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(clinical_effect="No significant interaction was observed.")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False


def test_normal_uncertainty_language_is_not_flagged():
    """Sanity check: legitimate hedged language must NOT trigger the forbidden-phrase check."""
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(
        interaction_assessment="supporting_evidence_found",
        evidence_summary="The evidence found does not establish a pair-specific connection between these two drugs; "
        "only drug-specific evidence for each was retrieved.",
    )
    result = validate(llm_output, evidence_map, "supporting_evidence_found")
    assert result.passed is True


# ---------------------------------------------------------------------------
# Fabricated URL detection
# ---------------------------------------------------------------------------

def test_fabricated_url_in_evidence_summary_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(evidence_summary="See https://example.com/study for more details.")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False
    assert any("url" in issue.lower() for issue in result.fatal_issues)


def test_fabricated_www_reference_is_fatal():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(mechanism="Described further at www.pubmed.example/12345.")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is False


def test_no_url_present_passes():
    evidence_map = {"EVIDENCE-001": _evidence()}
    llm_output = _output(evidence_summary="A normal summary referencing EVIDENCE-001 only.")

    result = validate(llm_output, evidence_map, "pair_specific_evidence_found")

    assert result.passed is True


# ---------------------------------------------------------------------------
# Multiple simultaneous fatal issues
# ---------------------------------------------------------------------------

def test_multiple_fatal_issues_all_reported():
    evidence_map = {"EVIDENCE-001": _evidence(classification=EvidenceClassification.DRUG_SPECIFIC)}
    llm_output = _output(
        interaction_assessment="pair_specific_evidence_found",  # upgrade violation
        cited_evidence_ids=["EVIDENCE-999"],  # hallucinated id
        evidence_summary="These drugs do not interact.",  # forbidden phrase
    )

    result = validate(llm_output, evidence_map, "insufficient_evidence")

    assert result.passed is False
    assert len(result.fatal_issues) >= 2
