"""
services/explanation_service.py
==================================
Ties the full pipeline together for one drug pair:

    Phase 2 RxNorm normalization
        |
        v
    Phase 3/4 DailyMed evidence (SQLite)
        |
        v
    Phase 5 ChromaDB retrieval
        |
        v
    Phase 6 pair-aware evidence assessment (PairRetrievalResult)
        |
        v
    THIS FILE: build EVIDENCE-XXX ids -> call Gemini -> validate -> final response
        |
        v
    ExplanationResponse (mode="llm_grounded" or mode="evidence_only")

THE LLM IS AN EXPLANATION LAYER, NOT THE DETECTION ENGINE
------------------------------------------------------------
Phase 6 already decided what evidence exists and how it's classified.
This file never lets the LLM override that: `generate_explanation()`
passes Phase 6's `evidence_status` into the validator as a ceiling the
LLM's own claimed assessment can never exceed (see
grounding_validator.py). If Phase 6 found nothing pair-specific, no
amount of eloquent LLM prose can turn it into a pair-specific claim that
survives validation.

WHY EVIDENCE-ONLY FALLBACK EXISTS, AND WHEN IT'S USED
------------------------------------------------------------
Per the project's core safety requirement: a broken or untrustworthy LLM
response must never silently become an invented-looking answer. Every
failure mode — missing/invalid API key, timeout, rate limit, service
unavailable, malformed JSON, or a validation FATAL issue — routes to the
exact same `build_evidence_only_response()`, which shows only what
Phase 6 actually retrieved (with full citations), no generated prose,
plus a `fallback_reason` explaining exactly why. The user always sees
something honest and traceable; never nothing, and never a guess dressed
up as an answer.

drug_not_found, invalid_input, and retrieval_error (Phase 6 statuses) never
even reach the LLM — there's nothing meaningful to explain in these cases, so
`generate_explanation()` short-circuits straight to the fallback path.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_explanation_service.py -v
    pytest tests/test_safety_regression.py -v

Uses a FakeLLMClient (tests/_fake_llm.py) and directly-constructed
PairRetrievalResult fixtures — no live Gemini call, no live retrieval
pipeline — to prove the ORCHESTRATION and FALLBACK logic in isolation.
"""

from __future__ import annotations

from app.exceptions import LLMServiceError
from app.schemas.evidence_assessment import PairEvidenceItem, PairRetrievalResult
from app.schemas.explanation import EvidenceCitation, ExplanationResponse, ValidationResult
from app.services.grounding_validator import validate
from app.services.llm_service import LLMClient, generate_grounded_response

_DEFAULT_SAFETY_NOTICE = (
    "This is an evidence-grounded information tool, not a substitute for professional clinical "
    "judgment. Verify any potential interaction with a pharmacist or physician before making "
    "treatment decisions."
)

# Statuses that never reach the LLM at all — nothing meaningful to explain.
_NO_LLM_STATUSES = {"drug_not_found", "invalid_input", "retrieval_error", "insufficient_evidence"}


def build_evidence_id_map(
    pair_result: PairRetrievalResult, max_items: int | None = None
) -> dict[str, PairEvidenceItem]:
    """
    Assigns stable EVIDENCE-NNN ids to the pair's evidence, pair_evidence
    first (strongest signal) then supporting_evidence, capped at
    max_items (defaults to settings.LLM_MAX_EVIDENCE_ITEMS) to keep
    prompts bounded. IDs are assigned in this fixed order every time for
    the same PairRetrievalResult, so they're stable/reproducible for a
    given retrieval — not randomly reshuffled between calls.
    """
    if max_items is None:
        from app.config import settings

        max_items = settings.LLM_MAX_EVIDENCE_ITEMS

    items = list(pair_result.pair_evidence) + list(pair_result.supporting_evidence)
    items = items[:max_items]
    return {f"EVIDENCE-{i + 1:03d}": item for i, item in enumerate(items)}


def _citation_from_item(evidence_id: str, item: PairEvidenceItem) -> EvidenceCitation:
    return EvidenceCitation(
        evidence_id=evidence_id,
        text=item.text,
        drug_name=item.drug_name,
        rxcui=item.rxcui,
        section_name=item.section_name,
        section_code=item.section_code,
        manufacturer=item.manufacturer,
        source_url=item.source_url,
        classification=item.classification.value,
    )


def build_evidence_only_response(
    pair_result: PairRetrievalResult,
    reason: str,
    validation: ValidationResult | None = None,
) -> ExplanationResponse:
    """
    The safe fallback: shows exactly what Phase 6 retrieved, with full
    citations, no LLM-generated prose, and an explicit reason. Used for
    every failure mode — see module docstring.
    """
    evidence_map = build_evidence_id_map(pair_result)
    cited = [_citation_from_item(eid, item) for eid, item in evidence_map.items()]

    return ExplanationResponse(
        drug_a=pair_result.drug_a,
        drug_b=pair_result.drug_b,
        mode="evidence_only",
        interaction_assessment=pair_result.evidence_status,
        evidence_summary=None,
        clinical_effect=None,
        mechanism=None,
        severity=None,
        cited_evidence=cited,
        limitations=list(pair_result.limitations),
        safety_notice=_DEFAULT_SAFETY_NOTICE,
        fallback_reason=reason,
        validation=validation,
    )


def generate_explanation(
    pair_result: PairRetrievalResult,
    llm_client: LLMClient | None = None,
) -> ExplanationResponse:
    """
    Main entry point. Never raises — every failure mode (missing
    evidence, LLM unavailable, validation failure) resolves to a valid
    ExplanationResponse, either llm_grounded or evidence_only.
    """
    if pair_result.evidence_status in _NO_LLM_STATUSES:
        return build_evidence_only_response(
            pair_result,
            reason=(
                "Cannot generate an explanation: no retrieved evidence was available."
                if pair_result.evidence_status == "insufficient_evidence"
                else f"Cannot generate an explanation: retrieval result was '{pair_result.evidence_status}'."
            ),
        )

    evidence_map = build_evidence_id_map(pair_result)
    drug_a_name = pair_result.drug_a.normalized_name or pair_result.drug_a.input_name
    drug_b_name = pair_result.drug_b.normalized_name or pair_result.drug_b.input_name

    try:
        llm_output = generate_grounded_response(drug_a_name, drug_b_name, evidence_map, client=llm_client)
    except LLMServiceError as exc:
        return build_evidence_only_response(pair_result, reason=f"LLM unavailable ({exc.source}): {exc}")

    validation = validate(llm_output, evidence_map, pair_result.evidence_status)
    if not validation.passed:
        return build_evidence_only_response(
            pair_result,
            reason="The model's response failed safety validation and was discarded: " + "; ".join(validation.fatal_issues),
            validation=validation,
        )

    cited = [
        _citation_from_item(eid, evidence_map[eid]) for eid in llm_output.cited_evidence_ids if eid in evidence_map
    ]

    return ExplanationResponse(
        drug_a=pair_result.drug_a,
        drug_b=pair_result.drug_b,
        mode="llm_grounded",
        interaction_assessment=llm_output.interaction_assessment,
        evidence_summary=llm_output.evidence_summary,
        clinical_effect=llm_output.clinical_effect,
        mechanism=llm_output.mechanism,
        severity=validation.corrected_severity,
        cited_evidence=cited,
        limitations=list(pair_result.limitations) + list(llm_output.limitations),
        safety_notice=llm_output.safety_notice or _DEFAULT_SAFETY_NOTICE,
        fallback_reason=None,
        validation=validation,
    )
