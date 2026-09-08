"""
schemas/explanation.py
=========================
The final, user-facing response schema for Phase 7 — either an
LLM-grounded explanation that passed safety validation, or a safe
evidence-only fallback. Both modes use the SAME schema, so callers don't
need to branch on which happened; `mode` and `fallback_reason` make the
distinction explicit and honest instead of hiding it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.evidence_assessment import DrugRef, PairEvidenceStatus

ExplanationMode = Literal["llm_grounded", "evidence_only"]


class EvidenceCitation(BaseModel):
    """One evidence item, resolved from its EVIDENCE-XXX id back to full source detail."""

    evidence_id: str
    text: str
    drug_name: str | None = None
    rxcui: str | None = None
    section_name: str | None = None
    section_code: str | None = None
    manufacturer: str | None = Field(
        default=None,
        description="Labeler/manufacturer of the source label, if known (Phase 3/4 provenance) — "
        "added in Phase 9 so the frontend's evidence cards can display it per the UI spec, "
        "without any change to how evidence is retrieved or classified.",
    )
    source_url: str | None = None
    classification: str = Field(description="pair_specific_evidence / drug_specific_evidence / class_level_evidence / general_label_evidence")


class ValidationResult(BaseModel):
    """The result of running the grounding validator over one LLM response."""

    passed: bool
    fatal_issues: list[str] = Field(
        default_factory=list, description="Issues that caused the response to be rejected entirely (triggers evidence-only fallback)."
    )
    corrections: list[str] = Field(
        default_factory=list, description="Non-fatal fixes applied to the response (e.g. an unsupported severity was stripped)."
    )
    corrected_severity: str | None = Field(
        default=None, description="The severity value to actually use downstream, after stripping any unsupported claim."
    )


class ExplanationResponse(BaseModel):
    """
    The final response for a drug pair, in EITHER mode. `evidence_summary`,
    `clinical_effect`, and `mechanism` are None in evidence_only mode —
    the caller/UI should render `cited_evidence` (or the full retrieval
    result) directly rather than expect prose in that mode.
    """

    drug_a: DrugRef
    drug_b: DrugRef

    mode: ExplanationMode
    interaction_assessment: PairEvidenceStatus = Field(
        description="pair_specific_evidence_found / supporting_evidence_found / insufficient_evidence "
        "(or drug_not_found / invalid_input / retrieval_error, passed through from Phase 6 when no LLM call was even attempted)."
    )

    evidence_summary: str | None = None
    clinical_effect: str | None = None
    mechanism: str | None = None
    severity: str | None = Field(
        default=None, description="Source-supported severity wording only, or null — never an invented/standardized category."
    )

    cited_evidence: list[EvidenceCitation] = Field(
        default_factory=list, description="Evidence actually cited (llm_grounded mode) or all available evidence (evidence_only mode)."
    )

    limitations: list[str] = Field(default_factory=list)
    safety_notice: str

    fallback_reason: str | None = Field(
        default=None, description="Set only when mode == 'evidence_only' — exactly why the LLM path wasn't used or was rejected."
    )
    validation: ValidationResult | None = Field(
        default=None, description="Attached for transparency/debugging — the validator's findings, when an LLM response was actually checked."
    )
