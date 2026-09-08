"""
schemas/llm_output.py
========================
The EXACT schema Gemini is instructed to return, before any safety
validation happens. This is deliberately narrow: the LLM is only allowed
to produce an assessment label, some short prose fields, a severity
string (or null), a list of EVIDENCE-XXX ids it's citing, limitations,
and a safety notice. It cannot invent structure beyond this.

WHY interaction_assessment EXCLUDES "no_interaction" AND NON-LLM STATUSES
------------------------------------------------------------------------------------------------
The three values here mirror Phase 6's PairEvidenceStatus, minus the three
values that never reach the LLM at all (drug_not_found, invalid_input, retrieval_error
are handled before an LLM call is ever made — see
services/explanation_service.py). "no interaction" was never a Phase 6
value and is not one here either — the LLM has no path to assert it
through this schema. If Gemini's raw JSON contains any other string for
this field, Pydantic validation fails and the whole response is treated
as malformed output (see services/llm_service.py), NOT silently coerced.

WHY severity IS A FREE STRING, NOT AN ENUM
-----------------------------------------------
The project requirement is that severity be reported using the SOURCE's
own wording (e.g. "Contraindicated") when present, never mapped to a
standardized category we invented. An enum would force exactly the kind
of invented categorization this project's requirements explicitly
prohibit. The grounding validator (services/grounding_validator.py)
checks that whatever string Gemini provides here actually appears in the
cited evidence text — if not, it's stripped to None, never displayed
unsupported.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LLMInteractionAssessment = Literal[
    "pair_specific_evidence_found",
    "supporting_evidence_found",
    "insufficient_evidence",
]


class LLMStructuredOutput(BaseModel):
    interaction_assessment: LLMInteractionAssessment

    evidence_summary: str = Field(
        description="A short, plain-language summary of what the cited evidence actually says. "
        "Must explicitly distinguish pair-specific evidence from drug-specific/class-level evidence."
    )
    clinical_effect: str | None = Field(
        default=None, description="What the cited evidence says the effect would be, if stated. Null if not stated."
    )
    mechanism: str | None = Field(
        default=None, description="What the cited evidence says the mechanism is, if stated. Null if not stated."
    )
    severity: str | None = Field(
        default=None,
        description="A short phrase copied from the cited evidence (e.g. 'Contraindicated'), or null if the "
        "evidence does not explicitly state a severity. Never a standardized category invented by the model.",
    )

    cited_evidence_ids: list[str] = Field(
        default_factory=list, description="EVIDENCE-XXX ids (from the supplied evidence list) that support the claims above."
    )
    limitations: list[str] = Field(
        default_factory=list, description="Any caveats the model wants to surface about the evidence it was given."
    )
    safety_notice: str = Field(
        description="A brief, standard note that this is not a substitute for professional clinical judgment."
    )
