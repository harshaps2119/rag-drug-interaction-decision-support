"""
tests/_fake_llm.py
=====================
NOT a test file (no test_ prefix, so pytest won't collect it) — a shared
test helper implementing the same LLMClient interface (`.generate(prompt) -> str`)
as GeminiLLMClient, without any network call or real model.

Used throughout tests/test_llm_service.py, tests/test_explanation_service.py,
and tests/test_safety_regression.py to test prompt building, response
parsing, and the full explanation pipeline's ORCHESTRATION and SAFETY
logic — deterministically and offline.
"""

from __future__ import annotations

import json


class FakeLLMClient:
    """Returns a fixed string (or raises a fixed exception) every time .generate() is called."""

    def __init__(self, response_text: str | None = None, raises: Exception | None = None):
        self.response_text = response_text
        self.raises = raises
        self.last_prompt: str | None = None
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        self.call_count += 1
        if self.raises is not None:
            raise self.raises
        return self.response_text


def make_llm_json(
    *,
    interaction_assessment: str = "pair_specific_evidence_found",
    evidence_summary: str = "The evidence indicates a potential interaction.",
    clinical_effect: str | None = "Increased bleeding risk.",
    mechanism: str | None = None,
    severity: str | None = None,
    cited_evidence_ids: list[str] | None = None,
    limitations: list[str] | None = None,
    safety_notice: str = "Consult a pharmacist or physician before making treatment decisions.",
) -> str:
    """Builds a valid (unless the caller deliberately breaks it) JSON string matching LLMStructuredOutput."""
    payload = {
        "interaction_assessment": interaction_assessment,
        "evidence_summary": evidence_summary,
        "clinical_effect": clinical_effect,
        "mechanism": mechanism,
        "severity": severity,
        "cited_evidence_ids": cited_evidence_ids or [],
        "limitations": limitations or [],
        "safety_notice": safety_notice,
    }
    return json.dumps(payload)
