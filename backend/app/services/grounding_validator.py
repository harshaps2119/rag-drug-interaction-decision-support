"""
services/grounding_validator.py
==================================
Runs AFTER Gemini, BEFORE anything from the LLM is shown to a user.
This is the actual safety enforcement layer — the prompt in
llm_service.py asks Gemini nicely to follow the rules; this module
checks, mechanically and independently, that it actually did, and
refuses to pass through anything that didn't.

WHAT IS CHECKED, AND WHY EACH ONE MATTERS
----------------------------------------------
1. EVERY CITED EVIDENCE ID MUST EXIST. If Gemini cites EVIDENCE-999 but
   only EVIDENCE-001/002 were ever supplied, that's a hallucinated
   citation — FATAL, triggers evidence-only fallback. There is no
   partial-trust option here: a model willing to invent one citation ID
   cannot be trusted on the surrounding claims either.

2. THE CLAIMED interaction_assessment MUST NOT EXCEED WHAT PHASE 6 ACTUALLY
   FOUND. If Phase 6's retrieval only supports "supporting_evidence_found"
   (no pair-specific evidence at all), the LLM is not allowed to report
   "pair_specific_evidence_found" — that would be the LLM inventing an
   interaction connection retrieval never established. FATAL.

3. NO "NO INTERACTION" LANGUAGE, ANYWHERE IN THE FREE TEXT. Phase 6 never
   offers "no interaction" as an outcome, and the LLM must not either —
   not as the assessment field (impossible; not a valid enum value) and
   not smuggled into prose ("these drugs do not interact", "safe to
   combine", etc.). A simple phrase-matching check catches the most
   direct forms. FATAL if found.

4. NO FABRICATED URLS. The model was told to cite only EVIDENCE-XXX ids.
   Any URL-shaped text anywhere in its free-text fields means it ignored
   that instruction. FATAL.

5. SEVERITY MUST BE VERBATIM-SUPPORTED BY THE CITED EVIDENCE TEXT. This
   one is NOT fatal — it's corrected: an unsupported severity claim is
   stripped to None (with a recorded correction), and the rest of the
   response can still be shown. This asymmetry is deliberate: a
   confidently wrong evidence_summary or an invented pair-connection is a
   structural trust failure worth discarding the whole response over; an
   overreaching one-word severity claim is a narrower, correctable
   mistake, and stripping just that field is safer AND more useful than
   discarding an otherwise-good grounded explanation. See
    docs/safety.md for the full reasoning on this split.

6. PAIR-SPECIFIC EFFECT AND MECHANISM CLAIMS MUST HAVE PAIR-SPECIFIC
   SUPPORT. Supporting, class-level, and general evidence cannot establish
   a claim about this specific drug pair, so such claims are fatal and the
   response falls back to evidence-only output.

WHAT THIS MODULE DOES NOT DO
---------------------------------
It does not check whether the evidence_summary is a GOOD summary, or
whether the mechanism explanation is medically accurate. Judging
quality/accuracy of grounded prose is not this module's job and not
something a mechanical validator can safely attempt — it only checks
the specific, checkable safety properties listed above.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_grounding_validator.py -v

Pure function tests, constructing LLMStructuredOutput and evidence maps
directly — no LLM, no network, runs instantly offline.
"""

from __future__ import annotations

import re

from app.schemas.evidence_assessment import PairEvidenceItem
from app.schemas.explanation import ValidationResult
from app.schemas.llm_output import LLMStructuredOutput

# Ordinal strength of each possible assessment — used to check the LLM
# never claims MORE than what Phase 6's retrieval actually supports.
_ASSESSMENT_RANK = {
    "insufficient_evidence": 0,
    "supporting_evidence_found": 1,
    "pair_specific_evidence_found": 2,
}

# Direct, common phrasings asserting the ABSENCE of an interaction.
# Deliberately not exhaustive (see docs/safety.md) — this is a mechanical
# safety net, not a substitute for the prompt-level instruction.
_FORBIDDEN_PHRASES = [
    "no interaction exists",
    "no known interaction",
    "does not interact",
    "do not interact",
    "safe to combine",
    "safe to take together",
    "safe to use together",
    "no interaction between",
    "there is no interaction",
    "not expected to interact",
    "no significant interaction",
    "these drugs are safe",
]

_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)

_NULL_SEVERITY_STRINGS = {"", "null", "none", "not stated in retrieved source", "not stated", "n/a"}


def validate(
    llm_output: LLMStructuredOutput,
    evidence_map: dict[str, PairEvidenceItem],
    phase6_evidence_status: str,
) -> ValidationResult:
    """
    Validates one LLM response against the evidence it was actually given
    and the evidence_status Phase 6's retrieval independently established.
    Returns a ValidationResult; callers must check `.passed` before
    displaying anything from `llm_output` — see services/explanation_service.py.
    """
    fatal: list[str] = []
    corrections: list[str] = []

    # 1. Every cited evidence id must exist in what was actually supplied.
    unknown_ids = [eid for eid in llm_output.cited_evidence_ids if eid not in evidence_map]
    if unknown_ids:
        fatal.append(
            f"Cited unknown evidence id(s) not present in the supplied evidence: {unknown_ids}. "
            "This is treated as a hallucinated citation."
        )

    # 2. Claimed assessment must not exceed what Phase 6 actually found.
    llm_rank = _ASSESSMENT_RANK.get(llm_output.interaction_assessment, 99)
    phase6_rank = _ASSESSMENT_RANK.get(phase6_evidence_status, -1)
    if llm_rank > phase6_rank:
        fatal.append(
            f"Model claimed interaction_assessment='{llm_output.interaction_assessment}' but retrieval only "
            f"supports '{phase6_evidence_status}' — refusing to let the model upgrade evidence strength."
        )

    # 3. No "no interaction" language anywhere in the free-text fields.
    combined_text = " ".join(
        filter(None, [llm_output.evidence_summary, llm_output.clinical_effect, llm_output.mechanism, llm_output.safety_notice])
    ).lower()
    matched_phrase = next((p for p in _FORBIDDEN_PHRASES if p in combined_text), None)
    if matched_phrase:
        fatal.append(f"Response contains prohibited 'no interaction' language: '{matched_phrase}'.")

    # 4. No fabricated URLs/citations anywhere in the free text.
    if combined_text and _URL_PATTERN.search(combined_text):
        fatal.append("Response contains a URL, which is not allowed — evidence must be cited by EVIDENCE-XXX id only.")

    # 5. Pair-specific prose requires a cited pair-specific evidence item.
    cited_pair_specific = any(
        evidence_map[eid].classification.value == "pair_specific_evidence"
        for eid in llm_output.cited_evidence_ids
        if eid in evidence_map
    )
    if not cited_pair_specific:
        unsupported_claims = [
            field
            for field, value in (("clinical_effect", llm_output.clinical_effect), ("mechanism", llm_output.mechanism))
            if value and value.strip()
        ]
        if unsupported_claims:
            fatal.append(
                "Pair-specific claim(s) in " + ", ".join(unsupported_claims)
                + " require at least one cited pair-specific evidence item; "
                "supporting/class-level/general evidence is insufficient."
            )

    # 6. Severity must be verbatim-supported by the CITED evidence text. Non-fatal: stripped, not rejected.
    corrected_severity = _validate_severity(llm_output, evidence_map, corrections)

    return ValidationResult(
        passed=len(fatal) == 0,
        fatal_issues=fatal,
        corrections=corrections,
        corrected_severity=corrected_severity if len(fatal) == 0 else None,
    )


def _validate_severity(
    llm_output: LLMStructuredOutput, evidence_map: dict[str, PairEvidenceItem], corrections: list[str]
) -> str | None:
    severity = llm_output.severity
    if severity is None or severity.strip().lower() in _NULL_SEVERITY_STRINGS:
        return None

    cited_texts = " ".join(
        evidence_map[eid].text.lower() for eid in llm_output.cited_evidence_ids if eid in evidence_map
    )
    if severity.strip().lower() not in cited_texts:
        corrections.append(
            f"Severity '{severity}' was not found verbatim in the cited evidence text — stripped to "
            "'Not stated in retrieved source' rather than shown unsupported."
        )
        return None

    return severity
