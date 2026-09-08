"""
scripts/test_gemini_live.py
==============================
Standalone verification that the REAL Gemini API works end-to-end with
this project's actual prompt and schema — not mocked.

>>> NOT EXECUTED BY Claude <<<
This sandbox cannot reach generativelanguage.googleapis.com. A direct
test was attempted during this project's build (using google-generativeai's
default gRPC transport) and HUNG past a 15-second bash-level timeout
rather than failing cleanly — the network here doesn't just return an
error for this host, the connection attempt itself doesn't complete.
That is exactly why services/llm_service.py uses transport="rest" (see
that file's module docstring) — REST at least fails/succeeds within the
configured timeout rather than hanging indefinitely, which matters for
real deployments behind restrictive proxies too, not just this sandbox.

Because of this, NOTHING about actual Gemini API behavior — whether a
real key works, what real latency looks like, or how the model responds
to this project's actual prompt — has been verified by Claude. Run this
yourself, with a real GEMINI_API_KEY in your .env, on a network that can
reach Google's API.

PREREQUISITE
--------------
    1. Get a free key at https://aistudio.google.com/apikey
    2. Put it in backend/../.env as GEMINI_API_KEY=your-key-here
       (copy .env.example to .env first if you haven't already)

HOW TO RUN
-----------
    cd backend
    python ../scripts/test_gemini_live.py

EXPECTED OUTPUT (what you SHOULD see)
------------------------------------------
    Calling the real Gemini API...
    Raw response received (first 300 chars):
    {"interaction_assessment": "pair_specific_evidence_found", ...

    Parsed successfully into LLMStructuredOutput:
      interaction_assessment: pair_specific_evidence_found
      evidence_summary: ...
      severity: ... (or None)
      cited_evidence_ids: ['EVIDENCE-001']

    Running grounding validation...
      passed: True
      corrections: []

If parsing or validation fails, that's genuinely useful signal — it
means either the prompt needs tightening for this specific model
version, or the model didn't follow instructions on this run. Compare
against docs/llm.md and docs/safety.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.schemas.evidence_assessment import DrugMentionResult, EvidenceClassification, PairEvidenceItem  # noqa: E402
from app.services.explanation_service import build_evidence_id_map  # noqa: E402
from app.services.grounding_validator import validate  # noqa: E402
from app.services.llm_service import GeminiLLMClient, generate_grounded_response  # noqa: E402
from app.schemas.evidence_assessment import PairRetrievalResult, DrugRef  # noqa: E402


def main() -> None:
    if not settings.GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set in your .env file. See this script's docstring.")
        sys.exit(1)

    # A small, realistic evidence set -- deliberately similar to what
    # Phase 6's real retrieval would produce for warfarin + ibuprofen.
    sample_item = PairEvidenceItem(
        chunk_id="link-1-section-1-chunk-0",
        text="Concomitant use of warfarin with NSAIDs such as ibuprofen may increase the risk of bleeding. "
             "Monitor INR closely if co-administration cannot be avoided.",
        distance=0.12,
        drug_name="warfarin", rxcui="11289",
        label_setid="sample-setid", spl_version="1",
        section_name="Drug Interactions", section_code="34073-7",
        source_url="https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=sample-setid",
        classification=EvidenceClassification.PAIR_SPECIFIC,
        drug_mentions=DrugMentionResult(mentions_drug_a=True, mentions_drug_b=True, matched_terms_b=["ibuprofen"]),
        evidence_status_of_label="interaction_evidence_found",
    )
    pair_result = PairRetrievalResult(
        drug_a=DrugRef(input_name="warfarin", rxcui="11289", normalized_name="warfarin", resolved=True),
        drug_b=DrugRef(input_name="ibuprofen", rxcui="5640", normalized_name="ibuprofen", resolved=True),
        evidence_status="pair_specific_evidence_found",
        pair_evidence=[sample_item],
    )
    evidence_map = build_evidence_id_map(pair_result)

    print(f"Calling the real Gemini API (model: {settings.GEMINI_MODEL})...")
    client = GeminiLLMClient()
    try:
        llm_output = generate_grounded_response("warfarin", "ibuprofen", evidence_map, client=client)
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        sys.exit(1)

    print("\nParsed successfully into LLMStructuredOutput:")
    print(f"  interaction_assessment: {llm_output.interaction_assessment}")
    print(f"  evidence_summary: {llm_output.evidence_summary}")
    print(f"  clinical_effect: {llm_output.clinical_effect}")
    print(f"  mechanism: {llm_output.mechanism}")
    print(f"  severity: {llm_output.severity}")
    print(f"  cited_evidence_ids: {llm_output.cited_evidence_ids}")
    print(f"  limitations: {llm_output.limitations}")
    print(f"  safety_notice: {llm_output.safety_notice}")

    print("\nRunning grounding validation...")
    result = validate(llm_output, evidence_map, pair_result.evidence_status)
    print(f"  passed: {result.passed}")
    print(f"  fatal_issues: {result.fatal_issues}")
    print(f"  corrections: {result.corrections}")
    print(f"  corrected_severity: {result.corrected_severity}")


if __name__ == "__main__":
    main()
