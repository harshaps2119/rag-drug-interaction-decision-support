"""
scripts/test_dailymed_live.py
================================
Manual smoke-test script that calls the REAL, LIVE DailyMed API (not mocked).

WHY THIS SCRIPT EXISTS SEPARATELY FROM THE PYTEST SUITE
------------------------------------------------------------
Same reasoning as test_rxnorm_live.py: tests/test_dailymed_service.py uses
mocked HTTP responses built from officially-documented DailyMed response
shapes, so it runs instantly and offline — but it can't tell you whether
the REAL API still behaves the way we assumed. This script is that check.

>>> NOT EXECUTED BY Claude <<<
This sandbox's network egress blocks dailymed.nlm.nih.gov (confirmed:
returns HTTP 403 "Host not in allowlist" from the sandbox's own egress
proxy, same as rxnav.nlm.nih.gov in Phase 2 — not a real DailyMed error).
So this script has NOT been run by Claude and its output has NOT been
seen. Run it yourself in VS Code's terminal.

HOW TO RUN
-----------
    cd backend
    python ../scripts/test_dailymed_live.py

EXPECTED OUTPUT (what you SHOULD see — real API/label content varies and
is not guaranteed to match exactly)
------------------------------------------------------------------------
For warfarin (rxcui 11289, from the Phase 2 live test):
    evidence_status: interaction_evidence_found
    sections['drug_interactions'].found: True
    sections['drug_interactions'].text: (should mention things like
        potentiation of anticoagulant effect, bleeding risk, or specific
        interacting drug classes)

For a drug with typically minimal generic OTC labeling (e.g. some
ibuprofen store-brand repackager labels), you may see:
    evidence_status: label_found_no_interaction_section
This is EXPECTED behavior, not a bug — see dailymed_service.py's
docstring on why not every label has a filled Drug Interactions section.

For a nonsense drug name with no matching rxcui/name:
    evidence_status: no_label_found

If results look very different in kind (e.g. exceptions instead of
graceful DrugLabelResult objects), something about the live API or SPL
structure may have changed — compare against the assumptions documented
in app/services/dailymed_service.py's docstring, especially the LOINC
section codes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.dailymed_service import get_drug_label_info  # noqa: E402
from app.services.rxnorm_service import normalize_drug_name  # noqa: E402

# (drug name, known/likely RxCUI or None to force live normalization first)
TEST_CASES = [
    "warfarin",
    "ibuprofen",
    "Tylenol",
    "amiodarone",
    "notarealdrugxyz123",
]


def main() -> None:
    print("=" * 70)
    print("LIVE DailyMed API test — run by YOU, not by Claude (see docstring)")
    print("=" * 70)

    for name in TEST_CASES:
        print(f"\n--- Input: '{name}' ---")

        # Chain Phase 2 -> Phase 3, exactly as the real pipeline will in
        # later phases: normalize first, then look up the label by RxCUI.
        norm = normalize_drug_name(name)
        print(f"  [RxNorm]  match_type: {norm.match_type}  rxcui: {norm.rxcui}")

        if norm.match_type == "error":
            print("  Skipping DailyMed lookup — RxNorm itself failed (see above).")
            continue

        result = get_drug_label_info(rxcui=norm.rxcui, drug_name=name)
        print(f"  [DailyMed] evidence_status:      {result.evidence_status}")
        print(f"  [DailyMed] setid:                {result.setid}")
        print(f"  [DailyMed] title:                {result.title}")
        print(f"  [DailyMed] candidates_checked:   {result.candidates_checked} / {result.total_candidates_found}")

        if result.sections:
            di = result.sections.get("drug_interactions")
            if di:
                print(f"  [DailyMed] drug_interactions.found: {di.found}")
                if di.found:
                    preview = di.text[:300] + ("..." if len(di.text) > 300 else "")
                    print(f"  [DailyMed] drug_interactions.text:  {preview}")

        if result.warnings:
            print(f"  [DailyMed] warnings: {result.warnings}")
        if result.error:
            print(f"  [DailyMed] ERROR: {result.error}")

    print("\n" + "=" * 70)
    print("Done. Compare results against the 'EXPECTED OUTPUT' section")
    print("in this script's docstring.")
    print("=" * 70)


if __name__ == "__main__":
    main()
