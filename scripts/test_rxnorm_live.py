"""
scripts/test_rxnorm_live.py
==============================
Manual smoke-test script that calls the REAL, LIVE RxNorm API (not mocked).

WHY THIS SCRIPT EXISTS SEPARATELY FROM THE PYTEST SUITE
------------------------------------------------------------
backend/tests/test_rxnorm_service.py uses mocked HTTP responses so it can
run instantly, offline, in CI, etc. That's great for catching regressions
in OUR code, but it can never tell you whether the REAL RxNorm API still
behaves the way we assumed (API changes, deprecations, downtime).

This script is the complementary "does it actually work against the real
world" check. Run it whenever you want to sanity-check the live
integration — e.g. before a demo or viva.

>>> NOT EXECUTED BY Claude <<<
This sandbox's network egress only allows a fixed list of package-registry
domains (pypi, npm, github, etc.) and explicitly blocks rxnav.nlm.nih.gov
(confirmed: it returns HTTP 403 "Host not in allowlist" from the sandbox's
own egress proxy — not a real RxNorm error). So this script has NOT been
run against the live API by Claude, and its output has NOT been seen.
Run it yourself in VS Code's terminal, where there is no such network
restriction.

HOW TO RUN
-----------
    cd backend
    python ../scripts/test_rxnorm_live.py

EXPECTED OUTPUT (what you SHOULD see, not a guarantee — real API behavior
can vary slightly over time)
------------------------------------------------------------------------
For "warfarin":
    match_type: exact
    rxcui: 11289
    normalized_name: warfarin
    brand_names: [...Coumadin or similar...]

For "worfarin" (deliberate typo):
    match_type: approximate
    rxcui: 11289 (or similar — RxNorm's fuzzy match should still find warfarin)
    match_score: some number, typically 70-100

For "Tylenol" (brand name):
    match_type: exact
    generic_name: acetaminophen

For "notarealdrugxyz123":
    match_type: none

If any of these look very different, RxNorm's API may have changed, or
there may be a bug — compare against the assumptions documented in
app/services/rxnorm_service.py's docstring.
"""

import sys
from pathlib import Path

# Allow running this script directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.services.rxnorm_service import normalize_drug_name  # noqa: E402

TEST_CASES = [
    "warfarin",           # generic name, should be exact
    "Tylenol",             # brand name, should be exact, generic_name populated
    "worfarin",            # misspelling, should be approximate
    "acetominophen",       # common misspelling of acetaminophen
    "ibuprofen",           # generic name, should be exact
    "notarealdrugxyz123",  # should be none
]


def main() -> None:
    print("=" * 70)
    print("LIVE RxNorm API test — run by YOU, not by Claude (see docstring)")
    print("=" * 70)

    for name in TEST_CASES:
        print(f"\n--- Input: '{name}' ---")
        result = normalize_drug_name(name)
        print(f"  match_type:      {result.match_type}")
        print(f"  rxcui:           {result.rxcui}")
        print(f"  normalized_name: {result.normalized_name}")
        print(f"  term_type:       {result.term_type}")
        if result.match_score is not None:
            print(f"  match_score:     {result.match_score}")
        if result.brand_names:
            print(f"  brand_names:     {result.brand_names}")
        if result.generic_name:
            print(f"  generic_name:    {result.generic_name}")
        if result.warnings:
            print(f"  warnings:        {result.warnings}")
        if result.error:
            print(f"  ERROR:           {result.error}")

    print("\n" + "=" * 70)
    print("Done. Compare results against the 'EXPECTED OUTPUT' section")
    print("in this script's docstring.")
    print("=" * 70)


if __name__ == "__main__":
    main()
