"""
scripts/ingest_labels.py
===========================
Targeted ingestion CLI: normalizes drug names (RxNorm), fetches their
DailyMed label evidence, and stores/updates everything in the local
SQLite knowledge base — idempotently.

>>> THIS SCRIPT MAKES REAL, LIVE NETWORK CALLS <<<
It calls both RxNorm and DailyMed for each drug. It is NOT executed by
Claude in this environment — this sandbox's network egress blocks both
rxnav.nlm.nih.gov and dailymed.nlm.nih.gov (confirmed with 403 responses
in Phase 2 and Phase 3; see scripts/test_rxnorm_live.py and
scripts/test_dailymed_live.py). Run this yourself in VS Code.

The DATABASE LAYER underneath this script (app/db.py, the SQLAlchemy
models, and knowledge_base_service.ingest_drug()'s upsert logic) HAS been
tested and run for real in this sandbox — see tests/test_db.py and
tests/test_knowledge_base_service.py, which use a real local SQLite
database and only mock the two network calls. What has NOT been verified
here is what actually comes back from the live APIs for these specific
drugs.

HOW TO RUN
-----------
    cd backend
    python ../scripts/setup_data.py                        # once, if you haven't already
    python ../scripts/ingest_labels.py --seed                 # ingest the dev seed set
    python ../scripts/ingest_labels.py --drug warfarin --drug ibuprofen
    python ../scripts/ingest_labels.py --seed                 # run again -> should show "unchanged", not duplicates

WHAT "IDEMPOTENT" MEANS HERE
-------------------------------
Running this script twice for the same drug will NOT create duplicate
rows — the second run should report drug_status/label_status/link_status
as "unchanged" (or "updated" only if the live label content genuinely
changed since the last run). See docs/database.md for exactly which
unique constraints enforce this.

EXPECTED OUTPUT (first run, illustrative — real label content varies)
------------------------------------------------------------------------
    --- Ingesting 'warfarin' ---
      rxcui:            11289
      drug_status:      added
      label_status:     added
      evidence_status:  interaction_evidence_found
      sections added/updated/unchanged: 7/0/0
      link_status:      added

Second run of the same command should instead show:
      drug_status:      unchanged
      label_status:     unchanged
      sections added/updated/unchanged: 0/0/7
      link_status:      unchanged   <- content (evidence_status/candidates)
                                        didn't change, so this is reported
                                        as unchanged even though the link's
                                        internal retrieved_at timestamp was
                                        still refreshed — see docs/database.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.data.seed_drugs import DEV_SEED_DRUGS  # noqa: E402
from app.db import get_engine, get_session_factory, init_db  # noqa: E402
from app.services.knowledge_base_service import ingest_drug  # noqa: E402


def run_ingestion(drug_names: list[str]) -> None:
    engine = get_engine()
    init_db(engine)
    Session = get_session_factory(engine)
    session = Session()
    try:
        for name in drug_names:
            print(f"\n--- Ingesting '{name}' ---")
            try:
                result = ingest_drug(name, session=session)
                session.commit()
            except Exception as exc:  # keep going even if one drug fails
                session.rollback()
                print(f"  UNEXPECTED FAILURE: {exc}")
                continue

            print(f"  rxcui:            {result.rxcui}")
            print(f"  drug_status:      {result.drug_status}")
            print(f"  label_status:     {result.label_status}")
            print(f"  evidence_status:  {result.evidence_status}")
            print(
                f"  sections added/updated/unchanged: "
                f"{result.sections_added}/{result.sections_updated}/{result.sections_unchanged}"
            )
            print(f"  link_status:      {result.link_status}")
            if result.warnings:
                print(f"  warnings: {result.warnings}")
            if result.error:
                print(f"  ERROR: {result.error}")
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest drugs (RxNorm normalization + DailyMed label evidence) into the local knowledge base."
    )
    parser.add_argument("--drug", action="append", default=[], help="Drug name to ingest (repeatable).")
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Ingest the development seed set (see app/data/seed_drugs.py). NOT a clinical dataset.",
    )
    args = parser.parse_args()

    drug_names = list(args.drug)
    if args.seed:
        drug_names.extend(DEV_SEED_DRUGS)

    if not drug_names:
        parser.error("Provide at least one --drug NAME, or use --seed.")

    run_ingestion(drug_names)


if __name__ == "__main__":
    main()
