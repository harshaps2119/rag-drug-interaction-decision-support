"""
data/seed_drugs.py
=====================
DEVELOPMENT / TEST SEED SET ONLY.

This is a small, fixed list of well-known drug names used to exercise the
ingestion pipeline during development (Phase 4) and later RAG testing
(Phase 5+). It exists purely so there's SOMETHING in the local knowledge
base to develop and test against without hand-typing drug names every
time.

>>> THIS IS NOT A CLINICAL FORMULARY OR A CURATED DDI DATASET. <<<
It is not exhaustive, not clinically reviewed, and must never be treated
as a complete or authoritative list of drugs the system "supports". The
system's actual drug coverage is exactly whatever RxNorm/DailyMed
provide at query time, for any drug name — this list is just a
convenient, familiar set for testing that pipeline.

The five drugs below were chosen because they are common, well-known,
and (based on general clinical knowledge) reasonably likely to have
non-trivial Drug Interactions sections in their FDA labels — useful for
exercising both the "interaction_evidence_found" and
"label_found_no_interaction_section" code paths during development.
"""

DEV_SEED_DRUGS: list[str] = [
    "warfarin",
    "ibuprofen",
    "aspirin",
    "amiodarone",
    "metformin",
]
