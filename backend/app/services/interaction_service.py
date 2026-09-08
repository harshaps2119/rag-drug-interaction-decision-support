"""
services/interaction_service.py
==================================
Orchestrates the full pipeline for the production API endpoints
(Phase 8):

    Input validation (schemas/api.py, done before this is called)
        |
        v
    ensure_drug_available()  -- Phase 2 RxNorm + Phase 3 DailyMed + Phase 4
                                  SQLite, ON DEMAND, only if not already
                                  known locally
        |
        v
    Phase 5 ChromaDB indexing, on demand (same "only if missing" rule)
        |
        v
    PairRetrievalEngine.retrieve_pair() / .retrieve_multi()  -- Phase 6
        |
        v
    generate_explanation()  -- Phase 7 (Gemini + grounding validation)

WHY "ON DEMAND" INGESTION
------------------------------
Earlier phases' scripts (ingest_labels.py, build_vector_index.py) ingest
a fixed dev seed set ahead of time. The production API should work for
ANY drug a caller asks about, not just ones already pre-ingested -- so
this service checks whether a drug is already known locally first, and
if not, runs the same ingestion pipeline those scripts use, live, before
continuing. This keeps the API self-sufficient without a manual
pre-ingestion step for every drug it might ever be asked about.

A first-time query for a brand-new drug is therefore slower (it pays the
RxNorm + DailyMed + embedding cost once) than a repeat query for an
already-known drug. This is an expected, reasonable trade-off, and is
exactly what Phase 4's idempotent-caching design was built for.

WHY THIS NEVER RAISES FOR "DRUG DOESN'T EXIST"
--------------------------------------------------
If a drug genuinely can't be normalized (RxNorm has nothing for it),
`ensure_drug_available()` simply returns without indexing anything --
Phase 6's `PairRetrievalEngine.resolve_drug()` will then correctly
report `drug_not_found`, exactly as it already did before this phase.
This function only lets genuine infrastructure errors (a network
failure, a DB error) propagate up, which the API layer's generic
exception handler then turns into a safe 500 response.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_interaction_service.py -v

Mocks ingest_drug/ingest_rag_index and uses controlled SQLite/ChromaDB
fixtures plus FakeLLMClient -- no live RxNorm/DailyMed/Gemini calls.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.rag.ingest import ingest_rag_index
from app.schemas.explanation import ExplanationResponse
from app.services.explanation_service import generate_explanation
from app.services.knowledge_base_service import get_drug_by_rxcui, ingest_drug, search_drugs_by_name
from app.services.pair_retrieval_service import PairRetrievalEngine


def _is_drug_known_locally(session: Session, name: str) -> bool:
    clean = name.strip()
    if clean.isdigit() and get_drug_by_rxcui(session, clean) is not None:
        return True
    return len(search_drugs_by_name(session, clean)) > 0


def ensure_drug_available(session: Session, embedding_model, collection, name: str) -> None:
    """Ensures `name` is both in SQLite (Phases 2-4) and indexed in ChromaDB (Phase 5). No-op if already available."""
    if _is_drug_known_locally(session, name):
        return

    result = ingest_drug(name, session=session)
    session.commit()
    if result.drug_status == "failed" or not result.rxcui:
        return  # genuinely unknown drug -- Phase 6 will report drug_not_found correctly

    ingest_rag_index(session, embedding_model, collection, rxcui=result.rxcui)


def check_pair(
    session: Session,
    embedding_model,
    collection,
    drug_a: str,
    drug_b: str,
    llm_client=None,
) -> ExplanationResponse:
    ensure_drug_available(session, embedding_model, collection, drug_a)
    ensure_drug_available(session, embedding_model, collection, drug_b)

    engine = PairRetrievalEngine(session, embedding_model, collection)
    pair_result = engine.retrieve_pair(drug_a, drug_b)
    return generate_explanation(pair_result, llm_client=llm_client)


def check_multiple(
    session: Session,
    embedding_model,
    collection,
    drug_names: list[str],
    llm_client=None,
) -> dict[tuple[str, str], ExplanationResponse]:
    for name in drug_names:
        ensure_drug_available(session, embedding_model, collection, name)

    engine = PairRetrievalEngine(session, embedding_model, collection)
    pair_results = engine.retrieve_multi(drug_names)  # Phase 6's query cache applies across these pairs
    return {pair: generate_explanation(result, llm_client=llm_client) for pair, result in pair_results.items()}
