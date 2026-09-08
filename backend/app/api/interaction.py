"""
api/interaction.py
=====================
POST /api/interaction/check           -- one drug pair
POST /api/interaction/check-multiple  -- N drugs, all unique pairs

Both endpoints are rate-limited (core/rate_limiter.py) and audit-logged
(core/audit_log.py). Business logic lives entirely in
services/interaction_service.py -- these route functions only handle
HTTP concerns (dependency wiring, timing, audit logging, response shape).

CLINICAL SAFETY RULE ENFORCED HERE (inherited from Phases 6-7, never
weakened at this layer): the response's `interaction_assessment` field
can only ever be one of the values Phase 6/7 already define --
pair_specific_evidence_found, supporting_evidence_found,
    insufficient_evidence, drug_not_found, invalid_input, or retrieval_error. "No
interaction exists" is not a value this API can return, for any reason,
including a Gemini failure or an empty retrieval result -- see
tests/test_interaction_router.py for an executed proof of this per
failure mode.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_chroma_collection, get_db_session, get_embedding_model, get_llm_client, rate_limit_dependency
from app.core.audit_log import log_interaction_check
from app.core.request_id import get_request_id
from app.schemas.api import InteractionCheckRequest, MultiDrugCheckRequest, MultiDrugCheckResponse, PairResultEntry
from app.schemas.explanation import ExplanationResponse
from app.services import interaction_service

router = APIRouter()


@router.post(
    "/interaction/check",
    response_model=ExplanationResponse,
    dependencies=[Depends(rate_limit_dependency)],
)
def check_interaction(
    body: InteractionCheckRequest,
    request: Request,
    session=Depends(get_db_session),
    embedding_model=Depends(get_embedding_model),
    collection=Depends(get_chroma_collection),
    llm_client=Depends(get_llm_client),
) -> ExplanationResponse:
    request_id = get_request_id(request)
    start = time.monotonic()

    response = interaction_service.check_pair(session, embedding_model, collection, body.drug_a, body.drug_b, llm_client=llm_client)

    latency_ms = (time.monotonic() - start) * 1000
    log_interaction_check(
        request_id=request_id,
        endpoint="/api/interaction/check",
        rxcui_a=response.drug_a.rxcui,
        rxcui_b=response.drug_b.rxcui,
        evidence_status=response.interaction_assessment,
        llm_mode=response.mode,
        latency_ms=latency_ms,
    )
    return response


@router.post(
    "/interaction/check-multiple",
    response_model=MultiDrugCheckResponse,
    dependencies=[Depends(rate_limit_dependency)],
)
def check_multiple_interactions(
    body: MultiDrugCheckRequest,
    request: Request,
    session=Depends(get_db_session),
    embedding_model=Depends(get_embedding_model),
    collection=Depends(get_chroma_collection),
    llm_client=Depends(get_llm_client),
) -> MultiDrugCheckResponse:
    request_id = get_request_id(request)
    start = time.monotonic()

    results = interaction_service.check_multiple(session, embedding_model, collection, body.drugs, llm_client=llm_client)

    latency_ms = (time.monotonic() - start) * 1000
    entries: list[PairResultEntry] = []
    for (drug_a, drug_b), result in results.items():
        entries.append(PairResultEntry(drug_a=drug_a, drug_b=drug_b, result=result))
        log_interaction_check(
            request_id=request_id,
            endpoint="/api/interaction/check-multiple",
            rxcui_a=result.drug_a.rxcui,
            rxcui_b=result.drug_b.rxcui,
            evidence_status=result.interaction_assessment,
            llm_mode=result.mode,
            latency_ms=latency_ms,
        )

    return MultiDrugCheckResponse(pairs=entries, total_pairs=len(entries), request_id=request_id)
