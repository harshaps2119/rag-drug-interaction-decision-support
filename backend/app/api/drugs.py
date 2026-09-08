"""
api/drugs.py
==============
GET /api/drugs/search -- read-only lookup against the local SQLite
knowledge base (Phase 4). No external calls; supports the frontend
phase's need for a drug-name autocomplete/search box.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db_session
from app.schemas.api import DrugSearchResult
from app.services.knowledge_base_service import search_drugs_by_name

router = APIRouter()


@router.get("/drugs/search", response_model=list[DrugSearchResult])
def search_drugs(
    q: str = Query(..., min_length=1, max_length=200, description="Drug name or partial name to search for."),
    session: Session = Depends(get_db_session),
) -> list[DrugSearchResult]:
    matches = search_drugs_by_name(session, q)
    return [
        DrugSearchResult(rxcui=d.rxcui, normalized_name=d.normalized_name, term_type=d.term_type)
        for d in matches
    ]
