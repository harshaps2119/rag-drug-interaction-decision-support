"""
services/retrieval_service.py
================================
Accepts a text query and returns the most relevant stored evidence
chunks from ChromaDB (Phase 5), with full source metadata attached.

WHAT THIS SERVICE DOES AND DOES NOT DO
-------------------------------------------
It performs SEMANTIC SIMILARITY SEARCH over previously indexed FDA label
text and returns the results with their metadata. That is the entirety
of its job.

It does NOT:
  - decide whether a drug interaction exists,
  - rank or score clinical severity,
  - filter results based on any notion of "correctness",
  - or produce any natural-language explanation of what it found.

Those are all Phase 6+ (RAG retrieval logic feeding an LLM) and later
responsibilities, deliberately not implemented here. This file is the
"find relevant text" step only — see the critical note below on what
the returned `distance` value means.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_retrieval_service.py -v

Uses the DeterministicFakeEmbeddingModel (rag/embeddings.py) and a real
ChromaDB instance in a pytest tmp_path — tests the retrieval PLUMBING
(filtering, top_k, empty results, metadata pass-through) fully offline.
It does NOT and CANNOT test retrieval QUALITY with fake embeddings — see
scripts/query_index_live.py for a script you run yourself with the real
model to sanity-check actual relevance.
"""

from __future__ import annotations

from app.rag.vector_store import build_where
from app.rag.vector_store import query as vector_query
from app.schemas.retrieval import RetrievedChunk


def retrieve(
    embedding_model,
    collection,
    query_text: str,
    top_k: int | None = None,
    rxcui: str | list[str] | None = None,
    section_key: str | None = None,
) -> list[RetrievedChunk]:
    """
    Embeds `query_text` and returns the top_k most similar stored chunks,
    optionally filtered to one or more drugs (by rxcui — pass a single
    string, or a list to match any of several drugs at once, e.g. when
    Phase 6 scopes a pair-specific query to either of two drugs' own
    indexed evidence) and/or one section type (e.g. "drug_interactions").

    top_k=None uses settings.RAG_DEFAULT_TOP_K (configurable — see
    Phase 5's requirement for a configurable retrieval parameter).

    Returns an empty list for an empty/whitespace query, or if the
    collection has no matching data — never raises for these cases.

    ============================================================
    CRITICAL — WHAT THE RETURNED `distance` MEANS, AND DOES NOT
    ============================================================
    `distance` is a VECTOR-SPACE distance between the query's embedding
    and a chunk's embedding (lower = more textually/semantically similar,
    according to whichever embedding model produced both vectors). It is
    a RETRIEVAL-RELEVANCE SIGNAL ONLY.

    It must NEVER be interpreted, displayed, or used downstream as:
      - a clinical interaction severity rating,
      - a probability that an interaction exists,
      - proof that two drugs interact.

    A low-distance (highly similar) chunk simply means "this stored text
    reads like it's about the same topic as the query" — nothing more.
    Phase 6+ (an LLM reasoning over the ACTUAL RETRIEVED TEXT, always
    citing it) is what will ever produce a clinically meaningful
    statement — never this distance number, and never this phase.
    """
    if not query_text or not query_text.strip():
        return []

    if top_k is None:
        from app.config import settings

        top_k = settings.RAG_DEFAULT_TOP_K

    query_embedding = embedding_model.embed([query_text])[0]
    where = build_where({"rxcui": rxcui, "section_key": section_key})

    raw = vector_query(collection, query_embedding, top_k=top_k, where=where)

    if not raw.get("ids") or not raw["ids"][0]:
        return []

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    results: list[RetrievedChunk] = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        meta = meta or {}
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                distance=distance,
                drug_name=meta.get("drug_name") or None,
                rxcui=meta.get("rxcui") or None,
                label_setid=meta.get("label_setid") or None,
                spl_version=meta.get("spl_version") or None,
                manufacturer=meta.get("manufacturer") or None,
                section_key=meta.get("section_key") or None,
                section_name=meta.get("section_name") or None,
                section_code=meta.get("section_code") or None,
                source_url=meta.get("source_url") or None,
                api_url=meta.get("api_url") or None,
                db_record_id=meta.get("db_record_id"),
                evidence_status=meta.get("evidence_status") or None,
            )
        )
    return results
