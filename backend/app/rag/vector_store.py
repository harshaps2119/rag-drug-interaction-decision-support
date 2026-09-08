"""
rag/vector_store.py
======================
Thin wrapper around ChromaDB — a local, persistent vector database — for
storing and querying chunk embeddings.

WHY WE ALWAYS SUPPLY EMBEDDINGS OURSELVES
----------------------------------------------
`get_collection()` explicitly passes `embedding_function=None`. This is
required, not optional: if left unset, ChromaDB falls back to its own
default embedding function, which tries to download its own small model
from HuggingFace on first use — the same network dependency Phase 5's
own model faces (see rag/embeddings.py). By always computing embeddings
ourselves via app/rag/embeddings.py and passing them explicitly to every
`add`/`upsert`/`query` call, ChromaDB itself never needs network access
at all — confirmed by running this module's tests fully offline in this
sandbox (chromadb the library needs no network for local persistence;
only its optional default embedding function does).

WHY UPSERT, NOT ADD (idempotency at the vector-store level)
------------------------------------------------------------------
ChromaDB's `collection.upsert()` natively replaces a document in place if
its `id` already exists, rather than creating a duplicate — this is
exactly the primitive Phase 5's idempotency requirement needs. Combined
with the DETERMINISTIC chunk IDs built in rag/ingest.py (derived from
stable database row IDs + chunk index, never randomly generated), running
ingestion twice with unchanged data upserts the exact same IDs with the
exact same content — a no-op in effect, even though upsert() is called.

WHY TELEMETRY IS DISABLED
------------------------------
`anonymized_telemetry=False` is passed explicitly. This is an appropriate
default for a health-adjacent local tool (avoid any unnecessary outbound
calls describing what a user is doing), and also avoids a harmless but
noisy stderr warning this project's ChromaDB version prints when its
telemetry call fails in this sandbox's restricted network.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_vector_store.py -v

These tests use a REAL ChromaDB instance persisted to a pytest tmp_path
directory — no mocking needed, and no network required, confirmed by
running them in this sandbox.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings as app_settings

# .../backend/app/rag/vector_store.py -> rag/ -> app/ -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

COLLECTION_NAME = "ddi_evidence"


def _resolve_persist_dir(raw_path: str) -> Path:
    """Same cwd-independence approach as app/db.py: anchor relative paths to backend/."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = BACKEND_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_client(persist_dir: str | None = None) -> chromadb.ClientAPI:
    """
    Returns a persistent local ChromaDB client.
    - persist_dir=None (default): uses settings.CHROMA_PERSIST_DIR, resolved
      relative to backend/.
    - persist_dir="/some/tmp/path": fully isolated, for tests.
    """
    if persist_dir is None:
        persist_dir = str(_resolve_persist_dir(app_settings.CHROMA_PERSIST_DIR))
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(client: chromadb.ClientAPI | None = None, persist_dir: str | None = None):
    """
    Returns (creating if needed) the single collection this project uses.
    embedding_function=None is REQUIRED — see module docstring.
    """
    client = client or get_client(persist_dir)
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=None)


def get_existing_documents(collection, ids: list[str]) -> dict[str, str]:
    """Returns {id: document_text} for whichever of the given ids already exist in the collection."""
    if not ids:
        return {}
    result = collection.get(ids=ids)
    return dict(zip(result["ids"], result["documents"]))


def get_ids_by_metadata(collection, where: dict) -> list[str]:
    """Returns all chunk ids matching a metadata filter — used to find stale chunks to delete."""
    result = collection.get(where=where)
    return result["ids"]


def upsert_chunks(
    collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """
    Upserts (insert-or-replace-by-id) a batch of chunks. No-op on an
    empty ids list.

    NOTE: ChromaDB rejects an EMPTY metadata dict ({}) with a ValueError
    ("Expected metadata to be a non-empty dict") — discovered by actually
    running this module's tests, not assumed. This is a non-issue for
    real ingestion (rag/ingest.py always builds a full metadata dict per
    chunk — drug name, rxcui, section info, etc. — never an empty one),
    but matters if you ever call this function directly with placeholder
    metadata.
    """
    if not ids:
        return
    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def delete_chunks(collection, ids: list[str]) -> None:
    """Deletes chunks by id. No-op on an empty ids list."""
    if ids:
        collection.delete(ids=ids)


def build_where(filters: dict[str, str | int | list | None]) -> dict | None:
    """
    Builds a ChromaDB-compatible `where` filter from a flat
    {field: value} dict, dropping None values and wrapping multiple
    fields in the `$and` operator ChromaDB >=0.5 requires for combining
    more than one filter condition (a bare multi-key dict raises an
    error in this version).

    A value that is a list/tuple/set is treated as "match any of these"
    and built as a ChromaDB `$in` condition — added in Phase 6 to support
    scoping a query to two drugs at once (e.g. rxcui in [rxcui_a, rxcui_b])
    without changing behavior for the plain single-value case Phase 5
    already relies on.
    """
    clean = {k: v for k, v in filters.items() if v is not None}
    if not clean:
        return None

    conditions = []
    for key, value in clean.items():
        if isinstance(value, (list, tuple, set)):
            conditions.append({key: {"$in": list(value)}})
        else:
            conditions.append({key: value})

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def query(collection, query_embedding: list[float], top_k: int = 5, where: dict | None = None) -> dict:
    """
    Runs a similarity search. Returns ChromaDB's raw result dict
    (ids/documents/metadatas/distances, each a list-of-lists — one inner
    list per query embedding; we always pass exactly one).
    """
    return collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
