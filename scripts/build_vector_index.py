"""
scripts/build_vector_index.py
================================
Builds (or rebuilds/updates) the REAL ChromaDB vector index from the
Phase 4 SQLite knowledge base, using the REAL embedding model.

>>> NOT EXECUTED BY Claude <<<
This script downloads and runs the real sentence-transformers model
('all-MiniLM-L6-v2' by default), which requires reaching huggingface.co.
This sandbox's network egress does not allow that domain (confirmed: a
direct attempt to load the model here failed with "couldn't connect to
'https://huggingface.co'" — see tests/test_embeddings.py, where the
equivalent test auto-skips for the same reason). Run this yourself in
VS Code, with normal network access.

PREREQUISITE
--------------
You need drugs already ingested into SQLite first — run Phase 4's
scripts/ingest_labels.py (with --seed or specific --drug names) before
this script has anything to index.

HOW TO RUN
-----------
    cd backend
    python ../scripts/ingest_labels.py --seed        # Phase 4, if not already done
    python ../scripts/build_vector_index.py             # index everything in SQLite
    python ../scripts/build_vector_index.py --rxcui 11289   # index just one drug

Run it again any time — it's idempotent (see app/rag/ingest.py's
docstring for exactly how): unchanged evidence reports "unchanged", not
duplicated; genuinely updated FDA text reports "updated" in place;
shrunk sections have their stale extra chunks cleaned up automatically.

EXPECTED OUTPUT (illustrative — real chunk counts depend on real label
content, which this sandbox cannot fetch or verify)
------------------------------------------------------------------------
    Loading embedding model 'all-MiniLM-L6-v2' (first run downloads ~90MB)...
    Embedding model ready. Dimension: 384

    --- Indexing SQLite evidence into ChromaDB ---
    Drugs processed: 5
    Sections processed: ~15-20 (varies by real label content)
    Chunks added: ...
    Chunks updated: 0
    Chunks unchanged: 0
    Chunks deleted (stale): 0
    Warnings: []

    Vector index persisted at: backend/data/chroma_store
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config import settings  # noqa: E402
from app.db import get_engine, get_session_factory, init_db  # noqa: E402
from app.rag.embeddings import SentenceTransformerEmbeddingModel  # noqa: E402
from app.rag.ingest import ingest_rag_index  # noqa: E402
from app.rag.vector_store import get_collection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the ChromaDB vector index from SQLite evidence.")
    parser.add_argument("--rxcui", default=None, help="Only index this drug's RxCUI (default: index everything).")
    args = parser.parse_args()

    engine = get_engine()
    init_db(engine)
    Session = get_session_factory(engine)
    session = Session()

    print(f"Loading embedding model '{settings.EMBEDDING_MODEL_NAME}' (first run downloads the model)...")
    embedding_model = SentenceTransformerEmbeddingModel()
    try:
        dimension = embedding_model.dimension  # triggers the actual load
    except Exception as exc:
        print(f"\nFAILED to load the embedding model: {exc}")
        print("Check your internet connection — this requires reaching huggingface.co.")
        sys.exit(1)
    print(f"Embedding model ready. Dimension: {dimension}\n")

    collection = get_collection()

    print("--- Indexing SQLite evidence into ChromaDB ---")
    try:
        result = ingest_rag_index(session, embedding_model, collection, rxcui=args.rxcui)
    finally:
        session.close()

    print(f"Drugs processed: {len(result.drugs_processed)} ({sorted(result.drugs_processed)})")
    print(f"Sections processed: {result.sections_processed}")
    print(f"Chunks added: {result.chunks_added}")
    print(f"Chunks updated: {result.chunks_updated}")
    print(f"Chunks unchanged: {result.chunks_unchanged}")
    print(f"Chunks deleted (stale): {result.chunks_deleted_stale}")
    print(f"Sections skipped (empty after cleaning): {result.skipped_empty_sections}")
    print(f"Malformed records skipped: {result.malformed_records_skipped}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    print(f"\nVector index persisted at: backend/{settings.CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
