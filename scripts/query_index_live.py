"""
scripts/query_index_live.py
==============================
Manual retrieval-quality check: runs a few sample queries against the
REAL, already-built ChromaDB vector index (see build_vector_index.py)
using the REAL embedding model, and prints what comes back.

>>> NOT EXECUTED BY Claude <<<
Same reason as build_vector_index.py — requires the real embedding
model, which requires network access to huggingface.co, which this
sandbox blocks. Run this yourself, after running build_vector_index.py.

WHAT TO LOOK FOR WHEN YOU RUN THIS
--------------------------------------
This is the first point in the whole pipeline where you can actually
judge whether "all-MiniLM-L6-v2 is good enough" for this project (see
docs/embeddings.md for the full trade-off discussion). Check:
  - Does a query about a drug's interactions actually return that
    drug's Drug Interactions section highly ranked (low distance)?
  - Do completely unrelated queries return clearly worse (higher
    distance) matches than relevant ones?
  - Do the rxcui/section_key filters correctly narrow results?

REMINDER: whatever distance values you see are a retrieval-relevance
signal only — never a severity/probability/interaction-proof number.
See services/retrieval_service.retrieve()'s docstring.

HOW TO RUN
-----------
    cd backend
    python ../scripts/build_vector_index.py     # if you haven't already
    python ../scripts/query_index_live.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.rag.embeddings import SentenceTransformerEmbeddingModel  # noqa: E402
from app.rag.vector_store import get_collection  # noqa: E402
from app.services.retrieval_service import retrieve  # noqa: E402

SAMPLE_QUERIES = [
    ("Does warfarin interact with NSAIDs like ibuprofen?", None, None),
    ("What are the contraindications for amiodarone?", None, "contraindications"),
    ("bleeding risk with blood thinners", None, None),
]


def main() -> None:
    print("Loading embedding model (first run downloads the model)...")
    embedding_model = SentenceTransformerEmbeddingModel()
    try:
        embedding_model.dimension  # triggers load
    except Exception as exc:
        print(f"FAILED to load the embedding model: {exc}")
        sys.exit(1)

    collection = get_collection()
    if collection.count() == 0:
        print("The vector index is empty — run scripts/build_vector_index.py first.")
        sys.exit(1)

    print(f"Vector index has {collection.count()} chunks.\n")

    for query_text, rxcui, section_key in SAMPLE_QUERIES:
        print("=" * 70)
        print(f"Query: {query_text!r}  (rxcui={rxcui}, section_key={section_key})")
        results = retrieve(embedding_model, collection, query_text, top_k=3, rxcui=rxcui, section_key=section_key)

        if not results:
            print("  No results.")
            continue

        for i, r in enumerate(results, 1):
            preview = r.text[:200] + ("..." if len(r.text) > 200 else "")
            print(f"\n  [{i}] distance={r.distance:.4f}  drug={r.drug_name}  section={r.section_name}")
            print(f"      source: {r.source_url}")
            print(f"      text: {preview}")

    print("\n" + "=" * 70)
    print("Done. Remember: 'distance' is a retrieval-relevance signal only —")
    print("never a clinical severity, probability, or interaction-proof value.")


if __name__ == "__main__":
    main()
