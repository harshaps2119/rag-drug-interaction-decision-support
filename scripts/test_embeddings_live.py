"""
scripts/test_embeddings_live.py
==================================
Standalone verification that the real embedding model ('all-MiniLM-L6-v2'
by default) downloads, loads, and produces sensible-looking embeddings —
including a basic semantic sanity check (similar sentences should be
closer together than unrelated ones).

>>> NOT EXECUTED BY Claude <<<
Requires downloading the model from huggingface.co, which this sandbox's
network egress blocks (confirmed directly — see docs/embeddings.md and
tests/test_embeddings.py). Run this yourself with normal network access.

HOW TO RUN
-----------
    cd backend
    python ../scripts/test_embeddings_live.py

EXPECTED OUTPUT (what you SHOULD see)
------------------------------------------
    Downloading/loading 'all-MiniLM-L6-v2'...
    Loaded. Dimension: 384

    Semantic sanity check:
      distance(warfarin-bleeding, warfarin-NSAID) = <small number>
      distance(warfarin-bleeding, unrelated weather sentence) = <larger number>
      PASS: related sentences are closer than the unrelated one.

If the "PASS" line doesn't appear, or the numbers look inverted, that's
worth investigating before relying on this model for retrieval — compare
against the discussion in docs/embeddings.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.rag.embeddings import SentenceTransformerEmbeddingModel  # noqa: E402


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main() -> None:
    model = SentenceTransformerEmbeddingModel()
    print(f"Downloading/loading '{model.model_name}'...")
    try:
        dimension = model.dimension
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        print("Check your internet connection — this requires reaching huggingface.co.")
        sys.exit(1)

    print(f"Loaded. Dimension: {dimension}\n")

    sentence_a = "Warfarin may increase bleeding risk when taken with NSAIDs."
    sentence_b = "Taking ibuprofen alongside warfarin can raise the risk of bleeding."
    sentence_c = "The weather in Bengaluru is pleasant during the monsoon season."

    vec_a, vec_b, vec_c = model.embed([sentence_a, sentence_b, sentence_c])

    dist_related = _euclidean_distance(vec_a, vec_b)
    dist_unrelated = _euclidean_distance(vec_a, vec_c)

    print("Semantic sanity check:")
    print(f"  distance(warfarin-bleeding, warfarin-NSAID)      = {dist_related:.4f}")
    print(f"  distance(warfarin-bleeding, unrelated sentence)  = {dist_unrelated:.4f}")

    if dist_related < dist_unrelated:
        print("  PASS: related sentences are closer than the unrelated one.")
    else:
        print("  UNEXPECTED: related sentences were NOT closer — investigate before relying on this model.")


if __name__ == "__main__":
    main()
