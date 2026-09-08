"""
rag/embeddings.py
====================
Wraps text embedding generation behind a small interface, with two
implementations:

  1. SentenceTransformerEmbeddingModel — the REAL model, used in
     production and by scripts you run locally.
  2. DeterministicFakeEmbeddingModel — NOT a real embedding model at
     all. A dependency-free, offline stand-in used ONLY by this
     project's automated test suite, in environments (like the sandbox
     this project was built in) where the real model cannot be
     downloaded. See its own docstring below for exactly what it can and
     cannot verify.

WHICH MODEL WAS SELECTED, AND WHY
------------------------------------
Selected: **sentence-transformers/all-MiniLM-L6-v2**

This is a GENERAL-PURPOSE embedding model, not a biomedical-specific one.
That was a deliberate, considered choice — see docs/embeddings.md for the
full trade-off analysis (general-purpose vs. biomedical models,
dimensions, resource cost, alternatives considered). Summary:

  - 384-dimension embeddings, ~90MB download, runs comfortably on CPU
    with no GPU required — appropriate for a student's laptop and for
    keeping this an accessible, reproducible academic project.
  - Officially maintained as part of the sentence-transformers library,
    extremely widely used and tested, permissive Apache 2.0 license.
  - Retrieval quality is helped substantially by the METADATA FILTERING
    already available from Phase 4 (filtering to one drug's rxcui and/or
    one section type before ranking by similarity) — the embedding model
    only needs to rank an already-narrowed set of candidates well, not
    carry the entire retrieval burden semantically.
  - The considered alternative — a biomedical-domain model such as
    pritamdeka/S-PubMedBert-MS-MARCO or NeuML/pubmedbert-base-embeddings
    — likely captures clinical synonym relationships (e.g. "MI" vs.
    "myocardial infarction") better, at the cost of a much larger
    download (400-900MB), heavier CPU inference, and less consistent
    tooling/community support than the mainstream sentence-transformers
    models. This is flagged in docs/embeddings.md as a documented future
    upgrade path for Phase 13's evaluation phase to A/B against the
    current choice — not dismissed, just not selected as the MVP default.

THIS MODULE'S DESIGN GOAL: LAZY LOADING
-------------------------------------------
SentenceTransformerEmbeddingModel does NOT download or load the model at
import time or even at construction time — only on the first call to
`.embed()` or access to `.dimension`. This means the rest of the app
(and every test that doesn't actually need real embeddings) can import
this module freely without triggering a network call or a slow model
load, which is exactly what makes the offline/fake-vs-real test split
required by this phase possible.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_embeddings.py -v

Tests using DeterministicFakeEmbeddingModel run fully offline, always.
The one test that loads the REAL model will automatically SKIP (not
fail) if it can't reach the model — see that test's own comment for how,
and scripts/test_embeddings_live.py for the version you should run
yourself with real network access to actually verify the real model.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from app.config import settings


class EmbeddingModel(Protocol):
    """The interface both the real and fake embedding models satisfy."""

    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingModel:
    """
    The real embedding model, backed by the `sentence-transformers`
    library. Lazily loads the underlying model on first use.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        # normalize_embeddings=True -> unit-length vectors, so ChromaDB's
        # default L2 distance behaves consistently with cosine similarity.
        vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


class DeterministicFakeEmbeddingModel:
    """
    NOT a real embedding model. A small, dependency-free, fully
    deterministic stand-in for offline testing ONLY.

    WHAT IT DOES: hashes each input text (SHA-256) into a fixed-size
    vector of floats in [-1, 1]. Same text always produces the exact
    same vector; different texts (almost certainly) produce different
    vectors. That's the full extent of its behavior.

    WHAT IT CAN VERIFY (used throughout this phase's test suite):
      - chunking -> embedding -> ChromaDB PLUMBING works end-to-end
      - deterministic chunk IDs behave correctly under re-ingestion
      - metadata is preserved through the pipeline
      - idempotent upsert / stale-chunk cleanup logic works
      - retrieval filtering (by rxcui, by section) works
      - top_k, empty-database, and malformed-record handling work

    WHAT IT CANNOT VERIFY, EVER (do not use it to draw ANY conclusion
    about these):
      - whether semantically similar medical text actually gets embedded
        close together in vector space
      - retrieval RELEVANCE/QUALITY for any real query
      - anything at all about which drugs, sections, or interactions are
        "similar" to each other in a meaningful sense

    Any test using this class is a plumbing test, not a quality test —
    and is named/commented accordingly throughout this project's test
    suite, per this phase's explicit requirement to keep these
    categories separate.
    """

    def __init__(self, dimension: int = 32):
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes
            repeats = (self.dimension // len(digest)) + 1
            raw_bytes = (digest * repeats)[: self.dimension]
            vectors.append([(b - 127.5) / 127.5 for b in raw_bytes])
        return vectors
