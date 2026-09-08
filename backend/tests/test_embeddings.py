"""
tests/test_embeddings.py
===========================
Tests for app/rag/embeddings.py.

THIS FILE DELIBERATELY SEPARATES TWO CATEGORIES OF TEST
-------------------------------------------------------------
1. Tests using DeterministicFakeEmbeddingModel — run fully offline,
   always, everywhere, no network or downloaded model needed. These
   verify the fake's own contract (determinism, dimensionality) since
   the rest of this project's test suite relies on that contract holding.

2. ONE test that loads the REAL sentence-transformers model
   (test_real_model_loads_and_embeds_if_available). It attempts the real
   download/load, and calls `pytest.skip()` with a clear reason if that
   fails — it does NOT fail the test run. In THIS sandbox (confirmed:
   huggingface.co is not in the allowed network egress list, and a
   direct attempt to load 'all-MiniLM-L6-v2' here failed with
   "couldn't connect to 'https://huggingface.co'"), this test WILL show
   as SKIPPED, not passed and not failed. Run it yourself, with normal
   network access, to see it actually pass — see also
   scripts/test_embeddings_live.py for a more thorough standalone check.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_embeddings.py -v
"""

import pytest

from app.rag.embeddings import DeterministicFakeEmbeddingModel, SentenceTransformerEmbeddingModel


# ---------------------------------------------------------------------------
# Offline tests: DeterministicFakeEmbeddingModel
# ---------------------------------------------------------------------------

def test_fake_model_is_deterministic():
    model = DeterministicFakeEmbeddingModel(dimension=32)
    v1 = model.embed(["warfarin may increase bleeding risk"])
    v2 = model.embed(["warfarin may increase bleeding risk"])
    assert v1 == v2


def test_fake_model_different_texts_produce_different_vectors():
    model = DeterministicFakeEmbeddingModel(dimension=32)
    v1, v2 = model.embed(["warfarin", "ibuprofen"])
    assert v1 != v2


def test_fake_model_respects_configured_dimension():
    model = DeterministicFakeEmbeddingModel(dimension=16)
    vectors = model.embed(["some text"])
    assert len(vectors[0]) == 16
    assert model.dimension == 16


def test_fake_model_empty_list_returns_empty_list():
    model = DeterministicFakeEmbeddingModel()
    assert model.embed([]) == []


def test_fake_model_vectors_are_bounded():
    model = DeterministicFakeEmbeddingModel(dimension=32)
    vectors = model.embed(["a fairly ordinary sentence about drug interactions"])
    assert all(-1.0 <= x <= 1.0 for x in vectors[0])


def test_fake_model_handles_batch_of_multiple_texts():
    model = DeterministicFakeEmbeddingModel(dimension=32)
    texts = ["warfarin interacts with NSAIDs", "aspirin affects platelet function", "metformin lactic acidosis risk"]
    vectors = model.embed(texts)
    assert len(vectors) == 3
    assert len({tuple(v) for v in vectors}) == 3  # all distinct


# ---------------------------------------------------------------------------
# Real model test — auto-skips if the model can't be downloaded/loaded
# ---------------------------------------------------------------------------

def test_real_model_loads_and_embeds_if_available():
    """
    Attempts to load the REAL sentence-transformers model and embed a
    sample sentence. Skips (does not fail) if the model can't be
    downloaded — e.g. no network access, as in the sandbox this project
    was built in.
    """
    model = SentenceTransformerEmbeddingModel()
    try:
        dimension = model.dimension
    except Exception as exc:  # broad: any download/load failure means "can't verify here"
        pytest.skip(f"Real embedding model unavailable in this environment: {exc}")
        return

    assert dimension == 384  # expected for all-MiniLM-L6-v2 — see docs/embeddings.md

    vectors = model.embed(["warfarin interacts with NSAIDs, increasing bleeding risk"])
    assert len(vectors) == 1
    assert len(vectors[0]) == dimension
    # Real embeddings are normalized (normalize_embeddings=True) -> unit length.
    magnitude = sum(x * x for x in vectors[0]) ** 0.5
    assert abs(magnitude - 1.0) < 1e-3
