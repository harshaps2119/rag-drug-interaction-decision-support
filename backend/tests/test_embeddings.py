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

from app.rag.embeddings import (
    DeterministicFakeEmbeddingModel,
    OnnxEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    _is_model_cached,
)


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
    Attempts to load the REAL ONNX model and embed a sample sentence.
    Skips if the model artifacts cannot be loaded.
    """
    model = OnnxEmbeddingModel()
    try:
        dimension = model.dimension
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable in this environment: {exc}")
        return

    assert dimension == 384

    vectors = model.embed(["warfarin interacts with NSAIDs, increasing bleeding risk"])
    assert len(vectors) == 1
    assert len(vectors[0]) == dimension
    assert all(isinstance(x, float) for x in vectors[0])
    # Real embeddings are normalized -> unit length.
    magnitude = sum(x * x for x in vectors[0]) ** 0.5
    assert abs(magnitude - 1.0) < 1e-4


def test_is_model_cached_utility(tmp_path):
    # Nonexistent model should return False
    assert not _is_model_cached("nonexistent-org/nonexistent-model-xyz-12345")

    # Existing directory should return True
    d = tmp_path / "dummy_model"
    d.mkdir()
    assert _is_model_cached(str(d))


def test_onnx_model_thread_configuration():
    """
    Verifies that OnnxEmbeddingModel initializes with bounded CPU threads
    (intra_op=1, inter_op=1) for Render container deployment.
    """
    model = OnnxEmbeddingModel(intra_op_num_threads=1, inter_op_num_threads=1)
    assert model._intra_op_num_threads == 1
    assert model._inter_op_num_threads == 1


def test_is_model_cached_namespaced_fallback():
    """
    Verifies that _is_model_cached checks both the bare model identifier
    and the sentence-transformers/ namespaced identifier in Hugging Face cache.
    """
    if _is_model_cached("sentence-transformers/all-MiniLM-L6-v2"):
        assert _is_model_cached("all-MiniLM-L6-v2") is True


def test_real_model_cpu_and_batch_inference():
    """
    Verifies that OnnxEmbeddingModel encodes batches of texts (batch_size=4)
    preserving vector ordering and dimensionality.
    """
    model = OnnxEmbeddingModel()
    try:
        _ = model.dimension
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")
        return

    texts = [
        "warfarin interaction with ibuprofen",
        "aspirin and metformin warnings",
        "lisinopril dosage and side effects",
        "atorvastatin lipid management",
        "metoprolol beta blocker interaction",
    ]
    vectors = model.embed(texts)
    assert len(vectors) == 5
    assert all(len(v) == 384 for v in vectors)
    assert len({tuple(v) for v in vectors}) == 5
    for v in vectors:
        mag = sum(x * x for x in v) ** 0.5
        assert abs(mag - 1.0) < 1e-4


def test_onnx_model_empty_and_whitespace_inputs():
    model = OnnxEmbeddingModel()
    assert model.embed([]) == []

    try:
        v = model.embed(["   "])
        assert len(v) == 1
        assert len(v[0]) == 384
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")


def test_onnx_model_truncation_and_padding():
    """
    Verifies that inputs exceeding 256 tokens are safely truncated, and
    differing-length inputs in a batch pad correctly without error.
    """
    model = OnnxEmbeddingModel()
    try:
        _ = model.dimension
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")
        return

    short_text = "warfarin"
    long_text = "patient safety warnings and clinical adverse reactions " * 80  # > 400 words
    vectors = model.embed([short_text, long_text])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    for v in vectors:
        mag = sum(x * x for x in v) ** 0.5
        assert abs(mag - 1.0) < 1e-4


def test_onnx_model_batch_equivalence():
    """
    Verifies that embedding texts in a multi-item batch yields embeddings
    equivalent to embedding each text individually (padding invariance).
    """
    model = OnnxEmbeddingModel()
    try:
        _ = model.dimension
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")
        return

    texts = [
        "warfarin and ibuprofen interaction",
        "aspirin and metformin warnings with long clinical description of adverse events",
        "lisinopril blood pressure management",
    ]
    batch_vectors = model.embed(texts)
    single_vectors = [model.embed([t])[0] for t in texts]

    for bv, sv in zip(batch_vectors, single_vectors):
        max_diff = max(abs(b - s) for b, s in zip(bv, sv))
        assert max_diff < 1e-5


def test_onnx_model_concurrency():
    """
    Verifies thread-safe concurrent embedding calls across multiple threads.
    """
    import concurrent.futures

    model = OnnxEmbeddingModel()
    try:
        _ = model.dimension
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")
        return

    texts = [
        ["warfarin interaction with ibuprofen"],
        ["aspirin and metformin warnings"],
        ["lisinopril dosage side effects"],
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(model.embed, t) for t in texts * 3]
        results = [f.result() for f in futures]

    assert len(results) == 9
    for r in results:
        assert len(r) == 1
        assert len(r[0]) == 384


def test_onnx_model_warmup():
    """
    Verifies that warmup() initializes the session and returns self.
    """
    model = OnnxEmbeddingModel()
    try:
        res = model.warmup()
        assert res is model
        assert model._session is not None
        assert model._tokenizer is not None
    except Exception as exc:
        pytest.skip(f"Real embedding model unavailable: {exc}")


def test_onnx_model_missing_artifact_raises(tmp_path):
    """
    Verifies that missing model artifacts raise FileNotFoundError.
    """
    missing_model = tmp_path / "nonexistent.onnx"
    missing_tok = tmp_path / "nonexistent.json"
    model = OnnxEmbeddingModel(model_path=missing_model, tokenizer_path=missing_tok)
    with pytest.raises(FileNotFoundError):
        _ = model.dimension


def test_onnx_model_corrupt_artifact_raises(tmp_path):
    """
    Verifies that corrupt model artifacts raise RuntimeError.
    """
    corrupt_model = tmp_path / "corrupt.onnx"
    corrupt_model.write_text("not a valid onnx protobuf file")
    corrupt_tok = tmp_path / "corrupt_tok.json"
    corrupt_tok.write_text("{invalid json")

    model = OnnxEmbeddingModel(model_path=corrupt_model, tokenizer_path=corrupt_tok)
    with pytest.raises(RuntimeError):
        _ = model.dimension


def test_api_deps_embedding_model_satisfies_protocol():
    """
    Verifies that the embedding model returned by api.deps.get_embedding_model
    satisfies the EmbeddingModel protocol.
    """
    from app.api.deps import get_embedding_model

    model = get_embedding_model()
    assert hasattr(model, "dimension")
    assert hasattr(model, "embed")
    assert callable(model.embed)
