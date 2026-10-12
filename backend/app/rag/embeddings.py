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
from pathlib import Path
from typing import Protocol

from app.config import settings


class EmbeddingModel(Protocol):
    """The interface both the real and fake embedding models satisfy."""

    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


PINNED_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_ONNX_FILENAME = "onnx/model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"

MODEL_ONNX_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
TOKENIZER_SHA256 = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"


def _resolve_onnx_model_files(model_name: str | None = None) -> tuple[Path, Path]:
    """
    Resolves the paths to the pinned ONNX model file and tokenizer.json file.
    Searches:
      1. Direct local path if model_name is a directory or .onnx file
      2. Environment-configured ONNX_MODEL_DIR
      3. Project-local backend/models/ directory
      4. Hugging Face hub cache directory (pinned revision)
    """
    import os
    from pathlib import Path

    target_name = model_name or settings.EMBEDDING_MODEL_NAME

    # 1. Direct path check on local filesystem
    p = Path(target_name)
    if p.is_dir():
        m_candidates = [p / "model.onnx", p / "onnx" / "model.onnx"]
        t_candidates = [p / "tokenizer.json"]
        m_found = next((c for c in m_candidates if c.is_file()), None)
        t_found = next((c for c in t_candidates if c.is_file()), None)
        if m_found and t_found:
            return m_found, t_found
    elif p.is_file() and p.suffix == ".onnx":
        t_cand = p.parent / "tokenizer.json"
        if not t_cand.exists():
            t_cand = p.parent.parent / "tokenizer.json"
        if t_cand.is_file():
            return p, t_cand

    is_pinned_model = target_name in {
        "all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        PINNED_REPO_ID,
    }

    if not is_pinned_model:
        # Check custom non-pinned HF repo
        try:
            from huggingface_hub import try_to_load_from_cache

            m_c = try_to_load_from_cache(target_name, MODEL_ONNX_FILENAME)
            t_c = try_to_load_from_cache(target_name, TOKENIZER_FILENAME)
            if isinstance(m_c, str) and isinstance(t_c, str):
                return Path(m_c), Path(t_c)
        except Exception:
            pass
        raise FileNotFoundError(f"Model files for '{target_name}' not found.")

    # 2. Project-local models directory for pinned model
    backend_root = Path(__file__).resolve().parent.parent.parent
    local_dirs = [
        backend_root / "models" / "all-MiniLM-L6-v2",
        backend_root / "data" / "models" / "all-MiniLM-L6-v2",
    ]
    for ld in local_dirs:
        m = ld / "model.onnx"
        t = ld / "tokenizer.json"
        if m.is_file() and t.is_file():
            return m, t

    # 3. Hugging Face hub cache resolution for pinned model
    hf_cache_dirs = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hf_cache_dirs.append(Path(hf_home) / "hub")
    hf_cache_dirs.append(Path.home() / ".cache" / "huggingface" / "hub")

    for hf_dir in hf_cache_dirs:
        repo_dir = hf_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        snap_dir = repo_dir / "snapshots" / PINNED_REVISION
        if snap_dir.is_dir():
            m = snap_dir / "onnx" / "model.onnx"
            t = snap_dir / "tokenizer.json"
            if m.is_file() and t.is_file():
                return m, t

    # Try huggingface_hub cache lookup utility
    try:
        from huggingface_hub import try_to_load_from_cache

        m_cached = try_to_load_from_cache(
            PINNED_REPO_ID,
            MODEL_ONNX_FILENAME,
            revision=PINNED_REVISION,
        )
        t_cached = try_to_load_from_cache(
            PINNED_REPO_ID,
            TOKENIZER_FILENAME,
            revision=PINNED_REVISION,
        )
        if isinstance(m_cached, str) and isinstance(t_cached, str):
            return Path(m_cached), Path(t_cached)
    except Exception:
        pass

    # If offline mode is NOT active, attempt to download pinned artifacts
    is_offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if not is_offline:
        try:
            from huggingface_hub import hf_hub_download

            m_down = hf_hub_download(PINNED_REPO_ID, MODEL_ONNX_FILENAME, revision=PINNED_REVISION)
            t_down = hf_hub_download(PINNED_REPO_ID, TOKENIZER_FILENAME, revision=PINNED_REVISION)
            return Path(m_down), Path(t_down)
        except Exception:
            pass

    raise FileNotFoundError(
        f"Pinned ONNX embedding model or tokenizer not found offline ({target_name}). "
        f"Run 'python scripts/download_onnx_model.py' to download and verify the pinned artifacts."
    )


def _is_model_cached(model_name: str | None = None) -> bool:
    """
    Checks whether the model files exist locally in the filesystem
    or in the Hugging Face cache.
    """
    from pathlib import Path

    if model_name:
        p = Path(model_name)
        if p.is_dir() or (p.is_file() and p.suffix == ".onnx"):
            return True

    try:
        m, t = _resolve_onnx_model_files(model_name)
        return m.is_file() and t.is_file()
    except Exception:
        return False


class OnnxEmbeddingModel:
    """
    The real embedding model, backed by ONNX Runtime CPUExecutionProvider
    and Hugging Face Tokenizers. Lazily loads the underlying session on first use.
    Uses the official pinned unquantized all-MiniLM-L6-v2 ONNX artifact (384 dimensions).
    """

    def __init__(
        self,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
        batch_size: int = 4,
    ):
        import threading

        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._model_path = Path(model_path) if model_path else None
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self._intra_op_num_threads = intra_op_num_threads
        self._inter_op_num_threads = inter_op_num_threads
        self._batch_size = batch_size

        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load(self):
        if self._session is None or self._tokenizer is None:
            with self._lock:
                if self._session is None or self._tokenizer is None:
                    if self._model_path and self._tokenizer_path:
                        m_path = self._model_path
                        t_path = self._tokenizer_path
                    else:
                        m_path, t_path = _resolve_onnx_model_files(self.model_name)

                    if not m_path.exists():
                        raise FileNotFoundError(f"ONNX model file not found: {m_path}")
                    if not t_path.exists():
                        raise FileNotFoundError(f"Tokenizer file not found: {t_path}")

                    try:
                        from tokenizers import Tokenizer

                        tok = Tokenizer.from_file(str(t_path))
                        tok.enable_truncation(max_length=256)
                        tok.enable_padding(pad_id=0, pad_token="[PAD]")
                        self._tokenizer = tok
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load tokenizer from {t_path}: {exc}") from exc

                    try:
                        import onnxruntime as ort

                        opts = ort.SessionOptions()
                        opts.intra_op_num_threads = self._intra_op_num_threads
                        opts.inter_op_num_threads = self._inter_op_num_threads
                        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                        sess = ort.InferenceSession(
                            str(m_path),
                            sess_options=opts,
                            providers=["CPUExecutionProvider"],
                        )
                        self._session = sess
                    except Exception as exc:
                        raise RuntimeError(f"Failed to load ONNX model session from {m_path}: {exc}") from exc

        return self._session, self._tokenizer

    def warmup(self):
        """Pre-loads the underlying ONNX session and tokenizer into memory."""
        self._load()
        _ = self.embed(["warmup"])
        return self

    @property
    def dimension(self) -> int:
        self._load()
        return 384

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        sess, tok = self._load()
        import numpy as np

        raw_texts = [str(t) for t in texts]
        all_embeddings: list[list[float]] = []

        for i in range(0, len(raw_texts), self._batch_size):
            batch = raw_texts[i : i + self._batch_size]
            encodings = tok.encode_batch(batch)

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            outputs = sess.run(None, inputs)
            last_hidden_state = outputs[0]  # shape: (batch_size, seq_len, 384)

            # Mean pooling with attention-mask weighting:
            # sum(last_hidden_state * mask) / clamp(sum(mask), min=1e-9)
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 normalization: v / clamp(norm, min=1e-12)
            norms = np.linalg.norm(mean_pooled, ord=2, axis=1, keepdims=True)
            normalized = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)

            all_embeddings.extend(normalized.tolist())

        return all_embeddings


# Backwards compatibility alias
SentenceTransformerEmbeddingModel = OnnxEmbeddingModel


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
