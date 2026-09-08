# Embedding Model Selection

## What was selected

**`sentence-transformers/all-MiniLM-L6-v2`**

| Property | Value |
|---|---|
| Type | General-purpose sentence embedding model |
| Embedding dimension | 384 |
| Parameters | ~22 million |
| Download size | ~90 MB |
| Max input length | 256 word pieces (longer input is truncated by the model itself — one more reason Phase 5 chunks text before embedding) |
| Hardware needed | CPU only — no GPU required |
| License | Apache 2.0 |
| Library | `sentence-transformers` (already in `requirements.txt` from Phase 1) |

## Why this model, not a biomedical one — the actual trade-off

This is a **general-purpose** model, not a biomedical/clinical-domain
one. That was a deliberate decision after weighing both options, not a
default picked because it's popular.

### The case for a general-purpose model (what was chosen)

- **Resource footprint.** ~90MB, runs comfortably on any CPU with no
  special setup. This matters concretely for this project: it needs to
  run on a student's laptop, be reproducible for a viva demonstration,
  and not require GPU infrastructure.
- **Maturity and support.** `all-MiniLM-L6-v2` is one of the most widely
  used and tested sentence-transformers models in existence — extensive
  community usage, stable behavior, well-documented.
- **Retrieval quality is helped by metadata filtering, not just
  semantics.** Phase 4's SQLite schema and Phase 5's ChromaDB metadata
  let retrieval filter to one specific drug (`rxcui`) and/or one section
  type (`section_key`) *before* semantic ranking even happens. The
  embedding model only has to rank an already-narrowed candidate set
  well — it doesn't have to carry the entire burden of understanding
  medical semantics across the whole corpus unaided.
- **License and reproducibility.** Apache 2.0, no usage restrictions,
  simple to cite and explain in an academic report.

### The case for a biomedical-domain model (considered, not selected)

Models such as **`pritamdeka/S-PubMedBert-MS-MARCO`** or
**`NeuML/pubmedbert-base-embeddings`** are trained specifically on
biomedical/clinical text, and would likely be *better* at:

- Recognizing that "MI" and "myocardial infarction" are the same
  concept, or that "renal impairment" and "kidney dysfunction" are
  related — general-purpose models see less of this kind of domain
  synonym during training.
- Ranking clinically-similar passages more accurately when the query
  uses different phrasing than the source label text.

**Costs of that choice**, which is why it wasn't selected as the MVP
default:

- Larger download (400–900MB depending on the specific model), slower
  CPU inference.
- Typically 768-dimension embeddings — more storage, slower similarity
  search at scale (not a concern at this project's current data volume,
  but worth naming).
- Less consistent tooling/community support than the mainstream
  sentence-transformers ecosystem — some biomedical embedding models
  need extra steps to work well with plain `sentence-transformers`
  loading conventions, or have less predictable version compatibility.
- Not obviously a large enough quality improvement to justify the extra
  weight for an MVP whose retrieval is already narrowed by structured
  metadata filtering (see above).

### The actual decision

Use `all-MiniLM-L6-v2` now, as the accessible, well-supported MVP
default — and treat a biomedical-model upgrade as a **documented future
option**, not a rejected idea. Phase 13 (Evaluation) is the natural
place to actually A/B these two options against a small labeled test set
and settle the question with evidence rather than assumption, once the
full pipeline (through the LLM layer) exists to evaluate end-to-end.

## Why embeddings are computed ourselves, not left to ChromaDB's default

ChromaDB can optionally embed text for you automatically via a "default
embedding function" — but that function also downloads its own small
model from HuggingFace on first use, which is the exact same network
dependency this project's own embedding step has. This project always
computes embeddings explicitly (via `app/rag/embeddings.py`) and passes
them directly into ChromaDB (`embedding_function=None` on the
collection), which means:

1. ChromaDB itself never needs network access — confirmed by running
   `tests/test_vector_store.py` fully offline in this project's build
   sandbox.
2. The embedding model is swappable in one place (this file) without
   touching ChromaDB integration code at all.

## Local resource requirements, concretely

- **Disk:** ~90MB for the model weights (downloaded once, cached locally
  by `sentence-transformers`/`huggingface_hub` after the first run).
- **RAM:** a few hundred MB during inference — trivial for any modern
  laptop.
- **CPU:** no GPU needed; encoding a single short sentence takes a
  fraction of a second on CPU.
- **Network:** required ONLY the first time the model is used (to
  download it) — after that, it's cached locally and inference is fully
  offline.

## What could NOT be verified in this project's build sandbox

This sandbox's network egress does not permit reaching `huggingface.co`
(confirmed directly — see the "FAILED" result documented in this
project's build history and in `tests/test_embeddings.py`'s
auto-skipping real-model test). That means the following could **not**
be executed or verified here, and must be run by you locally:

- Actually downloading `all-MiniLM-L6-v2`.
- Verifying its real embedding dimension (384) in practice, not just per
  its published model card.
- Any judgment about actual retrieval *quality* / semantic relevance —
  see `scripts/test_embeddings_live.py` and `scripts/query_index_live.py`,
  which you should run yourself.

What **was** verified in this sandbox, genuinely: the full chunking →
embedding-interface → ChromaDB pipeline, using a small deterministic
fake embedding model (`DeterministicFakeEmbeddingModel`, see
`app/rag/embeddings.py`) that exercises every piece of *plumbing*
(chunk IDs, metadata, idempotent upsert, stale-chunk cleanup, retrieval
filtering) without needing the real model at all. That distinction —
plumbing vs. quality — is maintained throughout this phase's test suite
and is the direct answer to this phase's requirement to keep these two
categories of testing separate and honestly labeled.

## Alternatives considered, summarized

| Model | Dim | Size | Domain | Verdict |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** (selected) | 384 | ~90MB | General | Best fit for MVP: light, mature, sufficient given metadata filtering |
| all-mpnet-base-v2 | 768 | ~420MB | General | Better general quality than MiniLM, but heavier — no clear win for this domain-narrowed use case |
| pritamdeka/S-PubMedBert-MS-MARCO | 768 | ~440MB | Biomedical | Likely better clinical semantic matching; documented as a future evaluation candidate |
| NeuML/pubmedbert-base-embeddings | 768 | ~440MB | Biomedical | Same trade-off profile as above |
| OpenAI/Gemini embedding APIs | varies | N/A (API) | General | Rejected: adds a paid external dependency and a network requirement for every single query, contradicting the project's "runs locally" requirement |
