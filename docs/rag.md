# RAG (Retrieval-Augmented Generation) — Phase 5

## The one-sentence explanation for your viva

> **SQLite stores the trusted evidence. ChromaDB helps us quickly find
> the most relevant evidence. The LLM comes later and explains that
> evidence.**

That's the whole idea. Everything below is detail on top of that.

## Why "retrieval-augmented" at all?

An LLM asked "does warfarin interact with ibuprofen?" with no other
information would answer from whatever it happened to learn during
training — which might be outdated, might be subtly wrong, and offers no
way to check where the answer came from. That's unacceptable for a
clinical-information tool.

RAG flips the order: **first** find the actual, real FDA label text that
might be relevant, **then** (in Phase 6+) have the LLM read *only* that
retrieved text and explain it — never inventing anything beyond it, and
always able to cite exactly which document it came from.

Phase 5 builds the "first" part: retrieval. It does not yet do the
"then" part — there is no LLM in this phase at all.

## The three-store mental model

| Store | Role | Analogy |
|---|---|---|
| **SQLite** (Phase 4) | The permanent, trusted record of every piece of evidence, with full source traceability | A library's card catalog *and* the actual books |
| **ChromaDB** (Phase 5, this phase) | A fast index for finding which evidence is *relevant* to a question | The library's search desk — points you to the right book fast |
| **LLM** (Phase 6+, not built yet) | Reads the specific pages the search desk pointed to, and explains them in plain language | A librarian who reads the exact pages you were pointed to and summarizes them for you — without making anything up |

ChromaDB never contains information that isn't already in SQLite — it's
a **derived, rebuildable index**, not a second source of truth. If it
were ever deleted or corrupted, `scripts/build_vector_index.py` can
regenerate it completely from SQLite alone, with no need to re-contact
RxNorm or DailyMed.

## The pipeline, step by step

```
SQLite: LabelSectionRecord.text (found=True sections only)
        |
        v
Section extraction + clinical-priority ordering
    (Drug Interactions and Drug/Lab Test Interactions processed first,
     then Contraindications, Warnings & Precautions, Clinical
     Pharmacology, Adverse Reactions, then anything else —
     see app/rag/ingest.py's CLINICAL_PRIORITY_ORDER)
        |
        v
Cleaning (whitespace normalization — defensive, on top of Phase 3's own cleaning)
        |
        v
Chunking (app/rag/chunking.py)
    Splits long section text into ~180-word overlapping windows, so each
    chunk is small enough to embed well and still captures a complete
    thought even near a boundary (40-word overlap).
        |
        v
Embedding (app/rag/embeddings.py)
    Converts each chunk's text into a 384-number vector using
    all-MiniLM-L6-v2 — see docs/embeddings.md for the full model choice
    rationale.
        |
        v
ChromaDB (app/rag/vector_store.py)
    Stores each chunk's vector alongside its original text AND full
    source metadata (drug, rxcui, label setid, section, source URL,
    database record id, chunk id) — never just the vector alone.
```

## What "relevant" means here — and its limits

ChromaDB finds chunks whose embeddings are *close* to a query's
embedding in vector space. This is a statistical measure of
textual/semantic similarity, computed entirely locally, with no
knowledge of pharmacology beyond whatever patterns the embedding model
picked up during its own training.

**This is the single most important thing to understand for the
viva:** a low "distance" (high similarity) score means *"this stored
text reads like it's about the same topic as the query"* — nothing more.
It is never treated, anywhere in this codebase, as:

- an interaction severity rating,
- a probability that an interaction exists,
- or proof that two drugs interact.

`app/services/retrieval_service.py`'s `retrieve()` function and
`app/schemas/retrieval.py`'s `RetrievedChunk.distance` field both carry
this warning directly in their docstrings/field descriptions, precisely
so this constraint stays visible to anyone (including future-you) who
extends this code later.

## Provenance: every chunk is traceable

Every chunk stored in ChromaDB carries this metadata, copied straight
from its Phase 4 SQLite record:

- `drug_name`, `rxcui`
- `label_setid`, `spl_version`, `manufacturer`
- `section_key`, `section_name`, `section_code` (LOINC)
- `source_url`, `api_url`
- `db_record_id` (the exact `LabelSectionRecord.id` it came from)
- `chunk_id`, `chunk_index`, `total_chunks`
- `evidence_status` (from Phase 3/4 — factual metadata about the label
  this chunk came from, not a claim about the chunk's own content)

Nothing is chunked or embedded without this metadata attached — see
`tests/test_rag_ingest.py::test_ingest_preserves_all_required_metadata_fields`
for a concrete, executed test proving every one of these fields survives
the full pipeline intact.

## Idempotency: how "run it twice, get the same result" actually works

Every chunk's ID is built deterministically:

```
f"link-{drug_label_link.id}-section-{label_section_record.id}-chunk-{chunk.index}"
```

All three components are stable — they come from database row IDs (which
never change once created) and a chunk's position (which is
deterministic given unchanged text and chunk-size settings). Re-running
`ingest_rag_index()` over unchanged data always recomputes the exact
same IDs, and ChromaDB's `upsert()` replaces content at an existing ID
rather than duplicating it — so nothing grows unboundedly.

Three specific behaviors, each with a dedicated executed test:

1. **Unchanged content** → reported `chunks_unchanged`, not re-embedded
   or re-written at all (`test_ingest_twice_does_not_duplicate_vectors`).
2. **Genuinely changed text** (e.g. FDA updated a label) → the same
   chunk ID now maps to different text, reported `chunks_updated`, and
   replaced in place (`test_ingest_detects_real_text_change_as_update_not_duplicate`).
3. **Shrunk text** (fewer chunks than before) → the now-unused
   higher-index chunk IDs from the previous run are actively found and
   deleted, reported `chunks_deleted_stale`
   (`test_ingest_removes_stale_chunks_when_section_shrinks`).

## Configurable retrieval parameters

`retrieve()` accepts:

- `top_k` — how many chunks to return (defaults to
  `settings.RAG_DEFAULT_TOP_K`, configurable via `.env`)
- `rxcui` — restrict results to one specific drug
- `section_key` — restrict results to one section type (e.g. only
  `"drug_interactions"` sections)

Both filters can be combined; see `app/rag/vector_store.py::build_where()`
for how they're translated into ChromaDB's filter syntax.

## What Phase 5 explicitly does NOT do

- **No LLM.** Nothing in this phase generates natural-language
  explanations. `retrieve()` returns raw chunk text and metadata only.
- **No interaction determination.** A vector similarity match is never
  interpreted as evidence that an interaction exists between two named
  drugs — that inference doesn't happen anywhere in this phase's code.
- **No severity of any kind.** Same principle as Phase 3/4: nothing here
  invents or infers a severity level.

Phase 6 will build the retrieval *logic* that decides what to query for
given a pair of drugs the user asks about (e.g. querying for each drug's
evidence, or for evidence mentioning both). Phase 7 will add the LLM
that reads what Phase 6 retrieves and produces a grounded, cited
explanation — still never going beyond what the retrieved text actually
supports.
