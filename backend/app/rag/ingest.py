"""
rag/ingest.py
================
The Phase 5 pipeline: reads evidence FROM the Phase 4 SQLite knowledge
base, chunks it, embeds it, and idempotently upserts it into ChromaDB.

    SQLite (LabelSectionRecord.text, found=True)
        |
        v
    section selection + clinical-priority ordering   [this file]
        |
        v
    chunk_text()                                       [rag/chunking.py]
        |
        v
    embedding_model.embed()                             [rag/embeddings.py]
        |
        v
    upsert_chunks() into ChromaDB                        [rag/vector_store.py]

WHY WE READ FROM SQLITE RATHER THAN CALLING RxNorm/DailyMed AGAIN
------------------------------------------------------------------------
SQLite (Phase 4) is the durable source of truth precisely so this step
doesn't need any network access at all — it's a pure local
transform-and-index operation over data already verified and stored.
This also means the ChromaDB index can always be safely deleted and
REBUILT from SQLite alone, without re-fetching anything from RxNorm or
DailyMed.

WHAT "CLINICAL PRIORITY" MEANS HERE
----------------------------------------
CLINICAL_PRIORITY_ORDER below determines the ORDER sections are
processed and reported in — it does NOT exclude any section. Every
section with found=True gets chunked, embedded, and indexed, regardless
of its priority rank; unlisted section types are processed last. This
matches the project requirement to "prioritize" these sections, not
restrict retrieval to only them.

HOW DETERMINISTIC CHUNK IDs MAKE THIS IDEMPOTENT
------------------------------------------------------
Each chunk's id is built as:
    f"link-{drug_label_link.id}-section-{label_section_record.id}-chunk-{chunk.index}"

All three components are stable database identifiers (or a chunk's
position, itself deterministic given unchanged input text and chunk
size). Re-running ingestion over unchanged data always recomputes the
exact same ids — so ChromaDB's upsert() replaces the same rows in place,
never creating duplicates. If a section's underlying text changes
between runs, that same id now maps to different chunk text, correctly
recorded as an "updated" chunk. If a section's text SHRINKS (produces
fewer chunks than before), the now-unused higher-numbered chunk ids from
the previous run are actively found (via a metadata filter on link_id +
section id) and deleted — otherwise they'd survive forever as orphaned,
outdated evidence. This stale-chunk cleanup is exercised directly in
tests/test_rag_ingest.py.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_rag_ingest.py -v

These tests use a REAL in-memory SQLite database, a REAL ChromaDB
instance (in a pytest tmp_path), and the DeterministicFakeEmbeddingModel
from rag/embeddings.py (see that module for exactly what fake-embedding
tests can and cannot verify) — so this file's PLUMBING is fully,
genuinely tested offline. Embedding QUALITY is a separate concern,
verified only by you, locally, with the real model (see
scripts/build_vector_index.py).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.db_models import Drug, DrugLabelLink, Label, LabelSectionRecord
from app.rag.chunking import chunk_text
from app.rag.vector_store import delete_chunks, get_existing_documents, get_ids_by_metadata, upsert_chunks
from app.schemas.rag_ingestion import RagIngestionResult

# Governs processing ORDER and reporting only — see module docstring.
CLINICAL_PRIORITY_ORDER = [
    "drug_interactions",
    "drug_lab_interactions",
    "contraindications",
    "warnings_and_precautions",
    "clinical_pharmacology",
    "adverse_reactions",
]


def _section_sort_key(section_key: str) -> tuple[int, str]:
    try:
        rank = CLINICAL_PRIORITY_ORDER.index(section_key)
    except ValueError:
        rank = len(CLINICAL_PRIORITY_ORDER)
    return (rank, section_key)


def _build_chunk_metadata(
    drug: Drug,
    label: Label,
    section: LabelSectionRecord,
    link: DrugLabelLink,
    chunk_index: int,
    total_chunks: int,
) -> dict:
    """
    Builds the full metadata dict attached to every chunk. Every field
    required by the project's provenance requirement is included, so any
    retrieved chunk is fully traceable back to its SQLite record and,
    from there (per docs/database.md), back to the original FDA document.
    ChromaDB metadata values must be str/int/float/bool — None values are
    normalized to "" for strings to keep filtering behavior predictable.
    """
    return {
        "drug_name": drug.normalized_name or "",
        "rxcui": drug.rxcui,
        "label_setid": label.setid or "",
        "spl_version": label.spl_version or "",
        "manufacturer": label.manufacturer or "",
        "section_key": section.section_key,
        "section_name": section.section_name,
        "section_code": section.section_code,
        "source_url": label.source_url or "",
        "api_url": label.api_url or "",
        "db_record_id": section.id,
        "link_id": link.id,
        "evidence_status": link.evidence_status,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }


def ingest_rag_index(
    session: Session,
    embedding_model,
    collection,
    rxcui: str | None = None,
) -> RagIngestionResult:
    """
    Reads found=True label sections from SQLite (optionally scoped to a
    single drug's rxcui), chunks + embeds their text, and idempotently
    upserts the result into the given ChromaDB collection.

    Never raises for "nothing to index" (empty database) — reported via
    RagIngestionResult.warnings instead, exactly like Phase 2-4's
    services handle their own "nothing found" cases.
    """
    result = RagIngestionResult()

    link_query = session.query(DrugLabelLink)
    if rxcui:
        link_query = link_query.join(Drug).filter(Drug.rxcui == rxcui)
    links = link_query.all()

    if not links:
        result.warnings.append(
            "No drug-label evidence found in the database to index."
            + (f" (rxcui={rxcui})" if rxcui else " Run drug ingestion (Phase 4 / scripts/ingest_labels.py) first.")
        )
        return result

    for link in links:
        drug = link.drug
        label = link.label

        if drug is None or label is None:
            # Data integrity anomaly: a link without its drug or label.
            # Should not happen given the FK constraints from Phase 4,
            # but handled defensively rather than crashing the whole run.
            result.malformed_records_skipped += 1
            result.warnings.append(
                f"Skipped drug_label_link id={link.id}: missing drug or label reference."
            )
            continue

        result.drugs_processed.add(drug.rxcui)

        sections = sorted(
            (s for s in label.sections if s.found),
            key=lambda s: _section_sort_key(s.section_key),
        )

        for section in sections:
            if not section.text or not section.text.strip():
                # found=True but empty text: shouldn't happen given Phase 3's
                # _is_meaningful_text() check, but defended against here
                # rather than silently embedding nothing or crashing.
                result.malformed_records_skipped += 1
                result.warnings.append(
                    f"Section id={section.id} ({section.section_key}) is marked found=True "
                    "but has empty text — skipped as a data integrity anomaly."
                )
                continue

            chunks = chunk_text(
                section.text,
                chunk_size_words=_chunk_size_words(),
                overlap_words=_chunk_overlap_words(),
            )
            if not chunks:
                result.skipped_empty_sections += 1
                continue

            ids = [f"link-{link.id}-section-{section.id}-chunk-{c.index}" for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [
                _build_chunk_metadata(drug, label, section, link, c.index, len(chunks)) for c in chunks
            ]

            existing = get_existing_documents(collection, ids)
            to_upsert_ids: list[str] = []
            to_upsert_docs: list[str] = []
            to_upsert_meta: list[dict] = []

            for chunk_id, doc, meta in zip(ids, documents, metadatas):
                if chunk_id not in existing:
                    result.chunks_added += 1
                elif existing[chunk_id] != doc:
                    result.chunks_updated += 1
                else:
                    result.chunks_unchanged += 1
                    continue  # identical content already stored — skip re-embedding it
                to_upsert_ids.append(chunk_id)
                to_upsert_docs.append(doc)
                to_upsert_meta.append(meta)

            if to_upsert_ids:
                embeddings = embedding_model.embed(to_upsert_docs)
                upsert_chunks(collection, to_upsert_ids, to_upsert_docs, embeddings, to_upsert_meta)

            # Stale chunk cleanup: if this section produced FEWER chunks
            # this run than a previous run did, remove the leftover
            # higher-index chunk ids so they don't survive as orphaned,
            # outdated evidence.
            existing_ids_for_section = get_ids_by_metadata(
                collection, {"$and": [{"link_id": link.id}, {"db_record_id": section.id}]}
            )
            stale_ids = [i for i in existing_ids_for_section if i not in ids]
            if stale_ids:
                delete_chunks(collection, stale_ids)
                result.chunks_deleted_stale += len(stale_ids)

            result.sections_processed += 1

    return result


def _chunk_size_words() -> int:
    from app.config import settings

    return settings.RAG_CHUNK_SIZE_WORDS


def _chunk_overlap_words() -> int:
    from app.config import settings

    return settings.RAG_CHUNK_OVERLAP_WORDS
