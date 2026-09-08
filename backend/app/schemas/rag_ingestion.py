"""
schemas/rag_ingestion.py
===========================
Report object returned by rag/ingest.py's ingest_rag_index(), so every
run reports exactly what happened — added/updated/unchanged/deleted —
never silently.
"""

from pydantic import BaseModel, Field


class RagIngestionResult(BaseModel):
    drugs_processed: set[str] = Field(
        default_factory=set, description="RxCUIs of drugs whose evidence was processed."
    )
    sections_processed: int = 0

    chunks_added: int = 0
    chunks_updated: int = 0
    chunks_unchanged: int = 0
    chunks_deleted_stale: int = Field(
        default=0,
        description="Chunks removed because a section's text shrank and produced fewer chunks than before.",
    )

    skipped_empty_sections: int = Field(
        default=0, description="Sections that produced zero chunks after cleaning (e.g. whitespace-only)."
    )
    malformed_records_skipped: int = Field(
        default=0,
        description="Section rows marked found=True but with empty/missing text, or links with missing drug/label — data integrity anomalies, not crashes.",
    )

    warnings: list[str] = Field(default_factory=list)
