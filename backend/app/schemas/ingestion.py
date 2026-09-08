"""
schemas/ingestion.py
=======================
The result object returned by knowledge_base_service.ingest_drug(), used
by ingestion scripts (and tests) to report exactly what happened —
per the project requirement to always report what was added/skipped/
updated, never silently.
"""

from typing import Literal

from pydantic import BaseModel, Field

UpsertStatus = Literal["added", "updated", "unchanged"]
DrugIngestStatus = Literal["added", "updated", "unchanged", "failed"]


class IngestionResult(BaseModel):
    drug_name: str = Field(description="The name that was submitted for ingestion.")
    rxcui: str | None = None

    drug_status: DrugIngestStatus = Field(
        description="'failed' means RxNorm normalization did not resolve to a usable RxCUI."
    )
    label_status: UpsertStatus | None = Field(
        default=None, description="None if no label was found/stored (see evidence_status)."
    )

    sections_added: int = 0
    sections_updated: int = 0
    sections_unchanged: int = 0

    link_status: UpsertStatus | None = None
    evidence_status: str | None = Field(
        default=None,
        description="Same evidence_status values as DrugLabelResult: interaction_evidence_found / "
        "label_found_no_interaction_section / no_label_found / error / None (drug lookup itself failed).",
    )

    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
