"""
schemas/retrieval.py
=======================
The result object returned by services/retrieval_service.retrieve().

CRITICAL: what `distance` is, and is not
--------------------------------------------
`distance` is a vector-space distance between a query embedding and a
chunk's embedding — a RETRIEVAL-RELEVANCE signal only (lower = more
textually/semantically similar per the embedding model used).

It must NEVER be interpreted, displayed, or used downstream as a
clinical interaction severity rating, a probability that an interaction
exists, or proof that two drugs interact. See
services/retrieval_service.retrieve()'s docstring for the full
explanation — repeated here on the field itself so this constraint stays
visible wherever this schema is used, including in Phase 6+.
"""

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str

    text: str = Field(description="The retrieved chunk's original evidence text, verbatim.")

    distance: float = Field(
        description=(
            "Vector-space distance between the query and this chunk — a retrieval-relevance "
            "signal ONLY. NOT a clinical severity score, NOT an interaction probability, and "
            "NOT proof that any interaction exists. Lower means more similar text, nothing more."
        )
    )

    drug_name: str | None = None
    rxcui: str | None = None

    label_setid: str | None = None
    spl_version: str | None = None
    manufacturer: str | None = None

    section_key: str | None = None
    section_name: str | None = None
    section_code: str | None = None

    source_url: str | None = None
    api_url: str | None = None

    db_record_id: int | None = Field(
        default=None, description="The LabelSectionRecord.id (Phase 4 SQLite) this chunk was derived from."
    )
    evidence_status: str | None = Field(
        default=None,
        description="The evidence_status (Phase 3/4) of the label this chunk came from — factual metadata, not a claim about this specific chunk's content.",
    )
