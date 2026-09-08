"""
models/db_models.py
======================
SQLAlchemy ORM models for the local knowledge base.

WHAT THIS DATABASE IS — AND ISN'T
------------------------------------
This is an EVIDENCE / DOCUMENT STORE, not a curated drug-interaction
database. It records:
  - which drugs we've normalized (via RxNorm),
  - which FDA labels (via DailyMed) we found for them,
  - the specific label SECTIONS we extracted (interactions, warnings,
    etc.) with their original source text,
  - and a link connecting a drug to the label evidence retrieved for it.

It does NOT contain:
  - any severity rating,
  - any "Drug A interacts with Drug B" relationship we invented,
  - any judgment about whether an interaction is clinically significant.

Those would require either a licensed curated database or an LLM
inference step over this evidence — neither belongs in the storage layer.
Storing a label's text is NOT the same as "we have identified an
interaction" — see LabelSectionRecord.found and DrugLabelLink.evidence_status
below, which is exactly how that distinction is preserved in the schema.

WHY FOUR TABLES (not one big flat table)
--------------------------------------------
1. Drug       — one row per unique RxCUI. A drug is looked up once,
                 reused across many later queries.
2. Label       — one row per unique DailyMed SPL (identified by setid +
                 spl_version). Multiple drugs could theoretically point at
                 related labels, and the same label should never be
                 re-fetched/re-stored just because it was looked up again.
3. LabelSectionRecord — one row per (label, section) pair. This is the
                 actual evidence text, with its LOINC code preserved for
                 traceability and later filtering.
4. DrugLabelLink — the join between a Drug and the Label DailyMed
                 resolved for it, carrying the evidence_status of that
                 specific lookup (interaction_evidence_found /
                 label_found_no_interaction_section / etc.) and when it
                 was retrieved.

This shape means: "give me all Drug Interactions section text for
warfarin" is a simple, indexed join — and "was this exact label already
stored" is a simple unique-constraint lookup, which is what makes
ingestion idempotent (see app/services/knowledge_base_service.py).

WHY THIS IS COMPATIBLE WITH THE LATER CHROMADB/RAG LAYER (Phase 5+)
------------------------------------------------------------------------
LabelSectionRecord.text is exactly the unit of content Phase 5 will chunk
and embed. LabelSectionRecord's id, section_code, section_name, and the
parent Label's setid/source_url/api_url are exactly the metadata Phase 5
will attach to each embedded chunk (see the project's METADATA
requirements: drug name, normalized identifier, source, document ID,
section, URL, chunk ID). Nothing about this schema needs to change to
support that — Phase 5 reads FROM this table, it doesn't replace it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Drug(Base):
    """One row per unique RxCUI — a normalized drug concept from RxNorm (Phase 2)."""

    __tablename__ = "drugs"

    id = Column(Integer, primary_key=True)
    rxcui = Column(String, nullable=False, unique=True, index=True)
    normalized_name = Column(String, nullable=False, index=True)
    term_type = Column(String, nullable=True)  # RxNorm TTY: IN, BN, SCD, SBD, etc.
    generic_name = Column(String, nullable=True)
    input_name = Column(String, nullable=False)  # what the user/ingestion script actually typed
    match_type = Column(String, nullable=False)  # "exact" | "approximate"
    match_score = Column(Float, nullable=True)
    source = Column(String, nullable=False, default="RxNorm")
    source_url = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    synonyms = relationship("DrugSynonym", back_populates="drug", cascade="all, delete-orphan")
    links = relationship("DrugLabelLink", back_populates="drug", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover - debug convenience only
        return f"<Drug rxcui={self.rxcui} name={self.normalized_name!r}>"


class DrugSynonym(Base):
    """Brand names and other RxNorm-related synonym names for a drug."""

    __tablename__ = "drug_synonyms"
    __table_args__ = (
        UniqueConstraint("drug_id", "synonym", "synonym_type", name="uq_drug_synonym"),
    )

    id = Column(Integer, primary_key=True)
    drug_id = Column(Integer, ForeignKey("drugs.id"), nullable=False, index=True)
    synonym = Column(String, nullable=False, index=True)
    synonym_type = Column(String, nullable=False)  # "brand" | "synonym"

    drug = relationship("Drug", back_populates="synonyms")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DrugSynonym {self.synonym_type}={self.synonym!r}>"


class Label(Base):
    """One row per unique DailyMed SPL (Structured Product Label), identified by setid + version."""

    __tablename__ = "labels"
    __table_args__ = (
        UniqueConstraint("setid", "spl_version", name="uq_label_setid_version"),
    )

    id = Column(Integer, primary_key=True)
    setid = Column(String, nullable=False, index=True)
    spl_version = Column(String, nullable=True)
    title = Column(String, nullable=True)
    manufacturer = Column(String, nullable=True, index=True)
    published_date = Column(String, nullable=True)
    source_url = Column(String, nullable=True)
    api_url = Column(String, nullable=True)

    retrieved_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    sections = relationship(
        "LabelSectionRecord", back_populates="label", cascade="all, delete-orphan"
    )
    links = relationship("DrugLabelLink", back_populates="label", cascade="all, delete-orphan")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Label setid={self.setid} v{self.spl_version} title={self.title!r}>"


class LabelSectionRecord(Base):
    """
    One row per (label, section) pair — the actual evidence text.

    found=False means we explicitly checked for this section (by its
    official LOINC code) and it was absent or contained only placeholder
    content. This row still gets stored (with empty text) so there is a
    permanent, queryable record of "we looked and it wasn't there" —
    which is itself useful provenance, distinct from "we never checked".
    """

    __tablename__ = "label_sections"
    __table_args__ = (
        UniqueConstraint("label_id", "section_code", name="uq_label_section"),
    )

    id = Column(Integer, primary_key=True)
    label_id = Column(Integer, ForeignKey("labels.id"), nullable=False, index=True)
    section_key = Column(String, nullable=False, index=True)  # internal key, e.g. "drug_interactions"
    section_code = Column(String, nullable=False, index=True)  # official LOINC code, e.g. "34073-7"
    section_name = Column(String, nullable=False)
    text = Column(Text, nullable=False, default="")
    found = Column(Boolean, nullable=False, default=False, index=True)

    updated_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    label = relationship("Label", back_populates="sections")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LabelSectionRecord {self.section_key} found={self.found}>"


class DrugLabelLink(Base):
    """
    Join between a Drug and the Label that DailyMed resolved for it,
    carrying the evidence_status of that specific lookup. This is where
    "we found this label for this drug, and here's what kind of evidence
    it gave us" is recorded — it is NOT a drug-drug interaction record.
    """

    __tablename__ = "drug_label_links"
    __table_args__ = (
        UniqueConstraint("drug_id", "label_id", name="uq_drug_label_link"),
    )

    id = Column(Integer, primary_key=True)
    drug_id = Column(Integer, ForeignKey("drugs.id"), nullable=False, index=True)
    label_id = Column(Integer, ForeignKey("labels.id"), nullable=False, index=True)

    evidence_status = Column(String, nullable=False, index=True)
    candidates_checked = Column(Integer, nullable=False, default=0)
    total_candidates_found = Column(Integer, nullable=False, default=0)

    retrieved_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow)

    drug = relationship("Drug", back_populates="links")
    label = relationship("Label", back_populates="links")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DrugLabelLink drug_id={self.drug_id} label_id={self.label_id} status={self.evidence_status}>"
