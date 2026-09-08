"""
services/knowledge_base_service.py
=====================================
Orchestrates RxNorm normalization (Phase 2) + DailyMed label retrieval
(Phase 3) into the local SQLite knowledge base (Phase 4), and provides
the retrieval helpers later phases (and this phase's tests) use to query
what's been stored.

WHAT WE ARE DOING
------------------
`ingest_drug(name)` is the single entry point that: normalizes a drug
name, fetches its FDA label evidence, and stores/updates the result in
SQLite — all as one idempotent operation.

WHY IDEMPOTENCY MATTERS HERE (and how it's implemented)
------------------------------------------------------------
This system will be re-run against the same drugs repeatedly — during
development, in scheduled refreshes, or just because a user searches a
drug that's already in the database. Every "insert" in this module is
actually an "upsert": look for an existing row by its natural unique key
first (RxCUI for drugs, setid+version for labels, label+section-code for
sections, drug+label for links), and only insert if it doesn't exist.
If it exists but the incoming data differs, we UPDATE it and report
"updated"; if it's identical, we report "unchanged" and touch nothing.
This is what makes "run ingestion twice" safe — see docs/database.md for
the exact unique constraints enforcing this at the database level, not
just in application logic (so it holds even under concurrent access).

WHAT WE DELIBERATELY DO NOT DO
-----------------------------------
- We do NOT create any "Drug A interacts with Drug B" record. The only
  relationship stored is Drug -> Label (via DrugLabelLink), which
  represents "this label is the evidence we found for this drug" — not
  a drug-drug interaction claim.
- We do NOT skip storing a section just because found=False. An explicit
  "we checked and it wasn't there" row is itself valuable provenance,
  and its absence must not later be misread as "we never checked".
- We do NOT touch label section TEXT once written except to correct it
  to match a fresh fetch — never to summarize, clean up, or "improve" it.
  Preserving the original evidence text verbatim is what makes later RAG
  answers traceable back to the real FDA source (a hard project
  requirement).

HOW THIS CONNECTS RxNorm -> DailyMed -> SQLite -> Phase 5
----------------------------------------------------------------
    normalize_drug_name()      [Phase 2, RxNorm]
            |
            v
    get_drug_label_info()      [Phase 3, DailyMed]
            |
            v
    ingest_drug()               [Phase 4, THIS FILE]
            |
            v
    Drug / Label / LabelSectionRecord / DrugLabelLink rows in SQLite
            |
            v
    Phase 5 will read LabelSectionRecord.text rows, chunk them, embed
    them, and store the embeddings + this same provenance metadata
    (section_code, source_url, setid, drug rxcui, etc.) in ChromaDB.
    SQLite remains the durable, traceable source of truth; ChromaDB will
    hold a derived, re-buildable search index over it.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_knowledge_base_service.py -v

These tests use a REAL in-memory SQLite database (no mocking needed for
the DB layer) and MOCK only the two upstream network calls
(normalize_drug_name, get_drug_label_info) with monkeypatch, since this
module's own job is orchestration + storage, not talking to the network
itself — that's already covered by test_rxnorm_service.py and
test_dailymed_service.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.models.db_models import Drug, DrugLabelLink, DrugSynonym, Label, LabelSectionRecord
from app.schemas.drug import DrugNormalizationResult
from app.schemas.ingestion import IngestionResult
from app.schemas.label import DrugLabelResult
from app.services.dailymed_service import get_drug_label_info
from app.services.rxnorm_service import normalize_drug_name


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------------------------

def _sync_synonyms(session: Session, drug: Drug, norm: DrugNormalizationResult) -> bool:
    """
    Reconciles drug.synonyms to exactly match the current brand_names +
    synonyms from a fresh RxNorm lookup: adds missing ones, removes ones
    no longer reported. Returns True if anything changed.
    """
    desired: set[tuple[str, str]] = {(b, "brand") for b in norm.brand_names} | {
        (s, "synonym") for s in norm.synonyms
    }
    existing_by_key = {(s.synonym, s.synonym_type): s for s in drug.synonyms}

    changed = False

    for key in existing_by_key.keys() - desired:
        session.delete(existing_by_key[key])
        changed = True

    for synonym, synonym_type in desired - existing_by_key.keys():
        session.add(DrugSynonym(drug_id=drug.id, synonym=synonym, synonym_type=synonym_type))
        changed = True

    return changed


def _get_or_create_drug(session: Session, norm: DrugNormalizationResult) -> tuple[Drug, str]:
    """Upserts a Drug row keyed by RxCUI. Returns (drug, status)."""
    existing = session.query(Drug).filter_by(rxcui=norm.rxcui).one_or_none()

    if existing is None:
        drug = Drug(
            rxcui=norm.rxcui,
            normalized_name=norm.normalized_name or norm.input_name,
            term_type=norm.term_type,
            generic_name=norm.generic_name,
            input_name=norm.input_name,
            match_type=norm.match_type,
            match_score=norm.match_score,
            source=norm.source,
            source_url=norm.source_url,
        )
        session.add(drug)
        session.flush()  # assigns drug.id without committing the transaction
        _sync_synonyms(session, drug, norm)
        return drug, "added"

    changed = False
    field_updates = {
        "normalized_name": norm.normalized_name or norm.input_name,
        "term_type": norm.term_type,
        "generic_name": norm.generic_name,
        "input_name": norm.input_name,
        "match_type": norm.match_type,
        "match_score": norm.match_score,
        "source_url": norm.source_url,
    }
    for field, new_value in field_updates.items():
        if getattr(existing, field) != new_value:
            setattr(existing, field, new_value)
            changed = True

    session.flush()
    synonyms_changed = _sync_synonyms(session, existing, norm)

    if changed or synonyms_changed:
        existing.updated_at = _utcnow()
        return existing, "updated"
    return existing, "unchanged"


def _get_or_create_label(session: Session, label_result: DrugLabelResult) -> tuple[Label, str]:
    """Upserts a Label row keyed by (setid, spl_version). Returns (label, status)."""
    existing = (
        session.query(Label)
        .filter_by(setid=label_result.setid, spl_version=label_result.spl_version)
        .one_or_none()
    )

    if existing is None:
        label = Label(
            setid=label_result.setid,
            spl_version=label_result.spl_version,
            title=label_result.title,
            manufacturer=label_result.manufacturer,
            published_date=label_result.published_date,
            source_url=label_result.source_url,
            api_url=label_result.api_url,
        )
        session.add(label)
        session.flush()
        return label, "added"

    changed = False
    field_updates = {
        "title": label_result.title,
        "manufacturer": label_result.manufacturer,
        "published_date": label_result.published_date,
        "source_url": label_result.source_url,
        "api_url": label_result.api_url,
    }
    for field, new_value in field_updates.items():
        if getattr(existing, field) != new_value:
            setattr(existing, field, new_value)
            changed = True

    if changed:
        existing.updated_at = _utcnow()
        return existing, "updated"
    return existing, "unchanged"


def _sync_sections(session: Session, label: Label, sections: dict) -> dict[str, int]:
    """
    Upserts LabelSectionRecord rows for every section in `sections`
    (keyed by our internal section_key, e.g. "drug_interactions"),
    matching existing rows by (label_id, section_code).
    Returns counts of {"added": n, "updated": n, "unchanged": n}.
    """
    counts = {"added": 0, "updated": 0, "unchanged": 0}
    existing_by_code = {s.section_code: s for s in label.sections}

    for key, section in sections.items():
        existing = existing_by_code.get(section.section_code)

        if existing is None:
            session.add(
                LabelSectionRecord(
                    label_id=label.id,
                    section_key=key,
                    section_code=section.section_code,
                    section_name=section.section_name,
                    text=section.text,
                    found=section.found,
                )
            )
            counts["added"] += 1
            continue

        changed = False
        if existing.text != section.text:
            existing.text = section.text
            changed = True
        if existing.found != section.found:
            existing.found = section.found
            changed = True
        if existing.section_name != section.section_name:
            existing.section_name = section.section_name
            changed = True

        if changed:
            existing.updated_at = _utcnow()
            counts["updated"] += 1
        else:
            counts["unchanged"] += 1

    return counts


def _get_or_create_link(
    session: Session, drug: Drug, label: Label, label_result: DrugLabelResult
) -> str:
    """Upserts a DrugLabelLink row keyed by (drug_id, label_id). Returns status."""
    existing = (
        session.query(DrugLabelLink).filter_by(drug_id=drug.id, label_id=label.id).one_or_none()
    )
    now = _utcnow()

    if existing is None:
        session.add(
            DrugLabelLink(
                drug_id=drug.id,
                label_id=label.id,
                evidence_status=label_result.evidence_status,
                candidates_checked=label_result.candidates_checked,
                total_candidates_found=label_result.total_candidates_found,
                retrieved_at=now,
            )
        )
        return "added"

    changed = (
        existing.evidence_status != label_result.evidence_status
        or existing.candidates_checked != label_result.candidates_checked
        or existing.total_candidates_found != label_result.total_candidates_found
    )
    if changed:
        existing.evidence_status = label_result.evidence_status
        existing.candidates_checked = label_result.candidates_checked
        existing.total_candidates_found = label_result.total_candidates_found

    # Always refresh the retrieval timestamp — re-checking evidence is
    # itself meaningful provenance ("last verified at ..."), even if the
    # content turned out to be unchanged.
    existing.retrieved_at = now

    return "updated" if changed else "unchanged"


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_drug(drug_name: str, session: Session) -> IngestionResult:
    """
    Normalizes `drug_name` via RxNorm, fetches DailyMed label evidence for
    it, and idempotently upserts everything into the knowledge base using
    the given SQLAlchemy session (caller owns the session's lifecycle —
    this function does not commit or close it, so callers can batch
    multiple ingestions in one transaction if desired).

    Never raises for "drug not found" or "no label found" — those are
    reported via the returned IngestionResult's status fields, exactly
    like the underlying Phase 2/3 services do.
    """
    norm = normalize_drug_name(drug_name)

    if norm.match_type == "error":
        return IngestionResult(
            drug_name=drug_name,
            rxcui=None,
            drug_status="failed",
            warnings=norm.warnings,
            error=norm.error,
        )
    if norm.match_type == "none" or not norm.rxcui:
        return IngestionResult(
            drug_name=drug_name,
            rxcui=None,
            drug_status="failed",
            warnings=norm.warnings or [f"'{drug_name}' could not be normalized via RxNorm."],
        )

    drug, drug_status = _get_or_create_drug(session, norm)

    label_result = get_drug_label_info(rxcui=norm.rxcui, drug_name=drug_name)

    if not label_result.setid:
        # no_label_found or error — nothing to store at the label/section/link level.
        return IngestionResult(
            drug_name=drug_name,
            rxcui=norm.rxcui,
            drug_status=drug_status,
            evidence_status=label_result.evidence_status,
            warnings=norm.warnings + label_result.warnings,
            error=label_result.error,
        )

    label, label_status = _get_or_create_label(session, label_result)
    section_counts = _sync_sections(session, label, label_result.sections)
    link_status = _get_or_create_link(session, drug, label, label_result)

    return IngestionResult(
        drug_name=drug_name,
        rxcui=norm.rxcui,
        drug_status=drug_status,
        label_status=label_status,
        sections_added=section_counts["added"],
        sections_updated=section_counts["updated"],
        sections_unchanged=section_counts["unchanged"],
        link_status=link_status,
        evidence_status=label_result.evidence_status,
        warnings=norm.warnings + label_result.warnings,
        error=None,
    )


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def get_drug_by_rxcui(session: Session, rxcui: str) -> Drug | None:
    """Direct lookup by RxCUI — the precise, indexed path."""
    return session.query(Drug).filter_by(rxcui=rxcui).one_or_none()


def search_drugs_by_name(session: Session, name: str) -> list[Drug]:
    """
    Case-insensitive search across normalized_name, input_name, and
    synonym/brand names. Returns matching Drug rows (deduplicated).

    This is a simple LIKE-based search suitable for the local dev
    knowledge base built in this phase. It is NOT the semantic retrieval
    Phase 6 will implement over ChromaDB — this is exact-ish structured
    lookup, useful for the API layer's "did we already ingest this drug"
    checks and for this phase's own tests.
    """
    pattern = f"%{name.strip().lower()}%"

    by_name = (
        session.query(Drug)
        .filter(
            (Drug.normalized_name.ilike(pattern))
            | (Drug.input_name.ilike(pattern))
            | (Drug.generic_name.ilike(pattern))
        )
        .all()
    )
    by_synonym = (
        session.query(Drug)
        .join(DrugSynonym)
        .filter(DrugSynonym.synonym.ilike(pattern))
        .all()
    )

    seen: dict[int, Drug] = {}
    for d in by_name + by_synonym:
        seen[d.id] = d
    return list(seen.values())


def get_sections_for_drug(
    session: Session, rxcui: str, section_key: str | None = None, only_found: bool = False
) -> list[LabelSectionRecord]:
    """
    Returns stored label sections for a drug (by RxCUI), across all labels
    linked to it. Optionally filter to one section_key (e.g.
    "drug_interactions") and/or only sections that were actually found
    with meaningful content.
    """
    query = (
        session.query(LabelSectionRecord)
        .join(Label, LabelSectionRecord.label_id == Label.id)
        .join(DrugLabelLink, DrugLabelLink.label_id == Label.id)
        .join(Drug, DrugLabelLink.drug_id == Drug.id)
        .filter(Drug.rxcui == rxcui)
    )
    if section_key:
        query = query.filter(LabelSectionRecord.section_key == section_key)
    if only_found:
        query = query.filter(LabelSectionRecord.found.is_(True))
    return query.all()
