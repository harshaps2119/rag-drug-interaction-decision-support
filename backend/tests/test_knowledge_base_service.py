"""
tests/test_knowledge_base_service.py
=======================================
Tests for app/services/knowledge_base_service.py — ingestion (upserts)
and retrieval helpers.

WHY WE MOCK normalize_drug_name AND get_drug_label_info HERE
------------------------------------------------------------------
This module's job is orchestration + storage, not talking to RxNorm or
DailyMed directly — that's already fully tested (with HTTP-level mocking)
in test_rxnorm_service.py and test_dailymed_service.py. Here we monkeypatch
those two functions at the point knowledge_base_service imports them, so
these tests exercise ONLY the upsert/idempotency/retrieval logic, against
a REAL in-memory SQLite database.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_knowledge_base_service.py -v
"""

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import init_db, make_engine
from app.models.db_models import Drug, DrugLabelLink, Label, LabelSectionRecord
from app.schemas.drug import DrugNormalizationResult
from app.schemas.label import DrugLabelResult, LabelSection
from app.services import knowledge_base_service as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _norm_result(
    *, name="warfarin", rxcui="11289", match_type="exact",
    brand_names=None, synonyms=None, normalized_name="warfarin",
) -> DrugNormalizationResult:
    return DrugNormalizationResult(
        input_name=name,
        match_type=match_type,
        rxcui=rxcui,
        normalized_name=normalized_name,
        term_type="IN",
        brand_names=brand_names or ["Coumadin", "Jantoven"],
        generic_name=None,
        synonyms=synonyms or [],
        source_url=f"https://mor.nlm.nih.gov/RxNav/search?searchBy=RXCUI&searchTerm={rxcui}",
    )


def _label_result(
    *, setid="aaaa-1111", spl_version="3", title="WARFARIN SODIUM TABLET",
    interaction_text="May increase INR with NSAIDs.",
    evidence_status="interaction_evidence_found",
) -> DrugLabelResult:
    sections = {
        "drug_interactions": LabelSection(
            section_code="34073-7", section_name="Drug Interactions",
            text=interaction_text or "", found=bool(interaction_text),
        ),
        "contraindications": LabelSection(
            section_code="34070-3", section_name="Contraindications",
            text="Active bleeding.", found=True,
        ),
    }
    return DrugLabelResult(
        evidence_status=evidence_status,
        setid=setid,
        spl_version=spl_version,
        title=title,
        manufacturer="Example Pharma Inc.",
        published_date="Jan 15, 2024",
        source_url=f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
        api_url=f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{setid}.xml",
        sections=sections,
        candidates_checked=1,
        total_candidates_found=1,
    )


# ---------------------------------------------------------------------------
# Tests: basic ingestion
# ---------------------------------------------------------------------------

def test_ingest_drug_first_time_adds_everything(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())

    result = kb.ingest_drug("warfarin", session=session)
    session.commit()

    assert result.drug_status == "added"
    assert result.label_status == "added"
    assert result.link_status == "added"
    assert result.sections_added == 2
    assert result.sections_updated == 0
    assert result.evidence_status == "interaction_evidence_found"

    assert session.query(Drug).count() == 1
    assert session.query(Label).count() == 1
    assert session.query(LabelSectionRecord).count() == 2
    assert session.query(DrugLabelLink).count() == 1


# ---------------------------------------------------------------------------
# Tests: idempotent re-ingestion (THE core Phase 4 requirement)
# ---------------------------------------------------------------------------

def test_ingest_drug_twice_does_not_duplicate(session, monkeypatch):
    """Running ingestion twice with IDENTICAL upstream data must not create
    any duplicate rows — second run should report 'unchanged' throughout
    (except the link's retrieved_at timestamp, which always refreshes)."""
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())

    first = kb.ingest_drug("warfarin", session=session)
    session.commit()
    second = kb.ingest_drug("warfarin", session=session)
    session.commit()

    assert first.drug_status == "added"
    assert second.drug_status == "unchanged"
    assert second.label_status == "unchanged"
    assert second.sections_added == 0
    assert second.sections_updated == 0
    assert second.sections_unchanged == 2
    # link_status reflects only meaningful content changes (evidence_status,
    # candidates_checked, total_candidates_found) — retrieved_at is always
    # refreshed internally as a "last verified" timestamp, but that alone
    # does not count as a change worth reporting, so "unchanged" here is
    # correct and intentional, not a bug.
    assert second.link_status == "unchanged"

    # The real assertion: row COUNTS must not have grown.
    assert session.query(Drug).count() == 1
    assert session.query(Label).count() == 1
    assert session.query(LabelSectionRecord).count() == 2
    assert session.query(DrugLabelLink).count() == 1


def test_ingest_drug_three_times_still_one_row_each(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())

    for _ in range(3):
        kb.ingest_drug("warfarin", session=session)
        session.commit()

    assert session.query(Drug).count() == 1
    assert session.query(Label).count() == 1
    assert session.query(LabelSectionRecord).count() == 2
    assert session.query(DrugLabelLink).count() == 1


def test_ingest_drug_detects_real_content_changes_as_updated(session, monkeypatch):
    """If the upstream label text genuinely changes between ingestions
    (e.g. FDA updated the label), the section row should be UPDATED in
    place, not duplicated."""
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())

    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result(interaction_text="Old text."))
    kb.ingest_drug("warfarin", session=session)
    session.commit()

    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result(interaction_text="Updated text about NSAIDs."))
    second = kb.ingest_drug("warfarin", session=session)
    session.commit()

    assert second.sections_updated == 1  # only drug_interactions text changed
    assert second.sections_unchanged == 1  # contraindications stayed the same
    assert session.query(LabelSectionRecord).count() == 2  # still just 2 rows, not 4

    stored = (
        session.query(LabelSectionRecord)
        .filter_by(section_code="34073-7")
        .one()
    )
    assert "Updated text" in stored.text


# ---------------------------------------------------------------------------
# Tests: drug normalization failure handling
# ---------------------------------------------------------------------------

def test_ingest_drug_unknown_drug_reports_failed_not_exception(session, monkeypatch):
    unknown = DrugNormalizationResult(input_name="notarealdrug123", match_type="none", warnings=["not found"])
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: unknown)

    result = kb.ingest_drug("notarealdrug123", session=session)

    assert result.drug_status == "failed"
    assert result.rxcui is None
    assert session.query(Drug).count() == 0


def test_ingest_drug_rxnorm_api_error_reports_failed(session, monkeypatch):
    errored = DrugNormalizationResult(
        input_name="warfarin", match_type="error", error="RxNorm API timed out"
    )
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: errored)

    result = kb.ingest_drug("warfarin", session=session)

    assert result.drug_status == "failed"
    assert result.error == "RxNorm API timed out"


def test_ingest_drug_no_label_found_still_stores_drug(session, monkeypatch):
    """Drug normalization can succeed even when DailyMed has no label —
    the Drug row should still be stored (we know the drug exists), just
    with no label/section/link data."""
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    no_label = DrugLabelResult(evidence_status="no_label_found", warnings=["no label"])
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: no_label)

    result = kb.ingest_drug("warfarin", session=session)
    session.commit()

    assert result.drug_status == "added"
    assert result.label_status is None
    assert result.evidence_status == "no_label_found"
    assert session.query(Drug).count() == 1
    assert session.query(Label).count() == 0


# ---------------------------------------------------------------------------
# Tests: retrieval by RxCUI
# ---------------------------------------------------------------------------

def test_get_drug_by_rxcui(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())
    kb.ingest_drug("warfarin", session=session)
    session.commit()

    found = kb.get_drug_by_rxcui(session, "11289")
    assert found is not None
    assert found.normalized_name == "warfarin"

    missing = kb.get_drug_by_rxcui(session, "99999999")
    assert missing is None


# ---------------------------------------------------------------------------
# Tests: retrieval by drug name (including synonyms/brand names)
# ---------------------------------------------------------------------------

def test_search_drugs_by_name_matches_normalized_name(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())
    kb.ingest_drug("warfarin", session=session)
    session.commit()

    results = kb.search_drugs_by_name(session, "warfarin")
    assert len(results) == 1
    assert results[0].rxcui == "11289"


def test_search_drugs_by_name_matches_brand_synonym(session, monkeypatch):
    """Searching for a BRAND name (Coumadin) must find the drug stored
    under its generic RxCUI, via the synonym table."""
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())
    kb.ingest_drug("warfarin", session=session)
    session.commit()

    results = kb.search_drugs_by_name(session, "Coumadin")
    assert len(results) == 1
    assert results[0].rxcui == "11289"


def test_search_drugs_by_name_case_insensitive_partial_match(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())
    kb.ingest_drug("warfarin", session=session)
    session.commit()

    assert len(kb.search_drugs_by_name(session, "WARF")) == 1
    assert len(kb.search_drugs_by_name(session, "warf")) == 1
    assert len(kb.search_drugs_by_name(session, "zzz_nonexistent")) == 0


# ---------------------------------------------------------------------------
# Tests: section filtering
# ---------------------------------------------------------------------------

def test_get_sections_for_drug_filters_by_section_key(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())
    kb.ingest_drug("warfarin", session=session)
    session.commit()

    all_sections = kb.get_sections_for_drug(session, "11289")
    assert len(all_sections) == 2

    interactions_only = kb.get_sections_for_drug(session, "11289", section_key="drug_interactions")
    assert len(interactions_only) == 1
    assert interactions_only[0].section_code == "34073-7"


def test_get_sections_for_drug_only_found_filter(session, monkeypatch):
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    # One section found, one not.
    label_with_gap = _label_result(interaction_text=None)
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: label_with_gap)

    kb.ingest_drug("warfarin", session=session)
    session.commit()

    all_sections = kb.get_sections_for_drug(session, "11289")
    assert len(all_sections) == 2  # both stored, even the not-found one

    found_only = kb.get_sections_for_drug(session, "11289", only_found=True)
    assert len(found_only) == 1
    assert found_only[0].section_code == "34070-3"  # contraindications, the one that was found


# ---------------------------------------------------------------------------
# Tests: provenance / source preservation
# ---------------------------------------------------------------------------

def test_provenance_fields_preserved_end_to_end(session, monkeypatch):
    """Every field required by the project's provenance requirement must
    be traceable from the stored records back to the original source."""
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: _label_result())

    kb.ingest_drug("warfarin", session=session)
    session.commit()

    drug = kb.get_drug_by_rxcui(session, "11289")
    assert drug.rxcui == "11289"
    assert drug.normalized_name == "warfarin"
    assert {s.synonym for s in drug.synonyms} == {"Coumadin", "Jantoven"}

    link = session.query(DrugLabelLink).filter_by(drug_id=drug.id).one()
    label = link.label

    assert label.setid == "aaaa-1111"
    assert label.spl_version == "3"
    assert label.manufacturer == "Example Pharma Inc."
    assert label.source_url == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=aaaa-1111"
    assert label.api_url == "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/aaaa-1111.xml"
    assert label.published_date == "Jan 15, 2024"

    interactions = [s for s in label.sections if s.section_code == "34073-7"][0]
    assert interactions.text == "May increase INR with NSAIDs."
    assert interactions.section_name == "Drug Interactions"
    assert interactions.found is True

    # And the evidence_status itself is preserved on the link, not lost.
    assert link.evidence_status == "interaction_evidence_found"
    assert link.retrieved_at is not None


def test_storing_a_label_is_not_conflated_with_identifying_an_interaction(session, monkeypatch):
    """Explicit regression test for the project's core safety requirement:
    storing label_found_no_interaction_section evidence must never look,
    in the stored data, like an actual interaction was identified."""
    monkeypatch.setattr(kb, "normalize_drug_name", lambda name: _norm_result())
    no_interactions = _label_result(interaction_text=None, evidence_status="label_found_no_interaction_section")
    monkeypatch.setattr(kb, "get_drug_label_info", lambda rxcui, drug_name: no_interactions)

    result = kb.ingest_drug("warfarin", session=session)
    session.commit()

    assert result.evidence_status == "label_found_no_interaction_section"

    link = session.query(DrugLabelLink).filter_by(drug_id=kb.get_drug_by_rxcui(session, "11289").id).one()
    assert link.evidence_status == "label_found_no_interaction_section"

    sections = kb.get_sections_for_drug(session, "11289", section_key="drug_interactions")
    assert sections[0].found is False
    assert sections[0].text == ""
