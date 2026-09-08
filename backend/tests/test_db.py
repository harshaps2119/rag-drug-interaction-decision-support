"""
tests/test_db.py
==================
Tests for app/db.py and app/models/db_models.py.

These use a REAL SQLite database (in-memory, isolated per test via the
`engine` fixture) — no mocking needed, since SQLite has no network
dependency. This directly verifies the schema itself: constraints,
indexes, and relationships actually behave as designed.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_db.py -v
"""

import pytest
from sqlalchemy.exc import IntegrityError

from app.db import init_db, make_engine
from app.models.db_models import Drug, DrugLabelLink, DrugSynonym, Label, LabelSectionRecord


@pytest.fixture
def engine():
    eng = make_engine("sqlite:///:memory:")
    init_db(eng)
    return eng


@pytest.fixture
def session(engine):
    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def test_init_db_creates_all_expected_tables(engine):
    from app.models.db_models import Base

    table_names = set(Base.metadata.tables.keys())
    assert table_names == {"drugs", "drug_synonyms", "labels", "label_sections", "drug_label_links"}


def test_init_db_is_idempotent(engine):
    """Calling init_db twice on the same engine must not raise or duplicate anything."""
    init_db(engine)  # second call
    init_db(engine)  # third call, just to be sure
    # No exception means success — CREATE TABLE IF NOT EXISTS semantics.


def test_insert_and_retrieve_drug(session):
    drug = Drug(
        rxcui="11289",
        normalized_name="warfarin",
        input_name="warfarin",
        match_type="exact",
        source="RxNorm",
    )
    session.add(drug)
    session.commit()

    fetched = session.query(Drug).filter_by(rxcui="11289").one()
    assert fetched.normalized_name == "warfarin"
    assert fetched.id is not None


def test_drug_rxcui_uniqueness_enforced(session):
    """Inserting two drugs with the same RxCUI must fail at the DB level."""
    session.add(Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact"))
    session.commit()

    session.add(Drug(rxcui="11289", normalized_name="warfarin sodium", input_name="warfarin", match_type="exact"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_label_setid_version_uniqueness_enforced(session):
    session.add(Label(setid="abc-123", spl_version="1", title="TEST DRUG"))
    session.commit()

    session.add(Label(setid="abc-123", spl_version="1", title="TEST DRUG DUPLICATE"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_label_different_version_same_setid_allowed(session):
    """The SAME setid with a DIFFERENT version is a legitimately different label revision."""
    session.add(Label(setid="abc-123", spl_version="1", title="TEST DRUG v1"))
    session.add(Label(setid="abc-123", spl_version="2", title="TEST DRUG v2"))
    session.commit()  # must not raise

    rows = session.query(Label).filter_by(setid="abc-123").all()
    assert len(rows) == 2


def test_label_section_unique_per_label_and_code(session):
    label = Label(setid="abc-123", spl_version="1", title="TEST DRUG")
    session.add(label)
    session.commit()

    session.add(
        LabelSectionRecord(
            label_id=label.id, section_key="drug_interactions", section_code="34073-7",
            section_name="Drug Interactions", text="some text", found=True,
        )
    )
    session.commit()

    session.add(
        LabelSectionRecord(
            label_id=label.id, section_key="drug_interactions", section_code="34073-7",
            section_name="Drug Interactions", text="duplicate insert attempt", found=True,
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_drug_label_link_unique_per_pair(session):
    drug = Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact")
    label = Label(setid="abc-123", spl_version="1", title="TEST DRUG")
    session.add_all([drug, label])
    session.commit()

    session.add(
        DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    )
    session.commit()

    session.add(
        DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_cascade_delete_drug_removes_synonyms_and_links(session):
    drug = Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact")
    session.add(drug)
    session.commit()
    session.add(DrugSynonym(drug_id=drug.id, synonym="Coumadin", synonym_type="brand"))
    session.commit()

    assert session.query(DrugSynonym).filter_by(drug_id=drug.id).count() == 1

    session.delete(drug)
    session.commit()

    assert session.query(DrugSynonym).filter_by(drug_id=drug.id).count() == 0


def test_relationships_navigable_both_directions(session):
    drug = Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact")
    label = Label(setid="abc-123", spl_version="1", title="TEST DRUG")
    session.add_all([drug, label])
    session.commit()

    section = LabelSectionRecord(
        label_id=label.id, section_key="drug_interactions", section_code="34073-7",
        section_name="Drug Interactions", text="text", found=True,
    )
    session.add(section)
    link = DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()

    session.refresh(label)
    session.refresh(drug)

    assert label.sections[0].section_code == "34073-7"
    assert drug.links[0].label.setid == "abc-123"
