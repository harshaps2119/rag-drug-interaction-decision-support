"""
tests/test_rag_ingest.py
===========================
Tests for app/rag/ingest.py — the SQLite -> chunk -> embed -> ChromaDB
pipeline.

Uses a REAL in-memory SQLite database (populated directly via the
SQLAlchemy models, not through Phase 2/3's network-calling services —
this file tests Phase 5's own logic in isolation), a REAL ChromaDB
instance in a pytest tmp_path, and DeterministicFakeEmbeddingModel (see
app/rag/embeddings.py for exactly what plumbing vs. quality this fake
can and cannot verify).

HOW TO RUN
-----------
    cd backend
    pytest tests/test_rag_ingest.py -v
"""

import pytest
from sqlalchemy.orm import sessionmaker

from app.db import init_db, make_engine
from app.models.db_models import Drug, DrugLabelLink, Label, LabelSectionRecord
from app.rag.embeddings import DeterministicFakeEmbeddingModel
from app.rag.ingest import ingest_rag_index
from app.rag.vector_store import get_client, get_collection


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


@pytest.fixture
def collection(tmp_path):
    client = get_client(persist_dir=str(tmp_path / "chroma"))
    return get_collection(client=client)


@pytest.fixture
def embedding_model():
    return DeterministicFakeEmbeddingModel(dimension=16)


def _seed_drug_with_sections(session, *, rxcui="11289", interaction_text="May increase INR when co-administered with NSAIDs.") -> DrugLabelLink:
    """Seeds one Drug + Label + two LabelSectionRecords (one found, one not) + one DrugLabelLink."""
    drug = Drug(rxcui=rxcui, normalized_name="warfarin", input_name="warfarin", match_type="exact")
    label = Label(setid="setid-1", spl_version="1", title="WARFARIN TABLET", manufacturer="Example Pharma")
    session.add_all([drug, label])
    session.commit()

    interactions = LabelSectionRecord(
        label_id=label.id, section_key="drug_interactions", section_code="34073-7",
        section_name="Drug Interactions", text=interaction_text, found=True,
    )
    contraindications_absent = LabelSectionRecord(
        label_id=label.id, section_key="contraindications", section_code="34070-3",
        section_name="Contraindications", text="", found=False,
    )
    session.add_all([interactions, contraindications_absent])
    session.commit()

    link = DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()
    return link


# ---------------------------------------------------------------------------
# Tests: basic ingestion + metadata preservation
# ---------------------------------------------------------------------------

def test_ingest_indexes_found_sections_only(session, collection, embedding_model):
    _seed_drug_with_sections(session)

    result = ingest_rag_index(session, embedding_model, collection)

    assert result.sections_processed == 1  # only the found=True interactions section
    assert result.chunks_added == 1
    assert "11289" in result.drugs_processed
    assert collection.count() == 1


def test_ingest_preserves_all_required_metadata_fields(session, collection, embedding_model):
    _seed_drug_with_sections(session)
    ingest_rag_index(session, embedding_model, collection)

    stored = collection.get(where={"section_key": "drug_interactions"})
    meta = stored["metadatas"][0]

    assert meta["drug_name"] == "warfarin"
    assert meta["rxcui"] == "11289"
    assert meta["label_setid"] == "setid-1"
    assert meta["spl_version"] == "1"
    assert meta["manufacturer"] == "Example Pharma"
    assert meta["section_name"] == "Drug Interactions"
    assert meta["section_code"] == "34073-7"
    assert meta["db_record_id"] is not None
    assert stored["ids"][0].startswith("link-")


def test_ingest_chunk_id_is_deterministic(session, collection, embedding_model):
    link = _seed_drug_with_sections(session)
    ingest_rag_index(session, embedding_model, collection)

    section = [s for s in link.label.sections if s.section_key == "drug_interactions"][0]
    expected_id = f"link-{link.id}-section-{section.id}-chunk-0"

    stored_ids = collection.get()["ids"]
    assert expected_id in stored_ids


# ---------------------------------------------------------------------------
# Tests: idempotency (the core Phase 5 requirement)
# ---------------------------------------------------------------------------

def test_ingest_twice_does_not_duplicate_vectors(session, collection, embedding_model):
    _seed_drug_with_sections(session)

    first = ingest_rag_index(session, embedding_model, collection)
    second = ingest_rag_index(session, embedding_model, collection)

    assert first.chunks_added == 1
    assert second.chunks_added == 0
    assert second.chunks_unchanged == 1
    assert collection.count() == 1  # NOT 2


def test_ingest_three_times_still_one_vector(session, collection, embedding_model):
    _seed_drug_with_sections(session)
    for _ in range(3):
        ingest_rag_index(session, embedding_model, collection)
    assert collection.count() == 1


def test_ingest_detects_real_text_change_as_update_not_duplicate(session, collection, embedding_model):
    _seed_drug_with_sections(session, interaction_text="Old interaction text.")
    ingest_rag_index(session, embedding_model, collection)
    assert collection.count() == 1

    # Simulate a label being re-fetched with updated FDA text (Phase 4 would do this).
    section = session.query(LabelSectionRecord).filter_by(section_key="drug_interactions").one()
    section.text = "Updated interaction text about a different NSAID."
    session.commit()

    second = ingest_rag_index(session, embedding_model, collection)

    assert second.chunks_updated == 1
    assert second.chunks_added == 0
    assert collection.count() == 1  # still one row, updated in place

    stored = collection.get()
    assert "Updated interaction text" in stored["documents"][0]


def test_ingest_removes_stale_chunks_when_section_shrinks(session, collection, embedding_model):
    """A long section producing multiple chunks, then shortened to fewer
    chunks on re-ingestion, must not leave orphaned old chunk ids behind."""
    long_text = " ".join(f"clinicalterm{i}" for i in range(400))  # will produce multiple chunks
    _seed_drug_with_sections(session, interaction_text=long_text)

    first = ingest_rag_index(session, embedding_model, collection)
    assert first.chunks_added > 1
    count_after_first = collection.count()
    assert count_after_first > 1

    # Shrink the text drastically -> should now produce just 1 chunk.
    section = session.query(LabelSectionRecord).filter_by(section_key="drug_interactions").one()
    section.text = "Short updated text."
    session.commit()

    second = ingest_rag_index(session, embedding_model, collection)

    assert second.chunks_deleted_stale == count_after_first - 1
    assert collection.count() == 1


# ---------------------------------------------------------------------------
# Tests: empty database / missing evidence
# ---------------------------------------------------------------------------

def test_ingest_empty_database_returns_gracefully(session, collection, embedding_model):
    result = ingest_rag_index(session, embedding_model, collection)

    assert result.sections_processed == 0
    assert collection.count() == 0
    assert len(result.warnings) == 1
    assert "No drug-label evidence" in result.warnings[0]


def test_ingest_scoped_to_nonexistent_rxcui_returns_gracefully(session, collection, embedding_model):
    _seed_drug_with_sections(session, rxcui="11289")

    result = ingest_rag_index(session, embedding_model, collection, rxcui="99999999")

    assert collection.count() == 0
    assert "99999999" in result.warnings[0]


def test_ingest_scoped_to_specific_rxcui_only_indexes_that_drug(session, collection, embedding_model):
    _seed_drug_with_sections(session, rxcui="11289")

    drug2 = Drug(rxcui="5640", normalized_name="ibuprofen", input_name="ibuprofen", match_type="exact")
    label2 = Label(setid="setid-2", spl_version="1", title="IBUPROFEN TABLET")
    session.add_all([drug2, label2])
    session.commit()
    section2 = LabelSectionRecord(
        label_id=label2.id, section_key="drug_interactions", section_code="34073-7",
        section_name="Drug Interactions", text="Ibuprofen interacts with warfarin too.", found=True,
    )
    session.add(section2)
    session.commit()
    link2 = DrugLabelLink(drug_id=drug2.id, label_id=label2.id, evidence_status="interaction_evidence_found")
    session.add(link2)
    session.commit()

    result = ingest_rag_index(session, embedding_model, collection, rxcui="11289")

    assert result.drugs_processed == {"11289"}
    stored = collection.get()
    assert all(m["rxcui"] == "11289" for m in stored["metadatas"])


# ---------------------------------------------------------------------------
# Tests: malformed / corrupted records
# ---------------------------------------------------------------------------

def test_ingest_skips_found_true_but_empty_text_section(session, collection, embedding_model):
    """Data integrity anomaly: found=True with empty text should never
    happen given Phase 3's checks, but must be handled gracefully, not crash."""
    drug = Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact")
    label = Label(setid="setid-1", spl_version="1", title="WARFARIN TABLET")
    session.add_all([drug, label])
    session.commit()

    bad_section = LabelSectionRecord(
        label_id=label.id, section_key="drug_interactions", section_code="34073-7",
        section_name="Drug Interactions", text="", found=True,  # anomalous: found=True, no text
    )
    session.add(bad_section)
    session.commit()
    link = DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()

    result = ingest_rag_index(session, embedding_model, collection)

    assert result.malformed_records_skipped == 1
    assert collection.count() == 0
    assert any("empty text" in w for w in result.warnings)


def test_ingest_skips_whitespace_only_section_text(session, collection, embedding_model):
    drug = Drug(rxcui="11289", normalized_name="warfarin", input_name="warfarin", match_type="exact")
    label = Label(setid="setid-1", spl_version="1", title="WARFARIN TABLET")
    session.add_all([drug, label])
    session.commit()

    whitespace_section = LabelSectionRecord(
        label_id=label.id, section_key="drug_interactions", section_code="34073-7",
        section_name="Drug Interactions", text="   \n\t  ", found=True,
    )
    session.add(whitespace_section)
    session.commit()
    link = DrugLabelLink(drug_id=drug.id, label_id=label.id, evidence_status="interaction_evidence_found")
    session.add(link)
    session.commit()

    result = ingest_rag_index(session, embedding_model, collection)

    assert result.malformed_records_skipped == 1
    assert collection.count() == 0


# ---------------------------------------------------------------------------
# Tests: clinical priority ordering (pure function)
# ---------------------------------------------------------------------------

def test_section_sort_key_orders_by_clinical_priority():
    from app.rag.ingest import _section_sort_key

    keys = ["adverse_reactions", "drug_interactions", "unknown_section", "contraindications"]
    ordered = sorted(keys, key=_section_sort_key)

    assert ordered == ["drug_interactions", "contraindications", "adverse_reactions", "unknown_section"]
