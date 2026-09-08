"""
tests/test_vector_store.py
=============================
Tests for app/rag/vector_store.py.

These use a REAL local ChromaDB instance, persisted to a pytest tmp_path
directory — no mocking needed, and no network required (confirmed: this
module never lets ChromaDB fall back to its own default embedding
function, so no HuggingFace download is ever triggered by these tests).

HOW TO RUN
-----------
    cd backend
    pytest tests/test_vector_store.py -v
"""

import pytest

from app.rag.vector_store import (
    build_where,
    delete_chunks,
    get_client,
    get_collection,
    get_existing_documents,
    get_ids_by_metadata,
    query,
    upsert_chunks,
)


@pytest.fixture
def collection(tmp_path):
    client = get_client(persist_dir=str(tmp_path / "chroma"))
    return get_collection(client=client)


def test_get_collection_creates_empty_collection(collection):
    assert collection.count() == 0


def test_upsert_and_count(collection):
    upsert_chunks(
        collection,
        ids=["c1", "c2"],
        documents=["warfarin text", "ibuprofen text"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"rxcui": "11289"}, {"rxcui": "5640"}],
    )
    assert collection.count() == 2


def test_upsert_same_id_replaces_not_duplicates(collection):
    """The core idempotency primitive: upserting the same id twice must
    not increase the row count, and must update the content in place."""
    upsert_chunks(
        collection, ids=["c1"], documents=["old text"], embeddings=[[0.1, 0.2]], metadatas=[{"v": 1}]
    )
    assert collection.count() == 1

    upsert_chunks(
        collection, ids=["c1"], documents=["new text"], embeddings=[[0.9, 0.9]], metadatas=[{"v": 2}]
    )
    assert collection.count() == 1  # still just one row

    fetched = collection.get(ids=["c1"])
    assert fetched["documents"][0] == "new text"
    assert fetched["metadatas"][0]["v"] == 2


def test_upsert_empty_ids_is_a_noop(collection):
    upsert_chunks(collection, ids=[], documents=[], embeddings=[], metadatas=[])
    assert collection.count() == 0


def test_get_existing_documents(collection):
    upsert_chunks(
        collection,
        ids=["c1", "c2"],
        documents=["doc1", "doc2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"note": "a"}, {"note": "b"}],
    )
    existing = get_existing_documents(collection, ["c1", "c2", "c3-does-not-exist"])
    assert existing == {"c1": "doc1", "c2": "doc2"}


def test_get_existing_documents_empty_ids_returns_empty_dict(collection):
    assert get_existing_documents(collection, []) == {}


def test_get_ids_by_metadata_filter(collection):
    upsert_chunks(
        collection,
        ids=["c1", "c2", "c3"],
        documents=["d1", "d2", "d3"],
        embeddings=[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        metadatas=[
            {"link_id": 1, "db_record_id": 10},
            {"link_id": 1, "db_record_id": 10},
            {"link_id": 2, "db_record_id": 20},
        ],
    )
    ids = get_ids_by_metadata(collection, {"$and": [{"link_id": 1}, {"db_record_id": 10}]})
    assert set(ids) == {"c1", "c2"}


def test_delete_chunks(collection):
    upsert_chunks(
        collection, ids=["c1", "c2"], documents=["d1", "d2"], embeddings=[[0.1], [0.2]], metadatas=[{"note": "a"}, {"note": "b"}]
    )
    delete_chunks(collection, ["c1"])
    assert collection.count() == 1
    assert collection.get(ids=["c1"])["ids"] == []


def test_delete_chunks_empty_ids_is_a_noop(collection):
    upsert_chunks(collection, ids=["c1"], documents=["d1"], embeddings=[[0.1]], metadatas=[{"note": "a"}])
    delete_chunks(collection, [])
    assert collection.count() == 1


def test_build_where_no_filters_returns_none():
    assert build_where({}) is None
    assert build_where({"rxcui": None, "section_key": None}) is None


def test_build_where_single_filter_returns_flat_dict():
    assert build_where({"rxcui": "11289", "section_key": None}) == {"rxcui": "11289"}


def test_build_where_multiple_filters_uses_and_operator():
    where = build_where({"rxcui": "11289", "section_key": "drug_interactions"})
    assert where == {"$and": [{"rxcui": "11289"}, {"section_key": "drug_interactions"}]}


def test_build_where_list_value_uses_in_operator():
    """Phase 6 needs to scope a query to either of two drugs at once."""
    where = build_where({"rxcui": ["11289", "5640"]})
    assert where == {"rxcui": {"$in": ["11289", "5640"]}}


def test_build_where_list_value_combined_with_scalar_filter():
    where = build_where({"rxcui": ["11289", "5640"], "section_key": "drug_interactions"})
    assert where == {"$and": [{"rxcui": {"$in": ["11289", "5640"]}}, {"section_key": "drug_interactions"}]}


def test_query_returns_nearest_by_embedding(collection):
    upsert_chunks(
        collection,
        ids=["near", "far"],
        documents=["close doc", "far doc"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"note": "a"}, {"note": "b"}],
    )
    result = query(collection, query_embedding=[1.0, 0.0], top_k=2)
    assert result["ids"][0][0] == "near"  # nearest match first
    assert result["distances"][0][0] < result["distances"][0][1]


def test_query_with_where_filter(collection):
    upsert_chunks(
        collection,
        ids=["a", "b"],
        documents=["doc a", "doc b"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],  # identical embeddings on purpose
        metadatas=[{"rxcui": "111"}, {"rxcui": "222"}],
    )
    result = query(collection, query_embedding=[1.0, 0.0], top_k=5, where={"rxcui": "222"})
    assert result["ids"][0] == ["b"]


def test_query_against_empty_collection_returns_empty_not_error(collection):
    result = query(collection, query_embedding=[1.0, 0.0], top_k=5)
    assert result["ids"] == [[]]


def test_collection_is_persistent_across_client_instances(tmp_path):
    """Real persistence check: data written by one client instance must
    be readable by a fresh client pointed at the same directory."""
    persist_dir = str(tmp_path / "chroma")

    client1 = get_client(persist_dir=persist_dir)
    collection1 = get_collection(client=client1)
    upsert_chunks(
        collection1, ids=["persisted"], documents=["persisted doc"], embeddings=[[0.5, 0.5]], metadatas=[{"note": "a"}]
    )

    client2 = get_client(persist_dir=persist_dir)
    collection2 = get_collection(client=client2)
    assert collection2.count() == 1
    assert collection2.get(ids=["persisted"])["documents"][0] == "persisted doc"
