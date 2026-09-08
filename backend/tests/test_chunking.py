"""
tests/test_chunking.py
=========================
Tests for app/rag/chunking.py — pure functions, no I/O, no dependencies.
Runs instantly and fully offline everywhere.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_chunking.py -v
"""

import pytest

from app.rag.chunking import MIN_CHUNK_CHARS, chunk_text


def test_short_text_returns_single_chunk():
    text = "May increase INR when co-administered with NSAIDs such as ibuprofen."
    chunks = chunk_text(text, chunk_size_words=180, overlap_words=40)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == text
    assert chunks[0].word_count == len(text.split())


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_whitespace_only_text_returns_no_chunks():
    assert chunk_text("   \n\t  ") == []


def test_long_text_splits_into_multiple_overlapping_chunks():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size_words=100, overlap_words=20)

    assert len(chunks) > 1
    # Every chunk should have at most 100 words.
    assert all(c.word_count <= 100 for c in chunks)
    # Indices are sequential starting at 0.
    assert [c.index for c in chunks] == list(range(len(chunks)))

    # Verify actual overlap: the last 20 words of chunk 0 should equal
    # the first 20 words of chunk 1.
    chunk0_words = chunks[0].text.split(" ")
    chunk1_words = chunks[1].text.split(" ")
    assert chunk0_words[-20:] == chunk1_words[:20]


def test_all_words_covered_by_chunks():
    """Every word in the original text must appear in at least one chunk — chunking must not silently drop content."""
    words = [f"tok{i}" for i in range(437)]  # deliberately not a clean multiple of chunk size
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size_words=100, overlap_words=25)

    covered = set()
    for c in chunks:
        covered.update(c.text.split(" "))

    assert covered == set(words)


def test_cleans_excess_whitespace():
    messy = "some    text\n\nwith   irregular\t\twhitespace"
    chunks = chunk_text(messy)
    assert len(chunks) == 1
    assert chunks[0].text == "some text with irregular whitespace"


def test_overlap_greater_than_or_equal_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size_words=50, overlap_words=50)
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size_words=50, overlap_words=60)


def test_tiny_leftover_fragment_is_dropped_as_junk():
    """A final sliding-window fragment shorter than MIN_CHUNK_CHARS should
    not produce a near-empty, useless chunk."""
    # Construct text where the very last window would be a single short word.
    words = ["word"] * 100 + ["x"]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size_words=100, overlap_words=10)
    # No chunk's text should be shorter than MIN_CHUNK_CHARS.
    assert all(len(c.text) >= MIN_CHUNK_CHARS for c in chunks)


def test_chunking_is_deterministic_across_calls():
    """Same input, same parameters -> byte-identical output every time —
    this determinism is what makes chunk IDs stable across ingestion runs."""
    text = " ".join(f"clinicalword{i}" for i in range(350))

    result1 = chunk_text(text, chunk_size_words=120, overlap_words=30)
    result2 = chunk_text(text, chunk_size_words=120, overlap_words=30)

    assert [c.text for c in result1] == [c.text for c in result2]
    assert [c.index for c in result1] == [c.index for c in result2]
