"""
rag/chunking.py
=================
Splits label section text (from SQLite, Phase 4) into smaller chunks
suitable for embedding.

WHY CHUNKING IS NEEDED AT ALL
--------------------------------
Embedding models have a maximum input length, and even within that limit,
very long text embedded as one vector tends to blur together multiple
distinct ideas into a single, less-useful representation. Splitting text
into smaller, overlapping windows lets each chunk capture a more focused
piece of meaning, which is what makes semantic search actually work well.

WHY WORD-COUNT-BASED SPLITTING (not sentence or paragraph splitting)
--------------------------------------------------------------------------
Phase 3's `dailymed_service._extract_section_text()` deliberately produces
FLAT text — it strips all paragraph/list structure from the original SPL
XML down to a single whitespace-normalized string (documented in that
file as a Phase 3 simplification, deferred to Phase 5). That means there
are no paragraph or list-item boundaries left to split on here. A simple
sentence splitter (splitting on ". ") is unreliable on medical text full
of abbreviations ("e.g.", "Dr.", "mg.", dosage ranges) that produce false
sentence breaks. A fixed-size word-count sliding window is simple,
predictable, dependency-free, and good enough for this project's MVP
scope — this trade-off is documented here rather than hidden.

WHY OVERLAP
-------------
Without overlap, a sentence that happens to fall exactly on a chunk
boundary gets split in half, and neither half alone captures its full
meaning. A moderate overlap (default 40 of 180 words, ~22%) means
boundary-straddling content is very likely to appear complete in at
least one chunk.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_chunking.py -v

This is a pure function with no I/O, no database, no network, and no ML
model — it runs instantly, fully offline, everywhere, including in this
sandbox.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE_WORDS = 180
DEFAULT_CHUNK_OVERLAP_WORDS = 40

# A chunk shorter than this (in characters, after cleaning) is treated as
# junk rather than real evidence — e.g. a stray leftover fragment from a
# boundary split — and is dropped rather than embedded.
MIN_CHUNK_CHARS = 15


@dataclass
class TextChunk:
    index: int
    text: str
    word_count: int


def _clean_text(text: str) -> str:
    """Whitespace-normalizes text. Defensive even though Phase 3 already
    does this once — chunking should not assume its caller always will."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(
    text: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[TextChunk]:
    """
    Splits `text` into a list of TextChunk, using a sliding window of
    `chunk_size_words` words with `overlap_words` words of overlap
    between consecutive chunks.

    Returns an empty list for empty/whitespace-only input — this is a
    normal, expected outcome (e.g. a section that turned out to have no
    real content), not an error.

    Chunk `index` is always 0-based and sequential for a single call,
    which is what makes chunk IDs built from (some stable record id,
    chunk.index) deterministic and reproducible across re-runs, as long
    as the input text and these two size parameters don't change.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) must be smaller than "
            f"chunk_size_words ({chunk_size_words})."
        )

    cleaned = _clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")

    if len(words) <= chunk_size_words:
        return [TextChunk(index=0, text=cleaned, word_count=len(words))]

    chunks: list[TextChunk] = []
    step = chunk_size_words - overlap_words
    start = 0
    index = 0

    while start < len(words):
        window = words[start : start + chunk_size_words]
        chunk_str = " ".join(window)

        if len(chunk_str) >= MIN_CHUNK_CHARS:
            chunks.append(TextChunk(index=index, text=chunk_str, word_count=len(window)))
            index += 1

        if start + chunk_size_words >= len(words):
            break
        start += step

    return chunks
