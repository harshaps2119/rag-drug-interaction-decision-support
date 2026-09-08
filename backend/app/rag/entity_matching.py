"""
rag/entity_matching.py
=========================
A deliberately simple method for determining whether a retrieved passage
actually refers to a specific drug, a known synonym/brand name for it, or
a drug class it belongs to.

WHAT THIS DOES
----------------
Case-insensitive, whole-word string matching of a drug's known name
variants (its normalized generic name plus any brand names/synonyms
stored in Phase 4's SQLite `drug_synonyms` table) against a chunk of
text. Separately, a small internal lookup table maps a handful of common
drug-CLASS terms (e.g. "NSAID") to representative member drug names, so
text that only names a class — never the specific drug — can still be
flagged as a weaker category of evidence (class_level_evidence) rather
than silently missed or wrongly counted as direct evidence.

WHAT THIS DOES NOT DO — READ THIS BEFORE TRUSTING ITS OUTPUT
--------------------------------------------------------------------
This is intentionally NOT a clinical entity-resolution system, and does
not pretend to be one. Concretely, it does NOT:

  - Understand NEGATION. "No interaction has been observed with
    ibuprofen" and "co-administration with ibuprofen increases bleeding
    risk" both register as "mentions ibuprofen" — this module has no
    idea which one is true. Downstream, that's exactly why a text
    mention is called "evidence to consider", never "proof of
    interaction" — see docs/retrieval.md.
  - Handle synonyms, abbreviations, or misspellings beyond what's
    already stored as a known synonym in SQLite (from RxNorm, Phase 2).
    A drug referred to by a name RxNorm doesn't know about (a rare
    regional brand name, a research code name, a typo in the source
    label itself) will not be matched.
  - Resolve pronoun references ("this drug", "the aforementioned
    medication") back to a specific drug name.
  - Understand dosage-form or route qualifiers ("oral warfarin" vs.
    "topical warfarin") as distinct from the base drug.
  - Have a clinically complete or curated drug-class vocabulary.
    DRUG_CLASS_TERMS below is a small, illustrative, hand-picked set of
    common classes — NOT sourced from RxNorm's actual class
    relationships (RxClass) or any authoritative classification system.
    Its ABSENCE of a match must never be read as "this drug has no
    class-level relationship" — only "this simple lookup didn't find
    one".

Given these limits, this module's output is used ONLY to help SORT and
LABEL already-retrieved evidence for human/LLM review later — never to
assert, on its own, that an interaction exists or doesn't.

HOW TO TEST IT
----------------
    cd backend
    pytest tests/test_entity_matching.py -v

Pure functions, no dependencies — runs instantly, fully offline.
"""

from __future__ import annotations

import re

# Small, illustrative, NON-EXHAUSTIVE mapping from common drug-class
# terms to representative generic ingredient names in that class.
# See module docstring — this is NOT a clinical-grade classification
# database (not derived from RxNorm/RxClass or any curated source).
DRUG_CLASS_TERMS: dict[str, set[str]] = {
    "nsaid": {"ibuprofen", "naproxen", "diclofenac", "celecoxib", "indomethacin", "aspirin", "ketorolac"},
    "nsaids": {"ibuprofen", "naproxen", "diclofenac", "celecoxib", "indomethacin", "aspirin", "ketorolac"},
    "ssri": {"fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram"},
    "maoi": {"phenelzine", "tranylcypromine", "isocarboxazid", "selegiline"},
    "anticoagulant": {"warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin"},
    "anticoagulants": {"warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin"},
    "beta blocker": {"metoprolol", "atenolol", "propranolol", "carvedilol"},
    "beta-blocker": {"metoprolol", "atenolol", "propranolol", "carvedilol"},
    "ace inhibitor": {"lisinopril", "enalapril", "ramipril", "captopril"},
}


def _word_pattern(term: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(term.strip()) + r"\b", re.IGNORECASE)


def find_mentions(text: str, terms: list[str]) -> list[str]:
    """
    Returns the subset of `terms` (e.g. a drug's normalized name plus
    known synonyms) that appear as whole-word, case-insensitive matches
    in `text`. Empty/blank terms are ignored. Order of the input `terms`
    list is preserved in the (possibly shorter) output.
    """
    matched = []
    for term in terms:
        if term and term.strip() and _word_pattern(term).search(text):
            matched.append(term)
    return matched


def find_class_mention_containing(text: str, generic_name: str) -> str | None:
    """
    Returns the first drug-class term found in `text` (from
    DRUG_CLASS_TERMS) whose member list includes `generic_name`
    (case-insensitive exact membership check), or None if no such class
    term is mentioned in the text.
    """
    if not generic_name:
        return None
    generic_lower = generic_name.strip().lower()
    for class_term, members in DRUG_CLASS_TERMS.items():
        if generic_lower in members and _word_pattern(class_term).search(text):
            return class_term
    return None
