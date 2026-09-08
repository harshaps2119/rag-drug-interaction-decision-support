"""
tests/test_entity_matching.py
================================
Tests for app/rag/entity_matching.py — pure functions, no dependencies.

HOW TO RUN
-----------
    cd backend
    pytest tests/test_entity_matching.py -v
"""

from app.rag.entity_matching import DRUG_CLASS_TERMS, find_class_mention_containing, find_mentions


# ---------------------------------------------------------------------------
# find_mentions
# ---------------------------------------------------------------------------

def test_find_mentions_exact_match():
    assert find_mentions("Warfarin may cause bleeding.", ["warfarin"]) == ["warfarin"]


def test_find_mentions_case_insensitive():
    assert find_mentions("WARFARIN increases INR.", ["warfarin"]) == ["warfarin"]


def test_find_mentions_whole_word_only_no_substring_match():
    """'war' should not match inside 'warfarin' — whole-word boundaries matter."""
    assert find_mentions("Warfarin therapy.", ["war"]) == []


def test_find_mentions_multiple_terms_returns_only_matched():
    text = "Ibuprofen may interact with warfarin."
    matched = find_mentions(text, ["warfarin", "Coumadin", "aspirin"])
    assert matched == ["warfarin"]


def test_find_mentions_multiple_matches_preserves_input_order():
    text = "Coumadin, also known as warfarin, requires monitoring."
    matched = find_mentions(text, ["warfarin", "Coumadin"])
    assert matched == ["warfarin", "Coumadin"]


def test_find_mentions_no_match_returns_empty_list():
    assert find_mentions("Ibuprofen dosage information.", ["warfarin"]) == []


def test_find_mentions_ignores_blank_terms():
    assert find_mentions("Warfarin text.", ["", "  ", "warfarin"]) == ["warfarin"]


def test_find_mentions_empty_text_returns_empty():
    assert find_mentions("", ["warfarin"]) == []


def test_find_mentions_does_not_understand_negation():
    """Documented limitation: a negated statement still registers as a mention."""
    text = "No significant interaction has been observed with ibuprofen."
    assert find_mentions(text, ["ibuprofen"]) == ["ibuprofen"]


# ---------------------------------------------------------------------------
# find_class_mention_containing
# ---------------------------------------------------------------------------

def test_find_class_mention_containing_finds_class_term():
    text = "Concomitant use with NSAIDs may increase bleeding risk."
    assert find_class_mention_containing(text, "ibuprofen") == "nsaids"


def test_find_class_mention_containing_case_insensitive():
    text = "Use caution with nsaid medications."
    assert find_class_mention_containing(text, "naproxen") == "nsaid"


def test_find_class_mention_containing_no_class_term_present():
    text = "Take with food."
    assert find_class_mention_containing(text, "ibuprofen") is None


def test_find_class_mention_containing_drug_not_a_member_of_any_mentioned_class():
    text = "Concomitant use with NSAIDs may increase bleeding risk."
    # metformin is not in any class term present in this text
    assert find_class_mention_containing(text, "metformin") is None


def test_find_class_mention_containing_empty_generic_name():
    assert find_class_mention_containing("Some NSAID text.", "") is None


def test_drug_class_terms_are_all_lowercase_for_consistent_membership_checks():
    for members in DRUG_CLASS_TERMS.values():
        for member in members:
            assert member == member.lower()
