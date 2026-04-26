"""Unit tests for split_compound_sentence — the second-pass clause splitter.

These tests guard the strictly-additive guarantee: simple sentences pass
through unchanged, and compound sentences are split only when every resulting
segment is long enough to be meaningful (_MIN_CLAUSE_LEN = 15 chars).
"""

from __future__ import annotations

import pytest

from app.services.inference import _MIN_CLAUSE_LEN, split_compound_sentence, split_into_sentences


# ── Regression: simple sentences must pass through unchanged ──────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        "I feel sad today",
        "tired",
        "Can't sleep at all",
        "I'm feeling hopeless",
        "sad and tired",  # "and" split would produce "sad" (3 chars) < 15 → no split
        "Feeling sad and blue",  # both fragments < 15 chars → no split
        "No appetite lately",
    ],
)
def test_simple_sentences_pass_through(sentence: str):
    result = split_compound_sentence(sentence)
    assert result == [sentence], f"Should not have split: {sentence!r}"


# ── Comma splits ──────────────────────────────────────────────────────────────


def test_comma_split_fatigue_cognitive():
    """The motivating case: FATIGUE + COGNITIVE compound sentence."""
    sentence = "Feeling overwhelmed and exhausted lately, can't seem to focus on anything"
    result = split_compound_sentence(sentence)
    assert len(result) == 2
    assert result[0] == "Feeling overwhelmed and exhausted lately"
    assert result[1] == "can't seem to focus on anything"


def test_comma_split_two_long_clauses():
    sentence = "I haven't been sleeping well at all, my appetite has completely disappeared"
    result = split_compound_sentence(sentence)
    assert len(result) == 2
    assert all(len(p) >= _MIN_CLAUSE_LEN for p in result)


def test_comma_no_split_if_fragment_too_short():
    """'crying most days' (16 chars) is borderline — verify it does split."""
    sentence = "I've been feeling very depressed lately, crying most days"
    result = split_compound_sentence(sentence)
    # "crying most days" = 16 chars >= 15, so this should split
    assert len(result) == 2


def test_comma_list_short_items_no_split():
    """Comma-separated short list items should not produce micro-fragments."""
    sentence = "Sad, tired, hopeless"  # all parts < 15 chars
    result = split_compound_sentence(sentence)
    assert result == [sentence]


# ── Semicolon splits ──────────────────────────────────────────────────────────


def test_semicolon_split():
    sentence = "I can't get out of bed in the morning; everything feels pointless"
    result = split_compound_sentence(sentence)
    assert len(result) == 2
    assert all(len(p) >= _MIN_CLAUSE_LEN for p in result)


def test_semicolon_rejected_when_fragment_too_short_falls_to_comma():
    # "I feel empty" = 12 chars < _MIN_CLAUSE_LEN → semicolon split rejected.
    # Falls through to comma split:
    #   "I feel empty; nothing brings me joy" (36 chars) + "not even the things I used to love" (34 chars)
    sentence = "I feel empty; nothing brings me joy, not even the things I used to love"
    result = split_compound_sentence(sentence)
    assert len(result) == 2
    assert result[0] == "I feel empty; nothing brings me joy"
    assert result[1] == "not even the things I used to love"


# ── Adversative conjunction splits ───────────────────────────────────────────


@pytest.mark.parametrize(
    "sentence",
    [
        "I want to feel better but nothing seems to help anymore",
        "I try to push through yet the exhaustion never lifts",
        "I used to enjoy cooking though now it feels like a chore",
        "I show up to work however inside I feel completely numb",
    ],
)
def test_adversative_conjunction_split(sentence: str):
    result = split_compound_sentence(sentence)
    assert len(result) == 2, f"Expected 2 clauses for: {sentence!r}"
    assert all(len(p) >= _MIN_CLAUSE_LEN for p in result)


# ── "and" splits (long sentences only) ───────────────────────────────────────


def test_and_split_long_sentence():
    """Long sentence with "and" joining two independent clauses should split."""
    sentence = "I've been feeling really exhausted and I can't concentrate at work at all"
    result = split_compound_sentence(sentence)
    assert len(result) == 2
    assert all(len(p) >= _MIN_CLAUSE_LEN for p in result)


def test_and_no_split_short_sentence():
    """Short sentence with "and" must NOT split (under _AND_MIN_LEN = 40)."""
    sentence = "Feeling sad and completely hopeless now"  # 39 chars — just under threshold
    result = split_compound_sentence(sentence)
    assert result == [sentence]


# ── Integration with split_into_sentences pipeline ───────────────────────────


def test_full_pipeline_compound_post():
    """Verify that the two-pass pipeline (sentences → compound split) expands
    a multi-clause post into the right number of analyzable segments."""
    text = (
        "I haven't been sleeping well lately. "
        "Feeling overwhelmed and exhausted, can't seem to focus on anything. "
        "I don't enjoy the things I used to love."
    )
    # First pass: 3 sentences (split on `. `)
    sentences = split_into_sentences(text)
    assert len(sentences) == 3

    # Second pass: middle sentence should split on comma
    expanded: list[str] = []
    for s in sentences:
        expanded.extend(split_compound_sentence(s))

    # Sentence 1: no split → 1 segment
    # Sentence 2: comma split → 2 segments
    # Sentence 3: no split → 1 segment
    assert len(expanded) == 4
    assert "Feeling overwhelmed and exhausted" in expanded[1]
    assert "can't seem to focus on anything" in expanded[2]


def test_all_segments_meet_min_length():
    """Any output of split_compound_sentence must be >= _MIN_CLAUSE_LEN chars."""
    cases = [
        "Feeling overwhelmed and exhausted lately, can't seem to focus on anything",
        "I can't sleep at night; I'm exhausted all day long",
        "I want to feel better but nothing seems to work",
        "sad and tired",
        "I feel completely empty",
    ]
    for sentence in cases:
        result = split_compound_sentence(sentence)
        for part in result:
            assert len(part) >= _MIN_CLAUSE_LEN or result == [sentence], (
                f"Fragment {part!r} too short (from {sentence!r})"
            )
