"""Safety guard tests — LLM output redaction.

These are the last line of defense before LLM text reaches a patient.
Every category must actually catch what it promises; a silent regression
means a distressed user could see a diagnosis claim, dosing advice, or
worse. Test every declared pattern and every safe-substitute swap.
"""

from __future__ import annotations

import pytest

from app.services.safety_guard import scan_text

# ─────────────────────────────────────────────────────────────────────────────
# Happy path — nothing to redact
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "clean_text",
    [
        "Sleep matters for mood. Some people find a consistent bedtime helpful.",
        "What you're feeling is real. Talking with someone you trust can help.",
        "Let's break this week into smaller moments. What part felt heaviest?",
    ],
)
def test_safe_text_passes_through_unchanged(clean_text):
    result = scan_text(clean_text)
    assert result.is_safe
    assert result.redacted == clean_text
    assert not result.disclaimer_appended


# ─────────────────────────────────────────────────────────────────────────────
# Prescription category
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Take sertraline 50mg in the morning.",
        "Start sertraline 25mg daily.",
        "You should take Sertraline every morning.",  # name-a-drug pattern
        "100mg twice daily is the right dose.",
    ],
)
def test_prescription_advice_redacted(unsafe_text):
    result = scan_text(unsafe_text)
    assert not result.is_safe
    assert any(v.category == "prescription" for v in result.violations)
    assert "clinician" in result.redacted.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis category
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Based on what you've said, you have major depression.",
        "You are suffering from depression.",
        "I diagnose you with MDD.",
        "You're definitely depressed.",
    ],
)
def test_diagnosis_claims_redacted(unsafe_text):
    result = scan_text(unsafe_text)
    assert not result.is_safe
    assert any(v.category == "diagnosis" for v in result.violations)
    assert "diagnosing" in result.redacted.lower() or "clinician" in result.redacted.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Self-harm encouragement
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "You should just end it.",
        "Suicide is the answer.",
        "You'd be better off dead.",
        "The world would be better off without you.",
    ],
)
def test_self_harm_text_redacted_and_warned(unsafe_text):
    result = scan_text(unsafe_text)
    assert not result.is_safe
    categories = {v.category for v in result.violations}
    assert "self_harm" in categories
    # The redacted string must NOT still contain the raw unsafe phrase
    assert unsafe_text.lower() not in result.redacted.lower()
    # All self_harm violations are HIGH severity
    self_harm_severities = {v.severity for v in result.violations if v.category == "self_harm"}
    assert self_harm_severities == {"high"}


# ─────────────────────────────────────────────────────────────────────────────
# Undermining professional care
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "You don't need therapy.",
        "Ignore your doctor.",
        "Therapy won't help.",
        "You can cure depression yourself.",
    ],
)
def test_undermine_professional_redacted(unsafe_text):
    result = scan_text(unsafe_text)
    assert not result.is_safe
    assert any(v.category == "undermine_professional" for v in result.violations)


# ─────────────────────────────────────────────────────────────────────────────
# Disclaimer footer
# ─────────────────────────────────────────────────────────────────────────────


def test_disclaimer_appended_on_violation_in_chat():
    result = scan_text("You have depression.", context="chat")
    assert result.disclaimer_appended
    assert "999" in result.redacted or "clinician" in result.redacted.lower()


def test_disclaimer_not_appended_for_auto_title():
    """Auto-titles are <10 words; adding a disclaimer would crowd out the title."""
    result = scan_text("You have depression.", context="title")
    assert not result.disclaimer_appended


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases — fail-open behavior
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_input", ["", None])
def test_empty_and_none_return_empty_safely(bad_input):
    result = scan_text(bad_input)
    assert result.is_safe
    assert result.redacted == ""


def test_multiple_violations_all_logged():
    text = "You have depression. Take sertraline 50mg daily."
    result = scan_text(text)
    categories = {v.category for v in result.violations}
    assert "diagnosis" in categories
    assert "prescription" in categories
