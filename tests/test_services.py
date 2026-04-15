"""Service-level tests — pure logic, no DB, no HTTP.

These guard the domain rules that our routes depend on. If a crisis
keyword stops being detected, a patient mid-crisis gets a generic reply
instead of 999. If CPR validation drifts, onboarding starts accepting
malformed IDs. Guard them directly, not indirectly through routes.
"""

from __future__ import annotations

import pytest

from app.core import localization as loc
from app.services import chat as chat_svc

# ─────────────────────────────────────────────────────────────────────────────
# Crisis keyword detection
# ─────────────────────────────────────────────────────────────────────────────


def _contains_crisis(text: str) -> bool:
    """Mirrors the same lowercase-substring check the route uses."""
    lowered = text.lower()
    return any(kw in lowered for kw in chat_svc.CRISIS_KEYWORDS)


@pytest.mark.parametrize(
    "text",
    [
        "i want to kill myself",
        "I'm suicidal today",
        "I want to die",
        "I've been thinking about ending it all",
        "I don't want to be here anymore",
        "life isn't worth living",
        "I made a plan for tonight",
        "I've been cutting myself",
        "I have the means",
    ],
)
def test_crisis_keyword_detected(text):
    assert _contains_crisis(text), f"Should have flagged: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "I feel sad today",
        "I'm tired of feeling this way",
        "work has been really hard",
        "I keep replaying that conversation",
        "my family doesn't understand",
        "I had a tough week",
    ],
)
def test_non_crisis_text_not_flagged(text):
    assert not _contains_crisis(text), f"Should NOT have flagged: {text!r}"


def test_crisis_response_is_bahrain_localized():
    """The static crisis fallback must mention Bahrain resources, not US ones."""
    resp = chat_svc.CRISIS_RESPONSE
    assert "999" in resp, "Bahrain national emergency number missing"
    # Must NOT carry US-centric fallbacks
    assert "988" not in resp
    assert "Crisis Text Line" not in resp


# ─────────────────────────────────────────────────────────────────────────────
# Bahrain localization helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "phone,expected",
    [
        ("+97332223333", "+97332223333"),  # already international
        ("32223333", "+97332223333"),  # local 8-digit
        ("00973 3222 3333", "+97332223333"),  # 00973 prefix with spaces
        ("+973 3222 3333", "+97332223333"),  # international with spaces
        ("973-3222-3333", "+97332223333"),  # 973 without +
    ],
)
def test_normalize_phone_bahrain(phone, expected):
    assert loc.normalize_phone(phone) == expected


@pytest.mark.parametrize(
    "bad_phone",
    [
        "+14155551234",  # US
        "not-a-phone",
        "12345",  # too short
        "",
    ],
)
def test_normalize_phone_rejects_non_bahrain(bad_phone):
    with pytest.raises(ValueError):
        loc.normalize_phone(bad_phone)


def test_format_phone_display_adds_spacing():
    assert loc.format_phone_display("+97332223333") == "+973 3222 3333"


def test_format_date_dd_mm_yyyy():
    from datetime import date

    assert loc.format_date(date(2026, 4, 15)) == "15/04/2026"


def test_format_datetime_dd_mm_yyyy_hhmm():
    from datetime import datetime

    assert loc.format_datetime(datetime(2026, 4, 15, 14, 30)) == "15/04/2026 14:30"


# ─────────────────────────────────────────────────────────────────────────────
# CPR (Bahrain National ID)
# ─────────────────────────────────────────────────────────────────────────────


def test_cpr_rejects_wrong_length():
    assert loc.validate_cpr("12345") is False
    assert loc.validate_cpr("1234567890") is False


def test_cpr_rejects_non_digits():
    assert loc.validate_cpr("85A423456") is False


def test_cpr_rejects_invalid_month():
    # Month 13 is impossible
    assert loc.validate_cpr("851323456") is False


def test_cpr_accepts_plausible_format():
    # April 1985 → plausible DOB
    assert loc.validate_cpr("850423456") is True


def test_cpr_accepts_separators():
    # Display format with dashes
    assert loc.validate_cpr("8504-2345-6") is True
    assert loc.validate_cpr("8504 2345 6") is True


def test_cpr_display_formatting():
    assert loc.format_cpr_display("850423456") == "8504-2345-6"


def test_cpr_extract_dob():
    result = loc.extract_dob_from_cpr("850423456")
    assert result is not None
    year, month = result
    assert month == 4
    # Year should be 1985 (since 85 > current 2-digit year)
    assert year == 1985


# ─────────────────────────────────────────────────────────────────────────────
# Age calculation
# ─────────────────────────────────────────────────────────────────────────────


def test_calculate_age_is_stable_for_past_dob():
    from datetime import date

    # Born in 1990 — age will be at least 30 for the remaining lifespan of
    # this test suite. Avoids a freezegun dependency just for age math.
    age = loc.calculate_age(date(1990, 1, 1))
    assert age >= 30


def test_calculate_age_handles_today_as_dob():
    from datetime import date

    # Someone born today is 0
    assert loc.calculate_age(date.today()) == 0
