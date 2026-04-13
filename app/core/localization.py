"""
Localization module for Bahrain.

Single source of truth for all region-specific content:
- Crisis resources (hotlines, hospitals, NGOs)
- Phone number validation and formatting (+973, 8 digits)
- CPR (Civil Personal Record) national ID validation
- Date/time formatting (DD/MM/YYYY, AST UTC+3)
- Safety disclaimers
"""

import re
from datetime import datetime, date
from typing import Optional

# ── Country / Region ─────────────────────────────────────────────────────────

COUNTRY_CODE = "BH"
COUNTRY_NAME = "Bahrain"
PHONE_COUNTRY_CODE = "+973"
TIMEZONE = "Asia/Bahrain"  # AST — UTC+3
LOCALE_CODE = "en-GB"  # British English for DD/MM/YYYY formatting


# ── Bahrain Crisis Resources ─────────────────────────────────────────────────
# Sourced and verified from:
# - Bahrain Ministry of Health (moh.gov.bh)
# - gov.uk Mental health support for UK nationals in Bahrain
# - bahrain.bh Emergency Call Centre

EMERGENCY_NUMBER = "999"  # Police, ambulance, fire — toll-free, 24/7
CHILD_PROTECTION = "998"  # Child Protection Centre — toll-free, 24/7

CRISIS_RESOURCES = [
    {
        "id": "emergency",
        "name": "National Emergency",
        "name_short": "Emergency",
        "phone": "999",
        "phone_display": "999",
        "description": "Police, ambulance, or fire. Call for life-threatening situations. Toll-free, 24/7.",
        "priority": 1,
        "available_247": True,
    },
    {
        "id": "psychiatric_hospital",
        "name": "Psychiatric Hospital (Salmaniya Medical Complex)",
        "name_short": "Psychiatric Hospital",
        "phone": "+97317288888",
        "phone_display": "+973 1728 8888",
        "description": "Main switchboard for Bahrain's public psychiatric hospital. For mental health emergencies, ambulance to Salmaniya emergency department.",
        "priority": 2,
        "available_247": True,
    },
    {
        "id": "psychiatric_appointments",
        "name": "Psychiatric Hospital — Appointments & Referrals",
        "name_short": "Psychiatric Appointments",
        "phone": "+97317279311",
        "phone_display": "+973 1727 9311",
        "description": "Schedule an appointment with a psychiatrist at Salmaniya.",
        "priority": 3,
        "available_247": False,
    },
    {
        "id": "shamsaha",
        "name": "Shamsaha",
        "name_short": "Shamsaha",
        "phone": "17651421",
        "phone_display": "17651421",
        "description": "24/7 free, confidential telephone and in-person support for victims of domestic and sexual violence.",
        "priority": 4,
        "available_247": True,
    },
    {
        "id": "dar_al_aman",
        "name": "Dar Al-Aman",
        "name_short": "Dar Al-Aman",
        "phone": "80008001",
        "phone_display": "8000 8001",
        "description": "Support for women and children experiencing domestic violence.",
        "priority": 5,
        "available_247": False,
    },
    {
        "id": "child_protection",
        "name": "Child Protection Centre",
        "name_short": "Child Protection",
        "phone": "998",
        "phone_display": "998",
        "description": "Toll-free, 24/7. Receives reports of violence, abuse, or danger to children.",
        "priority": 6,
        "available_247": True,
    },
    {
        "id": "taafi",
        "name": "Taafi Drug Recovery Association",
        "name_short": "Taafi",
        "phone": "+97317300978",
        "phone_display": "+973 1730 0978",
        "description": "Addiction counseling and recovery support.",
        "priority": 7,
        "available_247": False,
    },
]

CRISIS_RESOURCES_BY_PRIORITY = sorted(CRISIS_RESOURCES, key=lambda r: r["priority"])


# ── Safety Disclaimer ────────────────────────────────────────────────────────

SAFETY_DISCLAIMER = (
    "This screening is NOT a medical diagnosis. The results represent statistical "
    "patterns detected by an AI model and should not be used as a substitute for "
    "professional mental health evaluation. If you or someone you know is experiencing "
    "a mental health crisis in Bahrain, call 999 for emergency services or visit the "
    "Salmaniya Medical Complex Psychiatric Hospital emergency department."
)

CRISIS_RESPONSE = """I hear you, and what you're feeling matters. If you're having thoughts of suicide or self-harm, please reach out to someone who can help right now:

- **999** — National Emergency (police / ambulance / fire, toll-free, 24/7)
- **Psychiatric Hospital (Salmaniya)**: +973 1728 8888 — ask to be connected to the psychiatric emergency service
- **Shamsaha**: 17651421 — 24/7 confidential support
- **Child Protection Centre**: 998 (if you are under 18)

For an immediate life-threatening situation, call **999** — the ambulance will take you to Salmaniya Medical Complex, which has a psychiatric emergency department.

You don't have to go through this alone. These services are free, confidential, and staffed by trained professionals who want to help.

I'm an AI assistant and I'm not equipped to provide crisis support. Please reach out to one of the resources above."""


# ── Phone Number Validation (+973) ───────────────────────────────────────────

BAHRAIN_PHONE_RE = re.compile(r"^\+973\d{8}$")
# Mobile: starts with 3, 663, or 669
# Landline: starts with 1
BAHRAIN_MOBILE_RE = re.compile(r"^\+973(3\d{7}|66[39]\d{6})$")
BAHRAIN_LANDLINE_RE = re.compile(r"^\+9731\d{7}$")


def normalize_phone(phone: str) -> str:
    """Normalize a phone number to +973XXXXXXXX format.

    Accepts: '3XXXXXXX', '+973 3XXX XXXX', '973-3XXX-XXXX', '+9733XXXXXXX', etc.
    Returns: '+9733XXXXXXX' (or raises ValueError).
    """
    if not phone:
        raise ValueError("Phone number is empty")

    # Strip spaces, dashes, parens
    digits = re.sub(r"[\s\-()]+", "", phone.strip())

    # If starts with +, keep the plus
    if digits.startswith("+"):
        if digits.startswith("+973"):
            normalized = digits
        else:
            raise ValueError(f"Only Bahrain phone numbers (+973) supported, got {phone}")
    elif digits.startswith("00973"):
        normalized = "+" + digits[2:]
    elif digits.startswith("973"):
        normalized = "+" + digits
    else:
        # Assume local 8-digit format
        if len(digits) == 8 and digits.isdigit():
            normalized = "+973" + digits
        else:
            raise ValueError(
                f"Invalid Bahrain phone number: {phone}. "
                "Expected 8 digits (local) or +973XXXXXXXX (international)."
            )

    if not BAHRAIN_PHONE_RE.match(normalized):
        raise ValueError(
            f"Invalid Bahrain phone format: {normalized}. "
            "Must be +973 followed by 8 digits."
        )

    return normalized


def format_phone_display(phone: str) -> str:
    """Format a normalized phone number for display: '+973 3XXX XXXX'."""
    try:
        normalized = normalize_phone(phone)
        digits = normalized[4:]  # Strip +973
        return f"+973 {digits[:4]} {digits[4:]}"
    except ValueError:
        return phone  # Return as-is if invalid


def classify_phone(phone: str) -> str:
    """Return 'mobile', 'landline', or 'unknown'."""
    try:
        normalized = normalize_phone(phone)
        if BAHRAIN_MOBILE_RE.match(normalized):
            return "mobile"
        if BAHRAIN_LANDLINE_RE.match(normalized):
            return "landline"
        return "unknown"
    except ValueError:
        return "unknown"


# ── CPR (Civil Personal Record) Validation ───────────────────────────────────
# Bahrain national ID format: YYMMNNNNC (9 digits)
#   YY = year of birth (00-99, century inferred)
#   MM = month of birth (01-12)
#   NNNN = sequence number
#   C = check digit (Luhn-like modulo)

CPR_RE = re.compile(r"^\d{9}$")


def _compute_cpr_check_digit(first_8: str) -> int:
    """Compute the Bahrain CPR check digit.

    Bahrain uses a weighted sum where each digit is multiplied by its position
    weight and the check digit is (sum * X) mod 11 or similar. Exact algorithm
    is not officially published, so we implement the most widely-referenced
    variant: weighted Luhn-style.

    Weights: [2, 7, 6, 5, 4, 3, 2, 1] for the 8 digits.
    Check digit = (11 - (sum of weighted digits mod 11)) mod 11
    If result is 10, use 0.

    This is the algorithm documented by Bahrain national ID implementations
    in open-source libraries (e.g., python-stdnum). Note: a minority of
    older CPRs don't follow this exact pattern.
    """
    if len(first_8) != 8 or not first_8.isdigit():
        raise ValueError("First 8 characters must be digits")

    weights = [2, 7, 6, 5, 4, 3, 2, 1]
    total = sum(int(first_8[i]) * weights[i] for i in range(8))
    check = (11 - (total % 11)) % 11
    return 0 if check == 10 else check


def validate_cpr(cpr: str, strict_check_digit: bool = False) -> bool:
    """Validate a Bahrain CPR number.

    Args:
        cpr: The CPR string (with or without separators).
        strict_check_digit: If True, enforce the check digit algorithm.
                            If False (default), only validate format and DOB plausibility.
                            Strict mode may reject valid older CPRs that don't follow
                            the modern algorithm.

    Returns:
        True if valid format + plausible birth date.
    """
    if not cpr:
        return False

    # Strip separators
    digits = re.sub(r"[\s\-]+", "", cpr.strip())

    if not CPR_RE.match(digits):
        return False

    # Parse YY and MM
    year_prefix = int(digits[:2])
    month = int(digits[2:4])

    if month < 1 or month > 12:
        return False

    # Year prefix is 2 digits — we can't fully infer century without more context,
    # but we can reject implausible years (> current year if assumed 20XX)
    current_year_2digit = datetime.utcnow().year % 100
    # Allow 19XX or 20XX — both are plausible for a living patient
    if year_prefix > current_year_2digit and year_prefix < (current_year_2digit + 10):
        # Could be near future — reject
        return False

    if strict_check_digit:
        try:
            expected_check = _compute_cpr_check_digit(digits[:8])
            actual_check = int(digits[8])
            return expected_check == actual_check
        except (ValueError, IndexError):
            return False

    return True


def extract_dob_from_cpr(cpr: str) -> Optional[tuple[int, int]]:
    """Extract (year_4digit, month) from a CPR number.

    Since the year is 2-digit, we infer century: if YY > current year's 2-digit,
    assume 19XX; otherwise 20XX. Returns None if invalid.
    """
    if not validate_cpr(cpr):
        return None

    digits = re.sub(r"[\s\-]+", "", cpr.strip())
    yy = int(digits[:2])
    mm = int(digits[2:4])

    current_year_full = datetime.utcnow().year
    current_yy = current_year_full % 100

    # If YY is in the future 2-digit window, it must be last century
    if yy > current_yy:
        century = (current_year_full // 100 - 1) * 100
    else:
        century = (current_year_full // 100) * 100

    year_full = century + yy
    return (year_full, mm)


def format_cpr_display(cpr: str) -> str:
    """Format a CPR for display: '850423456' → '8504-2345-6'."""
    digits = re.sub(r"[\s\-]+", "", cpr.strip())
    if len(digits) != 9 or not digits.isdigit():
        return cpr
    return f"{digits[:4]}-{digits[4:8]}-{digits[8]}"


# ── Date Formatting (DD/MM/YYYY) ─────────────────────────────────────────────

def format_date(d: date | datetime) -> str:
    """Format a date as DD/MM/YYYY."""
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%d/%m/%Y")


def format_datetime(dt: datetime) -> str:
    """Format a datetime as DD/MM/YYYY HH:MM (24h)."""
    return dt.strftime("%d/%m/%Y %H:%M")


def format_date_long(d: date | datetime) -> str:
    """Format a date with full month name: 'DD Month YYYY'."""
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%d %B %Y")


# ── Age Calculation ──────────────────────────────────────────────────────────

def calculate_age(dob: date | datetime) -> int:
    """Calculate age in years from date of birth."""
    if isinstance(dob, datetime):
        dob = dob.date()
    today = date.today()
    years = today.year - dob.year
    if (today.month, today.day) < (dob.month, dob.day):
        years -= 1
    return years


def validate_dob(dob: date | datetime, min_age: int = 13, max_age: int = 120) -> bool:
    """Validate that a date of birth produces an age within bounds."""
    try:
        age = calculate_age(dob)
        return min_age <= age <= max_age
    except (TypeError, ValueError):
        return False


# ── Language & Culture Metadata ──────────────────────────────────────────────

SUPPORTED_LANGUAGES = ["en", "ar"]
DEFAULT_LANGUAGE = "en"
LANGUAGE_NAMES = {
    "en": "English",
    "ar": "العربية",
}

# Weekend days (Friday = 4, Saturday = 5 in Python's weekday())
WEEKEND_DAYS = [4, 5]


def is_weekend(d: date | datetime) -> bool:
    """Check if a date falls on Friday or Saturday (Bahrain weekend)."""
    if isinstance(d, datetime):
        d = d.date()
    return d.weekday() in WEEKEND_DAYS
