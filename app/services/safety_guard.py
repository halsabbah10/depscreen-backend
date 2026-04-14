"""
LLM Output Safety Guard.

Every LLM-generated response that's shown to a patient (chat, explanation,
auto-titles) passes through `scan_text` before delivery. The guard catches:

1. **Prescriptive medication advice** — "take 50mg of X", "stop taking Y",
   "switch to Z". Gemini's safety filters catch most of this, but we add
   deterministic checks as defense-in-depth.
2. **Diagnostic claims** — "you have depression", "you are suffering from…"
   DepScreen is a screening aid, not a diagnostic tool.
3. **Self-harm encouragement** — highly unlikely but non-zero risk with any
   LLM; pattern-detect and replace before sending.
4. **Direct medical instructions that contradict professional care** — e.g.
   "you don't need therapy", "ignore your doctor".

Design:
- Rule-based regex + keyword scan (fast, no LLM call, deterministic).
- On match, replace the problematic phrase with a safe substitute AND append
  a clarifying disclaimer.
- Always logs matches to the audit trail for review.
- Falls open (returns original) on any unexpected error — never breaks UX.

Why not a second LLM call? Latency + cost. A single scan runs in <1ms.
A dedicated AI guardrail would cost a second API round-trip and charge for
every chat message.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SafetyViolation:
    """Single safety issue found in LLM output."""

    category: str  # 'prescription', 'diagnosis', 'self_harm', 'undermine_professional'
    matched_text: str
    severity: str  # 'low', 'medium', 'high'


@dataclass
class SafetyResult:
    """Result of scanning a piece of LLM-generated text."""

    original: str
    redacted: str
    violations: list[SafetyViolation] = field(default_factory=list)
    disclaimer_appended: bool = False

    @property
    def is_safe(self) -> bool:
        return not self.violations


# ── Pattern definitions ──────────────────────────────────────────────────────

# Prescriptive dosing: "take 50mg", "500mg twice daily", "start with X mg"
_PRESCRIPTION_PATTERNS = [
    (
        re.compile(
            r"\b(?:take|start|stop|switch|increase|decrease|continue)\s+(?:taking\s+)?\w+\s+(?:\d+\s*mg|\d+\s*ml)",
            re.IGNORECASE,
        ),
        "high",
    ),
    (re.compile(r"\b\d+\s*mg\s+(?:twice|three|four|once)\s+(?:daily|a\s+day|per\s+day)", re.IGNORECASE), "high"),
    (
        re.compile(
            r"\bi\s+recommend\s+(?:taking|trying|starting|switching)\s+\w+\s+(?:\d+\s*mg|\d+\s*ml)", re.IGNORECASE
        ),
        "high",
    ),
    # Direct name-a-drug prescription advice
    (
        re.compile(
            r"\byou\s+should\s+(?:take|try|start|switch\s+to|stop\s+taking)\s+[A-Z][a-z]+(?:line|zac|pam|ide|pril|oxetine|pram)",
            re.IGNORECASE,
        ),
        "high",
    ),
]

# Diagnostic claims
_DIAGNOSIS_PATTERNS = [
    (
        re.compile(
            r"\byou\s+(?:have|are\s+suffering\s+from|are\s+clinically|definitely\s+have)\s+(?:major\s+)?(?:depression|MDD|bipolar|schizophrenia|anxiety\s+disorder)",
            re.IGNORECASE,
        ),
        "high",
    ),
    (
        re.compile(
            r"\byou'?re\s+(?:definitely|clearly|obviously)\s+(?:depressed|bipolar|schizophrenic)", re.IGNORECASE
        ),
        "high",
    ),
    (re.compile(r"\bi\s+diagnose\s+you\s+with", re.IGNORECASE), "high"),
    (re.compile(r"\byour\s+diagnosis\s+is", re.IGNORECASE), "medium"),
]

# Self-harm encouragement — extremely unlikely from a safety-trained LLM,
# but belt-and-suspenders. Any match is HIGH and replaced entirely.
_SELF_HARM_PATTERNS = [
    (
        re.compile(
            r"\byou\s+should\s+(?:just\s+)?(?:end\s+it|kill\s+yourself|harm\s+yourself|cut\s+yourself)", re.IGNORECASE
        ),
        "high",
    ),
    (
        re.compile(
            r"\b(?:suicide|self[\s-]?harm|cutting)\s+is\s+(?:the\s+answer|the\s+only\s+way|a\s+solution)", re.IGNORECASE
        ),
        "high",
    ),
    (re.compile(r"\byou(?:'?d|\s+would)\s+be\s+better\s+off\s+(?:dead|gone|without)", re.IGNORECASE), "high"),
    (re.compile(r"\bworld\s+would\s+be\s+better\s+off\s+without\s+you", re.IGNORECASE), "high"),
]

# Undermining professional care
_UNDERMINE_PATTERNS = [
    (re.compile(r"\byou\s+don'?t\s+need\s+(?:therapy|a\s+(?:therapist|doctor|psychiatrist))", re.IGNORECASE), "medium"),
    (re.compile(r"\bignore\s+(?:your\s+)?(?:doctor|therapist|psychiatrist|medication)", re.IGNORECASE), "high"),
    (
        re.compile(r"\b(?:professional\s+help|therapy|medication)\s+(?:won'?t|doesn'?t|cannot)\s+help", re.IGNORECASE),
        "medium",
    ),
    (
        re.compile(
            r"\byou\s+can\s+(?:cure|heal|fix)\s+(?:depression|anxiety)\s+(?:yourself|alone|on\s+your\s+own)",
            re.IGNORECASE,
        ),
        "medium",
    ),
]


# ── Replacement strings (safe substitutes) ───────────────────────────────────

# Tone principle: the patient may be distressed. Use warm, first-person-friend
# language — but keep it specific enough to preserve clinical context. Empty
# empathy without substance helps no one.
_SAFE_SUBSTITUTES = {
    "prescription": "(medication specifics like dosage or timing — that's something your clinician should guide, not me. They can tailor it to your full history.)",
    "diagnosis": "(I can describe what the screening noticed, but diagnosing is something only a licensed clinician can do — they have the training and the full picture.)",
    "self_harm": "(I'm not going to repeat that — what you're feeling matters, and you deserve real support right now)",
    "undermine_professional": "(evidence-based care — therapy, medication when indicated, or both — genuinely helps many people. You don't have to navigate this alone.)",
}


_DISCLAIMER_FOOTER = (
    "\n\n---\n_A gentle reminder: I'm here to listen and share information, but I'm not a doctor. "
    "For medication questions or a diagnosis, please talk with your clinician. "
    "And if things feel too heavy to carry, Bahrain 999 or Shamsaha 17651421 are there, 24/7."
)


# ── Main scan function ──────────────────────────────────────────────────────


def scan_text(text: str, context: str = "chat") -> SafetyResult:
    """Scan LLM-generated text for unsafe content.

    Args:
        text: The LLM output to scan.
        context: Hint about where the text is used ('chat', 'explanation', 'title').
                 Auto-titles get a lighter touch since they're short and low-risk.

    Returns:
        SafetyResult with original, redacted (safe-to-display), and violations list.
    """
    if not text or not isinstance(text, str):
        return SafetyResult(original=text or "", redacted=text or "")

    try:
        violations: list[SafetyViolation] = []
        redacted = text

        pattern_groups = [
            ("prescription", _PRESCRIPTION_PATTERNS),
            ("diagnosis", _DIAGNOSIS_PATTERNS),
            ("self_harm", _SELF_HARM_PATTERNS),
            ("undermine_professional", _UNDERMINE_PATTERNS),
        ]

        for category, patterns in pattern_groups:
            for pattern, severity in patterns:
                for match in pattern.finditer(text):
                    violations.append(
                        SafetyViolation(
                            category=category,
                            matched_text=match.group(0)[:200],
                            severity=severity,
                        )
                    )
                # Replace all matches with the safe substitute
                if pattern.search(redacted):
                    redacted = pattern.sub(_SAFE_SUBSTITUTES[category], redacted)

        # Append the disclaimer only when violations were found AND context
        # allows space for it (not auto-titles)
        disclaimer_appended = False
        if violations and context in ("chat", "explanation"):
            redacted += _DISCLAIMER_FOOTER
            disclaimer_appended = True

        if violations:
            logger.warning(
                f"[safety_guard] {len(violations)} violations in {context}: "
                f"{[(v.category, v.severity) for v in violations]}"
            )

        return SafetyResult(
            original=text,
            redacted=redacted,
            violations=violations,
            disclaimer_appended=disclaimer_appended,
        )
    except Exception as e:
        logger.error(f"[safety_guard] scan failed ({type(e).__name__}: {e}) — returning original")
        # Fail-open: never break the UX because the guardrail itself errored
        return SafetyResult(original=text or "", redacted=text or "")
