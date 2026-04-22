"""Robust JSON extraction from LLM responses.

Handles common LLM output artifacts: <think> tags, markdown fences,
preamble text, and truncated responses.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response that may contain artifacts or truncation.

    Strategies (in order):
    1. Strip <think> tags and markdown fences, try direct parse.
    2. Depth-tracking brace scan for complete JSON objects.
    3. Truncation repair — close unclosed braces and retry.
    """
    if not text:
        raise ValueError("Empty response from LLM")

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Strategy 1: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: depth-tracking brace scan for complete objects
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(cleaned):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    # Strategy 3: truncation repair — close unclosed braces
    if depth > 0 and start is not None:
        fragment = cleaned[start:]
        # Close any open strings, then close braces
        repaired = fragment + '"' * (fragment.count('"') % 2) + "}" * depth
        try:
            result = json.loads(repaired)
            logger.warning(
                "Repaired truncated JSON (closed %d brace(s)): %s...",
                depth,
                fragment[:80],
            )
            return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")
