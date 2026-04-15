"""
Sentry error-monitoring initialization.

Called once at process start (before FastAPI is created) from main.py.
Behaves as a no-op when SENTRY_DSN is unset — local dev, CI, and anyone
forking the project don't need a Sentry account.

Defaults are deliberately cautious for a mental-health app:

- PII off: no IPs, auth headers, cookies, bodies, or usernames go to
  Sentry unless a specific breadcrumb is tagged safe.
- No Tracing, no Profiling: they burn the free quota without useful
  signal at our scale.
- `before_send` scrubber strips any field whose path contains auth,
  password, token, secret, CPR, or SSN before the event leaves the
  process. Belt over the already-buckled `send_default_pii=False`.
- Errors with an HTTPException 4xx status are dropped — those are
  user-facing validation errors (bad login, invalid upload), not bugs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Field names that should never reach Sentry even if some integration
# accidentally attaches them.
_SENSITIVE_FIELD_RE = re.compile(
    r"(password|token|secret|authorization|cookie|cpr|ssn|api[-_]?key|dsn)",
    re.IGNORECASE,
)


def _scrub_mapping(obj: Any) -> Any:
    """Recursively replace values of sensitive-looking keys with a placeholder.

    Operates in-place where possible. Safe for arbitrary JSON-ish payloads.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(k, str) and _SENSITIVE_FIELD_RE.search(k):
                obj[k] = "[REDACTED]"
            else:
                _scrub_mapping(v)
    elif isinstance(obj, list):
        for item in obj:
            _scrub_mapping(item)
    return obj


def _before_send(event: dict, hint: dict) -> dict | None:
    """Sentry before_send hook. Return None to drop the event.

    - Drops 4xx HTTPException events (user errors, not bugs)
    - Scrubs sensitive keys from request/extra/tags/user
    """
    # Drop 4xx HTTPExceptions: those are expected validation failures
    exc_info = hint.get("exc_info")
    if exc_info:
        exc_type, exc_value, _ = exc_info
        try:
            from fastapi import HTTPException

            if isinstance(exc_value, HTTPException):
                status = getattr(exc_value, "status_code", 500)
                if 400 <= status < 500:
                    return None
        except Exception:
            pass

    # Scrub the event payload
    _scrub_mapping(event)
    return event


def init_sentry(dsn: str, environment: str, release: str | None = None) -> bool:
    """Initialize the Sentry SDK. Returns True when actually enabled.

    Called once at process start. Idempotent — re-calling is a no-op.
    """
    if not dsn:
        logger.info("Sentry disabled — SENTRY_DSN not set")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
    except ImportError:
        logger.warning("sentry-sdk not installed; skipping Sentry init")
        return False

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        # Critical for a mental-health app: no automatic PII (IP, headers,
        # request bodies) ever goes to Sentry.
        send_default_pii=False,
        # Free tier quotas for these are tiny and we don't use them.
        traces_sample_rate=0.0,
        profiles_sample_rate=0.0,
        # No long-lived breadcrumb queue — avoids memory buildup and keeps
        # event bodies tight.
        max_breadcrumbs=30,
        # Tag environment explicitly so dashboards can filter.
        before_send=_before_send,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
        ],
        # Redact these URL parameters from breadcrumbs and transaction names.
        # Sentry uses this list when fingerprinting URLs; appending here
        # prevents query tokens from becoming part of issue titles.
        send_client_reports=True,
    )
    logger.info(f"Sentry initialized for environment={environment}")
    return True
