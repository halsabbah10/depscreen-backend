"""
Terminology proxy routes.

Thin, rate-limited proxies to the NLM Clinical Tables API so the frontend
can do drug-name and ICD-10 autocomplete without leaking user queries to
a third-party origin directly, and without CORS pain.

Sources:
- RxTerms (drug names, from NLM/RxNorm):
  https://clinicaltables.nlm.nih.gov/api/rxterms/v3/search
- ICD-10-CM (diagnosis codes):
  https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search

Both are public, free, no auth required, maintained by the US National
Library of Medicine.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, Query, Request

from app.middleware.rate_limiter import limiter
from app.models.db import User
from app.services.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

RXTERMS_URL = "https://clinicaltables.nlm.nih.gov/api/rxterms/v3/search"
ICD10_URL = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"

_HTTP_TIMEOUT = 4.0  # seconds — keep tight; autocomplete must feel instant


async def _fetch_clinical_tables(url: str, params: dict) -> list[dict]:
    """Call a Clinical Tables endpoint and return normalized suggestion rows.

    The NLM API returns a 4-element array: [total, codes, extra_fields, display_fields].
    We normalize each row into {"value": code-or-term, "label": display text, "extra": ...}.
    """
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException:
        logger.warning(f"Terminology upstream timeout: {url}")
        return []
    except httpx.HTTPError as e:
        logger.warning(f"Terminology upstream error: {e}")
        return []

    # Defensive: NLM may change shape on error
    if not isinstance(data, list) or len(data) < 4:
        return []

    codes = data[1] or []
    display_rows = data[3] or []

    suggestions = []
    for i, code in enumerate(codes):
        row = display_rows[i] if i < len(display_rows) else []
        label_parts = [p for p in row if p]
        label = " — ".join(label_parts) if label_parts else str(code)
        suggestions.append({"value": str(code), "label": label})
    return suggestions


@router.get("/rxnorm")
@limiter.limit("60/minute")
async def rxnorm_autocomplete(
    request: Request,
    q: str = Query(..., min_length=2, max_length=60, description="Partial drug name"),
    limit: int = Query(8, ge=1, le=20),
    _user: User = Depends(get_current_user),
):
    """Autocomplete drug names via NLM RxTerms (RxNorm-backed)."""
    suggestions = await _fetch_clinical_tables(
        RXTERMS_URL,
        {
            "terms": q,
            "maxList": limit,
            "ef": "STRENGTHS_AND_FORMS,DISPLAY_NAME",
        },
    )
    return {"suggestions": suggestions}


@router.get("/icd10")
@limiter.limit("60/minute")
async def icd10_autocomplete(
    request: Request,
    q: str = Query(..., min_length=2, max_length=60, description="Partial term or code"),
    limit: int = Query(8, ge=1, le=20),
    _user: User = Depends(get_current_user),
):
    """Autocomplete ICD-10-CM codes via NLM Clinical Tables.

    Matches on both code and name, returns {value: code, label: "code — name"}.
    """
    suggestions = await _fetch_clinical_tables(
        ICD10_URL,
        {
            "terms": q,
            "maxList": limit,
            "sf": "code,name",
            "df": "code,name",
        },
    )
    return {"suggestions": suggestions}
