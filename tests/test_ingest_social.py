"""Integration tests for social media ingestion endpoints."""

from __future__ import annotations

from unittest.mock import patch

# ── Username validation ──────────────────────────────────────────────────────


def test_reddit_rejects_invalid_username(client, patient_headers):
    resp = client.post(
        "/api/ingest/reddit",
        json={"username": "../../etc/passwd"},
        headers=patient_headers,
    )
    assert resp.status_code == 400
    assert "Invalid" in resp.json()["detail"]


def test_reddit_rejects_too_short_username(client, patient_headers):
    # Field has min_length=3, so "ab" triggers a 422 from Pydantic before the
    # route even runs.
    resp = client.post(
        "/api/ingest/reddit",
        json={"username": "ab"},
        headers=patient_headers,
    )
    assert resp.status_code in (400, 422)


def test_x_rejects_invalid_username(client, patient_headers):
    resp = client.post(
        "/api/ingest/x",
        json={"username": "has spaces!"},
        headers=patient_headers,
    )
    assert resp.status_code in (400, 422)


def test_x_rejects_too_long_username(client, patient_headers):
    # Field has max_length=15, so 16-char input hits Pydantic validation (422).
    resp = client.post(
        "/api/ingest/x",
        json={"username": "a" * 16},
        headers=patient_headers,
    )
    assert resp.status_code in (400, 422)


# ── X unavailable (503) ─────────────────────────────────────────────────────


def test_x_returns_503_when_not_configured(client, patient_headers):
    """When get_x_client() returns None, the endpoint returns 503."""
    # The route imports get_x_client from app.services.container at call time,
    # so we patch at the source to ensure the mock takes effect regardless of
    # how (or when) the module was imported.
    with patch("app.services.container.get_x_client", return_value=None):
        resp = client.post(
            "/api/ingest/x",
            json={"username": "testuser"},
            headers=patient_headers,
        )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"].lower()
