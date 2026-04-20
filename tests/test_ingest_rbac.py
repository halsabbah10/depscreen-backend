"""Tests for ingest endpoint role-based access control."""

from __future__ import annotations


def test_clinician_cannot_submit_checkin(client, clinician_headers):
    """Ingest endpoints should only be accessible to patients."""
    resp = client.post(
        "/api/ingest/checkin",
        json={"responses": [{"question_id": "q1", "answer": 3}]},
        headers=clinician_headers,
    )
    assert resp.status_code == 403


def test_clinician_cannot_submit_reddit(client, clinician_headers):
    resp = client.post(
        "/api/ingest/reddit",
        json={"username": "testuser"},
        headers=clinician_headers,
    )
    assert resp.status_code == 403


def test_clinician_cannot_submit_x(client, clinician_headers):
    resp = client.post(
        "/api/ingest/x",
        json={"username": "testuser"},
        headers=clinician_headers,
    )
    assert resp.status_code == 403


def test_clinician_cannot_submit_bulk(client, clinician_headers):
    resp = client.post(
        "/api/ingest/bulk",
        json={"texts": ["some text"]},
        headers=clinician_headers,
    )
    assert resp.status_code == 403
