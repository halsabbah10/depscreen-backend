"""Integration tests for token denylist + auth flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch


def test_logout_denies_tokens(client, patient_user, patient_headers):
    """After logout, the access token should be checked against the denylist."""
    client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "test-password-123"},
    )

    mock_deny = AsyncMock()
    with patch("app.api.routes.auth.deny_token", mock_deny):
        resp = client.post("/api/auth/logout", headers=patient_headers)
        assert resp.status_code == 200
        assert mock_deny.call_count >= 1


def test_denied_token_rejected(client, patient_user, patient_headers):
    """A token whose JTI is in the denylist should be rejected."""
    with patch("app.services.auth.is_denied", new_callable=AsyncMock, return_value=True):
        resp = client.get("/api/auth/me", headers=patient_headers)
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()


def test_non_denied_token_accepted(client, patient_user, patient_headers):
    """A token whose JTI is NOT denied should work normally."""
    with patch("app.services.auth.is_denied", new_callable=AsyncMock, return_value=False):
        resp = client.get("/api/auth/me", headers=patient_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == patient_user.id
