"""Tests for httpOnly cookie-based refresh token flow."""

from __future__ import annotations


def test_login_sets_refresh_cookie(client, patient_user):
    """Login should set an httpOnly refresh_token cookie."""
    resp = client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "test-password-123"},
    )
    assert resp.status_code == 200
    cookie = resp.cookies.get("refresh_token")
    assert cookie is not None, "Expected refresh_token cookie to be set"
    data = resp.json()
    assert "refresh_token" not in data
    assert data["access_token"]
    assert data["user"]["id"] == patient_user.id


def test_register_sets_refresh_cookie(client):
    """Register should set an httpOnly refresh_token cookie."""
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "cookie-test@test.local",
            "password": "SecurePass123",
            "full_name": "Cookie Tester",
            "role": "patient",
        },
    )
    assert resp.status_code == 200
    cookie = resp.cookies.get("refresh_token")
    assert cookie is not None, "Expected refresh_token cookie to be set"
    data = resp.json()
    assert "refresh_token" not in data


def test_refresh_reads_from_cookie(client, patient_user):
    """Refresh should read the token from cookie, not request body."""
    login_resp = client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "test-password-123"},
    )
    assert login_resp.status_code == 200
    refresh_resp = client.post("/api/auth/refresh")
    assert refresh_resp.status_code == 200
    data = refresh_resp.json()
    assert data["access_token"]
    assert "refresh_token" not in data
    new_cookie = refresh_resp.cookies.get("refresh_token")
    assert new_cookie is not None


def test_refresh_without_cookie_returns_401(client):
    """Refresh without a cookie should 401."""
    resp = client.post("/api/auth/refresh")
    assert resp.status_code == 401


def test_logout_clears_cookie(client, patient_user, patient_headers):
    """Logout should delete the refresh_token cookie."""
    client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "test-password-123"},
    )
    resp = client.post("/api/auth/logout", headers=patient_headers)
    assert resp.status_code == 200
    cookie = resp.cookies.get("refresh_token")
    assert cookie is None or cookie == ""
