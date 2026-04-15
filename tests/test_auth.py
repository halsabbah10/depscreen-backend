"""Route-level tests for /api/auth — register, login, refresh, me, logout, RBAC."""

from __future__ import annotations


def test_register_patient_creates_user_and_returns_tokens(client):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "newbie@test.local",
            "password": "SecurePass123",
            "full_name": "New Patient",
            "role": "patient",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["access_token"]
    assert data["refresh_token"]
    assert data["user"]["email"] == "newbie@test.local"
    assert data["user"]["role"] == "patient"
    # Patients don't receive a clinician_code
    assert data["user"]["clinician_code"] is None


def test_register_clinician_gets_invite_code(client):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "doctor@test.local",
            "password": "SecurePass123",
            "full_name": "Dr. New",
            "role": "clinician",
        },
    )
    assert resp.status_code == 200
    code = resp.json()["user"]["clinician_code"]
    assert code and len(code) == 6


def test_register_rejects_duplicate_email(client, patient_user):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": patient_user.email,
            "password": "SecurePass123",
            "full_name": "Duplicate",
            "role": "patient",
        },
    )
    assert resp.status_code == 409


def test_register_rejects_short_password(client):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "short@test.local",
            "password": "abc",
            "full_name": "Short Pass",
            "role": "patient",
        },
    )
    # Pydantic min_length validation fires before the 400 from the service
    assert resp.status_code in (400, 422)


def test_register_rejects_invalid_role(client):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "wrongrole@test.local",
            "password": "SecurePass123",
            "full_name": "Wrong Role",
            "role": "hacker",
        },
    )
    assert resp.status_code in (400, 422)


def test_login_happy_path(client, patient_user):
    resp = client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "test-password-123"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"]
    assert data["user"]["id"] == patient_user.id


def test_login_wrong_password_returns_401(client, patient_user):
    resp = client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "wrong-password"},
    )
    assert resp.status_code == 401


def test_login_unknown_email_returns_401(client):
    resp = client.post(
        "/api/auth/login",
        json={"email": "ghost@test.local", "password": "whatever123"},
    )
    assert resp.status_code == 401


def test_me_requires_auth(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_returns_profile_for_valid_token(client, patient_user, patient_headers):
    resp = client.get("/api/auth/me", headers=patient_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == patient_user.id
    assert resp.json()["role"] == "patient"


def test_me_rejects_refresh_token_as_access(client, patient_user):
    """A refresh token shouldn't be accepted on authenticated routes."""
    from app.core.config import get_settings
    from app.services.auth import create_refresh_token

    refresh = create_refresh_token(patient_user.id, get_settings())
    resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {refresh}"})
    assert resp.status_code == 401


def test_refresh_rotates_tokens(client, patient_user):
    # First login to get a real refresh token
    login = client.post(
        "/api/auth/login",
        json={"email": patient_user.email, "password": "test-password-123"},
    )
    refresh = login.json()["refresh_token"]

    resp = client.post("/api/auth/refresh", json={"refresh_token": refresh})
    assert resp.status_code == 200
    new_access = resp.json()["access_token"]
    assert new_access
    # Should work on /me
    ok = client.get("/api/auth/me", headers={"Authorization": f"Bearer {new_access}"})
    assert ok.status_code == 200


def test_refresh_rejects_access_token(client, patient_headers):
    """Passing an access token where a refresh is expected must 401."""
    access = patient_headers["Authorization"].removeprefix("Bearer ")
    resp = client.post("/api/auth/refresh", json={"refresh_token": access})
    assert resp.status_code == 401


def test_logout_records_audit_event(client, patient_user, patient_headers, db):
    from app.models.db import AuditLog

    resp = client.post("/api/auth/logout", headers=patient_headers)
    assert resp.status_code == 200

    # Verify an audit row was written
    entries = (
        db.query(AuditLog)
        .filter(
            AuditLog.user_id == patient_user.id,
            AuditLog.action == "logout",
        )
        .all()
    )
    assert len(entries) == 1


def test_patient_cannot_register_with_invalid_clinician_code(client):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "orphan@test.local",
            "password": "SecurePass123",
            "full_name": "Orphan",
            "role": "patient",
            "clinician_code": "NOTREAL",
        },
    )
    assert resp.status_code == 404


def test_patient_register_links_to_clinician_via_code(client, clinician_user):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": "linked@test.local",
            "password": "SecurePass123",
            "full_name": "Linked Patient",
            "role": "patient",
            "clinician_code": clinician_user.clinician_code,
        },
    )
    assert resp.status_code == 200
    # The new patient should be linked — verify via /me after login
    login = client.post(
        "/api/auth/login",
        json={"email": "linked@test.local", "password": "SecurePass123"},
    )
    token = login.json()["access_token"]
    # Patient's clinician link isn't exposed on /me; check the DB directly
    from app.core.config import get_settings
    from app.models.db import User
    from app.services.auth import decode_token

    payload = decode_token(token, get_settings())
    user_id = payload["sub"]

    # Need a session to query — use the app's override
    # (we don't have direct access to the `db` fixture session here since
    # this test commits via the HTTP path)
    # Pull the user back via a fresh session on the same engine

    from app.models.db import get_db
    from tests.conftest import _make_user  # noqa: F401 — just to keep import path stable

    db_session = next(client.app.dependency_overrides[get_db]())
    try:
        created = db_session.query(User).filter(User.id == user_id).first()
        assert created is not None
        assert created.clinician_id == clinician_user.id
    finally:
        db_session.close()
