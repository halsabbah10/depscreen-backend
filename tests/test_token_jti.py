"""Tests for JTI claim in JWT tokens."""

from __future__ import annotations

from app.core.config import get_settings
from app.services.auth import create_access_token, create_refresh_token, decode_token


def test_access_token_contains_jti():
    settings = get_settings()
    token = create_access_token("user-123", "patient", settings)
    payload = decode_token(token, settings)
    assert "jti" in payload
    assert len(payload["jti"]) == 36  # UUID format


def test_refresh_token_contains_jti():
    settings = get_settings()
    token = create_refresh_token("user-123", settings)
    payload = decode_token(token, settings)
    assert "jti" in payload
    assert len(payload["jti"]) == 36


def test_each_token_gets_unique_jti():
    settings = get_settings()
    t1 = create_access_token("user-123", "patient", settings)
    t2 = create_access_token("user-123", "patient", settings)
    p1 = decode_token(t1, settings)
    p2 = decode_token(t2, settings)
    assert p1["jti"] != p2["jti"]
