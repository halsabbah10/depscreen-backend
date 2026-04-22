"""Tests for the token denylist service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from app.services.token_denylist import deny_token, is_denied


@pytest.mark.asyncio
async def test_deny_and_check_with_mock_redis():
    """With Redis available, denied tokens should be detected."""
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=1)

    with patch("app.services.token_denylist._get_redis", return_value=mock_redis):
        expires = datetime.now(UTC) + timedelta(hours=1)
        await deny_token("test-jti-123", expires)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "denylist:test-jti-123"

        result = await is_denied("test-jti-123")
        assert result is True


@pytest.mark.asyncio
async def test_is_denied_returns_false_for_unknown_jti():
    """A non-denied JTI should return False."""
    mock_redis = AsyncMock()
    mock_redis.exists = AsyncMock(return_value=0)

    with patch("app.services.token_denylist._get_redis", return_value=mock_redis):
        result = await is_denied("unknown-jti")
        assert result is False


@pytest.mark.asyncio
async def test_deny_skips_expired_token():
    """If the token is already expired, don't bother adding to denylist."""
    mock_redis = AsyncMock()
    mock_redis.setex = AsyncMock()

    with patch("app.services.token_denylist._get_redis", return_value=mock_redis):
        expires = datetime.now(UTC) - timedelta(hours=1)
        await deny_token("expired-jti", expires)

        mock_redis.setex.assert_not_called()


@pytest.mark.asyncio
async def test_graceful_degradation_no_redis():
    """Without Redis, deny_token is a no-op and is_denied returns False."""
    with patch("app.services.token_denylist._get_redis", return_value=None):
        await deny_token("jti-abc", datetime.now(UTC) + timedelta(hours=1))

        result = await is_denied("jti-abc")
        assert result is False
