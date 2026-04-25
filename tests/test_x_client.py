"""Unit tests for the twikit XClient wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from app.services.x_client import XClient


@dataclass
class FakeTwikitTweet:
    """Minimal mock of a twikit Tweet object."""

    id: str
    text: str
    created_at_datetime: datetime | None
    favorite_count: int
    retweet_count: int


def _make_fake_tweets(texts: list[str]) -> list[FakeTwikitTweet]:
    return [
        FakeTwikitTweet(
            id=f"tweet_{i}",
            text=text,
            created_at_datetime=datetime(2026, 4, 1, 12, 0, tzinfo=UTC),
            favorite_count=i * 5,
            retweet_count=i * 2,
        )
        for i, text in enumerate(texts)
    ]


@pytest.fixture
def x_client():
    return XClient(username="test", email="test@test.com", password="pass")


@pytest.mark.asyncio
async def test_fetch_tweets_maps_fields(x_client):
    """twikit tweet fields are mapped correctly to our Tweet dataclass."""
    fake_user = AsyncMock()
    fake_user.get_tweets = AsyncMock(return_value=_make_fake_tweets(["I feel so depressed and anxious today"]))

    with (
        patch.object(x_client, "_authenticated", True),
        patch.object(x_client._client, "get_user_by_screen_name", new_callable=AsyncMock, return_value=fake_user),
    ):
        tweets = await x_client.fetch_user_tweets("someone", mental_health_filter=False)

    assert len(tweets) == 1
    t = tweets[0]
    assert t.tweet_id == "tweet_0"
    assert t.text == "I feel so depressed and anxious today"
    assert t.created_at == "2026-04-01T12:00:00+00:00"
    assert t.like_count == 0
    assert t.retweet_count == 0


@pytest.mark.asyncio
async def test_mental_health_filter_includes_matching(x_client):
    """Only tweets containing mental health keywords pass the filter."""
    fake_user = AsyncMock()
    fake_user.get_tweets = AsyncMock(
        return_value=_make_fake_tweets(
            [
                "I feel so depressed and hopeless",
                "Just had a great day at the park with friends",
                "My anxiety is through the roof today",
            ]
        )
    )

    with (
        patch.object(x_client, "_authenticated", True),
        patch.object(x_client._client, "get_user_by_screen_name", new_callable=AsyncMock, return_value=fake_user),
    ):
        tweets = await x_client.fetch_user_tweets("someone", mental_health_filter=True)

    assert len(tweets) == 2
    assert "depressed" in tweets[0].text.lower()
    assert "anxiety" in tweets[1].text.lower()


@pytest.mark.asyncio
async def test_mental_health_filter_off_returns_all(x_client):
    """With filter off, all tweets (>=10 chars) are returned."""
    fake_user = AsyncMock()
    fake_user.get_tweets = AsyncMock(
        return_value=_make_fake_tweets(
            [
                "I feel so depressed",
                "Great day at the park with friends",
                "Short",
            ]
        )
    )

    with (
        patch.object(x_client, "_authenticated", True),
        patch.object(x_client._client, "get_user_by_screen_name", new_callable=AsyncMock, return_value=fake_user),
    ):
        tweets = await x_client.fetch_user_tweets("someone", mental_health_filter=False)

    assert len(tweets) == 2


@pytest.mark.asyncio
async def test_user_not_found_raises_valueerror(x_client):
    """Non-existent user raises ValueError."""
    with (
        patch.object(x_client, "_authenticated", True),
        patch.object(
            x_client._client,
            "get_user_by_screen_name",
            new_callable=AsyncMock,
            side_effect=Exception("User not found"),
        ),
        patch.object(x_client, "_login", new_callable=AsyncMock, side_effect=Exception("login failed")),
        pytest.raises(ValueError, match="not found"),
    ):
        await x_client.fetch_user_tweets("nonexistent_user_xyz")


@pytest.mark.asyncio
async def test_rate_limit_raises_valueerror(x_client):
    """TooManyRequests from twikit is mapped to a ValueError with retry time."""
    from twikit import TooManyRequests

    exc = TooManyRequests("rate limited")
    exc.rate_limit_reset = None

    with (
        patch.object(x_client, "_authenticated", True),
        patch.object(x_client._client, "get_user_by_screen_name", new_callable=AsyncMock, side_effect=exc),
        pytest.raises(ValueError, match="rate limit"),
    ):
        await x_client.fetch_user_tweets("someone")


@pytest.mark.asyncio
async def test_not_authenticated_raises(x_client):
    """Calling fetch before authenticate raises ValueError."""
    with pytest.raises(ValueError, match="not authenticated"):
        await x_client.fetch_user_tweets("someone")


@pytest.mark.asyncio
async def test_limit_caps_results(x_client):
    """Results are capped at the specified limit."""
    fake_user = AsyncMock()
    fake_user.get_tweets = AsyncMock(
        return_value=_make_fake_tweets([f"I feel depressed today number {i}" for i in range(20)])
    )

    with (
        patch.object(x_client, "_authenticated", True),
        patch.object(x_client._client, "get_user_by_screen_name", new_callable=AsyncMock, return_value=fake_user),
    ):
        tweets = await x_client.fetch_user_tweets("someone", limit=5, mental_health_filter=False)

    assert len(tweets) == 5
