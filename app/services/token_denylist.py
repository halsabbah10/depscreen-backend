"""Redis-backed JWT token denylist.

Stores revoked token JTIs with automatic TTL expiry. Gracefully
degrades to a no-op when Redis is unavailable — matches the rate
limiter pattern (additive security, not a hard dependency).
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy singleton — created on first use
_redis_client = None
_redis_checked = False


def _get_redis():
    """Get the shared async Redis client, or None if unavailable."""
    global _redis_client, _redis_checked

    if _redis_checked:
        return _redis_client

    _redis_checked = True

    try:
        from app.core.config import get_settings

        redis_url = get_settings().redis_url
        if not redis_url:
            logger.info("Token denylist: disabled (no REDIS_URL)")
            return None

        import redis.asyncio as aioredis

        _redis_client = aioredis.from_url(redis_url, decode_responses=True)
        logger.info("Token denylist: Redis-backed")
        return _redis_client
    except Exception as e:
        logger.warning("Token denylist: failed to connect to Redis: %s", e)
        return None


async def deny_token(jti: str, expires_at: datetime) -> None:
    """Add a token's JTI to the denylist.

    TTL is set to the remaining lifetime of the token. Once the token
    would have expired naturally, the denylist entry auto-deletes.
    """
    client = _get_redis()
    if client is None:
        return

    ttl = int((expires_at - datetime.utcnow()).total_seconds())
    if ttl <= 0:
        return  # Already expired, nothing to deny

    try:
        await client.setex(f"denylist:{jti}", ttl, "1")
    except Exception as e:
        logger.warning("Failed to deny token %s: %s", jti, e)


async def is_denied(jti: str) -> bool:
    """Check if a token's JTI has been revoked."""
    client = _get_redis()
    if client is None:
        return False

    try:
        return bool(await client.exists(f"denylist:{jti}"))
    except Exception as e:
        logger.warning("Failed to check denylist for %s: %s", jti, e)
        return False
