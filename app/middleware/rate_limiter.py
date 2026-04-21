"""
Rate limiting middleware using slowapi.

Protects against:
- Brute-force login attempts
- LLM API abuse (each screening triggers 4+ LLM calls)
- General abuse / DoS

Uses Redis (Upstash free tier) when REDIS_URL is set,
falls back to in-memory storage otherwise.
"""

import logging

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)


def _build_limiter() -> Limiter:
    """Build limiter with Redis if available, in-memory otherwise."""
    try:
        from app.core.config import get_settings

        redis_url = get_settings().redis_url
    except Exception:
        redis_url = ""

    if redis_url:
        try:
            storage_uri = redis_url
            limiter = Limiter(
                key_func=get_remote_address,
                storage_uri=storage_uri,
            )
            logger.info("Rate limiter: Redis-backed")
            return limiter
        except Exception as e:
            logger.warning("Redis rate-limiter init failed, falling back to in-memory: %s", e)

    logger.info("Rate limiter: in-memory (set REDIS_URL for persistence)")
    return Limiter(key_func=get_remote_address)


limiter = _build_limiter()


def get_limiter() -> Limiter:
    """Get the global limiter instance."""
    return limiter


def rate_limit_exceeded_handler(request, exc):
    """Custom handler for rate limit exceeded — gentle message."""
    return _rate_limit_exceeded_handler(request, exc)
