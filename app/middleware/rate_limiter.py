"""
Rate limiting middleware using slowapi.

Protects against:
- Brute-force login attempts
- LLM API abuse (each screening triggers 4+ LLM calls)
- General abuse / DoS

Limits are configured in app.core.config.Settings.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Create limiter instance — uses IP address for rate tracking
limiter = Limiter(key_func=get_remote_address)


def get_limiter() -> Limiter:
    """Get the global limiter instance."""
    return limiter


def rate_limit_exceeded_handler(request, exc):
    """Custom handler for rate limit exceeded — gentle message."""
    return _rate_limit_exceeded_handler(request, exc)
