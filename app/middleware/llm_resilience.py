"""
LLM resilience utilities.

Provides retry logic with exponential backoff for LLM API calls.
Uses tenacity for robust retry handling.

For a mental health platform: LLM failures must NEVER crash the system.
Fallbacks exist for every LLM call. Retries are silent — the user
should never see "retrying..." messages.
"""

import logging
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retry configuration
LLM_MAX_RETRIES = 3
LLM_MIN_WAIT_SECONDS = 1
LLM_MAX_WAIT_SECONDS = 8

# Exceptions that should trigger a retry (transient errors)
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    ConnectionError,
    TimeoutError,
)


def with_llm_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that adds retry logic to async LLM calls.

    Retries up to 3 times with exponential backoff (1s, 2s, 4s)
    on transient errors (connection, timeout, rate limit, 500).

    Non-retryable errors (400, 401, 403, 422) fail immediately.

    Usage:
        @with_llm_retry
        async def my_llm_call():
            response = await client.chat.completions.create(...)
            return response
    """

    @retry(
        stop=stop_after_attempt(LLM_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1,
            min=LLM_MIN_WAIT_SECONDS,
            max=LLM_MAX_WAIT_SECONDS,
        ),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


# Pre-built retry decorator for inline use
llm_retry = retry(
    stop=stop_after_attempt(LLM_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=LLM_MIN_WAIT_SECONDS,
        max=LLM_MAX_WAIT_SECONDS,
    ),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
