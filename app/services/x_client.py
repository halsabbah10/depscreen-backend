"""
X/Twitter integration via twikit.

Wraps twikit's async Client to fetch a user's public tweets for
depression screening. Uses cookie-based auth with the @depscreen
service account.

Cookie caching avoids re-login on every request. If cookies expire,
one re-login attempt is made before failing.

Workarounds for twikit 2.3.3 bugs:
- ClientTransaction.init() fails with "Couldn't get KEY_BYTE indices"
  due to X changing their anti-bot JS. We stub the transaction system.
- User.__init__ crashes on new accounts missing optional fields like
  'pinned_tweet_ids_str', 'withheld_in_countries'. We wrap legacy data
  in SafeDict that returns sensible defaults for missing keys.
"""

import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from twikit import Client, TooManyRequests
from twikit.user import User as TwikitUser

from app.services.ingestion import MENTAL_HEALTH_KEYWORDS, Tweet

logger = logging.getLogger(__name__)

# Resolve cookie path relative to this file → backend/.x_cookies.json
_COOKIES_PATH = str(Path(__file__).resolve().parent.parent.parent / ".x_cookies.json")


# ── twikit 2.3.3 workarounds ────────────────────────────────────────────────


class _SafeDict(dict):
    """Dict that returns sensible defaults for missing keys.

    twikit's User.__init__ accesses many optional fields with raw dict[]
    lookups. New X accounts lack fields like 'pinned_tweet_ids_str',
    'withheld_in_countries', 'profile_banner_url'. This prevents KeyError
    crashes without modifying the twikit source.
    """

    def __missing__(self, key: str):
        if any(key.endswith(s) for s in ("_str", "_ids", "urls", "countries")):
            return []
        if key.endswith("_count") or key.endswith("_int"):
            return 0
        if key.endswith("_url") or key.endswith("_https"):
            return ""
        if key in ("entities", "description", "url"):
            return _SafeDict()
        return None


def _safe_wrap(data):
    """Recursively wrap dicts in _SafeDict."""
    if isinstance(data, dict):
        return _SafeDict({k: _safe_wrap(v) for k, v in data.items()})
    if isinstance(data, list):
        return [_safe_wrap(i) for i in data]
    return data


# Monkey-patch twikit User to handle missing legacy fields
_original_user_init = TwikitUser.__init__


def _patched_user_init(self, client, data, **kwargs):
    if "legacy" in data:
        data["legacy"] = _safe_wrap(data["legacy"])
    _original_user_init(self, client, data, **kwargs)


TwikitUser.__init__ = _patched_user_init


# ── XClient ──────────────────────────────────────────────────────────────────


class XClient:
    """Singleton wrapper around twikit for fetching X/Twitter user tweets."""

    def __init__(self, username: str, email: str, password: str) -> None:
        self._username = username
        self._email = email
        self._password = password
        self._client = Client("en-US")
        self._authenticated = False

    async def initialize(self) -> None:
        """Authenticate with X. Load cookies and stub the broken transaction system."""
        # Stub twikit's broken ClientTransaction (KEY_BYTE indices bug)
        self._client.client_transaction.init = AsyncMock()
        self._client.client_transaction.generate_transaction_id = MagicMock(return_value="depscreen")

        try:
            self._client.load_cookies(_COOKIES_PATH)
            self._authenticated = True
            logger.info("X/Twitter: loaded cached cookies")
        except Exception:
            logger.info("X/Twitter: no cached cookies, attempting login")
            await self._login()

    async def _login(self) -> None:
        """Perform a fresh login and save cookies."""
        try:
            await self._client.login(
                auth_info_1=self._username,
                auth_info_2=self._email,
                password=self._password,
            )
            self._client.save_cookies(_COOKIES_PATH)
            self._authenticated = True
            logger.info("X/Twitter: login successful, cookies saved")
        except Exception as e:
            logger.error(f"X/Twitter login failed: {e}")
            self._authenticated = False
            raise ValueError(
                "X/Twitter authentication failed. "
                "If login() is broken (KEY_BYTE bug), provide cookies via .x_cookies.json instead."
            ) from e

    async def fetch_user_tweets(
        self,
        username: str,
        limit: int = 50,
        mental_health_filter: bool = True,
    ) -> list[Tweet]:
        """Fetch a user's public tweets, optionally filtered for mental health content.

        Args:
            username: X handle without the @ prefix.
            limit: Maximum tweets to return.
            mental_health_filter: If True, only return tweets containing
                mental health keywords.

        Returns:
            List of Tweet dataclass instances with real metadata.

        Raises:
            ValueError: On user not found, rate limit, or auth failure.
        """
        if not self._authenticated:
            raise ValueError("X/Twitter client is not authenticated.")

        try:
            user = await self._client.get_user_by_screen_name(username)
        except TooManyRequests as e:
            minutes = self._rate_limit_minutes(e)
            raise ValueError(f"X rate limit reached — please try again in {minutes} minutes.") from e
        except Exception as e:
            # Try one re-login in case cookies expired
            if "auth" in str(e).lower() or "login" in str(e).lower():
                try:
                    await self._login()
                    user = await self._client.get_user_by_screen_name(username)
                except Exception:
                    raise ValueError(f"X/Twitter user '@{username}' not found or profile is private.") from e
            else:
                raise ValueError(f"X/Twitter user '@{username}' not found or profile is private.") from e

        try:
            tweets_result = await user.get_tweets("Tweets", count=limit)
        except TooManyRequests as e:
            minutes = self._rate_limit_minutes(e)
            raise ValueError(f"X rate limit reached — please try again in {minutes} minutes.") from e
        except Exception as e:
            raise ValueError(f"Could not fetch tweets for @{username}. Please try again.") from e

        tweets: list[Tweet] = []
        for tweet in tweets_result:
            text = tweet.text or ""
            if len(text) < 10:
                continue

            if mental_health_filter:
                text_lower = text.lower()
                if not any(kw in text_lower for kw in MENTAL_HEALTH_KEYWORDS):
                    continue

            created_at = ""
            if tweet.created_at_datetime:
                created_at = tweet.created_at_datetime.isoformat()

            tweets.append(
                Tweet(
                    tweet_id=str(tweet.id),
                    text=text,
                    created_at=created_at,
                    like_count=tweet.favorite_count or 0,
                    retweet_count=tweet.retweet_count or 0,
                )
            )

            if len(tweets) >= limit:
                break

        logger.info(f"Fetched {len(tweets)} tweets for @{username}")
        return tweets

    @staticmethod
    def _rate_limit_minutes(exc: TooManyRequests) -> int:
        """Extract minutes until rate limit reset from a TooManyRequests exception."""
        try:
            reset_time = exc.rate_limit_reset
            if reset_time:
                now = datetime.now(UTC).timestamp()
                return max(1, math.ceil((reset_time - now) / 60))
        except Exception:
            logger.debug("Could not parse rate limit reset time", exc_info=True)
        return 15  # Default to 15 minutes if we can't parse
