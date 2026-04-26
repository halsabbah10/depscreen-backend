"""
X/Twitter integration via twikit (phin fork).

Deployment-friendly auth strategy:
1. On startup: try login() for fresh cookies (auto-refresh).
2. If login fails (new account, error 399): fall back to X_COOKIES env var.
3. Every 12h: background task retries login(). The moment X accepts it
   (account ages past anti-spam gate), cookie refresh becomes automatic.

Cookies are stored in the X_COOKIES env var as base64-encoded JSON,
not on disk — works in stateless containers (HuggingFace Spaces, Docker).

Workaround for twikit User parsing: new X accounts are missing optional
fields like 'withheld_in_countries'. SafeDict returns sensible defaults.
"""

import base64
import json
import logging
import math
from datetime import UTC, datetime

from twikit import Client, TooManyRequests
from twikit.user import User as TwikitUser

from app.services.ingestion import MENTAL_HEALTH_KEYWORDS, Tweet

logger = logging.getLogger(__name__)


# ── twikit User parsing workaround ───────────────────────────────────────────


class _SafeDict(dict):
    """Dict that returns sensible defaults for missing keys.

    twikit's User.__init__ accesses optional fields with raw dict[] lookups.
    New X accounts lack fields like 'pinned_tweet_ids_str',
    'withheld_in_countries'. This prevents KeyError crashes.
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


_original_user_init = TwikitUser.__init__


def _patched_user_init(self, client, data, **kwargs):
    if "legacy" in data:
        data["legacy"] = _safe_wrap(data["legacy"])
    _original_user_init(self, client, data, **kwargs)


TwikitUser.__init__ = _patched_user_init


# ── XClient ──────────────────────────────────────────────────────────────────


class XClient:
    """Singleton wrapper around twikit for fetching X/Twitter user tweets."""

    def __init__(self, username: str, email: str, password: str, cookies_b64: str = "") -> None:
        self._username = username
        self._email = email
        self._password = password
        self._cookies_b64 = cookies_b64
        self._client = Client("en-US")
        self._authenticated = False

    async def initialize(self) -> None:
        """Authenticate with X. Try login() first, fall back to env var cookies."""
        # Try login for fresh cookies (auto-refresh path)
        if await self._try_login():
            return

        # Fall back to stored cookies from X_COOKIES env var
        if self._cookies_b64:
            try:
                cookie_json = base64.b64decode(self._cookies_b64).decode()
                cookies = json.loads(cookie_json)
                self._client.set_cookies(cookies)
                self._authenticated = True
                logger.info("X/Twitter: loaded cookies from X_COOKIES env var")
                return
            except Exception as e:
                logger.error(f"X/Twitter: failed to parse X_COOKIES: {e}")

        raise ValueError(
            "X/Twitter: could not authenticate. login() failed and no valid X_COOKIES provided. "
            "Set X_COOKIES env var with base64-encoded JSON: {\"auth_token\": \"...\", \"ct0\": \"...\"}"
        )

    async def _try_login(self) -> bool:
        """Attempt login(). Returns True on success, False on failure (non-fatal)."""
        try:
            await self._client.login(
                auth_info_1=self._username,
                auth_info_2=self._email,
                password=self._password,
            )
            self._authenticated = True
            logger.info("X/Twitter: login() succeeded — cookies auto-refreshed")
            return True
        except Exception as e:
            logger.info(f"X/Twitter: login() failed (will try stored cookies): {e}")
            return False

    async def refresh_cookies(self) -> None:
        """Background task: retry login() to auto-refresh cookies.

        Called by APScheduler every 12 hours. When the account ages past
        X's anti-spam gate, this starts succeeding automatically.
        """
        if await self._try_login():
            logger.info("X/Twitter: background cookie refresh succeeded")
        else:
            logger.debug("X/Twitter: background cookie refresh failed (not critical)")

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
