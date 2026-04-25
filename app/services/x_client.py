"""
X/Twitter integration via twikit.

Wraps twikit's async Client to fetch a user's public tweets for
depression screening. Uses cookie-based auth with the @depscreen
service account.

Cookie caching avoids re-login on every request. If cookies expire,
one re-login attempt is made before failing.
"""

import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from twikit import Client, TooManyRequests

from app.services.ingestion import MENTAL_HEALTH_KEYWORDS, Tweet

logger = logging.getLogger(__name__)

# Resolve cookie path relative to this file → backend/.x_cookies.json
_COOKIES_PATH = str(Path(__file__).resolve().parent.parent.parent / ".x_cookies.json")


class XClient:
    """Singleton wrapper around twikit for fetching X/Twitter user tweets."""

    def __init__(self, username: str, email: str, password: str) -> None:
        self._username = username
        self._email = email
        self._password = password
        self._client = Client("en-US")
        self._authenticated = False

    async def initialize(self) -> None:
        """Authenticate with X. Try cookies first, fall back to login."""
        try:
            self._client.load_cookies(_COOKIES_PATH)
            self._authenticated = True
            logger.info("X/Twitter: loaded cached cookies")
        except Exception:
            logger.info("X/Twitter: no cached cookies, performing login")
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
            raise ValueError("X/Twitter authentication failed. Please check credentials.") from e

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
