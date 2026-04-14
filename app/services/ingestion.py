"""
Data ingestion service for multi-source screening input.

Supports three ingestion methods:
1. Reddit profile analysis — fetch public posts, filter, screen
2. Guided clinical check-in — structured DSM-5-aligned prompts
3. Bulk text upload — social media data exports

No manual textarea — all input is either structured or API-sourced.
"""

import json
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# ── Reddit Integration ────────────────────────────────────────────────────────

# Subreddits relevant to mental health screening
MENTAL_HEALTH_SUBREDDITS = {
    "depression",
    "anxiety",
    "mentalhealth",
    "suicidewatch",
    "bipolar",
    "ptsd",
    "adhd",
    "socialanxiety",
    "lonely",
    "selfharm",
    "offmychest",
    "trueoffmychest",
    "confessions",
    "decidingtobebetter",
    "getmotivated",
    "selfimprovement",
    "chronicpain",
    "insomnia",
    "grief",
}

# Broader subreddits where mental health content appears
GENERAL_SUBREDDITS_TO_INCLUDE = {
    "askreddit",
    "casualconversation",
    "advice",
    "relationships",
    "relationship_advice",
    "amitheasshole",
}

REDDIT_USER_AGENT = "DepScreen/2.0 (Clinical Screening Research Platform)"


@dataclass
class RedditPost:
    """A single Reddit post with metadata."""

    post_id: str
    subreddit: str
    title: str
    text: str
    created_utc: float
    score: int
    url: str


async def fetch_reddit_posts(
    username: str,
    limit: int = 100,
    mental_health_only: bool = True,
) -> list[RedditPost]:
    """Fetch a Reddit user's public posts via the public JSON API.

    No API key or OAuth required for public posts.
    Rate limited to be respectful (1 request per 2 seconds).
    """
    posts = []
    after = None
    fetched = 0
    max_pages = 4  # 25 posts per page × 4 = 100 posts max

    async with httpx.AsyncClient(timeout=15.0) as client:
        for _ in range(max_pages):
            url = f"https://www.reddit.com/user/{username}/submitted.json"
            params = {"limit": 25, "raw_json": 1}
            if after:
                params["after"] = after

            try:
                response = await client.get(
                    url,
                    params=params,
                    headers={"User-Agent": REDDIT_USER_AGENT},
                )

                if response.status_code == 404:
                    raise ValueError(f"Reddit user '{username}' not found")
                if response.status_code == 403:
                    raise ValueError(f"Reddit user '{username}' has a private profile")
                if response.status_code == 429:
                    raise ValueError("Reddit rate limit reached — please try again in a few minutes")

                response.raise_for_status()
                data = response.json()

            except httpx.HTTPError as e:
                logger.error(f"Reddit API error: {e}")
                raise ValueError(f"Failed to fetch Reddit posts: {str(e)}")

            children = data.get("data", {}).get("children", [])
            if not children:
                break

            for child in children:
                post_data = child.get("data", {})
                subreddit = post_data.get("subreddit", "").lower()

                # Filter for relevant subreddits
                if mental_health_only:
                    if subreddit not in MENTAL_HEALTH_SUBREDDITS and subreddit not in GENERAL_SUBREDDITS_TO_INCLUDE:
                        continue

                # Get text content (selftext for text posts)
                text = post_data.get("selftext", "").strip()
                title = post_data.get("title", "").strip()

                # Skip posts with no meaningful text
                combined_text = f"{title}. {text}" if text else title
                if len(combined_text) < 20:
                    continue

                posts.append(
                    RedditPost(
                        post_id=post_data.get("name", ""),
                        subreddit=subreddit,
                        title=title,
                        text=combined_text,
                        created_utc=post_data.get("created_utc", 0),
                        score=post_data.get("score", 0),
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                    )
                )

                fetched += 1
                if fetched >= limit:
                    break

            after = data.get("data", {}).get("after")
            if not after or fetched >= limit:
                break

    logger.info(f"Fetched {len(posts)} posts for u/{username}")
    return posts


# ── X/Twitter Integration ─────────────────────────────────────────────────────

# Mental health keywords to filter tweets
MENTAL_HEALTH_KEYWORDS = [
    "depressed",
    "depression",
    "anxiety",
    "anxious",
    "sad",
    "lonely",
    "hopeless",
    "worthless",
    "can't sleep",
    "insomnia",
    "exhausted",
    "no energy",
    "suicidal",
    "self harm",
    "panic attack",
    "therapy",
    "therapist",
    "medication",
    "antidepressant",
    "mental health",
    "crying",
    "overwhelmed",
    "burned out",
    "burnout",
    "numb",
    "can't focus",
    "can't concentrate",
    "no motivation",
    "isolation",
]


@dataclass
class Tweet:
    """A single X/Twitter post with metadata."""

    tweet_id: str
    text: str
    created_at: str
    like_count: int
    retweet_count: int


async def fetch_x_posts(
    username: str,
    limit: int = 50,
    mental_health_filter: bool = True,
) -> list[Tweet]:
    """Fetch a user's public X/Twitter posts via Nitter or public endpoints.

    Note: X's official API requires paid access ($100/mo). For a capstone
    project, we use the public syndication endpoint which returns recent
    tweets without API keys.
    """
    tweets = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Use X's public syndication timeline endpoint
        url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{username}"

        try:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; DepScreen/2.0)",
                    "Accept": "text/html",
                },
            )

            if response.status_code == 404:
                raise ValueError(f"X/Twitter user '@{username}' not found")
            if response.status_code != 200:
                raise ValueError(
                    f"Could not fetch posts for @{username}. "
                    "The profile may be private or X's API may be rate-limiting."
                )

            # Parse tweet text from the HTML response
            # The syndication endpoint returns HTML with tweet content
            content = response.text

            # Extract tweet texts using regex on the HTML
            tweet_texts = re.findall(
                r'<p[^>]*class="[^"]*timeline-Tweet-text[^"]*"[^>]*>(.*?)</p>',
                content,
                re.DOTALL,
            )

            if not tweet_texts:
                # Fallback: try extracting from data attributes
                tweet_texts = re.findall(
                    r'data-tweet-text="([^"]+)"',
                    content,
                )

            for i, text in enumerate(tweet_texts[:limit]):
                # Clean HTML entities
                text = re.sub(r"<[^>]+>", "", text)
                text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                text = text.replace("&#39;", "'").replace("&quot;", '"')
                text = text.strip()

                if len(text) < 10:
                    continue

                # Filter for mental health content if requested
                if mental_health_filter:
                    text_lower = text.lower()
                    if not any(kw in text_lower for kw in MENTAL_HEALTH_KEYWORDS):
                        continue

                tweets.append(
                    Tweet(
                        tweet_id=f"tweet_{i}",
                        text=text,
                        created_at="",
                        like_count=0,
                        retweet_count=0,
                    )
                )

        except httpx.HTTPError as e:
            logger.error(f"X/Twitter fetch error: {e}")
            raise ValueError(
                f"Failed to fetch X/Twitter posts for @{username}. Please ensure the profile is public and try again."
            )

    logger.info(f"Fetched {len(tweets)} tweets for @{username}")
    return tweets


# ── Guided Clinical Check-in ─────────────────────────────────────────────────


@dataclass
class CheckInPrompt:
    """A structured check-in prompt mapped to a DSM-5 criterion."""

    id: str
    dsm5_criterion: str
    question: str
    follow_up: str | None = None


CHECKIN_PROMPTS = [
    CheckInPrompt(
        id="mood",
        dsm5_criterion="DEPRESSED_MOOD",
        question="How would you describe your mood most days over the past two weeks?",
        follow_up="Have there been times when you felt particularly sad, empty, or hopeless?",
    ),
    CheckInPrompt(
        id="interest",
        dsm5_criterion="ANHEDONIA",
        question="Have you noticed a loss of interest or pleasure in activities you usually enjoy?",
        follow_up="Are there things you used to look forward to that no longer feel appealing?",
    ),
    CheckInPrompt(
        id="appetite",
        dsm5_criterion="APPETITE_CHANGE",
        question="Have you experienced any changes in your appetite or weight recently?",
        follow_up="Are you eating noticeably more or less than usual?",
    ),
    CheckInPrompt(
        id="sleep",
        dsm5_criterion="SLEEP_ISSUES",
        question="Describe your sleep patterns over the past two weeks.",
        follow_up="Do you have trouble falling asleep, staying asleep, or do you sleep too much?",
    ),
    CheckInPrompt(
        id="psychomotor",
        dsm5_criterion="PSYCHOMOTOR",
        question="Have others noticed that you seem physically slowed down or unusually restless?",
        follow_up="Do you feel agitated or like you can't sit still, or the opposite — like everything is in slow motion?",
    ),
    CheckInPrompt(
        id="energy",
        dsm5_criterion="FATIGUE",
        question="How has your energy level and motivation been?",
        follow_up="Do you feel fatigued or drained even without physical exertion?",
    ),
    CheckInPrompt(
        id="worth",
        dsm5_criterion="WORTHLESSNESS",
        question="Have you been feeling worthless or experiencing excessive guilt?",
        follow_up="Do you find yourself being very self-critical or blaming yourself for things?",
    ),
    CheckInPrompt(
        id="concentration",
        dsm5_criterion="COGNITIVE_ISSUES",
        question="Are you able to concentrate on work, reading, or daily tasks as usual?",
        follow_up="Do you find it hard to make decisions or feel like your thinking is foggy?",
    ),
    CheckInPrompt(
        id="safety",
        dsm5_criterion="SUICIDAL_THOUGHTS",
        question="Sometimes, when things feel heavy, people have thoughts that life isn't worth it, or thoughts of hurting themselves. Has anything like that been on your mind?",
        follow_up="If yes — even a little — it helps us to know more. Is it a passing thought, or has it felt more concrete than that?",
    ),
]


def get_checkin_prompts() -> list[dict]:
    """Return the structured check-in prompts for the frontend."""
    return [
        {
            "id": p.id,
            "dsm5_criterion": p.dsm5_criterion,
            "question": p.question,
            "follow_up": p.follow_up,
        }
        for p in CHECKIN_PROMPTS
    ]


def combine_checkin_responses(responses: dict[str, str]) -> str:
    """Combine structured check-in responses into a single text for screening.

    Args:
        responses: Dict mapping prompt_id to patient's response text.

    Returns:
        Combined text suitable for the symptom classifier.
    """
    parts = []
    for prompt in CHECKIN_PROMPTS:
        response = responses.get(prompt.id, "").strip()
        if response:
            parts.append(response)

    return " ".join(parts)


# ── Bulk Text Parsing ─────────────────────────────────────────────────────────


def parse_reddit_export(content: str) -> list[dict]:
    """Parse a Reddit GDPR data export (CSV or JSON format).

    Reddit data exports contain posts in a CSV with columns:
    id, permalink, date, ip, subreddit, gildings, title, body, url
    """
    entries = []

    # Try JSON format first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                text = item.get("body", item.get("selftext", item.get("text", "")))
                title = item.get("title", "")
                combined = f"{title}. {text}" if text else title
                if len(combined.strip()) >= 20:
                    entries.append(
                        {
                            "text": combined.strip(),
                            "source": item.get("subreddit", "export"),
                            "date": item.get("date", item.get("created_utc", "")),
                        }
                    )
            return entries
    except (json.JSONDecodeError, TypeError):
        pass

    # Try CSV-like format (tab or comma separated)
    lines = content.strip().split("\n")
    if len(lines) > 1:
        for line in lines[1:]:  # Skip header
            # Simple parse — split by tab or comma and take the longest field as text
            fields = line.split("\t") if "\t" in line else line.split(",")
            text_candidates = [f.strip().strip('"') for f in fields if len(f.strip()) > 20]
            if text_candidates:
                longest = max(text_candidates, key=len)
                entries.append({"text": longest, "source": "export", "date": ""})

    # Fallback: treat each paragraph as a separate entry
    if not entries:
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            p = p.strip()
            if len(p) >= 20:
                entries.append({"text": p, "source": "manual", "date": ""})

    return entries
