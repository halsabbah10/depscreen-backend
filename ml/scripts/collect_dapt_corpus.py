"""
Collect unlabeled mental health Reddit posts for Domain-Adaptive Pre-Training (DAPT).

Uses Reddit's public JSON API to fetch posts from mental health subreddits.
No authentication needed — public endpoints only.

Target: 50,000–100,000 posts across multiple subreddits.
Output: One text file per subreddit + combined corpus file.

Usage:
    python collect_dapt_corpus.py [--output-dir PATH] [--target-per-sub N]
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Subreddits with high volume of depression/mental-health-related self-posts
DAPT_SUBREDDITS = [
    "depression",         # Core — highest relevance
    "mentalhealth",       # Broad mental health discussion
    "anxiety",            # Anxiety often co-occurs with depression
    "SuicideWatch",       # Crisis/suicidal ideation content
    "lonely",             # Isolation, social withdrawal
    "offmychest",         # Personal venting, emotional expression
    "TrueOffMyChest",     # Same
    "selfharm",           # Self-harm discussion
    "bipolar",            # Mood disorders
    "ptsd",               # Trauma-related
    "socialanxiety",      # Social withdrawal
    "grief",              # Loss and bereavement
    "insomnia",           # Sleep issues (DSM-5 criterion)
    "chronicpain",        # Fatigue, psychomotor (DSM-5 criteria)
    "decidingtobebetter", # Recovery language
]

USER_AGENT = "DepScreen-DAPT/1.0 (Academic Research; Domain Pre-training Corpus Collection)"


def clean_text(text: str) -> str:
    """Light cleaning — preserve natural language for MLM pre-training."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove Reddit formatting artifacts
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#x200B;", "", text)  # zero-width space
    # Collapse excessive whitespace/newlines but keep paragraph structure
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch_subreddit_posts(
    subreddit: str,
    target_count: int,
    min_text_length: int = 50,
) -> list[str]:
    """Fetch self-posts from a subreddit using Reddit's public JSON API.

    Paginates using the 'after' parameter. Reddit limits to ~1000 posts
    per listing endpoint, so we fetch from multiple sort orders.
    """
    collected_texts = []
    seen_ids = set()

    # Fetch from multiple sort orders to maximize unique posts
    sort_orders = ["hot", "new", "top", "rising"]
    time_filters = ["all", "year", "month"]  # for 'top' sort

    headers = {"User-Agent": USER_AGENT}

    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        for sort in sort_orders:
            if len(collected_texts) >= target_count:
                break

            params_variants = [{}]
            if sort == "top":
                params_variants = [{"t": t} for t in time_filters]

            for extra_params in params_variants:
                if len(collected_texts) >= target_count:
                    break

                after = None
                consecutive_empty = 0

                while len(collected_texts) < target_count and consecutive_empty < 3:
                    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
                    params = {"limit": 100, "raw_json": 1, **extra_params}
                    if after:
                        params["after"] = after

                    try:
                        resp = client.get(url, params=params)

                        if resp.status_code == 429:
                            wait = int(resp.headers.get("Retry-After", 60))
                            logger.warning(f"  Rate limited on r/{subreddit}, waiting {wait}s...")
                            time.sleep(wait)
                            continue

                        if resp.status_code != 200:
                            logger.warning(f"  r/{subreddit}/{sort}: HTTP {resp.status_code}")
                            break

                        data = resp.json()
                        posts = data.get("data", {}).get("children", [])

                        if not posts:
                            break

                        new_this_page = 0
                        for post in posts:
                            pd = post.get("data", {})
                            post_id = pd.get("id", "")

                            # Skip if already seen, not a self-post, or removed
                            if post_id in seen_ids:
                                continue
                            if not pd.get("is_self", False):
                                continue
                            if pd.get("removed_by_category"):
                                continue

                            text = pd.get("selftext", "")
                            if text in ("[removed]", "[deleted]", ""):
                                continue

                            cleaned = clean_text(text)
                            if len(cleaned) < min_text_length:
                                continue

                            seen_ids.add(post_id)
                            collected_texts.append(cleaned)
                            new_this_page += 1

                            if len(collected_texts) >= target_count:
                                break

                        if new_this_page == 0:
                            consecutive_empty += 1
                        else:
                            consecutive_empty = 0

                        after = data.get("data", {}).get("after")
                        if not after:
                            break

                        # Rate limit: Reddit asks for 1 req/sec for unauthenticated
                        time.sleep(1.5)

                    except (httpx.TimeoutException, httpx.ConnectError) as e:
                        logger.warning(f"  r/{subreddit}/{sort}: {type(e).__name__}, retrying...")
                        time.sleep(5)
                        continue

    return collected_texts


def main():
    parser = argparse.ArgumentParser(description="Collect DAPT corpus from Reddit")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--target-per-sub", type=int, default=5000,
        help="Target posts per subreddit (default: 5000)"
    )
    parser.add_argument(
        "--min-text-length", type=int, default=50,
        help="Minimum text length in characters (default: 50)"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "dapt_corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAPT Corpus Collection")
    print(f"Target: {args.target_per_sub} posts × {len(DAPT_SUBREDDITS)} subreddits")
    print(f"Output: {output_dir}")
    print("=" * 60)

    all_texts = []
    stats = {}

    for i, subreddit in enumerate(DAPT_SUBREDDITS):
        logger.info(f"\n[{i+1}/{len(DAPT_SUBREDDITS)}] Collecting r/{subreddit}...")

        texts = fetch_subreddit_posts(
            subreddit=subreddit,
            target_count=args.target_per_sub,
            min_text_length=args.min_text_length,
        )

        # Save per-subreddit file
        sub_file = output_dir / f"{subreddit.lower()}.txt"
        with open(sub_file, "w") as f:
            f.write("\n\n".join(texts))

        stats[subreddit] = len(texts)
        all_texts.extend(texts)
        logger.info(f"  r/{subreddit}: {len(texts)} posts collected")

    # Save combined corpus
    corpus_file = output_dir / "dapt_corpus.txt"
    with open(corpus_file, "w") as f:
        f.write("\n\n".join(all_texts))

    # Save stats
    stats_file = output_dir / "collection_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_posts": len(all_texts),
            "total_chars": sum(len(t) for t in all_texts),
            "avg_chars_per_post": sum(len(t) for t in all_texts) / max(len(all_texts), 1),
            "per_subreddit": stats,
            "min_text_length": args.min_text_length,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total posts: {len(all_texts)}")
    print(f"Total chars: {sum(len(t) for t in all_texts):,}")
    print("\nPer subreddit:")
    for sub, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  r/{sub}: {count}")
    print(f"\nSaved to: {corpus_file}")
    print(f"Stats:    {stats_file}")


if __name__ == "__main__":
    main()
