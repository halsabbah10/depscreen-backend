"""
DAPT corpus validation — run BEFORE training on collected Reddit data.

Checks:
1. Data leakage: DAPT corpus must NOT overlap with ReDSM5 labeled data
2. Deduplication: exact and near-duplicate removal
3. Language quality: filter non-English, bot/spam, too-short posts
4. Corpus statistics: length distribution, vocabulary
5. Provenance documentation

Usage:
    python validate_dapt_corpus.py
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def load_redsm5_texts(redsm5_dir: Path) -> set[str]:
    """Load all labeled ReDSM5 text for leakage detection."""
    posts = pd.read_csv(redsm5_dir / "redsm5_posts.csv")
    annotations = pd.read_csv(redsm5_dir / "redsm5_annotations.csv")

    # Collect all text that appears in the labeled dataset
    texts = set()
    for t in posts["text"].dropna():
        texts.add(t.strip().lower())
    for t in annotations["sentence_text"].dropna():
        texts.add(t.strip().lower())

    return texts


def normalize_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate detection."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text


def detect_bot_spam(text: str) -> bool:
    """Heuristic bot/spam detection."""
    # Repetitive character patterns
    if re.search(r"(.)\1{10,}", text):
        return True
    # All caps posts (shouting/spam)
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) > 20 and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.8:
        return True
    # Promotional patterns
    promo_patterns = [
        r"check out my",
        r"subscribe to",
        r"use code",
        r"discount",
        r"buy now",
        r"click here",
        r"free trial",
        r"www\.\S+\.(com|org|net)",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in promo_patterns)


def detect_non_english(text: str) -> bool:
    """Simple heuristic: if <60% of words are common English words, flag it.
    Uses a small set of the most common English words as a proxy.
    """
    common_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "am", "been", "was", "is", "are",
        "don", "feel", "really", "life", "much", "very", "still", "never",
        "going", "help", "being", "too", "need", "had", "did", "has",
    }
    words = text.lower().split()
    if len(words) < 5:
        return False  # too short to judge
    common_count = sum(1 for w in words if w in common_words)
    ratio = common_count / len(words)
    return ratio < 0.15  # very low overlap with English


def main():
    project_root = Path(__file__).parent.parent.parent.parent
    base_dir = Path(__file__).parent.parent
    corpus_dir = base_dir / "data" / "dapt_corpus"
    redsm5_dir = project_root / "redsm5"
    corpus_file = corpus_dir / "dapt_corpus.txt"

    print("=" * 60)
    print("DAPT CORPUS VALIDATION")
    print("=" * 60)

    # Load corpus
    with open(corpus_file) as f:
        raw = f.read()
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 0]
    print(f"\nRaw chunks loaded: {len(chunks)}")

    # ── CHECK 1: Data Leakage ──────────────────────────────────────────────
    print("\n── CHECK 1: Data Leakage Against ReDSM5 ──")
    redsm5_texts = load_redsm5_texts(redsm5_dir)
    print(f"  ReDSM5 reference texts: {len(redsm5_texts)}")

    leaked = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.strip().lower()
        # Exact match
        if chunk_lower in redsm5_texts:
            leaked.append((i, chunk[:100]))
            continue
        # Substring match — check if any ReDSM5 sentence appears in the DAPT chunk
        for ref in redsm5_texts:
            if len(ref) >= 30 and ref in chunk_lower:
                leaked.append((i, f"contains: '{ref[:80]}...'"))
                break

    if leaked:
        print(f"  LEAKAGE DETECTED: {len(leaked)} chunks overlap with ReDSM5!")
        for idx, preview in leaked[:5]:
            print(f"    chunk {idx}: {preview}")
        if len(leaked) > 5:
            print(f"    ... and {len(leaked) - 5} more")
    else:
        print("  NO LEAKAGE — zero overlap with ReDSM5 labeled data")

    # ── CHECK 2: Exact Deduplication ───────────────────────────────────────
    print("\n── CHECK 2: Deduplication ──")
    seen_hashes = {}
    exact_dupes = 0
    unique_chunks = []
    dupe_indices = set()

    for i, chunk in enumerate(chunks):
        h = hashlib.md5(chunk.encode()).hexdigest()
        if h in seen_hashes:
            exact_dupes += 1
            dupe_indices.add(i)
        else:
            seen_hashes[h] = i
            unique_chunks.append(chunk)

    print(f"  Exact duplicates: {exact_dupes}")
    print(f"  Unique chunks: {len(unique_chunks)}")

    # Near-duplicate detection (normalized text)
    seen_normalized = {}
    near_dupes = 0
    final_chunks = []

    for chunk in unique_chunks:
        norm = normalize_for_dedup(chunk)
        if norm in seen_normalized:
            near_dupes += 1
        else:
            seen_normalized[norm] = True
            final_chunks.append(chunk)

    print(f"  Near-duplicates (after normalization): {near_dupes}")
    print(f"  After dedup: {len(final_chunks)} chunks")

    # ── CHECK 3: Bot/Spam Detection ────────────────────────────────────────
    print("\n── CHECK 3: Bot/Spam Detection ──")
    spam_count = 0
    spam_indices = []
    clean_chunks = []

    for i, chunk in enumerate(final_chunks):
        if detect_bot_spam(chunk):
            spam_count += 1
            spam_indices.append(i)
        else:
            clean_chunks.append(chunk)

    print(f"  Bot/spam detected: {spam_count}")
    if spam_indices[:3]:
        for idx in spam_indices[:3]:
            print(f"    example: '{final_chunks[idx][:100]}...'")
    print(f"  After spam removal: {len(clean_chunks)} chunks")

    # ── CHECK 4: Language Quality ──────────────────────────────────────────
    print("\n── CHECK 4: Language Quality ──")
    non_english = 0
    too_short = 0
    quality_chunks = []

    for chunk in clean_chunks:
        if len(chunk) < 30:
            too_short += 1
            continue
        if detect_non_english(chunk):
            non_english += 1
            continue
        quality_chunks.append(chunk)

    print(f"  Too short (<30 chars): {too_short}")
    print(f"  Non-English (heuristic): {non_english}")
    print(f"  After quality filter: {len(quality_chunks)} chunks")

    # ── CHECK 5: Corpus Statistics ─────────────────────────────────────────
    print("\n── CHECK 5: Corpus Statistics ──")
    lengths = [len(c) for c in quality_chunks]
    word_counts = [len(c.split()) for c in quality_chunks]
    total_chars = sum(lengths)
    total_words = sum(word_counts)

    print(f"  Final chunks: {len(quality_chunks)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total words: {total_words:,}")
    print(f"  Estimated tokens (~0.75 words/token): {int(total_words / 0.75):,}")
    print(f"  Avg chars/chunk: {total_chars / len(quality_chunks):.0f}")
    print(f"  Avg words/chunk: {total_words / len(quality_chunks):.0f}")
    print(f"  Median chars: {sorted(lengths)[len(lengths)//2]}")
    print(f"  Min chars: {min(lengths)}, Max chars: {max(lengths)}")

    # Length distribution buckets
    buckets = Counter()
    for l in lengths:
        if l < 100:
            buckets["<100"] += 1
        elif l < 300:
            buckets["100-300"] += 1
        elif l < 500:
            buckets["300-500"] += 1
        elif l < 1000:
            buckets["500-1000"] += 1
        else:
            buckets["1000+"] += 1

    print("\n  Length distribution:")
    for bucket in ["<100", "100-300", "300-500", "500-1000", "1000+"]:
        count = buckets.get(bucket, 0)
        pct = count / len(quality_chunks) * 100
        bar = "#" * int(pct / 2)
        print(f"    {bucket:>10}: {count:>6} ({pct:>5.1f}%) {bar}")

    # ── Save cleaned corpus ────────────────────────────────────────────────
    print("\n── Saving cleaned corpus ──")
    cleaned_file = corpus_dir / "dapt_corpus_cleaned.txt"
    with open(cleaned_file, "w") as f:
        f.write("\n\n".join(quality_chunks))

    # Save validation report
    report = {
        "raw_chunks": len(chunks),
        "leakage_detected": len(leaked),
        "exact_duplicates_removed": exact_dupes,
        "near_duplicates_removed": near_dupes,
        "spam_removed": spam_count,
        "too_short_removed": too_short,
        "non_english_removed": non_english,
        "final_chunks": len(quality_chunks),
        "total_characters": total_chars,
        "total_words": total_words,
        "estimated_tokens": int(total_words / 0.75),
        "avg_chars_per_chunk": round(total_chars / len(quality_chunks)),
        "avg_words_per_chunk": round(total_words / len(quality_chunks)),
        "validation_passed": len(leaked) == 0,
    }

    with open(corpus_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Cleaned corpus: {cleaned_file}")
    print(f"  Validation report: {corpus_dir / 'validation_report.json'}")

    # ── VERDICT ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if len(leaked) > 0:
        print("VERDICT: FAIL — Data leakage detected. Must remove leaked chunks before DAPT.")
    else:
        removed = len(chunks) - len(quality_chunks)
        print(f"VERDICT: PASS — No leakage. Removed {removed} chunks ({removed/len(chunks)*100:.1f}%) for quality.")
        print(f"Cleaned corpus: {len(quality_chunks)} chunks, {total_words:,} words, ~{int(total_words/0.75):,} tokens")
    print("=" * 60)


if __name__ == "__main__":
    main()
