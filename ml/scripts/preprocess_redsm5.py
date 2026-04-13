"""
ReDSM5 preprocessing pipeline for sentence-level DSM-5 symptom classification.

Loads the ReDSM5 dataset (1,484 posts, 2,058 annotations), creates an 11-class
sentence-level classification dataset, and splits by post_id to prevent data leakage.

Classes:
    9 DSM-5 symptoms + SPECIAL_CASE + NO_SYMPTOM

Usage:
    python preprocess_redsm5.py [--redsm5-dir PATH] [--output-dir PATH]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ── Label configuration ──────────────────────────────────────────────────────

SYMPTOM_LABELS = {
    "DEPRESSED_MOOD": 0,
    "ANHEDONIA": 1,
    "APPETITE_CHANGE": 2,
    "SLEEP_ISSUES": 3,
    "PSYCHOMOTOR": 4,
    "FATIGUE": 5,
    "WORTHLESSNESS": 6,
    "COGNITIVE_ISSUES": 7,
    "SUICIDAL_THOUGHTS": 8,
    "SPECIAL_CASE": 9,
    "NO_SYMPTOM": 10,
}

SYMPTOM_READABLE = {
    "DEPRESSED_MOOD": "Depressed Mood",
    "ANHEDONIA": "Loss of Interest / Pleasure",
    "APPETITE_CHANGE": "Appetite / Weight Change",
    "SLEEP_ISSUES": "Sleep Disturbance",
    "PSYCHOMOTOR": "Psychomotor Changes",
    "FATIGUE": "Fatigue / Loss of Energy",
    "WORTHLESSNESS": "Worthlessness / Guilt",
    "COGNITIVE_ISSUES": "Difficulty Concentrating",
    "SUICIDAL_THOUGHTS": "Suicidal Ideation",
    "SPECIAL_CASE": "Other Clinical Indicator",
    "NO_SYMPTOM": "No Symptom Detected",
}


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_sentence(text: str) -> str:
    """Clean a single sentence for model input."""
    if not isinstance(text, str):
        return ""
    # Replace URLs
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)
    # Replace Reddit usernames and subreddits
    text = re.sub(r"u/\w+", "[USER]", text)
    text = re.sub(r"r/\w+", "[SUBREDDIT]", text)
    # Normalize unicode quotes and dashes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " -- ").replace("\u2013", " - ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter for Reddit-style text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    # Split on sentence-ending punctuation followed by space + uppercase or end
    # Handles abbreviations like Dr., Mr., etc. imperfectly but good enough
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)
    sentences = []
    for part in parts:
        part = part.strip()
        if len(part) >= 5:  # Skip very short fragments
            sentences.append(part)
    # If no splits happened and text is long, return as single sentence
    if not sentences and len(text.strip()) >= 5:
        sentences = [text.strip()]
    return sentences


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(redsm5_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load posts and annotations CSVs."""
    posts_path = redsm5_dir / "redsm5_posts.csv"
    annot_path = redsm5_dir / "redsm5_annotations.csv"

    if not posts_path.exists():
        raise FileNotFoundError(f"Posts file not found: {posts_path}")
    if not annot_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annot_path}")

    posts = pd.read_csv(posts_path)
    annotations = pd.read_csv(annot_path)

    print(f"Loaded {len(posts)} posts, {len(annotations)} annotations")
    return posts, annotations


# ── Positive samples ──────────────────────────────────────────────────────────

def create_positive_samples(annotations: pd.DataFrame) -> pd.DataFrame:
    """Create training samples from status=1 (symptom present) annotations.

    Deduplicates: if the same (sentence_id, DSM5_symptom) pair appears multiple
    times (multiple annotator explanations), keep only the first.
    """
    positives = annotations[annotations["status"] == 1].copy()

    # Deduplicate same sentence + same symptom (keep first annotator)
    before = len(positives)
    positives = positives.drop_duplicates(subset=["sentence_id", "DSM5_symptom"], keep="first")
    after = len(positives)
    if before != after:
        print(f"  Deduplicated {before - after} duplicate (sentence, symptom) pairs")

    positives["label"] = positives["DSM5_symptom"]
    positives["label_id"] = positives["DSM5_symptom"].map(SYMPTOM_LABELS)
    positives["clean_text"] = positives["sentence_text"].apply(clean_sentence)

    # Remove empty after cleaning
    positives = positives[positives["clean_text"].str.len() >= 5]

    print(f"  Positive samples: {len(positives)}")
    print(f"  Per symptom:")
    for symptom, count in positives["label"].value_counts().items():
        print(f"    {symptom}: {count}")

    return positives[["post_id", "sentence_id", "sentence_text", "clean_text",
                       "label", "label_id"]].reset_index(drop=True)


# ── Negative samples ─────────────────────────────────────────────────────────

def create_negative_samples(
    posts: pd.DataFrame,
    annotations: pd.DataFrame,
    max_negatives: int = 400,
) -> pd.DataFrame:
    """Create NO_SYMPTOM training samples from two sources:

    1. Sentences that only appear with status=0 (never status=1) in annotations.
    2. Sentences extracted from completely unannotated posts.
    """
    # Source 1: True negative sentences (only status=0, never status=1)
    positive_sentence_ids = set(annotations[annotations["status"] == 1]["sentence_id"].unique())
    all_sentence_ids = set(annotations["sentence_id"].unique())
    negative_only_ids = all_sentence_ids - positive_sentence_ids

    neg_from_annotations = annotations[
        annotations["sentence_id"].isin(negative_only_ids)
    ].drop_duplicates(subset=["sentence_id"], keep="first")

    print(f"  Negative sentences from annotations (status=0 only): {len(neg_from_annotations)}")

    # Source 2: Sentences from unannotated posts
    annotated_post_ids = set(annotations["post_id"].unique())
    all_post_ids = set(posts["post_id"].unique())
    unannotated_post_ids = all_post_ids - annotated_post_ids

    unannotated_posts = posts[posts["post_id"].isin(unannotated_post_ids)]
    neg_from_posts_rows = []
    for _, row in unannotated_posts.iterrows():
        sentences = split_into_sentences(row["text"])
        for i, sent in enumerate(sentences):
            neg_from_posts_rows.append({
                "post_id": row["post_id"],
                "sentence_id": f"{row['post_id']}_neg_{i}",
                "sentence_text": sent,
            })

    neg_from_posts = pd.DataFrame(neg_from_posts_rows)
    print(f"  Negative sentences from unannotated posts: {len(neg_from_posts)}")

    # Combine both sources
    neg_combined_rows = []

    for _, row in neg_from_annotations.iterrows():
        neg_combined_rows.append({
            "post_id": row["post_id"],
            "sentence_id": row["sentence_id"],
            "sentence_text": row["sentence_text"],
        })

    for _, row in neg_from_posts.iterrows():
        neg_combined_rows.append({
            "post_id": row["post_id"],
            "sentence_id": row["sentence_id"],
            "sentence_text": row["sentence_text"],
        })

    negatives = pd.DataFrame(neg_combined_rows)
    negatives["clean_text"] = negatives["sentence_text"].apply(clean_sentence)
    negatives = negatives[negatives["clean_text"].str.len() >= 5]

    # Cap negatives to prevent class domination
    if len(negatives) > max_negatives:
        negatives = negatives.sample(n=max_negatives, random_state=42)
        print(f"  Capped NO_SYMPTOM to {max_negatives} samples")

    negatives["label"] = "NO_SYMPTOM"
    negatives["label_id"] = SYMPTOM_LABELS["NO_SYMPTOM"]

    print(f"  Total negative samples: {len(negatives)}")

    return negatives[["post_id", "sentence_id", "sentence_text", "clean_text",
                       "label", "label_id"]].reset_index(drop=True)


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_by_post_id(
    df: pd.DataFrame,
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by post_id to prevent data leakage.

    Uses GroupShuffleSplit so no sentences from the same post appear in
    different splits.
    """
    # First split: train+val vs test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df["post_id"]))
    train_val = df.iloc[train_val_idx]
    test = df.iloc[test_idx]

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(train_val, groups=train_val["post_id"]))
    train = train_val.iloc[train_idx]
    val = train_val.iloc[val_idx]

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def compute_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = train_df["label_id"].value_counts().sort_index()
    total = len(train_df)
    n_classes = len(counts)
    weights = {}
    for label_id, count in counts.items():
        weights[int(label_id)] = total / (n_classes * count)
    return weights


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess ReDSM5 dataset")
    parser.add_argument("--redsm5-dir", type=str, default=None,
                        help="Path to redsm5 directory (with CSV files)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Path to output directory for processed splits")
    parser.add_argument("--max-negatives", type=int, default=400,
                        help="Maximum NO_SYMPTOM samples (default: 400)")
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent.parent.parent  # backend/ml/scripts → project root
    redsm5_dir = Path(args.redsm5_dir) if args.redsm5_dir else project_root / "redsm5"
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "data" / "redsm5" / "processed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ReDSM5 Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input:  {redsm5_dir}")
    print(f"Output: {output_dir}")

    # ── Load ──
    print("\n── Loading data ──")
    posts, annotations = load_data(redsm5_dir)

    # ── Create samples ──
    print("\n── Creating positive samples (status=1) ──")
    positives = create_positive_samples(annotations)

    print("\n── Creating negative samples (NO_SYMPTOM) ──")
    negatives = create_negative_samples(posts, annotations, max_negatives=args.max_negatives)

    # ── Combine ──
    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    print(f"\nTotal dataset: {len(combined)} samples across {combined['post_id'].nunique()} posts")

    # ── Split ──
    print("\n── Splitting by post_id (80/10/10) ──")
    train, val, test = split_by_post_id(combined)

    print(f"  Train: {len(train)} samples ({train['post_id'].nunique()} posts)")
    print(f"  Val:   {len(val)} samples ({val['post_id'].nunique()} posts)")
    print(f"  Test:  {len(test)} samples ({test['post_id'].nunique()} posts)")

    # Verify no post leakage
    train_posts = set(train["post_id"])
    val_posts = set(val["post_id"])
    test_posts = set(test["post_id"])
    assert len(train_posts & val_posts) == 0, "Post leakage: train ∩ val"
    assert len(train_posts & test_posts) == 0, "Post leakage: train ∩ test"
    assert len(val_posts & test_posts) == 0, "Post leakage: val ∩ test"
    print("  ✓ No post_id leakage across splits")

    # ── Per-split label distribution ──
    print("\n── Label distribution per split ──")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"\n  {name}:")
        for label, count in split["label"].value_counts().sort_index().items():
            print(f"    {label}: {count}")

    # ── Class weights ──
    class_weights = compute_class_weights(train)
    print("\n── Class weights (inverse frequency) ──")
    for label_id, weight in sorted(class_weights.items()):
        label_name = [k for k, v in SYMPTOM_LABELS.items() if v == label_id][0]
        print(f"  {label_name} ({label_id}): {weight:.3f}")

    # ── Save ──
    print("\n── Saving splits ──")
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        "label_map": SYMPTOM_LABELS,
        "label_readable": SYMPTOM_READABLE,
        "class_weights": class_weights,
        "num_classes": len(SYMPTOM_LABELS),
        "total_samples": len(combined),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "train_posts": train["post_id"].nunique(),
        "val_posts": val["post_id"].nunique(),
        "test_posts": test["post_id"].nunique(),
        "label_distribution": {
            "train": train["label"].value_counts().to_dict(),
            "val": val["label"].value_counts().to_dict(),
            "test": test["label"].value_counts().to_dict(),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {output_dir}:")
    print(f"  train.csv  ({len(train)} rows)")
    print(f"  val.csv    ({len(val)} rows)")
    print(f"  test.csv   ({len(test)} rows)")
    print(f"  metadata.json")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Next step: python train_redsm5_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
