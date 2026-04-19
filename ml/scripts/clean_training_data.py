"""
Training data quality audit and cleaning.

Fixes three categories of noise in the ReDSM5 training data:
1. Conflicting labels: Same text appears with 2 different labels (53 cases)
   → Pick primary symptom based on clinical salience hierarchy
2. Exact duplicates: Same text + same label appearing twice (19 pairs)
   → Deduplicate
3. Mislabeled short sentences: Very short texts with questionable labels
   → Flag for review, remove clearly wrong ones

This is NOT modifying the original ReDSM5 annotations — it's creating a
cleaned training split. The original data is preserved.

Usage:
    python clean_training_data.py
"""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clinical salience hierarchy for resolving multi-label conflicts.
# When a sentence expresses two symptoms, pick the one that is:
# 1. More clinically urgent (SUICIDAL_THOUGHTS > everything)
# 2. More specific (APPETITE_CHANGE > DEPRESSED_MOOD)
# 3. Rarer in the dataset (helps balance)
SALIENCE_PRIORITY = {
    "SUICIDAL_THOUGHTS": 10,   # Always prioritize — safety critical
    "PSYCHOMOTOR": 9,          # Rare + specific observable behavior
    "APPETITE_CHANGE": 8,      # Rare + specific physical symptom
    "COGNITIVE_ISSUES": 7,     # Rare + specific cognitive symptom
    "SLEEP_ISSUES": 6,         # Specific physical symptom
    "FATIGUE": 5,              # Specific but overlaps with many
    "ANHEDONIA": 4,            # Core DSM-5 criterion
    "WORTHLESSNESS": 3,        # Common but specific
    "SPECIAL_CASE": 2,         # Catch-all
    "DEPRESSED_MOOD": 1,       # Most general — everything overlaps with this
    "NO_SYMPTOM": 0,           # Lowest priority — if ANY symptom is present, it's not "no symptom"
}

# Sentences that are clearly mislabeled based on manual review.
# Format: (clean_text_prefix, wrong_label, correct_label_or_None_to_remove)
MANUAL_FIXES = [
    # These were found in the distillation analysis + data audit
    ("I've literally made the best financial decision", "COGNITIVE_ISSUES", "NO_SYMPTOM"),
    ("I like that I can make decisions that affect only me", "COGNITIVE_ISSUES", "NO_SYMPTOM"),
    ("I work a lot and make decisions all day", "COGNITIVE_ISSUES", "NO_SYMPTOM"),
    ("Ive missed a lot of work", "COGNITIVE_ISSUES", None),  # Remove — no cognitive symptom evidenced
    ("Insecurities are getting to me", "COGNITIVE_ISSUES", "WORTHLESSNESS"),
    ("I feel successful", "WORTHLESSNESS", None),  # Remove — no symptom, possibly sarcastic but ambiguous
    ("I feel happiness", "SPECIAL_CASE", "NO_SYMPTOM"),
    ("I love meeting new people", "ANHEDONIA", "NO_SYMPTOM"),
    ("Now i get paid more and have more time", "SPECIAL_CASE", "NO_SYMPTOM"),
]


def resolve_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve conflicting labels for multi-label sentences.

    For each sentence that appears with multiple labels, keep only the
    label with highest clinical salience priority.
    """
    # Find conflicting texts
    text_groups = df.groupby("clean_text")["label"].apply(set).reset_index()
    conflicts = text_groups[text_groups["label"].apply(lambda x: len(x) > 1)]

    if len(conflicts) == 0:
        logger.info("  No conflicting labels found")
        return df

    logger.info(f"  Found {len(conflicts)} sentences with conflicting labels")

    resolved_count = 0
    rows_removed = 0
    indices_to_drop = []

    for _, conflict in conflicts.iterrows():
        text = conflict["clean_text"]
        labels = conflict["label"]

        # Pick highest-priority label
        primary = max(labels, key=lambda l: SALIENCE_PRIORITY.get(l, -1))

        # Find all rows with this text and drop the non-primary ones
        matching_rows = df[df["clean_text"] == text]
        for idx, row in matching_rows.iterrows():
            if row["label"] != primary:
                indices_to_drop.append(idx)
                rows_removed += 1

        resolved_count += 1

    df_clean = df.drop(indices_to_drop)
    logger.info(f"  Resolved {resolved_count} conflicts → removed {rows_removed} conflicting rows")
    logger.info(f"  Primary label chosen by clinical salience hierarchy")

    return df_clean.reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (same text + same label)."""
    before = len(df)
    df_clean = df.drop_duplicates(subset=["clean_text", "label"], keep="first")
    removed = before - len(df_clean)
    if removed > 0:
        logger.info(f"  Removed {removed} exact duplicates")
    return df_clean.reset_index(drop=True)


def apply_manual_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manual label corrections from expert review."""
    fixed = 0
    removed = 0
    indices_to_drop = []

    for text_prefix, wrong_label, correct_label in MANUAL_FIXES:
        mask = (
            df["clean_text"].str.startswith(text_prefix, na=False)
            & (df["label"] == wrong_label)
        )
        matching = df[mask]

        if len(matching) == 0:
            continue

        if correct_label is None:
            # Remove the row entirely
            indices_to_drop.extend(matching.index.tolist())
            removed += len(matching)
        else:
            # Fix the label
            from preprocess_redsm5 import SYMPTOM_LABELS
            df.loc[mask, "label"] = correct_label
            df.loc[mask, "label_id"] = SYMPTOM_LABELS[correct_label]
            fixed += len(matching)

    if indices_to_drop:
        df = df.drop(indices_to_drop)

    logger.info(f"  Manual fixes: {fixed} labels corrected, {removed} rows removed")
    return df.reset_index(drop=True)


def flag_suspicious_short(df: pd.DataFrame, min_length: int = 15) -> list[dict]:
    """Flag very short sentences that may have insufficient signal."""
    short = df[df["clean_text"].str.len() < min_length]
    flagged = []
    for _, row in short.iterrows():
        flagged.append({
            "text": row["clean_text"],
            "label": row["label"],
            "length": len(row["clean_text"]),
        })
    if flagged:
        logger.info(f"  Flagged {len(flagged)} very short sentences (<{min_length} chars) — kept but noted")
    return flagged


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "redsm5" / "processed"
    output_dir = base_dir / "data" / "redsm5" / "cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAINING DATA CLEANING")
    print("=" * 60)

    # Load original training data
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    test = pd.read_csv(data_dir / "test.csv")

    logger.info(f"\nOriginal: train={len(train)}, val={len(val)}, test={len(test)}")

    # ── Step 1: Resolve conflicting labels ──
    print("\n── Step 1: Resolve conflicting labels ──")
    train = resolve_conflicts(train)

    # Also clean val/test for consistency
    val = resolve_conflicts(val)
    test = resolve_conflicts(test)

    # ── Step 2: Remove exact duplicates ──
    print("\n── Step 2: Remove exact duplicates ──")
    train = remove_duplicates(train)
    val = remove_duplicates(val)
    test = remove_duplicates(test)

    # ── Step 3: Apply manual label fixes ──
    print("\n── Step 3: Apply manual label fixes ──")
    train = apply_manual_fixes(train)

    # ── Step 4: Flag suspicious short sentences ──
    print("\n── Step 4: Flag suspicious short sentences ──")
    flagged = flag_suspicious_short(train)

    # ── Recompute class weights ──
    from preprocess_redsm5 import SYMPTOM_LABELS, SYMPTOM_READABLE
    counts = train["label_id"].value_counts().sort_index()
    total = len(train)
    n_classes = len(SYMPTOM_LABELS)
    class_weights = {}
    for label_id, count in counts.items():
        class_weights[int(label_id)] = total / (n_classes * count)

    # ── Save cleaned data ──
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    metadata = {
        "label_map": SYMPTOM_LABELS,
        "label_readable": SYMPTOM_READABLE,
        "class_weights": class_weights,
        "num_classes": n_classes,
        "total_samples": len(train) + len(val) + len(test),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "cleaning_applied": {
            "conflicts_resolved": 53,
            "duplicates_removed": True,
            "manual_fixes": len(MANUAL_FIXES),
            "flagged_short_sentences": len(flagged),
        },
        "label_distribution": {
            "train": train["label"].value_counts().to_dict(),
            "val": val["label"].value_counts().to_dict(),
            "test": test["label"].value_counts().to_dict(),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "flagged_short.json", "w") as f:
        json.dump(flagged, f, indent=2)

    # ── Report ──
    print(f"\n{'='*60}")
    print("CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"Original training samples: 1591")
    print(f"After cleaning:            {len(train)}")
    print(f"Removed:                   {1591 - len(train)}")
    print(f"\nCleaned class distribution:")
    for label, count in train["label"].value_counts().sort_values().items():
        orig_count = pd.read_csv(data_dir / "train.csv")["label"].value_counts().get(label, 0)
        delta = count - orig_count
        print(f"  {label:<22} {count:>4} (was {orig_count}, {'+' if delta >= 0 else ''}{delta})")
    print(f"\nSaved to: {output_dir}")
    print(f"Use --data-dir {output_dir} in training scripts")


if __name__ == "__main__":
    main()
