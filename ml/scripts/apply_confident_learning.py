"""
Apply confident learning findings to clean training data.

Strategy:
1. HIGH confidence mislabels (>0.85): relabel to model's prediction
   EXCEPT negation cases ("I'm NOT suicidal") — these get removed
2. MEDIUM confidence (0.7-0.85): remove (too ambiguous to relabel)
3. LOW confidence (<0.7): keep as-is (model isn't sure enough)

Also handles:
- Negation detection: sentences that discuss symptoms in the negative
- Removes samples that are too ambiguous for any label

Usage:
    python apply_confident_learning.py
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patterns that indicate negation — discussing a symptom but NOT experiencing it
NEGATION_PATTERNS = [
    r"\b(not|never|no longer|don'?t|doesn'?t|isn'?t|wasn'?t|aren'?t|haven'?t|hasn'?t|won'?t|can'?t)\b.{0,20}\b(suicid|kill|die|death|depress|sad|cry|sleep|tired|energy|appetite|focus|concentrat|worthless|guilt|burden)",
    r"\b(suicid|kill|die|death|depress|sad|cry|sleep|tired|energy|appetite|focus|concentrat|worthless|guilt|burden).{0,20}\b(not|never|no longer|don'?t|doesn'?t|isn'?t|wasn'?t|aren'?t|haven'?t|hasn'?t|won'?t|can'?t)\b",
    r"\b(before|used to|in the past|years ago|stopped|quit|no more)\b.{0,30}\b(suicid|depress|cry|sad)",
    r"\bI'?m not\b",
    r"\bI never\b",
    r"\bI don'?t (have|feel|think|want)\b",
]


def is_negation(text: str) -> bool:
    """Check if text discusses a symptom in the negative/past tense."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in NEGATION_PATTERNS)


def main():
    base_dir = Path(__file__).parent.parent
    cleaned_dir = base_dir / "data" / "redsm5" / "cleaned"
    output_dir = base_dir / "data" / "redsm5" / "cleaned_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load current cleaned data
    train = pd.read_csv(cleaned_dir / "train.csv")
    val = pd.read_csv(cleaned_dir / "val.csv")
    test = pd.read_csv(cleaned_dir / "test.csv")

    with open(cleaned_dir / "metadata.json") as f:
        metadata = json.load(f)
    label_map = metadata["label_map"]

    # Load confident learning suspects
    with open(cleaned_dir / "confident_learning_suspects.json") as f:
        suspects = json.load(f)

    logger.info(f"Original cleaned training samples: {len(train)}")
    logger.info(f"Total suspects: {len(suspects)}")

    # Categorize suspects
    relabeled = 0
    removed = 0
    kept = 0
    indices_to_drop = []
    relabel_map = {}  # index → new label

    for s in suspects:
        idx = int(s["index"])
        conf = float(s["pred_conf"])
        true_label = s["true_label"]
        pred_label = s["pred_label"]
        text = train.iloc[idx]["clean_text"] if idx < len(train) else ""

        if conf > 0.85:
            # High confidence — model is very sure this is mislabeled
            if is_negation(text):
                # Negation case: "I'm NOT suicidal" — remove, it confuses the model
                indices_to_drop.append(idx)
                removed += 1
            else:
                # Genuine mislabel — relabel to model's prediction
                relabel_map[idx] = pred_label
                relabeled += 1

        elif conf > 0.7:
            # Medium confidence — too ambiguous, remove
            indices_to_drop.append(idx)
            removed += 1

        else:
            # Low confidence — keep original label
            kept += 1

    logger.info("\nActions:")
    logger.info(f"  Relabeled (conf >0.85, not negation): {relabeled}")
    logger.info(f"  Removed (negation or ambiguous 0.7-0.85): {removed}")
    logger.info(f"  Kept as-is (conf <0.7): {kept}")

    # Apply relabeling
    for idx, new_label in relabel_map.items():
        if idx < len(train):
            train.at[idx, "label"] = new_label
            train.at[idx, "label_id"] = label_map[new_label]

    # Apply removal
    train = train.drop(index=[i for i in indices_to_drop if i < len(train)])
    train = train.reset_index(drop=True)

    logger.info(f"\nAfter confident learning: {len(train)} training samples")

    # Recompute class weights
    from preprocess_redsm5 import SYMPTOM_LABELS, SYMPTOM_READABLE
    counts = train["label_id"].value_counts().sort_index()
    total = len(train)
    n_classes = len(SYMPTOM_LABELS)
    class_weights = {}
    for label_id, count in counts.items():
        class_weights[int(label_id)] = total / (n_classes * count)

    # Save
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    new_metadata = {
        "label_map": label_map,
        "label_readable": SYMPTOM_READABLE,
        "class_weights": class_weights,
        "num_classes": n_classes,
        "total_samples": len(train) + len(val) + len(test),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "confident_learning": {
            "relabeled": relabeled,
            "removed": removed,
            "kept": kept,
            "threshold_high": 0.85,
            "threshold_medium": 0.7,
        },
        "label_distribution": {
            "train": train["label"].value_counts().to_dict(),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(new_metadata, f, indent=2)

    # Report
    print(f"\n{'='*60}")
    print("CONFIDENT LEARNING APPLIED")
    print(f"{'='*60}")
    print("Original cleaned: 1514")
    print(f"After CL:         {len(train)}")
    print(f"  Relabeled:      {relabeled}")
    print(f"  Removed:        {removed}")
    print(f"  Kept:           {kept}")
    print("\nNew class distribution:")
    for label, count in train["label"].value_counts().sort_values().items():
        print(f"  {label:<22} {count:>4}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
