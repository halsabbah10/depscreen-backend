"""
Text preprocessing script for Suicide-Watch dataset.

This script:
1. Loads the raw CSV dataset
2. Cleans text (removes URLs, usernames, normalizes whitespace)
3. Maps labels to binary (risk/no-risk) or multi-class
4. Creates stratified train/val/test splits
5. Saves processed data and statistics

Usage:
    python preprocess_text.py [--multiclass]
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Clean a single text sample."""
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "[URL]", text)

    # Remove Reddit-style usernames
    text = re.sub(r"u/\w+", "[USER]", text)
    text = re.sub(r"@\w+", "[USER]", text)

    # Remove subreddit references
    text = re.sub(r"r/\w+", "[SUBREDDIT]", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def load_and_explore(data_dir: Path) -> pd.DataFrame:
    """Load dataset and explore its structure."""
    print("\n" + "=" * 50)
    print("Loading dataset...")
    print("=" * 50)

    # Find CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Load the main dataset (usually the largest file)
    main_file = max(csv_files, key=lambda f: f.stat().st_size)
    print(f"Loading: {main_file.name}")

    df = pd.read_csv(main_file)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nFirst few rows:\n{df.head()}")

    # Identify text and label columns
    text_col = None
    label_col = None

    # Common text column names
    for col in ["text", "Text", "content", "Content", "post", "Post", "body"]:
        if col in df.columns:
            text_col = col
            break

    # Common label column names
    for col in ["class", "Class", "label", "Label", "category", "target"]:
        if col in df.columns:
            label_col = col
            break

    if text_col is None:
        # Use first string column
        for col in df.columns:
            if df[col].dtype == "object":
                text_col = col
                break

    if label_col is None:
        # Use last column that's not the text column
        for col in reversed(df.columns.tolist()):
            if col != text_col:
                label_col = col
                break

    print("\nIdentified columns:")
    print(f"  Text column: {text_col}")
    print(f"  Label column: {label_col}")

    if label_col:
        print("\nLabel distribution:")
        print(df[label_col].value_counts())

    return df, text_col, label_col


def map_labels(df: pd.DataFrame, label_col: str, multiclass: bool = False) -> pd.DataFrame:
    """Map original labels to standardized format."""
    print("\n" + "=" * 50)
    print("Mapping labels...")
    print("=" * 50)

    # Get unique labels
    unique_labels = df[label_col].unique()
    print(f"Original labels: {unique_labels}")

    # Create label mapping
    if multiclass:
        # Multi-class: keep separate categories
        label_map = {}
        risk_labels = ["suicide", "Suicide", "depression", "Depression", "SuicideWatch"]
        neutral_labels = ["non-suicide", "normal", "Normal", "teenager", "teenagers"]

        for label in unique_labels:
            label_str = str(label).lower()
            if any(risk in label_str for risk in ["suicide", "depression"]):
                if "suicide" in label_str:
                    label_map[label] = "suicide"
                else:
                    label_map[label] = "depression"
            else:
                label_map[label] = "neutral"

        df["label"] = df[label_col].map(label_map)
        df["label_id"] = df["label"].map({"neutral": 0, "depression": 1, "suicide": 2})

    else:
        # Binary: risk vs no-risk
        label_map = {}
        for label in unique_labels:
            label_str = str(label).lower()
            if any(risk in label_str for risk in ["suicide", "depression", "suicidewatch"]):
                label_map[label] = "high_risk"
            else:
                label_map[label] = "low_risk"

        df["label"] = df[label_col].map(label_map)
        df["label_id"] = df["label"].map({"low_risk": 0, "high_risk": 1})

    print(f"\nLabel mapping: {label_map}")
    print("\nNew label distribution:")
    print(df["label"].value_counts())

    return df


def preprocess_dataset(
    df: pd.DataFrame, text_col: str, output_dir: Path, test_size: float = 0.15, val_size: float = 0.15
) -> dict:
    """Preprocess and split the dataset."""
    print("\n" + "=" * 50)
    print("Preprocessing text...")
    print("=" * 50)

    # Clean text
    print("Cleaning text...")
    df["clean_text"] = df[text_col].apply(clean_text)

    # Remove empty texts
    original_len = len(df)
    df = df[df["clean_text"].str.len() > 10]  # Min 10 chars
    print(f"Removed {original_len - len(df)} samples with empty/short text")

    # Calculate text statistics
    df["text_length"] = df["clean_text"].str.len()
    df["word_count"] = df["clean_text"].str.split().str.len()

    print("\nText length stats:")
    print(df["text_length"].describe())

    print("\nWord count stats:")
    print(df["word_count"].describe())

    # Stratified split
    print("\n" + "=" * 50)
    print("Creating stratified splits...")
    print("=" * 50)

    # First split: train+val vs test
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label_id"], random_state=42)

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val_size, stratify=train_val["label_id"], random_state=42
    )

    print(f"Train set: {len(train)} samples")
    print(f"Val set: {len(val)} samples")
    print(f"Test set: {len(test)} samples")

    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)

    train[["clean_text", "label", "label_id"]].to_csv(output_dir / "train.csv", index=False)
    val[["clean_text", "label", "label_id"]].to_csv(output_dir / "val.csv", index=False)
    test[["clean_text", "label", "label_id"]].to_csv(output_dir / "test.csv", index=False)

    print(f"\nSaved to: {output_dir}")

    # Create metadata
    label_counts = df["label"].value_counts().to_dict()
    label_map = {label: int(df[df["label"] == label]["label_id"].iloc[0]) for label in df["label"].unique()}

    metadata = {
        "total_samples": len(df),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "label_distribution": label_counts,
        "label_map": label_map,
        "avg_text_length": float(df["text_length"].mean()),
        "avg_word_count": float(df["word_count"].mean()),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_dir / 'metadata.json'}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess Suicide-Watch dataset")
    parser.add_argument("--multiclass", action="store_true", help="Use multi-class labels instead of binary")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output directory")
    args = parser.parse_args()

    # Set paths
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "suicide_watch"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "data" / "suicide_watch" / "processed"

    print("=" * 60)
    print("Text Data Preprocessing")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {'multi-class' if args.multiclass else 'binary'}")

    # Load and explore
    df, text_col, label_col = load_and_explore(data_dir)

    # Map labels
    df = map_labels(df, label_col, multiclass=args.multiclass)

    # Preprocess and split
    metadata = preprocess_dataset(df, text_col, output_dir)

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("\nNext step: python train_text_model.py")


if __name__ == "__main__":
    main()
