"""
Comprehensive evaluation script for the DSM-5 symptom classifier.

Generates all metrics, visualizations, and error analysis for the capstone report.

Usage:
    python evaluate_redsm5.py [--model-path PATH] [--data-dir PATH] [--output-dir PATH]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse model and dataset from training script
from train_redsm5_model import SymptomClassifier, SymptomDataset, collate_fn


def per_symptom_breakdown(labels, preds, label_names) -> pd.DataFrame:
    """Create a per-symptom F1 breakdown table."""
    p, r, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(label_names))), zero_division=0
    )
    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "Symptom": name,
            "Precision": round(p[i], 4),
            "Recall": round(r[i], 4),
            "F1-Score": round(f1[i], 4),
            "Support": int(support[i]),
        })
    return pd.DataFrame(rows)


def error_analysis(test_df, preds, label_names) -> pd.DataFrame:
    """Identify misclassified examples with true vs predicted labels."""
    errors = []
    for i, (true_label, pred_label) in enumerate(zip(test_df["label_id"], preds)):
        if true_label != pred_label:
            errors.append({
                "sentence_text": test_df.iloc[i]["clean_text"],
                "true_label": label_names[true_label],
                "predicted_label": label_names[pred_label],
                "post_id": test_df.iloc[i]["post_id"],
            })
    return pd.DataFrame(errors)


def post_level_evaluation(test_df, preds, label_names) -> dict:
    """Aggregate sentence predictions to post level and evaluate."""
    test_df = test_df.copy()
    test_df["predicted_label"] = [label_names[p] for p in preds]

    results_per_post = {}
    for post_id, group in test_df.groupby("post_id"):
        true_symptoms = set(
            group[group["label"] != "NO_SYMPTOM"]["label"].unique()
        )
        pred_symptoms = set(
            group[group["predicted_label"] != "NO_SYMPTOM"]["predicted_label"].unique()
        )
        results_per_post[post_id] = {
            "true_symptom_count": len(true_symptoms),
            "pred_symptom_count": len(pred_symptoms),
            "true_symptoms": sorted(true_symptoms),
            "pred_symptoms": sorted(pred_symptoms),
            "symptom_overlap": len(true_symptoms & pred_symptoms),
        }

    df = pd.DataFrame(results_per_post).T

    # Severity accuracy (based on count thresholds)
    def severity(count):
        if count == 0: return "none"
        if count <= 2: return "mild"
        if count <= 4: return "moderate"
        return "severe"

    df["true_severity"] = df["true_symptom_count"].apply(severity)
    df["pred_severity"] = df["pred_symptom_count"].apply(severity)
    severity_accuracy = (df["true_severity"] == df["pred_severity"]).mean()

    # Average symptom count error
    count_mae = (df["true_symptom_count"] - df["pred_symptom_count"]).abs().mean()

    return {
        "num_posts": len(df),
        "severity_accuracy": round(severity_accuracy, 4),
        "symptom_count_mae": round(count_mae, 4),
        "avg_true_symptoms": round(df["true_symptom_count"].mean(), 2),
        "avg_pred_symptoms": round(df["pred_symptom_count"].mean(), 2),
        "avg_symptom_overlap": round(df["symptom_overlap"].mean(), 2),
    }


def baseline_comparison(our_micro_f1: float) -> pd.DataFrame:
    """Compare to published ReDSM5 baselines."""
    baselines = {
        "SVM (TF-IDF)": 0.39,
        "CNN (10 epochs)": 0.25,
        "BERT (fine-tuned)": 0.51,
        "LLaMA 3.2-1B (fine-tuned)": 0.54,
        "DepScreen DistilBERT (ours)": our_micro_f1,
    }
    rows = []
    for name, score in baselines.items():
        delta = our_micro_f1 - score if name != f"DepScreen DistilBERT (ours)" else 0
        rows.append({
            "Model": name,
            "Micro-F1": round(score, 4),
            "Delta vs Ours": f"+{delta:.4f}" if delta > 0 else ("—" if delta == 0 else f"{delta:.4f}"),
        })
    return pd.DataFrame(rows)


def generate_markdown_report(
    test_metrics: dict,
    per_class_df: pd.DataFrame,
    error_df: pd.DataFrame,
    post_metrics: dict,
    baseline_df: pd.DataFrame,
    model_name: str,
    output_path: Path,
):
    """Generate a markdown evaluation report."""
    lines = [
        "# DepScreen — Model Evaluation Report",
        "",
        f"**Model**: {model_name}",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "## Overall Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {test_metrics['accuracy']:.4f} |",
        f"| Micro-F1 | {test_metrics['micro_f1']:.4f} |",
        f"| Macro-F1 | {test_metrics['macro_f1']:.4f} |",
        f"| Micro-Precision | {test_metrics['micro_precision']:.4f} |",
        f"| Micro-Recall | {test_metrics['micro_recall']:.4f} |",
        "",
        "## Per-Symptom Breakdown",
        "",
        per_class_df.to_markdown(index=False),
        "",
        "## Baseline Comparison",
        "",
        baseline_df.to_markdown(index=False),
        "",
        "## Post-Level Aggregation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Posts evaluated | {post_metrics['num_posts']} |",
        f"| Severity classification accuracy | {post_metrics['severity_accuracy']:.2%} |",
        f"| Symptom count MAE | {post_metrics['symptom_count_mae']:.2f} |",
        f"| Avg true symptoms/post | {post_metrics['avg_true_symptoms']} |",
        f"| Avg predicted symptoms/post | {post_metrics['avg_pred_symptoms']} |",
        f"| Avg symptom overlap/post | {post_metrics['avg_symptom_overlap']} |",
        "",
        "## Error Analysis",
        "",
        f"Total misclassifications: {len(error_df)} / {test_metrics.get('total_samples', '?')}",
        "",
        "### Sample Errors (first 20)",
        "",
    ]

    if len(error_df) > 0:
        sample = error_df.head(20)
        lines.append(sample.to_markdown(index=False))
    else:
        lines.append("No errors found.")

    lines.extend([
        "",
        "## Notes",
        "",
        "- PSYCHOMOTOR has only 1 test sample — F1 unreliable for this class",
        "- APPETITE_CHANGE has only 4 test samples — F1 should be interpreted with caution",
        "- NO_SYMPTOM has low recall (model prefers detecting symptoms) — safe bias for screening",
        "- All baselines are from the ReDSM5 paper (CIKM 2025, arXiv:2508.03399)",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DSM-5 symptom classifier")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "processed"
    model_path = Path(args.model_path) if args.model_path else base_dir / "models" / "symptom_classifier.pt"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load metadata
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    label_map = metadata["label_map"]
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = metadata["num_classes"]

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = SymptomClassifier(num_classes=num_classes, model_name=args.model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    test_df = pd.read_csv(data_dir / "test.csv")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_dataset = SymptomDataset(
        test_df["clean_text"].tolist(), test_df["label_id"].tolist(),
        tokenizer, args.max_length,
    )
    num_workers = 0 if device.type == "mps" else 2
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    # Evaluate
    logger.info("Running evaluation...")
    criterion = torch.nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    test_metrics = {
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "total_samples": len(all_labels),
        "loss": total_loss / len(test_loader),
    }

    # Per-symptom breakdown
    per_class_df = per_symptom_breakdown(all_labels, all_preds, label_names)
    print("\n" + "=" * 60)
    print("Per-Symptom F1 Breakdown:")
    print("=" * 60)
    print(per_class_df.to_string(index=False))

    # Error analysis
    error_df = error_analysis(test_df, all_preds, label_names)
    print(f"\nMisclassified: {len(error_df)} / {len(all_labels)} ({len(error_df)/len(all_labels):.1%})")

    # Post-level
    post_metrics = post_level_evaluation(test_df, all_preds, label_names)
    print(f"\nPost-level severity accuracy: {post_metrics['severity_accuracy']:.2%}")
    print(f"Symptom count MAE: {post_metrics['symptom_count_mae']:.2f}")

    # Baseline comparison
    baseline_df = baseline_comparison(micro_f1)
    print("\n" + "=" * 60)
    print("Baseline Comparison:")
    print("=" * 60)
    print(baseline_df.to_string(index=False))

    # Save everything
    per_class_df.to_csv(output_dir / "per_symptom_f1.csv", index=False)
    error_df.to_csv(output_dir / "error_analysis.csv", index=False)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    with open(output_dir / "post_level_metrics.json", "w") as f:
        json.dump(post_metrics, f, indent=2)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=str)

    # Confusion matrix as CSV
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Markdown report
    generate_markdown_report(
        test_metrics, per_class_df, error_df, post_metrics,
        baseline_df, args.model_name, output_dir / "evaluation_report.md",
    )

    print(f"\nAll evaluation artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
