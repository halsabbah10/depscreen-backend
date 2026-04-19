"""
Proper ensemble CV: trains all 3 models per fold, averages softmax
probabilities, then evaluates. Also does aggregated threshold tuning.

Usage:
    python ensemble_cv.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_redsm5 import SYMPTOM_LABELS
from train_redsm5_model import SymptomClassifier, SymptomDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ENSEMBLE_MODELS = [
    {"name": "./ml/models/v2_dapt_base", "label": "DAPT-DistilBERT", "has_distill": True},
    {"name": "roberta-base", "label": "RoBERTa", "has_distill": False},
    {"name": "microsoft/deberta-base", "label": "DeBERTa", "has_distill": False, "batch_size": 4},
]


def build_post_label_matrix(df):
    label_names = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
    post_symptoms = df.groupby("post_id")["label"].apply(set).reset_index()
    post_symptoms.columns = ["post_id", "symptoms"]
    mlb = MultiLabelBinarizer(classes=label_names)
    label_matrix = mlb.fit_transform(post_symptoms["symptoms"])
    return post_symptoms, label_matrix


def train_single_model(train_df, val_df, model_name, epochs, batch_size, lr, max_length, device):
    """Train one model and return softmax probabilities on val set."""
    label_names = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
    num_classes = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = SymptomDataset(
        train_df["clean_text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_length
    )
    val_dataset = SymptomDataset(val_df["clean_text"].tolist(), val_df["label_id"].tolist(), tokenizer, max_length)

    num_workers = 0 if device.type == "mps" else 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    model = SymptomClassifier(num_classes=num_classes, model_name=model_name, pooling="mean")
    model.to(device)

    # Effective-number weights
    from distillation_utils import compute_effective_number_weights

    class_counts = train_df["label_id"].value_counts().to_dict()
    weight_tensor = compute_effective_number_weights(class_counts, num_classes, 0.999).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"  {model_name.split('/')[-1]} E{epoch + 1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.extend(batch["label"].numpy())
        _, _, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")

        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Get softmax probabilities from best model
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    del model, best_state
    import gc

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return np.array(all_probs), np.array(all_labels), best_val_f1


def evaluate_predictions(all_labels, all_preds, num_classes, label_names):
    """Compute all metrics from predictions."""
    accuracy = accuracy_score(all_labels, all_preds)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "f1": float(per_class_f1[i]),
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "support": int(per_class_support[i]),
        }

    return {"accuracy": accuracy, "micro_f1": micro_f1, "macro_f1": macro_f1, "per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--augmented", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "cleaned_v2"
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load data
    train_full = pd.read_csv(data_dir / "train.csv")
    val_full = pd.read_csv(data_dir / "val.csv")
    combined = (
        pd.concat([train_full, val_full], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    )

    # Load augmented
    augmented_df = None
    if args.augmented:
        augmented_df = pd.read_csv(args.augmented)
        logger.info(f"Loaded {len(augmented_df)} augmented samples")

    label_names = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
    num_classes = len(label_names)

    # Build stratification matrix
    post_df, label_matrix = build_post_label_matrix(combined)
    mskf = MultilabelStratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    # Collect ALL predictions across all folds for aggregated threshold tuning
    all_fold_probs = []  # ensemble probabilities
    all_fold_labels = []
    fold_results = []

    for fold_idx, (train_post_idx, val_post_idx) in enumerate(mskf.split(post_df["post_id"], label_matrix)):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FOLD {fold_idx + 1}/{args.k}")
        logger.info(f"{'=' * 60}")

        train_post_ids = set(post_df.iloc[train_post_idx]["post_id"])
        val_post_ids = set(post_df.iloc[val_post_idx]["post_id"])
        train_df = combined[combined["post_id"].isin(train_post_ids)].reset_index(drop=True)
        val_df = combined[combined["post_id"].isin(val_post_ids)].reset_index(drop=True)

        # Add augmented to training
        if augmented_df is not None:
            aug_cols = ["post_id", "sentence_id", "sentence_text", "clean_text", "label", "label_id"]
            train_df = pd.concat([train_df, augmented_df[aug_cols]], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=42 + fold_idx).reset_index(drop=True)

        logger.info(f"  Train: {len(train_df)}, Val: {len(val_df)}")

        # Train all 3 models and collect probabilities
        import gc

        model_probs = []
        for model_cfg in ENSEMBLE_MODELS:
            logger.info(f"  Training {model_cfg['label']}...")
            bs = model_cfg.get("batch_size", args.batch_size)
            probs, labels, best_f1 = train_single_model(
                train_df, val_df, model_cfg["name"], args.epochs, bs, args.lr, 128, device
            )
            model_probs.append(probs)
            logger.info(f"    Best val micro-F1: {best_f1:.4f}")
            # Aggressive memory cleanup between models
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

        # Soft-vote: average probabilities
        ensemble_probs = np.mean(model_probs, axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)

        # Also get individual model predictions for comparison
        individual_metrics = {}
        for i, model_cfg in enumerate(ENSEMBLE_MODELS):
            preds = np.argmax(model_probs[i], axis=1)
            metrics = evaluate_predictions(labels, preds, num_classes, label_names)
            individual_metrics[model_cfg["label"]] = metrics
            logger.info(f"    {model_cfg['label']}: micro={metrics['micro_f1']:.4f} macro={metrics['macro_f1']:.4f}")

        # Ensemble metrics
        ens_metrics = evaluate_predictions(labels, ensemble_preds, num_classes, label_names)
        logger.info(f"    ENSEMBLE: micro={ens_metrics['micro_f1']:.4f} macro={ens_metrics['macro_f1']:.4f}")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "individual": individual_metrics,
                "ensemble": ens_metrics,
            }
        )

        # Collect for aggregated threshold tuning
        all_fold_probs.append(ensemble_probs)
        all_fold_labels.append(labels)

        # Aggressive cleanup between folds
        del model_probs, ensemble_probs, ensemble_preds
        import gc

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Aggregate results
    print(f"\n{'=' * 70}")
    print("ENSEMBLE CV RESULTS (SOFT-VOTE)")
    print(f"{'=' * 70}")

    ens_micros = [f["ensemble"]["micro_f1"] for f in fold_results]
    ens_macros = [f["ensemble"]["macro_f1"] for f in fold_results]

    print(
        f"\nEnsemble Micro-F1: {np.mean(ens_micros):.4f} ± {np.std(ens_micros):.4f}  [{', '.join(f'{v:.3f}' for v in ens_micros)}]"
    )
    print(
        f"Ensemble Macro-F1: {np.mean(ens_macros):.4f} ± {np.std(ens_macros):.4f}  [{', '.join(f'{v:.3f}' for v in ens_macros)}]"
    )

    # Per-model comparison
    print("\nPer-model averages:")
    for model_cfg in ENSEMBLE_MODELS:
        label = model_cfg["label"]
        micros = [f["individual"][label]["micro_f1"] for f in fold_results]
        macros = [f["individual"][label]["macro_f1"] for f in fold_results]
        print(
            f"  {label:<20} micro={np.mean(micros):.4f}±{np.std(micros):.4f}  macro={np.mean(macros):.4f}±{np.std(macros):.4f}"
        )

    # Per-class ensemble results
    print("\nEnsemble Per-Class F1:")
    print(f"{'Symptom':<25} {'F1 Mean':>8} {'± Std':>8}")
    print("-" * 45)
    for cls in label_names:
        f1s = [f["ensemble"]["per_class"][cls]["f1"] for f in fold_results]
        print(f"{cls:<25} {np.mean(f1s):>8.4f} {np.std(f1s):>8.4f}")

    # Aggregated threshold tuning
    print(f"\n{'=' * 70}")
    print("AGGREGATED THRESHOLD TUNING")
    print(f"{'=' * 70}")

    all_probs = np.concatenate(all_fold_probs, axis=0)
    all_labels_flat = np.concatenate(all_fold_labels, axis=0)

    best_thresholds = np.zeros(num_classes)
    for cls_id in range(num_classes):
        best_f1 = -1
        cls_true = (all_labels_flat == cls_id).astype(int)
        if cls_true.sum() == 0:
            continue
        for t in np.arange(0.05, 0.95, 0.05):
            cls_pred = (all_probs[:, cls_id] >= t).astype(int)
            if cls_pred.sum() == 0:
                continue
            _, _, f, _ = precision_recall_fscore_support(cls_true, cls_pred, average="binary", zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresholds[cls_id] = t

    # Apply thresholds
    adjusted = all_probs - best_thresholds[np.newaxis, :]
    tuned_preds = np.argmax(adjusted, axis=1)
    tuned_metrics = evaluate_predictions(all_labels_flat, tuned_preds, num_classes, label_names)

    print(f"\nThresholds: {dict(zip(label_names, [f'{t:.2f}' for t in best_thresholds]))}")
    print("\nWith threshold tuning:")
    print(f"  Micro-F1: {tuned_metrics['micro_f1']:.4f}")
    print(f"  Macro-F1: {tuned_metrics['macro_f1']:.4f}")
    print("\nPer-class (tuned):")
    for cls in label_names:
        m = tuned_metrics["per_class"][cls]
        print(f"  {cls:<25} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

    # Save
    output = {
        "models": [m["name"] for m in ENSEMBLE_MODELS],
        "ensemble_micro": {"mean": float(np.mean(ens_micros)), "std": float(np.std(ens_micros))},
        "ensemble_macro": {"mean": float(np.mean(ens_macros)), "std": float(np.std(ens_macros))},
        "thresholds": {label_names[i]: float(best_thresholds[i]) for i in range(num_classes)},
        "tuned_micro": tuned_metrics["micro_f1"],
        "tuned_macro": tuned_metrics["macro_f1"],
        "tuned_per_class": tuned_metrics["per_class"],
        "per_fold": fold_results,
    }
    output_path = base_dir / "evaluation" / "cv_results" / "ensemble_cv_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
