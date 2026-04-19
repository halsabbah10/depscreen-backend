"""
K-Fold Cross-Validation harness for the DSM-5 symptom classifier.

Replaces single train/val/test split with stratified K-fold CV at the post level.
All metrics reported as mean ± std across folds.

Usage:
    python kfold_evaluate.py [options]

Options:
    --model-name: Base model (default: distilbert-base-uncased)
    --k: Number of folds (default: 5)
    --epochs: Training epochs per fold (default: 5)
    --batch-size: Batch size (default: 16)
    --lr: Learning rate (default: 2e-5)
    --max-length: Max token length (default: 128)
    --output-dir: Where to save CV results
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

# Reuse existing model/dataset/collate from training script
sys.path.insert(0, str(Path(__file__).parent))
# Reuse preprocessing functions
from preprocess_redsm5 import (
    SYMPTOM_LABELS,
    SYMPTOM_READABLE,
    clean_sentence,
    create_negative_samples,
    create_positive_samples,
    load_data,
)
from train_redsm5_model import SymptomClassifier, SymptomDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_post_label_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a multi-label binary matrix at the post level for stratification.

    Each post gets a binary vector indicating which symptoms appear in it.
    This lets iterstrat stratify by symptom distribution across posts.

    Returns:
        post_df: DataFrame with one row per unique post_id
        label_matrix: (n_posts, n_classes) binary matrix
    """
    label_names = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])

    # Group sentences by post, collect unique symptoms per post
    post_symptoms = df.groupby("post_id")["label"].apply(set).reset_index()
    post_symptoms.columns = ["post_id", "symptoms"]

    # Build binary matrix
    mlb = MultiLabelBinarizer(classes=label_names)
    label_matrix = mlb.fit_transform(post_symptoms["symptoms"])

    return post_symptoms, label_matrix


def compute_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    """Compute inverse-frequency class weights for a training fold."""
    counts = train_df["label_id"].value_counts().sort_index()
    total = len(train_df)
    n_classes = len(counts)
    weights = {}
    for label_id, count in counts.items():
        weights[int(label_id)] = total / (n_classes * count)
    return weights


def train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    device: torch.device,
    fold_idx: int,
    soft_labels: list[list[float]] | None = None,
    distill_alpha: float = 0.5,
    distill_temperature: float = 3.0,
    loss_type: str = "ce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    effective_num_beta: float = 0.999,
    hard_label_classes: list[int] | None = None,
    use_llrd: bool = False,
    use_fgm: bool = False,
    fgm_epsilon: float = 0.5,
    use_rdrop: bool = False,
    rdrop_alpha: float = 0.1,
    **kwargs,
) -> dict:
    """Train and evaluate a single fold. Returns metrics dict.

    Args:
        soft_labels: If provided, uses knowledge distillation loss.
                     List of probability vectors aligned to train_df rows.
        distill_alpha: Weight for hard labels (1-alpha for soft). Default 0.5.
        distill_temperature: Softening temperature. Default 3.0.
    """

    label_names = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
    num_classes = len(label_names)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Datasets — use distillation-aware dataset if soft labels available
    if soft_labels is not None:
        from distillation_utils import DistillationDataset, DistillationLoss, collate_fn_distill

        train_dataset = DistillationDataset(
            train_df["clean_text"].tolist(),
            train_df["label_id"].tolist(),
            soft_labels,
            tokenizer,
            max_length,
        )
        train_collate = collate_fn_distill
    else:
        train_dataset = SymptomDataset(
            train_df["clean_text"].tolist(),
            train_df["label_id"].tolist(),
            tokenizer,
            max_length,
        )
        train_collate = collate_fn

    val_dataset = SymptomDataset(
        val_df["clean_text"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        max_length,
    )

    num_workers = 0 if device.type == "mps" else 2
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=train_collate, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    # Model (fresh for each fold)
    # Pooling strategy: "cls", "mean", or "cls_mean" (concatenated)
    pooling = kwargs.get("pooling", "mean")
    model = SymptomClassifier(num_classes=num_classes, model_name=model_name, pooling=pooling)
    model.to(device)

    # Class weights for this fold's training data
    if loss_type == "effective_num" or loss_type == "focal":
        from distillation_utils import compute_effective_number_weights
        class_counts = train_df["label_id"].value_counts().to_dict()
        weight_tensor = compute_effective_number_weights(class_counts, num_classes, effective_num_beta)
        weight_tensor = weight_tensor.to(device)
        logger.info(f"  Using effective-number weights (β={effective_num_beta})")
    else:
        class_weights = compute_class_weights(train_df)
        weight_tensor = torch.zeros(num_classes)
        for label_id_str, weight in class_weights.items():
            weight_tensor[int(label_id_str)] = weight
        weight_tensor = weight_tensor.to(device)

    # Loss function selection
    if loss_type == "focal":
        if soft_labels is not None:
            from distillation_utils import FocalDistillationLoss
            criterion = FocalDistillationLoss(
                alpha=distill_alpha,
                temperature=distill_temperature,
                gamma=focal_gamma,
                class_weights=weight_tensor,
                label_smoothing=label_smoothing,
            )
            logger.info(f"  Using focal+distillation loss (γ={focal_gamma}, α={distill_alpha}, T={distill_temperature}, ls={label_smoothing})")
        else:
            from distillation_utils import FocalLoss
            criterion = FocalLoss(gamma=focal_gamma, class_weights=weight_tensor, label_smoothing=label_smoothing)
            logger.info(f"  Using focal loss (γ={focal_gamma}, ls={label_smoothing})")
    elif soft_labels is not None:
        from distillation_utils import DistillationLoss
        per_class_alpha = None
        if hard_label_classes:
            per_class_alpha = {cls_id: 1.0 for cls_id in hard_label_classes}
            logger.info(f"  Per-class hard labels for class IDs: {hard_label_classes}")
        criterion = DistillationLoss(
            alpha=distill_alpha,
            temperature=distill_temperature,
            class_weights=weight_tensor,
            per_class_alpha=per_class_alpha,
        )
        if label_smoothing > 0:
            criterion.ce_loss = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
        logger.info(f"  Using {loss_type}+distillation loss (α={distill_alpha}, T={distill_temperature}, ls={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
        logger.info(f"  Using CE loss (ls={label_smoothing})")

    # Optimizer — LLRD or standard AdamW
    if use_llrd:
        from distillation_utils import build_llrd_param_groups
        param_groups = build_llrd_param_groups(model, lr=lr, decay_factor=0.8, weight_decay=0.01)
        optimizer = AdamW(param_groups)
        logger.info("  Using LLRD (decay=0.8, wd=0.01)")
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # FGM adversarial training
    fgm = None
    if use_fgm:
        from distillation_utils import FGM
        fgm = FGM(model, epsilon=fgm_epsilon)
        logger.info(f"  Using FGM adversarial training (ε={fgm_epsilon})")

    if use_rdrop:
        from distillation_utils import compute_rdrop_loss
        logger.info(f"  Using R-Drop regularization (α={rdrop_alpha})")

    # SWA: accumulate weights from good epochs
    use_swa = kwargs.get("swa", False)
    swa_start_epoch = max(2, epochs // 2)  # Start averaging from halfway
    swa_states = []
    if use_swa:
        logger.info(f"  Using SWA (averaging from epoch {swa_start_epoch+1})")

    # Threshold tuning
    use_threshold_tuning = kwargs.get("threshold_tuning", False)

    # Training loop
    best_val_f1 = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            # Primary loss
            if soft_labels is not None and "soft_label" in batch:
                teacher_soft = batch["soft_label"].to(device)
                loss = criterion(logits, labels, teacher_soft)
            else:
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss = criterion(logits, labels)
                else:
                    loss = criterion(logits, labels, None)

            # R-Drop: second forward pass with different dropout mask
            if use_rdrop:
                logits2 = model(input_ids, attention_mask)
                loss = loss + compute_rdrop_loss(logits, logits2, alpha=rdrop_alpha)

            loss.backward()

            # FGM: adversarial perturbation step
            if fgm is not None:
                fgm.attack()
                logits_adv = model(input_ids, attention_mask)
                if soft_labels is not None and "soft_label" in batch:
                    loss_adv = criterion(logits_adv, labels, teacher_soft)
                else:
                    if isinstance(criterion, nn.CrossEntropyLoss):
                        loss_adv = criterion(logits_adv, labels)
                    else:
                        loss_adv = criterion(logits_adv, labels, None)
                loss_adv.backward()
                fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        _, _, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
        _, _, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

        logger.info(
            f"  Fold {fold_idx+1} Epoch {epoch+1}: "
            f"loss={total_loss/len(train_loader):.4f} "
            f"micro_f1={micro_f1:.4f} macro_f1={macro_f1:.4f}"
        )

        if micro_f1 > best_val_f1:
            best_val_f1 = micro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # SWA: collect weights from later epochs
        if use_swa and epoch >= swa_start_epoch:
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

    # SWA: average collected weights
    if use_swa and len(swa_states) >= 2:
        avg_state = {}
        for key in swa_states[0]:
            avg_state[key] = torch.stack([s[key].float() for s in swa_states]).mean(0)
            if swa_states[0][key].dtype != torch.float32:
                avg_state[key] = avg_state[key].to(swa_states[0][key].dtype)

        # Evaluate SWA model vs best checkpoint — keep whichever is better
        model.load_state_dict(avg_state)
        model.to(device)
        model.eval()
        swa_preds, swa_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                swa_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                swa_labels.extend(batch["label"].numpy())
        _, _, swa_micro, _ = precision_recall_fscore_support(swa_labels, swa_preds, average="micro")
        _, _, swa_macro, _ = precision_recall_fscore_support(swa_labels, swa_preds, average="macro")
        logger.info(f"  SWA: micro={swa_micro:.4f} macro={swa_macro:.4f} (best checkpoint: micro={best_val_f1:.4f})")

        if swa_micro >= best_val_f1:
            best_state = avg_state
            logger.info("  SWA model selected (better than best checkpoint)")
        else:
            logger.info("  Best checkpoint kept (better than SWA)")

    # Load best model and compute final metrics on this fold's val set
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Threshold tuning: optimize per-class thresholds to maximize per-class F1
    if use_threshold_tuning:
        best_thresholds = np.full(num_classes, 0.0)
        for cls_id in range(num_classes):
            best_cls_f1 = -1
            best_t = 0.0
            cls_true = (all_labels == cls_id).astype(int)
            if cls_true.sum() == 0:
                continue
            for t in np.arange(0.05, 0.95, 0.05):
                cls_pred = (all_probs[:, cls_id] >= t).astype(int)
                if cls_pred.sum() == 0:
                    continue
                p, r, f, _ = precision_recall_fscore_support(cls_true, cls_pred, average="binary", zero_division=0)
                if f > best_cls_f1:
                    best_cls_f1 = f
                    best_t = t
            best_thresholds[cls_id] = best_t

        # Apply tuned thresholds: for each sample, pick the class with highest
        # (probability - threshold) as the prediction
        adjusted_scores = all_probs - best_thresholds[np.newaxis, :]
        all_preds = np.argmax(adjusted_scores, axis=1)
        logger.info(f"  Threshold tuning: {dict(zip(label_names, [f'{t:.2f}' for t in best_thresholds]))}")
    else:
        all_preds = np.argmax(all_probs, axis=1)

    accuracy = accuracy_score(all_labels, all_preds)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0,
    )

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }

    fold_metrics = {
        "fold": fold_idx + 1,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "per_class": per_class,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }

    # Clean up GPU memory
    del model, best_state
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return fold_metrics


def aggregate_cv_results(fold_results: list[dict]) -> dict:
    """Aggregate per-fold metrics into mean ± std."""

    scalar_keys = [
        "accuracy", "micro_f1", "macro_f1",
        "micro_precision", "micro_recall",
        "macro_precision", "macro_recall",
    ]

    aggregated = {}
    for key in scalar_keys:
        values = [f[key] for f in fold_results]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values,
        }

    # Per-class aggregation
    all_class_names = list(fold_results[0]["per_class"].keys())
    per_class_agg = {}
    for cls in all_class_names:
        cls_metrics = {}
        for metric in ["precision", "recall", "f1", "support"]:
            values = [f["per_class"][cls][metric] for f in fold_results]
            cls_metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        per_class_agg[cls] = cls_metrics

    aggregated["per_class"] = per_class_agg
    return aggregated


def print_cv_report(aggregated: dict, fold_results: list[dict]):
    """Print a formatted CV report."""
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Mean':>10} {'± Std':>10}   Per-fold values")
    print("-" * 70)
    for key in ["accuracy", "micro_f1", "macro_f1", "micro_precision", "micro_recall"]:
        m = aggregated[key]
        vals = "  ".join(f"{v:.3f}" for v in m["values"])
        print(f"{key:<25} {m['mean']:>10.4f} {m['std']:>10.4f}   [{vals}]")

    print(f"\n{'Symptom':<25} {'F1 Mean':>10} {'± Std':>10} {'Prec Mean':>10} {'Rec Mean':>10} {'Avg Support':>12}")
    print("-" * 80)
    per_class = aggregated["per_class"]
    # Sort by F1 descending
    sorted_classes = sorted(per_class.keys(), key=lambda c: per_class[c]["f1"]["mean"], reverse=True)
    for cls in sorted_classes:
        m = per_class[cls]
        print(
            f"{cls:<25} {m['f1']['mean']:>10.4f} {m['f1']['std']:>10.4f} "
            f"{m['precision']['mean']:>10.4f} {m['recall']['mean']:>10.4f} "
            f"{m['support']['mean']:>12.1f}"
        )

    print("\nSamples per fold:")
    for f in fold_results:
        print(f"  Fold {f['fold']}: train={f['train_samples']}, val={f['val_samples']}")


def main():
    parser = argparse.ArgumentParser(description="K-Fold CV for DSM-5 symptom classifier")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--k", type=int, default=5, help="Number of folds")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-negatives", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--redsm5-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to pre-split cleaned data dir (with train.csv, val.csv, metadata.json)")
    parser.add_argument("--distill", type=str, default=None,
                        help="Path to train_distilled.csv for knowledge distillation")
    parser.add_argument("--distill-alpha", type=float, default=0.5,
                        help="Weight for hard labels in distillation (default: 0.5)")
    parser.add_argument("--distill-temperature", type=float, default=3.0,
                        help="Temperature for softening distributions (default: 3.0)")
    parser.add_argument("--loss-type", type=str, default="ce",
                        choices=["ce", "effective_num", "focal"],
                        help="Loss type: ce (inverse-freq), effective_num (Cui et al.), focal (Lin et al.)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (default: 0.0, recommended: 0.1)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--effective-num-beta", type=float, default=0.999,
                        help="Effective number beta (default: 0.999)")
    parser.add_argument("--augmented", type=str, default=None,
                        help="Path to augmented_samples.csv (added to training only, never validation)")
    parser.add_argument("--hard-label-classes", type=str, default=None,
                        help="Comma-separated class IDs to use hard labels only (e.g. '4,7,9' for PSYCHOMOTOR,COGNITIVE,SPECIAL)")
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean", "cls_mean"],
                        help="Pooling strategy: cls, mean (default), or cls_mean (concatenated)")
    parser.add_argument("--llrd", action="store_true", help="Enable layer-wise learning rate decay")
    parser.add_argument("--fgm", action="store_true", help="Enable FGM adversarial training")
    parser.add_argument("--fgm-epsilon", type=float, default=0.5, help="FGM perturbation epsilon")
    parser.add_argument("--rdrop", action="store_true", help="Enable R-Drop regularization")
    parser.add_argument("--rdrop-alpha", type=float, default=0.1, help="R-Drop KL weight")
    parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging")
    parser.add_argument("--threshold-tuning", action="store_true", help="Enable per-class threshold tuning")
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    redsm5_dir = Path(args.redsm5_dir) if args.redsm5_dir else project_root / "redsm5"
    base_dir = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "evaluation" / "cv_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ── Build full dataset ──
    # If --data-dir is provided with pre-split CSVs, load directly instead of re-preprocessing
    data_dir_path = Path(args.data_dir) if args.data_dir else None
    if data_dir_path and (data_dir_path / "train.csv").exists() and (data_dir_path / "metadata.json").exists():
        logger.info(f"Loading pre-split data from {data_dir_path}...")
        train_presplit = pd.read_csv(data_dir_path / "train.csv")
        val_presplit = pd.read_csv(data_dir_path / "val.csv")
        combined = pd.concat([train_presplit, val_presplit], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Total dataset: {len(combined)} samples from pre-split CSVs")
    else:
        logger.info("Loading and preprocessing ReDSM5 dataset...")
        posts, annotations = load_data(redsm5_dir)
        positives = create_positive_samples(annotations)
        negatives = create_negative_samples(posts, annotations, max_negatives=args.max_negatives)
        combined = pd.concat([positives, negatives], ignore_index=True)
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Total dataset: {len(combined)} samples, {combined['post_id'].nunique()} posts")

    # ── Build post-level multi-label matrix for stratification ──
    logger.info("Building post-level label matrix for stratification...")
    post_df, label_matrix = build_post_label_matrix(combined)

    logger.info(f"Posts: {len(post_df)}, Label matrix shape: {label_matrix.shape}")

    # Verify label matrix has signal
    label_sums = label_matrix.sum(axis=0)
    label_names_sorted = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
    for i, name in enumerate(label_names_sorted):
        logger.info(f"  {name}: {int(label_sums[i])} posts")

    # ── Load augmented data if provided ──
    augmented_df = None
    if args.augmented:
        augmented_df = pd.read_csv(args.augmented)
        # Ensure required columns exist
        required_cols = {"clean_text", "label", "label_id", "sentence_id"}
        if not required_cols.issubset(set(augmented_df.columns)):
            logger.error(f"Augmented CSV missing columns: {required_cols - set(augmented_df.columns)}")
            return
        logger.info(f"Loaded {len(augmented_df)} augmented samples (training-only)")

    # ── Load soft labels if distillation enabled ──
    distill_lookup = None
    if args.distill:
        from distillation_utils import load_soft_labels_for_df
        logger.info(f"Distillation enabled: loading soft labels from {args.distill}")
        # We load the full distilled CSV once, then subset per fold
        distilled_df = pd.read_csv(args.distill)
        label_names_sorted = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
        soft_columns = [f"soft_{name}" for name in label_names_sorted]
        # Build lookup: sentence_id → soft label vector
        distill_lookup = {}
        for _, row in distilled_df.iterrows():
            if row.get("soft_label_valid", True):
                sid = row["sentence_id"]
                distill_lookup[sid] = [float(row[c]) for c in soft_columns]
        logger.info(f"  Loaded {len(distill_lookup)} soft label vectors")

    # ── K-Fold stratified splitting at post level ──
    mskf = MultilabelStratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_post_idx, val_post_idx) in enumerate(mskf.split(post_df["post_id"], label_matrix)):
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{args.k}")
        logger.info(f"{'='*60}")

        # Map post indices back to sentence-level data
        train_post_ids = set(post_df.iloc[train_post_idx]["post_id"])
        val_post_ids = set(post_df.iloc[val_post_idx]["post_id"])

        # Verify no leakage
        assert len(train_post_ids & val_post_ids) == 0, "Post leakage between train and val!"

        train_df = combined[combined["post_id"].isin(train_post_ids)].reset_index(drop=True)
        val_df = combined[combined["post_id"].isin(val_post_ids)].reset_index(drop=True)

        # Add augmented data to training only (never validation)
        if augmented_df is not None:
            aug_with_cols = augmented_df[["post_id", "sentence_id", "sentence_text", "clean_text", "label", "label_id"]].copy()
            train_df = pd.concat([train_df, aug_with_cols], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=42 + fold_idx).reset_index(drop=True)

        logger.info(f"  Train: {len(train_df)} sentences ({len(train_post_ids)} posts + augmented)")
        logger.info(f"  Val:   {len(val_df)} sentences ({len(val_post_ids)} posts, no augmented)")

        # Log per-class distribution in val fold
        val_dist = val_df["label"].value_counts()
        min_class = val_dist.min()
        min_class_name = val_dist.idxmin()
        logger.info(f"  Smallest val class: {min_class_name} = {min_class} samples")

        # Build fold-specific soft labels if distillation enabled
        fold_soft_labels = None
        if distill_lookup is not None:
            label_names_sorted = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
            fold_soft_labels = []
            missing = 0
            for _, row in train_df.iterrows():
                sid = row["sentence_id"]
                if sid in distill_lookup:
                    fold_soft_labels.append(distill_lookup[sid])
                else:
                    # One-hot fallback
                    one_hot = [0.0] * len(label_names_sorted)
                    one_hot[int(row["label_id"])] = 1.0
                    fold_soft_labels.append(one_hot)
                    missing += 1
            if missing > 0:
                logger.warning(f"  {missing}/{len(train_df)} samples missing soft labels — one-hot fallback")

        fold_metrics = train_one_fold(
            train_df=train_df,
            val_df=val_df,
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            device=device,
            fold_idx=fold_idx,
            soft_labels=fold_soft_labels,
            distill_alpha=args.distill_alpha,
            distill_temperature=args.distill_temperature,
            loss_type=args.loss_type,
            label_smoothing=args.label_smoothing,
            focal_gamma=args.focal_gamma,
            effective_num_beta=args.effective_num_beta,
            hard_label_classes=[int(x) for x in args.hard_label_classes.split(",")] if args.hard_label_classes else None,
            pooling=args.pooling,
            swa=args.swa,
            threshold_tuning=args.threshold_tuning,
            use_llrd=args.llrd,
            use_fgm=args.fgm,
            fgm_epsilon=args.fgm_epsilon,
            use_rdrop=args.rdrop,
            rdrop_alpha=args.rdrop_alpha,
        )

        fold_results.append(fold_metrics)

        logger.info(
            f"  Fold {fold_idx+1} DONE: "
            f"micro_f1={fold_metrics['micro_f1']:.4f}, "
            f"macro_f1={fold_metrics['macro_f1']:.4f}"
        )

    # ── Aggregate results ──
    aggregated = aggregate_cv_results(fold_results)
    print_cv_report(aggregated, fold_results)

    # ── Save ──
    cv_output = {
        "config": {
            "model_name": args.model_name,
            "k_folds": args.k,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "max_negatives": args.max_negatives,
            "loss_type": args.loss_type,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma if args.loss_type == "focal" else None,
            "effective_num_beta": args.effective_num_beta if args.loss_type in ("effective_num", "focal") else None,
            "distillation": args.distill is not None,
            "distill_alpha": args.distill_alpha if args.distill else None,
            "distill_temperature": args.distill_temperature if args.distill else None,
            "total_samples": len(combined),
            "total_posts": combined["post_id"].nunique(),
        },
        "aggregated": aggregated,
        "per_fold": fold_results,
    }

    output_file = output_dir / f"cv_results_{args.model_name.replace('/', '_')}_{args.k}fold.json"
    with open(output_file, "w") as f:
        json.dump(cv_output, f, indent=2, default=str)

    logger.info(f"\nCV results saved to: {output_file}")

    # Summary line for quick reference
    micro = aggregated["micro_f1"]
    macro = aggregated["macro_f1"]
    print(f"\nFINAL: Micro-F1 = {micro['mean']:.4f} ± {micro['std']:.4f}, "
          f"Macro-F1 = {macro['mean']:.4f} ± {macro['std']:.4f}")


if __name__ == "__main__":
    main()
