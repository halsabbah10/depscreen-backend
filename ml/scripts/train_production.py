"""
Train final production models on 100% of cleaned data.

Trains all 3 ensemble models (DAPT'd DistilBERT, RoBERTa, DeBERTa)
on the FULL dataset (train + val combined), saves weights, thresholds,
and metadata for deployment.

No validation holdout — CV already provided the performance estimate.
The production models see ALL available data for maximum performance.

Usage:
    python train_production.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_redsm5 import SYMPTOM_LABELS, SYMPTOM_READABLE
from train_redsm5_model import SymptomClassifier, SymptomDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODELS = [
    {
        "name": "./ml/models/v2_dapt_base",
        "label": "dapt_distilbert",
        "batch_size": 16,
    },
    {
        "name": "roberta-base",
        "label": "roberta",
        "batch_size": 16,
    },
    {
        "name": "microsoft/deberta-base",
        "label": "deberta",
        "batch_size": 4,
    },
]


def train_model(train_df, model_name, label, epochs, batch_size, lr, max_length, device, output_dir):
    """Train a single model on full data and save weights."""
    label_names = sorted(SYMPTOM_LABELS.keys(), key=lambda x: SYMPTOM_LABELS[x])
    num_classes = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = SymptomDataset(train_df["clean_text"].tolist(), train_df["label_id"].tolist(), tokenizer, max_length)

    num_workers = 0 if device.type == "mps" else 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)

    model = SymptomClassifier(num_classes=num_classes, model_name=model_name, pooling="mean")
    model.to(device)

    # Effective-number weights
    from distillation_utils import compute_effective_number_weights
    class_counts = train_df["label_id"].value_counts().to_dict()
    weight_tensor = compute_effective_number_weights(class_counts, num_classes, 0.999).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    # Train — no validation (full data, CV already gave performance estimate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in tqdm(loader, desc=f"  {label} epoch {epoch+1}/{epochs}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

        from sklearn.metrics import accuracy_score
        train_acc = accuracy_score(all_labels, all_preds)
        logger.info(f"  {label} epoch {epoch+1}: loss={total_loss/len(loader):.4f} train_acc={train_acc:.4f}")

    # Save model
    model_dir = output_dir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    tokenizer.save_pretrained(str(model_dir))

    # Save model config
    config = {
        "model_name": model_name,
        "label": label,
        "num_classes": num_classes,
        "pooling": "mean",
        "max_length": max_length,
        "hidden_size": model.encoder.config.hidden_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "training_samples": len(train_df),
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"  Saved {label} to {model_dir}")

    del model
    import gc
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    return model_dir


def main():
    base_dir = Path(__file__).parent.parent
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load ALL data (train + val combined)
    data_dir = base_dir / "data" / "redsm5" / "cleaned_v2"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    # Combine — production model sees everything
    cols = ["post_id", "sentence_id", "sentence_text", "clean_text", "label", "label_id"]
    full_data = pd.concat([train_df[cols], val_df[cols]], ignore_index=True)

    # Add augmented data
    aug_path = base_dir / "data" / "redsm5" / "augmented" / "augmented_samples_v2.csv"
    if aug_path.exists():
        aug_df = pd.read_csv(aug_path)
        full_data = pd.concat([full_data, aug_df[cols]], ignore_index=True)

    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Training on {len(full_data)} samples (train + val + augmented)")
    logger.info("Class distribution:")
    for label, count in full_data["label"].value_counts().sort_values().items():
        logger.info(f"  {label}: {count}")

    # Output directory
    output_dir = base_dir / "models" / "v_production_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train each model
    model_dirs = []
    for model_cfg in MODELS:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_cfg['label']} on full data")
        model_dir = train_model(
            full_data, model_cfg["name"], model_cfg["label"],
            epochs=7, batch_size=model_cfg["batch_size"],
            lr=3e-5, max_length=128, device=device,
            output_dir=output_dir,
        )
        model_dirs.append(model_dir)

    # Save ensemble metadata
    # Load thresholds from CV results
    cv_results_path = base_dir / "evaluation" / "cv_results" / "ensemble_cv_results.json"
    thresholds = {}
    if cv_results_path.exists():
        with open(cv_results_path) as f:
            cv_results = json.load(f)
        thresholds = cv_results.get("thresholds", {})

    ensemble_meta = {
        "type": "soft_vote_ensemble",
        "models": [
            {"name": m["name"], "label": m["label"], "dir": str(output_dir / m["label"])}
            for m in MODELS
        ],
        "label_map": SYMPTOM_LABELS,
        "label_readable": SYMPTOM_READABLE,
        "num_classes": len(SYMPTOM_LABELS),
        "pooling": "mean",
        "max_length": 128,
        "thresholds": thresholds,
        "cv_performance": {
            "ensemble_micro_f1": "0.813 ± 0.010",
            "ensemble_macro_f1": "0.770 ± 0.017",
            "threshold_tuned_micro_f1": 0.820,
            "threshold_tuned_macro_f1": 0.792,
            "note": "Threshold-tuned metrics have slight optimistic bias (tuned on eval data). True performance is between raw ensemble and tuned metrics.",
        },
        "training_config": {
            "data": "cleaned_v2 (train + val) + augmented_v2 (196 samples)",
            "total_samples": len(full_data),
            "epochs": 7,
            "lr": 3e-5,
            "loss": "CrossEntropyLoss (effective-number weights, label_smoothing=0.1)",
            "pooling": "mean",
        },
        "data_provenance": {
            "original_dataset": "ReDSM5 (CIKM 2025), 1,484 Reddit posts, 2,058 annotations",
            "cleaning": "Conflict resolution (53 sentences), dedup (20), confident learning (66 relabeled, 96 removed), manual fixes (9)",
            "augmentation": "196 samples via Gemini 2.5 Flash paraphrasing, similarity filtered [0.70, 0.95]",
            "dapt": "Domain-adaptive pre-training on 39K Reddit mental health posts (perplexity 16.90→7.59)",
        },
    }

    with open(output_dir / "ensemble_metadata.json", "w") as f:
        json.dump(ensemble_meta, f, indent=2)

    print(f"\n{'='*60}")
    print("PRODUCTION MODELS TRAINED")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Models: {', '.join(m['label'] for m in MODELS)}")
    print(f"Training samples: {len(full_data)}")
    print(f"Thresholds saved: {bool(thresholds)}")
    print("\nFiles:")
    for m in MODELS:
        d = output_dir / m["label"]
        print(f"  {d}/model.pt")
    print(f"  {output_dir}/ensemble_metadata.json")


if __name__ == "__main__":
    main()
