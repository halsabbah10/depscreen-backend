"""
Training script for the sentence-level DSM-5 symptom classifier.

Trains a DistilBERT or MentalBERT classifier on the preprocessed ReDSM5 dataset
for 11-class symptom detection (9 DSM-5 + SPECIAL_CASE + NO_SYMPTOM).

Usage:
    python train_redsm5_model.py [options]

Options:
    --model-name: Base model (default: distilbert-base-uncased)
    --epochs: Number of training epochs (default: 5)
    --batch-size: Batch size (default: 16)
    --lr: Learning rate (default: 2e-5)
    --max-length: Max token length (default: 128)
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────


class SymptomClassifier(nn.Module):
    """Transformer-based sentence-level DSM-5 symptom classifier."""

    def __init__(self, num_classes: int = 11, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for both DistilBERT and BERT
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits


# ── Dataset ───────────────────────────────────────────────────────────────────


class SymptomDataset(Dataset):
    """Dataset for sentence-level symptom classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    """Dynamic padding — pad to longest sequence in batch, not max_length."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_masks, "label": labels}


# ── Training ──────────────────────────────────────────────────────────────────


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
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

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, label_names: list[str]):
    """Evaluate the model and return comprehensive metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="micro")
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(label_names)))
    )

    per_class_metrics = {}
    for i, name in enumerate(label_names):
        per_class_metrics[name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "per_class": per_class_metrics,
    }

    return metrics, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Train DSM-5 symptom classifier")
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="HuggingFace model name (default: distilbert-base-uncased)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "redsm5" / "processed"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device (MPS → CUDA → CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ── Load data ──
    logger.info("Loading preprocessed data...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)

    label_map = metadata["label_map"]
    label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = metadata["num_classes"]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Classes: {num_classes} ({', '.join(label_names)})")

    # ── Tokenizer ──
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Datasets ──
    train_dataset = SymptomDataset(
        train_df["clean_text"].tolist(),
        train_df["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    val_dataset = SymptomDataset(
        val_df["clean_text"].tolist(),
        val_df["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )
    test_dataset = SymptomDataset(
        test_df["clean_text"].tolist(),
        test_df["label_id"].tolist(),
        tokenizer,
        args.max_length,
    )

    # Workers: 0 for MPS (shared memory issues), 2 otherwise
    num_workers = 0 if device.type == "mps" else 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # ── Model ──
    logger.info(f"Creating model: {args.model_name} → {num_classes} classes")
    model = SymptomClassifier(num_classes=num_classes, model_name=args.model_name)
    model.to(device)

    # ── Class weights ──
    class_weights_dict = metadata["class_weights"]
    # Build weight tensor in label_id order (0, 1, 2, ..., 10)
    weight_tensor = torch.zeros(num_classes)
    for label_id_str, weight in class_weights_dict.items():
        weight_tensor[int(label_id_str)] = weight
    weight_tensor = weight_tensor.to(device)
    logger.info(f"Class weights: {weight_tensor.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── Training loop ──
    logger.info("Starting training...")
    best_val_f1 = 0
    training_history = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, label_names)
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Micro-F1: {val_metrics['micro_f1']:.4f}, "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_micro_f1": val_metrics["micro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            torch.save(model.state_dict(), output_dir / "symptom_classifier.pt")
            logger.info(f"  Saved best model (Micro-F1: {best_val_f1:.4f})")

    # ── Test evaluation ──
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "symptom_classifier.pt", map_location=device))
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, criterion, device, label_names)

    logger.info("\nTest Results:")
    logger.info(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    logger.info(f"  Micro-F1:       {test_metrics['micro_f1']:.4f}")
    logger.info(f"  Macro-F1:       {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Micro-Precision: {test_metrics['micro_precision']:.4f}")
    logger.info(f"  Micro-Recall:    {test_metrics['micro_recall']:.4f}")

    # Per-class breakdown
    print("\n" + "=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(test_labels, test_preds, target_names=label_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))
    print(cm)

    # ── Baseline comparison ──
    print("\n" + "=" * 60)
    print("Baseline Comparison (Micro-F1):")
    print("=" * 60)
    baselines = {"SVM (TF-IDF)": 0.39, "CNN": 0.25, "BERT": 0.51, "LLaMA-1B (FT)": 0.54}
    for name, score in baselines.items():
        delta = test_metrics["micro_f1"] - score
        marker = "✓" if delta > 0 else "✗"
        print(
            f"  {name}: {score:.2f}  →  Ours: {test_metrics['micro_f1']:.2f}  ({'+' if delta > 0 else ''}{delta:.2f}) {marker}"
        )

    # ── Save results ──
    results = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "best_val_micro_f1": best_val_f1,
        "test_metrics": test_metrics,
        "training_history": training_history,
        "label_map": label_map,
        "baseline_comparison": baselines,
    }

    results_name = f"redsm5_training_results_{args.model_name.replace('/', '_')}.json"
    with open(output_dir / results_name, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save metadata for inference
    inference_metadata = {
        "model_name": args.model_name,
        "num_classes": num_classes,
        "label_map": label_map,
        "label_readable": metadata["label_readable"],
        "max_length": args.max_length,
    }
    with open(output_dir / "redsm5_metadata.json", "w") as f:
        json.dump(inference_metadata, f, indent=2)

    logger.info(f"\nModel saved to: {output_dir / 'symptom_classifier.pt'}")
    logger.info(f"Results saved to: {output_dir / results_name}")
    logger.info(f"Metadata saved to: {output_dir / 'redsm5_metadata.json'}")


if __name__ == "__main__":
    main()
