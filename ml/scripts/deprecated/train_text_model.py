"""
Training script for the text classifier model.

Trains a DistilBERT-based classifier on the preprocessed Suicide-Watch dataset.

Usage:
    python train_text_model.py [options]

Options:
    --epochs: Number of training epochs (default: 3)
    --batch-size: Batch size (default: 32)
    --lr: Learning rate (default: 2e-5)
    --max-length: Max token length (default: 256)
    --model-name: Base model name (default: distilbert-base-uncased)
    --subset: Use only N samples per class for fast iteration (default: 0 = all)
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    """DistilBERT-based text classifier."""

    def __init__(self, num_classes: int = 2, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits


class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch):
    """Dynamic padding — pad to longest sequence in batch, not max_length."""
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'label': labels}


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

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

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    return metrics, all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description='Train text classifier')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--subset', type=int, default=0,
                        help='Use N samples per class for fast iteration (0 = all data)')
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "suicide_watch" / "processed"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device — prefer MPS (Apple Silicon GPU), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')

    # Subset sampling for fast iteration
    if args.subset > 0:
        logger.info(f"Subsetting to {args.subset} samples per class...")
        train_dfs = [g.sample(n=min(args.subset, len(g)), random_state=42) for _, g in train_df.groupby('label_id')]
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        val_dfs = [g.sample(n=min(args.subset // 4, len(g)), random_state=42) for _, g in val_df.groupby('label_id')]
        val_df = pd.concat(val_dfs).reset_index(drop=True)
        test_dfs = [g.sample(n=min(args.subset // 4, len(g)), random_state=42) for _, g in test_df.groupby('label_id')]
        test_df = pd.concat(test_dfs).reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create datasets
    train_dataset = TextDataset(
        train_df['clean_text'].tolist(),
        train_df['label_id'].tolist(),
        tokenizer,
        args.max_length
    )
    val_dataset = TextDataset(
        val_df['clean_text'].tolist(),
        val_df['label_id'].tolist(),
        tokenizer,
        args.max_length
    )
    test_dataset = TextDataset(
        test_df['clean_text'].tolist(),
        test_df['label_id'].tolist(),
        tokenizer,
        args.max_length
    )

    # Create dataloaders with dynamic padding and parallel workers
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=False
    )

    # Create model
    logger.info("Creating model...")
    num_classes = len(train_df['label_id'].unique())
    model = TextClassifier(num_classes=num_classes, model_name=args.model_name)
    model.to(device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training...")
    best_val_f1 = 0
    training_history = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_f1': val_metrics['f1'],
            'val_roc_auc': val_metrics['roc_auc']
        })

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), output_dir / 'text_classifier.pt')
            logger.info(f"Saved best model with F1: {best_val_f1:.4f}")

    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / 'text_classifier.pt', map_location=device))
    test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device
    )

    logger.info("\nTest Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['low_risk', 'high_risk']))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    # Save training results
    results = {
        'model_name': args.model_name,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'label_map': {'low_risk': 0, 'high_risk': 1}
    }

    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nModel saved to: {output_dir / 'text_classifier.pt'}")
    logger.info(f"Results saved to: {output_dir / 'training_results.json'}")


if __name__ == "__main__":
    main()
