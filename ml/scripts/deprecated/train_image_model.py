"""
Training script for the image classifier model.

Trains a ResNet/EfficientNet-based classifier on extracted LMVD frames.

Usage:
    python train_image_model.py [options]

Options:
    --epochs: Number of training epochs (default: 10)
    --batch-size: Batch size (default: 32)
    --lr: Learning rate (default: 1e-4)
    --backbone: Model backbone (resnet50, efficientnet_b0)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageClassifier(nn.Module):
    """CNN-based image classifier with transfer learning."""

    def __init__(self, num_classes: int = 2, backbone: str = "resnet50"):
        super().__init__()

        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone.classifier = nn.Identity()
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class FrameDataset(Dataset):
    """Dataset for frame-level classification."""

    def __init__(self, data_dir: Path, split_file: Path, transform=None):
        with open(split_file) as f:
            self.samples = json.load(f)

        self.data_dir = data_dir
        self.transform = transform
        self.label_map = {"control": 0, "depressed": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = self.data_dir / sample["path"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[sample["label"]]

        return {"image": image, "label": torch.tensor(label, dtype=torch.long), "video": sample["video"]}


def get_transforms(train: bool = True):
    """Get image transforms for training or validation."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits, _ = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device, aggregate_clips: bool = True):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_videos = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            videos = batch["video"]

            logits, _ = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_videos.extend(videos)

    avg_loss = total_loss / len(dataloader)

    # Frame-level metrics
    frame_accuracy = accuracy_score(all_labels, all_preds)
    frame_precision, frame_recall, frame_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )

    try:
        frame_roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        frame_roc_auc = 0.0

    metrics = {
        "loss": avg_loss,
        "frame_accuracy": frame_accuracy,
        "frame_precision": frame_precision,
        "frame_recall": frame_recall,
        "frame_f1": frame_f1,
        "frame_roc_auc": frame_roc_auc,
    }

    # Clip-level metrics (aggregate by video)
    if aggregate_clips:
        video_probs = {}
        video_labels = {}

        for video, prob, label in zip(all_videos, all_probs, all_labels):
            if video not in video_probs:
                video_probs[video] = []
                video_labels[video] = label
            video_probs[video].append(prob)

        # Aggregate predictions (mean pooling)
        clip_preds = []
        clip_labels = []
        clip_probs = []

        for video in video_probs:
            mean_prob = np.mean(video_probs[video])
            clip_probs.append(mean_prob)
            clip_preds.append(1 if mean_prob > 0.5 else 0)
            clip_labels.append(video_labels[video])

        if len(clip_labels) > 0:
            clip_accuracy = accuracy_score(clip_labels, clip_preds)
            clip_precision, clip_recall, clip_f1, _ = precision_recall_fscore_support(
                clip_labels, clip_preds, average="binary"
            )
            try:
                clip_roc_auc = roc_auc_score(clip_labels, clip_probs)
            except ValueError:
                clip_roc_auc = 0.0

            metrics.update(
                {
                    "clip_accuracy": clip_accuracy,
                    "clip_precision": clip_precision,
                    "clip_recall": clip_recall,
                    "clip_f1": clip_f1,
                    "clip_roc_auc": clip_roc_auc,
                    "num_clips": len(clip_labels),
                }
            )

    return metrics, all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights initially")
    parser.add_argument("--unfreeze-epoch", type=int, default=3, help="Epoch to unfreeze backbone")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "lmvd" / "frames"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check for split files
    if not (data_dir / "train.json").exists():
        logger.error(f"Split files not found in {data_dir}")
        logger.error("Run extract_frames.py first to create the dataset.")
        return

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = FrameDataset(data_dir, data_dir / "train.json", get_transforms(train=True))
    val_dataset = FrameDataset(data_dir, data_dir / "val.json", get_transforms(train=False))
    test_dataset = FrameDataset(data_dir, data_dir / "test.json", get_transforms(train=False))

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Create model
    logger.info(f"Creating model with {args.backbone} backbone...")
    model = ImageClassifier(num_classes=2, backbone=args.backbone)
    model.to(device)

    # Optionally freeze backbone
    if args.freeze_backbone:
        logger.info("Freezing backbone weights...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training loop
    logger.info("Starting training...")
    best_val_f1 = 0
    training_history = []

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Unfreeze backbone if needed
        if args.freeze_backbone and epoch == args.unfreeze_epoch:
            logger.info("Unfreezing backbone weights...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            # Reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * 0.1

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Frame F1: {val_metrics['frame_f1']:.4f}, "
            f"Clip F1: {val_metrics.get('clip_f1', 0):.4f}"
        )

        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_frame_f1": val_metrics["frame_f1"],
                "val_clip_f1": val_metrics.get("clip_f1", 0),
            }
        )

        # Save best model (using clip-level F1 if available, else frame-level)
        current_f1 = val_metrics.get("clip_f1", val_metrics["frame_f1"])
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            torch.save(model.state_dict(), output_dir / "image_classifier.pt")
            logger.info(f"Saved best model with F1: {best_val_f1:.4f}")

    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "image_classifier.pt"))
    test_metrics, test_preds, test_labels, _ = evaluate(model, test_loader, criterion, device)

    logger.info("\nTest Results (Frame-level):")
    logger.info(f"  Accuracy: {test_metrics['frame_accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['frame_precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['frame_recall']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['frame_f1']:.4f}")
    logger.info(f"  ROC-AUC: {test_metrics['frame_roc_auc']:.4f}")

    if "clip_f1" in test_metrics:
        logger.info("\nTest Results (Clip-level):")
        logger.info(f"  Accuracy: {test_metrics['clip_accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['clip_precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['clip_recall']:.4f}")
        logger.info(f"  F1 Score: {test_metrics['clip_f1']:.4f}")
        logger.info(f"  ROC-AUC: {test_metrics['clip_roc_auc']:.4f}")
        logger.info(f"  Number of clips: {test_metrics['num_clips']}")

    # Print classification report
    print("\nClassification Report (Frame-level):")
    print(classification_report(test_labels, test_preds, target_names=["control", "depressed"]))

    # Save training results
    results = {
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "training_history": training_history,
        "label_map": {"control": 0, "depressed": 1},
    }

    with open(output_dir / "image_training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nModel saved to: {output_dir / 'image_classifier.pt'}")
    logger.info(f"Results saved to: {output_dir / 'image_training_results.json'}")


if __name__ == "__main__":
    main()
