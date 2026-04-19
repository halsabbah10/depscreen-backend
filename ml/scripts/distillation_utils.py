"""
Training utilities for the DSM-5 symptom classifier.

Provides:
- DistillationDataset: Dataset that returns both hard labels and soft teacher labels
- DistillationLoss: Combined CE (hard) + KL (soft) loss per Hinton et al. (2015)
- FocalLoss: Focuses on hard examples per Lin et al. (2017)
- compute_effective_number_weights: Cui et al. (CVPR 2019)
- collate_fn_distill: Collate function that handles soft labels

References:
- Hinton, Vinyals, Dean (2015) — "Distilling the Knowledge in a Neural Network"
- Lin et al. (2017) — "Focal Loss for Dense Object Detection"
- Cui et al. (2019) — "Class-Balanced Loss Based on Effective Number of Samples"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class DistillationDataset(Dataset):
    """Dataset that returns hard labels AND soft teacher distributions."""

    def __init__(
        self,
        texts: list[str],
        hard_labels: list[int],
        soft_labels: list[list[float]] | None,
        tokenizer,
        max_length: int = 128,
    ):
        """
        Args:
            texts: Input sentences
            hard_labels: Integer class labels (0-10)
            soft_labels: Teacher probability distributions (11 floats per sample).
                         If None, falls back to hard-label-only training.
            tokenizer: HuggingFace tokenizer
            max_length: Max token length
        """
        self.texts = texts
        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_soft = soft_labels is not None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.hard_labels[idx], dtype=torch.long),
        }
        if self.has_soft:
            item["soft_label"] = torch.tensor(self.soft_labels[idx], dtype=torch.float)
        return item


def collate_fn_distill(batch):
    """Dynamic padding collate that handles optional soft labels."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    result = {"input_ids": input_ids, "attention_mask": attention_masks, "label": labels}

    if "soft_label" in batch[0]:
        result["soft_label"] = torch.stack([item["soft_label"] for item in batch])

    return result


class DistillationLoss(nn.Module):
    """Combined hard-label CE + soft-label KL divergence loss.

    L = α * CE(student_logits, hard_label)
      + (1-α) * T² * KL(student_soft/T, teacher_soft/T)

    where:
      - α: weight for hard labels (default 0.5)
      - T: temperature for softening distributions (default 3.0)
      - T² scaling compensates for the reduced gradient magnitude from softened distributions

    Higher T → softer distributions → more inter-class information transferred.
    Hinton recommends T=3-20. We default to 3 (conservative for 11-class problem).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 3.0,
        class_weights: torch.Tensor | None = None,
        per_class_alpha: dict[int, float] | None = None,
    ):
        """
        Args:
            per_class_alpha: Override alpha for specific classes.
                e.g. {7: 1.0, 4: 1.0} → use hard labels only for COGNITIVE_ISSUES (7) and PSYCHOMOTOR (4).
                Classes not in this dict use the default alpha.
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.per_class_alpha = per_class_alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_soft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: Raw logits from student model (batch, num_classes)
            hard_labels: Integer class labels (batch,)
            teacher_soft: Teacher probability distribution (batch, num_classes).
                          If None, falls back to CE-only.

        Returns:
            Combined loss scalar
        """
        # Hard label loss (always computed)
        ce = self.ce_loss(student_logits, hard_labels)

        if teacher_soft is None:
            return ce

        # Per-class alpha: mask out distillation for unreliable classes
        if self.per_class_alpha is not None:
            # Build per-sample alpha based on each sample's hard label
            batch_alpha = torch.full((hard_labels.size(0),), self.alpha, device=hard_labels.device)
            for cls_id, cls_alpha in self.per_class_alpha.items():
                mask = hard_labels == cls_id
                batch_alpha[mask] = cls_alpha
            # Average alpha for this batch
            effective_alpha = batch_alpha.mean().item()
        else:
            effective_alpha = self.alpha

        # Soft label loss via KL divergence
        T = self.temperature

        # Student: log_softmax at temperature T
        student_log_soft = F.log_softmax(student_logits / T, dim=1)

        # Teacher: already probabilities, soften with temperature
        teacher_log = torch.log(teacher_soft.clamp(min=1e-8))
        teacher_tempered = F.softmax(teacher_log / T, dim=1)

        # KL divergence (input=log_probs, target=probs)
        kl = F.kl_div(student_log_soft, teacher_tempered, reduction="batchmean")

        # T² scaling per Hinton et al.
        kl_scaled = kl * (T * T)

        # Combined loss with effective alpha
        loss = effective_alpha * ce + (1 - effective_alpha) * kl_scaled
        return loss


def load_soft_labels_for_df(
    train_df,
    distilled_path,
    label_order: list[str],
) -> list[list[float]] | None:
    """Load soft labels from distilled CSV, aligned to train_df rows.

    Args:
        train_df: Training DataFrame with sentence_id column
        distilled_path: Path to train_distilled.csv
        label_order: Ordered list of label names matching label_id 0-10

    Returns:
        List of soft label vectors (one per row in train_df), or None if not available.
    """
    from pathlib import Path

    import pandas as pd

    path = Path(distilled_path)
    if not path.exists():
        return None

    distilled_df = pd.read_csv(path)

    # Build lookup: sentence_id → soft label vector
    soft_columns = [f"soft_{name}" for name in label_order]

    # Check all columns exist
    missing = [c for c in soft_columns if c not in distilled_df.columns]
    if missing:
        return None

    lookup = {}
    for _, row in distilled_df.iterrows():
        if row.get("soft_label_valid", True):
            sid = row["sentence_id"]
            probs = [float(row[c]) for c in soft_columns]
            lookup[sid] = probs

    # Align to train_df order
    soft_labels = []
    missing_count = 0
    for _, row in train_df.iterrows():
        sid = row["sentence_id"]
        if sid in lookup:
            soft_labels.append(lookup[sid])
        else:
            # Fallback: one-hot from hard label (no distillation benefit, but no crash)
            one_hot = [0.0] * len(label_order)
            one_hot[int(row["label_id"])] = 1.0
            soft_labels.append(one_hot)
            missing_count += 1

    if missing_count > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"  {missing_count}/{len(train_df)} samples missing soft labels — using one-hot fallback"
        )

    return soft_labels


# ── Effective Number Weights (Cui et al., CVPR 2019) ─────────────────────────


def compute_effective_number_weights(
    class_counts: dict[int, int],
    num_classes: int,
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute class weights using the effective number of samples.

    w_i = (1 - β) / (1 - β^n_i)

    where n_i is the number of samples in class i.
    β=0.999 is the standard choice (Cui et al.).

    Less aggressive than inverse-frequency: doesn't over-weight tiny classes
    or over-penalize large classes. Rescues collapsed classes like NO_SYMPTOM.
    """
    weights = torch.zeros(num_classes)
    for label_id in range(num_classes):
        n = class_counts.get(label_id, 1)
        effective_n = 1.0 - (beta ** n)
        weights[label_id] = (1.0 - beta) / effective_n

    # Normalize so weights sum to num_classes (same scale as inverse-freq)
    weights = weights / weights.sum() * num_classes
    return weights


# ── Focal Loss (Lin et al., 2017) ───────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    L = -α_t * (1 - p_t)^γ * log(p_t)

    where:
      - p_t is the predicted probability for the true class
      - γ (gamma) is the focusing parameter (default 2.0)
      - α_t is the class weight (optional)

    γ=0 reduces to standard CE. Higher γ → more focus on hard examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (batch, num_classes)
            labels: Integer class labels (batch,)
        """
        num_classes = logits.size(1)

        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_labels = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth_labels = F.one_hot(labels, num_classes).float()

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - probs) ** self.gamma

        # Per-sample loss
        loss = -focal_weight * smooth_labels * log_probs

        # Apply class weights
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(logits.device)
            loss = loss * weight_tensor.unsqueeze(0)

        return loss.sum(dim=1).mean()


class FocalDistillationLoss(nn.Module):
    """Focal Loss + KL Distillation combined.

    L = α * FocalLoss(student, hard_label)
      + (1-α) * T² * KL(student/T, teacher/T)
    """

    def __init__(
        self,
        alpha: float = 0.6,
        temperature: float = 3.0,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.focal_loss = FocalLoss(gamma=gamma, class_weights=class_weights, label_smoothing=label_smoothing)

    def forward(
        self,
        student_logits: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_soft: torch.Tensor | None = None,
    ) -> torch.Tensor:
        focal = self.focal_loss(student_logits, hard_labels)

        if teacher_soft is None:
            return focal

        T = self.temperature
        student_log_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_log = torch.log(teacher_soft.clamp(min=1e-8))
        teacher_tempered = F.softmax(teacher_log / T, dim=1)
        kl = F.kl_div(student_log_soft, teacher_tempered, reduction="batchmean")
        kl_scaled = kl * (T * T)

        return self.alpha * focal + (1 - self.alpha) * kl_scaled


# ── Layer-wise Learning Rate Decay (LLRD) ────────────────────────────────────


def build_llrd_param_groups(
    model,
    lr: float = 2e-5,
    decay_factor: float = 0.8,
    weight_decay: float = 0.01,
) -> list[dict]:
    """Build parameter groups with layer-wise learning rate decay.

    Lower encoder layers get smaller learning rates (they encode general
    language knowledge), higher layers get larger rates (task-specific).

    For DistilBERT (6 layers):
      Layer 0: lr * decay^5 = lr * 0.328  (most general)
      Layer 1: lr * decay^4 = lr * 0.410
      Layer 2: lr * decay^3 = lr * 0.512
      Layer 3: lr * decay^2 = lr * 0.640
      Layer 4: lr * decay^1 = lr * 0.800
      Layer 5: lr * decay^0 = lr * 1.000  (most task-specific)
      Classifier: lr * 1.0

    Reference: Standard transformer fine-tuning practice.
    """
    param_groups = []
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}

    # Encoder layers
    num_layers = 6  # DistilBERT has 6 transformer layers
    for layer_idx in range(num_layers):
        layer_lr = lr * (decay_factor ** (num_layers - 1 - layer_idx))
        layer_name = f"encoder.transformer.layer.{layer_idx}"

        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if layer_name in name:
                if any(nd in name for nd in no_decay):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        if decay_params:
            param_groups.append({"params": decay_params, "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0})

    # Embeddings (lowest lr)
    emb_lr = lr * (decay_factor ** num_layers)
    emb_decay = []
    emb_no_decay = []
    for name, param in model.named_parameters():
        if "embeddings" in name:
            if any(nd in name for nd in no_decay):
                emb_no_decay.append(param)
            else:
                emb_decay.append(param)
    if emb_decay:
        param_groups.append({"params": emb_decay, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_no_decay:
        param_groups.append({"params": emb_no_decay, "lr": emb_lr, "weight_decay": 0.0})

    # Classifier head (full lr)
    clf_decay = []
    clf_no_decay = []
    for name, param in model.named_parameters():
        if "classifier" in name or "dropout" in name:
            if any(nd in name for nd in no_decay):
                clf_no_decay.append(param)
            else:
                clf_decay.append(param)
    if clf_decay:
        param_groups.append({"params": clf_decay, "lr": lr, "weight_decay": weight_decay})
    if clf_no_decay:
        param_groups.append({"params": clf_no_decay, "lr": lr, "weight_decay": 0.0})

    return param_groups


# ── Fast Gradient Method (FGM) Adversarial Training ──────────────────────────


class FGM:
    """Fast Gradient Method for adversarial training.

    Adds small perturbations to word embeddings during training,
    making the model robust to input variations.

    Usage:
        fgm = FGM(model)
        # normal forward + backward
        loss.backward()
        fgm.attack()  # perturb embeddings
        loss_adv = criterion(model(input), label)
        loss_adv.backward()
        fgm.restore()  # restore original embeddings
        optimizer.step()

    Reference: Miyato et al. (2017) — Adversarial Training Methods
    """

    def __init__(self, model, epsilon: float = 0.5, emb_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Add adversarial perturbation to embedding weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """Restore original embedding weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# ── R-Drop (Regularized Dropout) ─────────────────────────────────────────────


def compute_rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Compute R-Drop KL divergence regularization.

    Runs the same input through the model twice with different dropout masks,
    then minimizes KL divergence between the two outputs.

    L_rdrop = alpha * (KL(p1 || p2) + KL(p2 || p1)) / 2

    Reference: Liang et al. (2021) — "R-Drop: Regularized Dropout for Neural Networks"
    """
    p1 = F.log_softmax(logits1, dim=1)
    p2 = F.log_softmax(logits2, dim=1)

    kl_1 = F.kl_div(p1, p2.exp(), reduction="batchmean")
    kl_2 = F.kl_div(p2, p1.exp(), reduction="batchmean")

    return alpha * (kl_1 + kl_2) / 2
