"""
Model inference service for DSM-5 symptom detection.

Loads the trained SymptomClassifier and provides sentence-level symptom
prediction with post-level aggregation.
"""

import asyncio
import json
import logging
import re

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from app.core.config import Settings
from app.schemas.analysis import PostSymptomSummary, SymptomDetection

logger = logging.getLogger(__name__)


# ── Model Architecture ────────────────────────────────────────────────────────


class SymptomClassifier(nn.Module):
    """Sentence-level DSM-5 symptom classifier (matches training architecture)."""

    def __init__(self, num_classes: int = 11, model_name: str = "distilbert-base-uncased", pooling: str = "mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.pooling = pooling

        if pooling == "cls_mean":
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0]

        if self.pooling == "mean" or self.pooling == "cls_mean":
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            mean_output = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

        if self.pooling == "cls_mean":
            pooled = torch.cat([cls_output, mean_output], dim=1)
        elif self.pooling == "mean":
            pooled = mean_output
        else:
            pooled = cls_output

        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits


# ── Sentence Splitting ────────────────────────────────────────────────────────


def split_into_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter for English and Arabic informal text."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    # Normalize line breaks into sentence boundaries
    text = re.sub(r"\n+", ". ", text)
    # Split on sentence-ending punctuation followed by space + start of new sentence.
    # Supports English (.!?) and Arabic (؟) punctuation.
    # \u0600-\u06FF covers Arabic Unicode block.
    parts = re.split(r"(?<=[.!?؟])\s+(?=[A-Z\"\u0600-\u06FF])", text)

    sentences = []
    for part in parts:
        part = part.strip()
        if len(part) >= 5:
            sentences.append(part)

    if not sentences and len(text.strip()) >= 5:
        sentences = [text.strip()]

    return sentences


# ── Human-Readable Labels ─────────────────────────────────────────────────────

SYMPTOM_READABLE = {
    "DEPRESSED_MOOD": "Depressed Mood",
    "ANHEDONIA": "Loss of Interest / Pleasure",
    "APPETITE_CHANGE": "Appetite / Weight Change",
    "SLEEP_ISSUES": "Sleep Disturbance",
    "PSYCHOMOTOR": "Psychomotor Changes",
    "FATIGUE": "Fatigue / Loss of Energy",
    "WORTHLESSNESS": "Worthlessness / Guilt",
    "COGNITIVE_ISSUES": "Difficulty Concentrating",
    "SUICIDAL_THOUGHTS": "Suicidal Ideation",
    "SPECIAL_CASE": "Other Clinical Indicator",
    "NO_SYMPTOM": "No Symptom Detected",
}

DSM5_CRITERIA = [
    "DEPRESSED_MOOD",
    "ANHEDONIA",
    "APPETITE_CHANGE",
    "SLEEP_ISSUES",
    "PSYCHOMOTOR",
    "FATIGUE",
    "WORTHLESSNESS",
    "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS",
]

# Safety floor: if SUICIDAL_THOUGHTS raw probability exceeds this, always
# predict it regardless of per-class threshold adjustment.  A depression
# screening tool must NEVER down-rank suicidal ideation.
SUICIDAL_SAFETY_FLOOR = 0.15

SYMPTOM_KEYWORDS = {
    "DEPRESSED_MOOD": ["sad", "depressed", "crying", "hopeless", "miserable", "unhappy"],
    "ANHEDONIA": ["no interest", "don't enjoy", "can't enjoy", "nothing matters", "don't care"],
    "APPETITE_CHANGE": ["not eating", "no appetite", "eating too much", "weight"],
    "SLEEP_ISSUES": ["can't sleep", "insomnia", "sleeping too much", "tired all day", "sleep"],
    "PSYCHOMOTOR": ["can't sit still", "restless", "slowed down", "can't move"],
    "FATIGUE": ["exhausted", "no energy", "fatigue", "tired", "drained"],
    "WORTHLESSNESS": ["worthless", "guilty", "failure", "hate myself", "useless"],
    "COGNITIVE_ISSUES": ["can't concentrate", "can't focus", "can't think", "indecisive", "brain fog"],
    "SUICIDAL_THOUGHTS": ["kill myself", "suicide", "want to die", "end it", "not worth living"],
}


# ── Severity Computation ──────────────────────────────────────────────────────


def compute_severity(unique_dsm5_count: int) -> dict:
    """Map symptom count to screening severity level.

    Note: These are screening indicators, not clinical diagnoses.
    The level keys (none/mild/moderate/severe) are used throughout
    the codebase as enum-like values — do not rename them.
    """
    if unique_dsm5_count == 0:
        return {
            "level": "none",
            "explanation": "No depression-related patterns were detected in this text.",
        }
    elif unique_dsm5_count <= 2:
        return {
            "level": "mild",
            "explanation": (
                f"{unique_dsm5_count} pattern(s) detected. "
                "Below the threshold that typically warrants further evaluation. "
                "Continued self-monitoring is encouraged."
            ),
        }
    elif unique_dsm5_count <= 4:
        return {
            "level": "moderate",
            "explanation": (
                f"{unique_dsm5_count} patterns detected. This suggests it may be helpful to speak with a professional."
            ),
        }
    else:
        return {
            "level": "severe",
            "explanation": (
                f"{unique_dsm5_count} patterns detected. "
                "This screening suggests speaking with a mental health professional "
                "would be beneficial."
            ),
        }


# ── Model Service ─────────────────────────────────────────────────────────────


class ModelService:
    """Service for loading the trained model(s) and running symptom inference.

    Supports both single-model and ensemble mode.
    Ensemble mode is activated when model_path contains ensemble_metadata.json.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self._get_device()
        self.symptom_model: SymptomClassifier | None = None
        self.tokenizer = None
        self.label_map: dict[int, str] = {}
        self.num_classes: int = 11
        self.max_length: int = 128
        self.model_name: str = "distilbert-base-uncased"

        # Ensemble support
        self.is_ensemble: bool = False
        self.ensemble_models: list[SymptomClassifier] = []
        self.ensemble_tokenizers: list = []
        self.thresholds: np.ndarray | None = None

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    async def _download_models_if_needed(self):
        """Download ensemble models from HuggingFace if not present locally."""
        model_dir = self.settings.model_path
        ensemble_meta = model_dir / "ensemble_metadata.json"

        if not ensemble_meta.exists():
            logger.info("No ensemble metadata found locally — skipping HF download")
            return

        with open(ensemble_meta) as f:
            meta = json.load(f)

        missing = []
        for model_info in meta.get("models", []):
            pt_path = model_dir / model_info["label"] / "model.pt"
            if not pt_path.exists():
                missing.append(model_info["label"])

        if not missing:
            logger.info("All ensemble model weights present locally")
            return

        logger.info(f"Downloading missing model weights from {self.settings.hf_model_repo}: {missing}")
        try:
            from huggingface_hub import hf_hub_download

            for label in missing:
                pt_file = f"{label}/model.pt"
                logger.info(f"  Downloading {pt_file}...")
                hf_hub_download(
                    repo_id=self.settings.hf_model_repo,
                    filename=pt_file,
                    local_dir=str(model_dir),
                )
                logger.info(f"  Downloaded {pt_file}")
        except Exception as e:
            logger.error(f"Failed to download models from HF: {e}")

    async def load_models(self):
        """Load the trained symptom classifier(s) and metadata.

        If model_path contains ensemble_metadata.json, loads all ensemble
        models and enables soft-vote averaging at inference time.
        Downloads missing weights from HuggingFace if needed.
        """
        await self._download_models_if_needed()
        model_dir = self.settings.model_path

        # Check for ensemble
        ensemble_meta_path = model_dir / "ensemble_metadata.json"
        if ensemble_meta_path.exists():
            await self._load_ensemble(model_dir, ensemble_meta_path)
            return

        # Load metadata (single model)
        metadata_path = model_dir / self.settings.symptom_metadata_name
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.label_map = {int(v): k for k, v in metadata["label_map"].items()}
            self.num_classes = metadata.get("num_classes", 11)
            self.max_length = metadata.get("max_length", 128)
            self.model_name = metadata.get("model_name", "distilbert-base-uncased")
            logger.info(f"Loaded metadata: {self.num_classes} classes, model={self.model_name}")
        else:
            logger.warning(f"Metadata not found at {metadata_path}, using defaults")
            self.label_map = {
                i: name
                for name, i in {
                    "DEPRESSED_MOOD": 0,
                    "ANHEDONIA": 1,
                    "APPETITE_CHANGE": 2,
                    "SLEEP_ISSUES": 3,
                    "PSYCHOMOTOR": 4,
                    "FATIGUE": 5,
                    "WORTHLESSNESS": 6,
                    "COGNITIVE_ISSUES": 7,
                    "SUICIDAL_THOUGHTS": 8,
                    "SPECIAL_CASE": 9,
                    "NO_SYMPTOM": 10,
                }.items()
            }

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info(f"Tokenizer loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

        # Load model weights
        model_path = model_dir / self.settings.symptom_model_name
        if model_path.exists():
            try:
                self.symptom_model = SymptomClassifier(
                    num_classes=self.num_classes,
                    model_name=self.model_name,
                )
                state_dict = torch.load(model_path, map_location=self.device)
                self.symptom_model.load_state_dict(state_dict)
                self.symptom_model.to(self.device)
                self.symptom_model.eval()
                logger.info(f"Symptom classifier loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load symptom classifier: {e}")
                self.symptom_model = None
        else:
            logger.warning(f"Model not found at {model_path} — using demo mode")

    async def _load_ensemble(self, model_dir, meta_path):
        """Load all ensemble models and thresholds."""
        with open(meta_path) as f:
            meta = json.load(f)

        self.is_ensemble = True
        self.num_classes = meta.get("num_classes", 11)
        self.max_length = meta.get("max_length", 128)

        # Label map
        label_map_raw = meta.get("label_map", {})
        self.label_map = {int(v): k for k, v in label_map_raw.items()}

        # Thresholds
        thresholds_dict = meta.get("thresholds", {})
        if thresholds_dict:
            label_names = sorted(label_map_raw.keys(), key=lambda x: label_map_raw[x])
            self.thresholds = np.array([float(thresholds_dict.get(n, 0.0)) for n in label_names])
            logger.info("Loaded per-class thresholds")

        # Load each model
        for model_info in meta.get("models", []):
            model_name = model_info["name"]
            model_label = model_info["label"]
            sub_dir = model_dir / model_label

            try:
                # Load tokenizer from saved dir, model architecture from original name
                tokenizer = AutoTokenizer.from_pretrained(str(sub_dir))
                model = SymptomClassifier(
                    num_classes=self.num_classes,
                    model_name=model_name,  # Use original model name for architecture
                    pooling=meta.get("pooling", "mean"),
                )
                # Load to CPU first, then move to device (avoids MPS alignment bug)
                state_dict = torch.load(sub_dir / "model.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)
                model.eval()

                self.ensemble_models.append(model)
                self.ensemble_tokenizers.append(tokenizer)
                logger.info(f"Loaded ensemble model: {model_label} from {sub_dir}")
            except Exception as e:
                import traceback

                logger.error(f"Failed to load ensemble model {model_label}: {e}")
                logger.error(traceback.format_exc())

        if self.ensemble_models:
            # Set primary tokenizer/model for backward compatibility
            self.tokenizer = self.ensemble_tokenizers[0]
            self.symptom_model = self.ensemble_models[0]
            logger.info(f"Ensemble loaded: {len(self.ensemble_models)} models")
        else:
            logger.error("No ensemble models loaded — falling back to demo mode")

    async def unload_models(self):
        """Release model resources."""
        self.symptom_model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self.symptom_model is not None and self.tokenizer is not None

    async def predict_symptoms(self, text: str) -> PostSymptomSummary:
        """Async entry point: offloads blocking PyTorch inference to a thread."""
        return await asyncio.to_thread(self._predict_symptoms_sync, text)

    def _predict_symptoms_sync(self, text: str) -> PostSymptomSummary:
        """Main inference: split text → classify each sentence → aggregate."""
        if not self.is_loaded:
            logger.warning("Model not loaded — falling back to demo prediction")
            return self._demo_symptom_prediction(text)

        sentences = split_into_sentences(text)
        if not sentences:
            return PostSymptomSummary(
                symptoms_detected=[],
                unique_symptom_count=0,
                total_sentences_analyzed=0,
                severity_level="none",
                severity_explanation="No analyzable text found.",
                dsm5_criteria_met=[],
            )

        detections: list[SymptomDetection] = []

        for i, sentence in enumerate(sentences):
            if self.is_ensemble and len(self.ensemble_models) > 1:
                # Ensemble: average softmax probabilities across models
                all_probs = []
                for model, tok in zip(self.ensemble_models, self.ensemble_tokenizers, strict=False):
                    encoding = tok(
                        sentence,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_ids = encoding["input_ids"].to(self.device)
                    attention_mask = encoding["attention_mask"].to(self.device)
                    with torch.no_grad():
                        logits = model(input_ids, attention_mask)
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    all_probs.append(probs)

                avg_probs = np.mean(all_probs, axis=0)

                # Safety override: if SUICIDAL_THOUGHTS raw probability
                # exceeds the safety floor, force that prediction.
                suicidal_idx = next(
                    (idx for idx, name in self.label_map.items() if name == "SUICIDAL_THOUGHTS"),
                    None,
                )
                if suicidal_idx is not None and avg_probs[suicidal_idx] >= SUICIDAL_SAFETY_FLOOR:
                    # Suicidal ideation detected — never allow thresholds to override this
                    pred_class = suicidal_idx
                elif self.thresholds is not None:
                    adjusted = avg_probs - self.thresholds
                    pred_class = int(np.argmax(adjusted))
                else:
                    pred_class = int(np.argmax(avg_probs))

                confidence = float(avg_probs[pred_class])
            else:
                # Single model
                encoding = self.tokenizer(
                    sentence,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    padding=True,
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)
                with torch.no_grad():
                    logits = self.symptom_model(input_ids, attention_mask)
                    probs = torch.softmax(logits, dim=1)

                    # Safety override: same suicidal safety floor as ensemble path
                    probs_np = probs[0].cpu().numpy()
                    suicidal_idx = next(
                        (idx for idx, name in self.label_map.items() if name == "SUICIDAL_THOUGHTS"),
                        None,
                    )
                    if suicidal_idx is not None and probs_np[suicidal_idx] >= SUICIDAL_SAFETY_FLOOR:
                        pred_class = suicidal_idx
                        confidence = float(probs_np[suicidal_idx])
                    else:
                        pred_class = torch.argmax(probs, dim=1).item()
                        confidence = probs[0, pred_class].item()

            symptom_name = self.label_map.get(pred_class, "NO_SYMPTOM")

            # Minimum confidence threshold to reduce false positives
            # (NO_SYMPTOM class has weak F1 0.508 — low-confidence predictions are unreliable)
            MIN_SYMPTOM_CONFIDENCE = 0.40
            if symptom_name != "NO_SYMPTOM" and confidence >= MIN_SYMPTOM_CONFIDENCE:
                detections.append(
                    SymptomDetection(
                        symptom=symptom_name,
                        symptom_label=SYMPTOM_READABLE.get(symptom_name, symptom_name),
                        status=1,
                        confidence=round(confidence, 4),
                        sentence_text=sentence,
                        sentence_id=f"s_{i}",
                    )
                )

        unique_symptoms = set(d.symptom for d in detections)
        dsm5_met = sorted(s for s in unique_symptoms if s in DSM5_CRITERIA)
        severity = compute_severity(len(dsm5_met))

        return PostSymptomSummary(
            symptoms_detected=detections,
            unique_symptom_count=len(unique_symptoms),
            total_sentences_analyzed=len(sentences),
            severity_level=severity["level"],
            severity_explanation=severity["explanation"],
            dsm5_criteria_met=dsm5_met,
        )

    def _demo_symptom_prediction(self, text: str) -> PostSymptomSummary:
        """Keyword heuristic fallback when model is not loaded."""
        sentences = split_into_sentences(text)
        detections: list[SymptomDetection] = []
        text_lower = text.lower()

        for symptom, keywords in SYMPTOM_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matching_sent = text
                    for sent in sentences:
                        if keyword in sent.lower():
                            matching_sent = sent
                            break
                    detections.append(
                        SymptomDetection(
                            symptom=symptom,
                            symptom_label=SYMPTOM_READABLE[symptom],
                            status=1,
                            confidence=0.60,
                            sentence_text=matching_sent,
                            sentence_id="demo",
                        )
                    )
                    break

        unique_symptoms = set(d.symptom for d in detections)
        dsm5_met = sorted(s for s in unique_symptoms if s in DSM5_CRITERIA)
        severity = compute_severity(len(dsm5_met))

        return PostSymptomSummary(
            symptoms_detected=detections,
            unique_symptom_count=len(unique_symptoms),
            total_sentences_analyzed=len(sentences),
            severity_level=severity["level"],
            severity_explanation=severity["explanation"] + " (Demo mode — model not loaded)",
            dsm5_criteria_met=dsm5_met,
        )
