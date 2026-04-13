"""
Model inference service for DSM-5 symptom detection.

Loads the trained SymptomClassifier and provides sentence-level symptom
prediction with post-level aggregation.
"""

import json
import logging
import re

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from app.core.config import Settings
from app.schemas.analysis import PostSymptomSummary, SymptomDetection

logger = logging.getLogger(__name__)


# ── Model Architecture ────────────────────────────────────────────────────────


class SymptomClassifier(nn.Module):
    """Sentence-level DSM-5 symptom classifier (matches training architecture)."""

    def __init__(self, num_classes: int = 11, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token
        dropped = self.dropout(pooled)
        logits = self.classifier(dropped)
        return logits


# ── Sentence Splitting ────────────────────────────────────────────────────────


def split_into_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter for informal text (Reddit-style)."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []

    # Normalize line breaks into sentence boundaries
    text = re.sub(r"\n+", ". ", text)
    # Split on sentence-ending punctuation followed by space + uppercase or end
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", text)

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
    """Map symptom count to DSM-5 severity level."""
    if unique_dsm5_count == 0:
        return {
            "level": "none",
            "explanation": "No DSM-5 depression symptoms were detected in the text.",
        }
    elif unique_dsm5_count <= 2:
        return {
            "level": "mild",
            "explanation": (
                f"{unique_dsm5_count} DSM-5 symptom(s) detected. "
                "Below the clinical threshold for Major Depressive Episode. "
                "Monitoring recommended."
            ),
        }
    elif unique_dsm5_count <= 4:
        return {
            "level": "moderate",
            "explanation": (
                f"{unique_dsm5_count} DSM-5 symptoms detected. "
                "Approaching clinical threshold. "
                "Professional evaluation is recommended."
            ),
        }
    else:
        return {
            "level": "severe",
            "explanation": (
                f"{unique_dsm5_count} DSM-5 symptoms detected. "
                "Meets or exceeds the DSM-5 threshold for Major Depressive Episode "
                "(5+ of 9 criteria). Prompt professional evaluation is strongly recommended."
            ),
        }


# ── Model Service ─────────────────────────────────────────────────────────────


class ModelService:
    """Service for loading the trained model and running symptom inference."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self._get_device()
        self.symptom_model: SymptomClassifier | None = None
        self.tokenizer = None
        self.label_map: dict[int, str] = {}
        self.num_classes: int = 11
        self.max_length: int = 128
        self.model_name: str = "distilbert-base-uncased"

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    async def load_models(self):
        """Load the trained symptom classifier and metadata."""
        model_dir = self.settings.model_path

        # Load metadata
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
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()

            symptom_name = self.label_map.get(pred_class, "NO_SYMPTOM")

            if symptom_name != "NO_SYMPTOM":
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
