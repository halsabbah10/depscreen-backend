"""Unit tests for DecisionService — pure logic, no external deps.

This is the safety-critical layer that combines DL predictions with LLM
verification. A bug here silently downgrades clinical flags, so we guard
each branch of the adjustment logic explicitly.
"""

from __future__ import annotations

import pytest

from app.schemas.analysis import (
    AdversarialCheck,
    ConfidenceAnalysis,
    EvidenceValidation,
    PerSymptomVerdict,
    PostSymptomSummary,
    SymptomDetection,
    VerificationReport,
)
from app.services.decision import DecisionService


def _summary(severity: str, symptoms: list[str] | None = None) -> PostSymptomSummary:
    detected = [
        SymptomDetection(
            symptom=s,
            symptom_label=s.replace("_", " ").title(),
            status=1,
            confidence=0.8,
            sentence_text=f"evidence for {s}",
            sentence_id=f"s{i}",
        )
        for i, s in enumerate(symptoms or ["DEPRESSED_MOOD"])
    ]
    return PostSymptomSummary(
        symptoms_detected=detected,
        unique_symptom_count=len(detected),
        total_sentences_analyzed=len(detected),
        severity_level=severity,
        severity_explanation="test",
        dsm5_criteria_met=symptoms or ["DEPRESSED_MOOD"],
    )


def _verification(
    *,
    evidence_supports: bool = True,
    adversarial: bool = False,
    trust: str = "high",
    flagged: bool = False,
) -> VerificationReport:
    return VerificationReport(
        evidence_validation=EvidenceValidation(
            evidence_supports_prediction=evidence_supports,
            coherence_score=0.9,
            flagged_for_review=flagged,
        ),
        confidence_analysis=ConfidenceAnalysis(
            should_trust_prediction=trust,
            reasoning="test",
        ),
        adversarial_check=AdversarialCheck(
            likely_adversarial=adversarial,
            adversarial_type="prompt_injection" if adversarial else None,
            authenticity_score=0.3 if adversarial else 0.95,
        ),
    )


@pytest.fixture
def service() -> DecisionService:
    return DecisionService()


# ─────────────────────────────────────────────────────────────────────────────
# Severity → prediction label mapping
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "severity,expected",
    [
        ("none", "no_indicators"),
        ("mild", "some_indicators"),
        ("moderate", "some_indicators"),
        ("severe", "significant_indicators"),
    ],
)
def test_prediction_label_tracks_severity(service, severity, expected):
    pred, _, _, _ = service.compute_final_prediction(_summary(severity), _verification())
    assert pred == expected


# ─────────────────────────────────────────────────────────────────────────────
# Confidence adjustments
# ─────────────────────────────────────────────────────────────────────────────


def test_happy_path_no_adjustment(service):
    _, conf, adjusted, _ = service.compute_final_prediction(_summary("mild"), _verification())
    assert not adjusted
    assert conf == pytest.approx(0.8, rel=1e-3)  # mean of canned symptom confidences


def test_evidence_mismatch_reduces_confidence(service):
    _, conf, adjusted, _ = service.compute_final_prediction(
        _summary("mild"),
        _verification(evidence_supports=False),
    )
    assert adjusted
    assert conf == pytest.approx(0.8 * 0.7, rel=1e-3)


def test_adversarial_heavily_reduces_confidence(service):
    _, conf, adjusted, _ = service.compute_final_prediction(
        _summary("mild"),
        _verification(adversarial=True),
    )
    assert adjusted
    assert conf == pytest.approx(0.8 * 0.3, rel=1e-3)


def test_low_trust_reduces_confidence(service):
    _, conf, adjusted, _ = service.compute_final_prediction(
        _summary("mild"),
        _verification(trust="low"),
    )
    assert adjusted
    assert conf == pytest.approx(0.8 * 0.8, rel=1e-3)


def test_confidence_clamped_to_unit_range(service):
    """Even with stacked reductions, confidence never goes negative."""
    _, conf, _, _ = service.compute_final_prediction(
        _summary("mild"),
        _verification(evidence_supports=False, adversarial=True, trust="low"),
    )
    assert 0.0 <= conf <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Flagging rules — the clinical-safety contract
# ─────────────────────────────────────────────────────────────────────────────


def test_severe_severity_always_flagged(service):
    _, _, _, flagged = service.compute_final_prediction(_summary("severe"), _verification())
    assert flagged is True


def test_suicidal_thoughts_always_flagged_regardless_of_severity(service):
    summary = _summary("mild", symptoms=["SUICIDAL_THOUGHTS"])
    _, _, _, flagged = service.compute_final_prediction(summary, _verification())
    assert flagged is True


def test_adversarial_input_flagged(service):
    _, _, _, flagged = service.compute_final_prediction(_summary("mild"), _verification(adversarial=True))
    assert flagged is True


def test_low_trust_flagged(service):
    _, _, _, flagged = service.compute_final_prediction(_summary("mild"), _verification(trust="low"))
    assert flagged is True


def test_evidence_flagged_for_review_propagates(service):
    _, _, _, flagged = service.compute_final_prediction(_summary("mild"), _verification(flagged=True))
    assert flagged is True


def test_clean_mild_case_not_flagged(service):
    _, _, _, flagged = service.compute_final_prediction(_summary("mild"), _verification())
    assert flagged is False


# ─────────────────────────────────────────────────────────────────────────────
# Empty symptoms
# ─────────────────────────────────────────────────────────────────────────────


def test_no_symptoms_gives_high_baseline_confidence(service):
    summary = PostSymptomSummary(
        symptoms_detected=[],
        unique_symptom_count=0,
        total_sentences_analyzed=5,
        severity_level="none",
        severity_explanation="No symptoms",
        dsm5_criteria_met=[],
    )
    pred, conf, adjusted, flagged = service.compute_final_prediction(summary, _verification())
    assert pred == "no_indicators"
    assert conf == pytest.approx(0.9, rel=1e-3)
    assert not adjusted
    assert not flagged


# ─────────────────────────────────────────────────────────────────────────────
# Verification summary (free-text helper used by the explanation LLM)
# ─────────────────────────────────────────────────────────────────────────────


def test_verification_summary_all_clean(service):
    out = service.get_verification_summary(_verification())
    assert "reliable" in out.lower()


def test_verification_summary_mentions_adversarial(service):
    out = service.get_verification_summary(_verification(adversarial=True))
    assert "adversarial" in out.lower() or "injection" in out.lower()


def test_verification_summary_mentions_evidence_mismatch(service):
    out = service.get_verification_summary(_verification(evidence_supports=False))
    assert "evidence" in out.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Per-symptom false positive filtering
# ─────────────────────────────────────────────────────────────────────────────


def _verification_with_verdicts(verdicts: list[PerSymptomVerdict]) -> VerificationReport:
    return VerificationReport(
        evidence_validation=EvidenceValidation(
            evidence_supports_prediction=True,
            coherence_score=0.9,
            per_symptom_verdicts=verdicts,
        ),
        confidence_analysis=ConfidenceAnalysis(should_trust_prediction="high", reasoning="test"),
        adversarial_check=AdversarialCheck(authenticity_score=0.95),
    )


def test_filter_removes_unsupported_symptoms():
    summary = _summary("moderate", symptoms=["DEPRESSED_MOOD", "ANHEDONIA", "SLEEP_ISSUES"])
    verdicts = [
        PerSymptomVerdict(symptom="DEPRESSED_MOOD", supported=True, reason="genuine"),
        PerSymptomVerdict(symptom="ANHEDONIA", supported=False, reason="negated statement"),
        PerSymptomVerdict(symptom="SLEEP_ISSUES", supported=True, reason="genuine"),
    ]
    result = DecisionService.filter_false_positives(summary, _verification_with_verdicts(verdicts))
    assert result.unique_symptom_count == 2
    symptom_codes = [d.symptom for d in result.symptoms_detected]
    assert "ANHEDONIA" not in symptom_codes
    assert "DEPRESSED_MOOD" in symptom_codes
    assert "SLEEP_ISSUES" in symptom_codes
    assert result.severity_level == "mild"  # 2 symptoms = mild


def test_filter_never_removes_suicidal_thoughts():
    summary = _summary("moderate", symptoms=["DEPRESSED_MOOD", "SUICIDAL_THOUGHTS"])
    verdicts = [
        PerSymptomVerdict(symptom="DEPRESSED_MOOD", supported=True, reason="genuine"),
        PerSymptomVerdict(symptom="SUICIDAL_THOUGHTS", supported=False, reason="LLM thinks false positive"),
    ]
    result = DecisionService.filter_false_positives(summary, _verification_with_verdicts(verdicts))
    assert result.unique_symptom_count == 2
    assert "SUICIDAL_THOUGHTS" in [d.symptom for d in result.symptoms_detected]


def test_filter_empty_verdicts_returns_unchanged():
    summary = _summary("moderate", symptoms=["DEPRESSED_MOOD", "ANHEDONIA"])
    result = DecisionService.filter_false_positives(summary, _verification())
    assert result is summary  # exact same object, not a copy


def test_filter_all_rejected():
    summary = _summary("mild", symptoms=["DEPRESSED_MOOD", "ANHEDONIA"])
    verdicts = [
        PerSymptomVerdict(symptom="DEPRESSED_MOOD", supported=False, reason="negated"),
        PerSymptomVerdict(symptom="ANHEDONIA", supported=False, reason="negated"),
    ]
    result = DecisionService.filter_false_positives(summary, _verification_with_verdicts(verdicts))
    assert result.unique_symptom_count == 0
    assert result.severity_level == "none"
    assert result.symptoms_detected == []


def test_filter_recomputes_severity():
    symptoms = ["DEPRESSED_MOOD", "ANHEDONIA", "SLEEP_ISSUES", "FATIGUE", "WORTHLESSNESS"]
    summary = _summary("severe", symptoms=symptoms)
    verdicts = [
        PerSymptomVerdict(symptom="DEPRESSED_MOOD", supported=True, reason="genuine"),
        PerSymptomVerdict(symptom="ANHEDONIA", supported=False, reason="negated"),
        PerSymptomVerdict(symptom="SLEEP_ISSUES", supported=True, reason="genuine"),
        PerSymptomVerdict(symptom="FATIGUE", supported=False, reason="negated"),
        PerSymptomVerdict(symptom="WORTHLESSNESS", supported=False, reason="negated"),
    ]
    result = DecisionService.filter_false_positives(summary, _verification_with_verdicts(verdicts))
    assert result.unique_symptom_count == 2
    assert result.severity_level == "mild"  # 2 symptoms = mild, not severe


def test_filter_unknown_symptom_in_verdict():
    summary = _summary("mild", symptoms=["DEPRESSED_MOOD"])
    verdicts = [
        PerSymptomVerdict(symptom="DEPRESSED_MOOD", supported=True, reason="genuine"),
        PerSymptomVerdict(symptom="NONEXISTENT", supported=False, reason="unknown"),
    ]
    result = DecisionService.filter_false_positives(summary, _verification_with_verdicts(verdicts))
    assert result.unique_symptom_count == 1
    assert result.symptoms_detected[0].symptom == "DEPRESSED_MOOD"
