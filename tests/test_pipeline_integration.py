"""Integration tests for the screening decision pipeline.

Tests DecisionService.compute_final_prediction() with realistic inputs,
verifying confidence adjustment, flagging logic, and severity mapping.
No network calls, no DB, no LLM — pure logic testing.
"""

from __future__ import annotations

from app.schemas.analysis import (
    AdversarialCheck,
    ConfidenceAnalysis,
    EvidenceValidation,
    PostSymptomSummary,
    SymptomDetection,
    VerificationReport,
)
from app.services.decision import DecisionService

# ── Helpers ───────────────────────────────────────────────────────────────────


def _symptom(code: str, confidence: float, sentence: str = "test sentence") -> SymptomDetection:
    return SymptomDetection(
        symptom=code,
        symptom_label=code.replace("_", " ").title(),
        status=1,
        confidence=confidence,
        sentence_text=sentence,
    )


def _verification(
    evidence_supports: bool = True,
    coherence: float = 0.9,
    flagged_for_review: bool = False,
    adversarial: bool = False,
    adversarial_type: str | None = None,
    authenticity: float = 0.95,
    trust: str = "high",
    reasoning: str = "Prediction appears reliable.",
    confounders: list[str] | None = None,
) -> VerificationReport:
    return VerificationReport(
        evidence_validation=EvidenceValidation(
            evidence_supports_prediction=evidence_supports,
            coherence_score=coherence,
            flagged_for_review=flagged_for_review,
        ),
        adversarial_check=AdversarialCheck(
            likely_adversarial=adversarial,
            adversarial_type=adversarial_type,
            authenticity_score=authenticity,
        ),
        confidence_analysis=ConfidenceAnalysis(
            should_trust_prediction=trust,
            reasoning=reasoning,
            potential_confounders=confounders or [],
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDecisionServicePipeline:
    """Integration tests for DecisionService.compute_final_prediction()."""

    def setup_method(self):
        self.svc = DecisionService()

    # ── Test 1: Moderate depression, clean verification ───────────────────────

    def test_moderate_depression_no_flags(self):
        """Four moderate-severity symptoms with clean verification.

        Expected: some_indicators, confidence = mean(0.92, 0.88, 0.85, 0.79),
        no adjustment, not flagged.
        """
        symptoms = [
            _symptom("DEPRESSED_MOOD", 0.92),
            _symptom("SLEEP_ISSUES", 0.88),
            _symptom("ANHEDONIA", 0.85),
            _symptom("WORTHLESSNESS", 0.79),
        ]
        summary = PostSymptomSummary(
            symptoms_detected=symptoms,
            unique_symptom_count=4,
            total_sentences_analyzed=10,
            severity_level="moderate",
            severity_explanation="Four DSM-5 criteria met at moderate severity.",
            dsm5_criteria_met=["DEPRESSED_MOOD", "SLEEP_ISSUES", "ANHEDONIA", "WORTHLESSNESS"],
        )
        verification = _verification()

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        expected_conf = (0.92 + 0.88 + 0.85 + 0.79) / 4  # 0.86
        assert prediction == "some_indicators"
        assert abs(confidence - expected_conf) < 1e-9
        assert adjusted is False
        assert flagged is False

    # ── Test 2: Severe depression, auto-flagged ───────────────────────────────

    def test_severe_depression_auto_flagged(self):
        """Severe severity with six symptoms including SUICIDAL_THOUGHTS.

        Both the severity=="severe" and SUICIDAL_THOUGHTS conditions independently
        trigger flagged=True.
        """
        symptoms = [
            _symptom("DEPRESSED_MOOD", 0.95),
            _symptom("SLEEP_ISSUES", 0.90),
            _symptom("ANHEDONIA", 0.88),
            _symptom("WORTHLESSNESS", 0.87),
            _symptom("FATIGUE", 0.85),
            _symptom("SUICIDAL_THOUGHTS", 0.82),
        ]
        summary = PostSymptomSummary(
            symptoms_detected=symptoms,
            unique_symptom_count=6,
            total_sentences_analyzed=15,
            severity_level="severe",
            severity_explanation="Six DSM-5 criteria met including suicidal ideation.",
            dsm5_criteria_met=[
                "DEPRESSED_MOOD",
                "SLEEP_ISSUES",
                "ANHEDONIA",
                "WORTHLESSNESS",
                "FATIGUE",
                "SUICIDAL_THOUGHTS",
            ],
        )
        verification = _verification()

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        assert prediction == "significant_indicators"
        assert flagged is True

    # ── Test 3: Adversarial input, confidence crushed ─────────────────────────

    def test_adversarial_input_crushes_confidence(self):
        """Three symptoms, but adversarial flag set.

        Confidence should be multiplied by 0.3 (adversarial_factor).
        adjusted=True, flagged=True.
        """
        symptoms = [
            _symptom("DEPRESSED_MOOD", 0.80),
            _symptom("SLEEP_ISSUES", 0.75),
            _symptom("ANHEDONIA", 0.70),
        ]
        summary = PostSymptomSummary(
            symptoms_detected=symptoms,
            unique_symptom_count=3,
            total_sentences_analyzed=8,
            severity_level="mild",
            severity_explanation="Three DSM-5 criteria met.",
            dsm5_criteria_met=["DEPRESSED_MOOD", "SLEEP_ISSUES", "ANHEDONIA"],
        )
        verification = _verification(
            adversarial=True,
            adversarial_type="keyword_stuffing",
            authenticity=0.15,
        )

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        base_conf = (0.80 + 0.75 + 0.70) / 3
        expected_conf = base_conf * 0.3  # adversarial_factor
        assert abs(confidence - expected_conf) < 1e-9
        assert adjusted is True
        assert flagged is True

    # ── Test 4: Evidence mismatch ─────────────────────────────────────────────

    def test_evidence_mismatch_reduces_confidence(self):
        """Four symptoms but evidence_supports_prediction=False and flagged_for_review=True.

        Confidence should be multiplied by 0.7 (evidence_mismatch_factor).
        adjusted=True, flagged=True.
        """
        symptoms = [
            _symptom("DEPRESSED_MOOD", 0.78),
            _symptom("SLEEP_ISSUES", 0.74),
            _symptom("ANHEDONIA", 0.72),
            _symptom("CONCENTRATION", 0.68),
        ]
        summary = PostSymptomSummary(
            symptoms_detected=symptoms,
            unique_symptom_count=4,
            total_sentences_analyzed=12,
            severity_level="moderate",
            severity_explanation="Four criteria met but evidence is ambiguous.",
            dsm5_criteria_met=["DEPRESSED_MOOD", "SLEEP_ISSUES", "ANHEDONIA", "CONCENTRATION"],
        )
        verification = _verification(
            evidence_supports=False,
            coherence=0.45,
            flagged_for_review=True,
        )

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        base_conf = (0.78 + 0.74 + 0.72 + 0.68) / 4
        expected_conf = base_conf * 0.7  # evidence_mismatch_factor
        assert abs(confidence - expected_conf) < 1e-9
        assert adjusted is True
        assert flagged is True

    # ── Test 5: Low trust ─────────────────────────────────────────────────────

    def test_low_trust_reduces_confidence(self):
        """Three symptoms with trust="low".

        Confidence should be multiplied by 0.8 (low_trust_factor).
        adjusted=True, flagged=True.
        """
        symptoms = [
            _symptom("DEPRESSED_MOOD", 0.85),
            _symptom("WORTHLESSNESS", 0.80),
            _symptom("FATIGUE", 0.75),
        ]
        summary = PostSymptomSummary(
            symptoms_detected=symptoms,
            unique_symptom_count=3,
            total_sentences_analyzed=9,
            severity_level="mild",
            severity_explanation="Three criteria met.",
            dsm5_criteria_met=["DEPRESSED_MOOD", "WORTHLESSNESS", "FATIGUE"],
        )
        verification = _verification(
            trust="low",
            reasoning="Input contains heavy use of metaphorical language.",
            confounders=["sarcasm", "metaphor"],
        )

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        base_conf = (0.85 + 0.80 + 0.75) / 3
        expected_conf = base_conf * 0.8  # low_trust_factor
        assert abs(confidence - expected_conf) < 1e-9
        assert adjusted is True
        assert flagged is True

    # ── Test 6: No symptoms detected ─────────────────────────────────────────

    def test_no_symptoms_detected(self):
        """Zero symptoms detected, severity='none'.

        Base confidence defaults to 0.9 (high confidence in "no symptoms").
        prediction='no_indicators', flagged=False, adjusted=False.
        """
        summary = PostSymptomSummary(
            symptoms_detected=[],
            unique_symptom_count=0,
            total_sentences_analyzed=5,
            severity_level="none",
            severity_explanation="No DSM-5 criteria detected.",
            dsm5_criteria_met=[],
        )
        verification = _verification()

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        assert prediction == "no_indicators"
        assert abs(confidence - 0.9) < 1e-9
        assert adjusted is False
        assert flagged is False

    # ── Test 7: Suicidal ideation alone ──────────────────────────────────────

    def test_suicidal_ideation_alone_always_flagged(self):
        """Single symptom: SUICIDAL_THOUGHTS at mild severity.

        SUICIDAL_THOUGHTS in dsm5_criteria_met must trigger flagged=True
        regardless of severity level or verification quality.
        """
        symptoms = [
            _symptom("SUICIDAL_THOUGHTS", 0.91),
        ]
        summary = PostSymptomSummary(
            symptoms_detected=symptoms,
            unique_symptom_count=1,
            total_sentences_analyzed=4,
            severity_level="mild",
            severity_explanation="Single criterion: suicidal ideation.",
            dsm5_criteria_met=["SUICIDAL_THOUGHTS"],
        )
        verification = _verification()  # clean verification — still must flag

        prediction, confidence, adjusted, flagged = self.svc.compute_final_prediction(summary, verification)

        assert prediction == "some_indicators"
        assert flagged is True
        assert adjusted is False
        assert abs(confidence - 0.91) < 1e-9
