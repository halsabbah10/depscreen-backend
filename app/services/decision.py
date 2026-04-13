"""
Decision service for combining symptom predictions with LLM verification.

The LLM does NOT override symptom detections, but CAN adjust confidence
and flag cases for clinician review based on verification results.
"""

import logging

from app.schemas.analysis import PostSymptomSummary, VerificationReport

logger = logging.getLogger(__name__)


class DecisionService:
    """Service for computing final screening assessment from DL + LLM verification."""

    def __init__(self):
        self.evidence_mismatch_factor = 0.7  # 30% reduction
        self.adversarial_factor = 0.3  # 70% reduction
        self.low_trust_factor = 0.8  # 20% reduction

    def compute_final_prediction(
        self,
        symptom_analysis: PostSymptomSummary,
        verification: VerificationReport,
    ) -> tuple[str, float, bool, bool]:
        """Combine symptom analysis with LLM verification.

        Returns:
            (final_prediction, final_confidence, was_adjusted, flagged_for_review)
        """
        adjusted = False
        reasons = []

        # Base confidence: average of top symptom detection confidences
        if symptom_analysis.symptoms_detected:
            confidences = [d.confidence for d in symptom_analysis.symptoms_detected]
            base_conf = sum(confidences) / len(confidences)
        else:
            base_conf = 0.9  # High confidence in "no symptoms" when none detected

        final_conf = base_conf

        # 1. Evidence validation adjustment
        if not verification.evidence_validation.evidence_supports_prediction:
            final_conf *= self.evidence_mismatch_factor
            adjusted = True
            reasons.append("evidence mismatch")

        # 2. Adversarial detection adjustment
        if verification.adversarial_check.likely_adversarial:
            final_conf *= self.adversarial_factor
            adjusted = True
            reasons.append("adversarial input detected")
            logger.warning("Confidence significantly reduced — adversarial input")

        # 3. Trust level adjustment
        trust = verification.confidence_analysis.should_trust_prediction
        if trust == "low":
            final_conf *= self.low_trust_factor
            adjusted = True
            reasons.append("low trust assessment")
        elif trust == "medium":
            final_conf *= 0.95
            if base_conf != final_conf:
                adjusted = True

        # Map severity to prediction label
        severity = symptom_analysis.severity_level
        if severity == "none":
            final_prediction = "no_indicators"
        elif severity in ("mild", "moderate"):
            final_prediction = "some_indicators"
        else:
            final_prediction = "significant_indicators"

        # Flag for review
        flagged = (
            verification.evidence_validation.flagged_for_review
            or verification.adversarial_check.likely_adversarial
            or verification.confidence_analysis.should_trust_prediction == "low"
            # Always flag severe cases and suicidal ideation
            or severity == "severe"
            or "SUICIDAL_THOUGHTS" in symptom_analysis.dsm5_criteria_met
        )

        if adjusted:
            logger.info(f"Confidence adjusted from {base_conf:.2%} to {final_conf:.2%}. Reasons: {', '.join(reasons)}")

        final_conf = max(0.0, min(1.0, final_conf))
        return final_prediction, final_conf, adjusted, flagged

    def get_verification_summary(self, verification: VerificationReport) -> str:
        """Generate a text summary of verification findings for the explanation LLM."""
        parts = []

        ev = verification.evidence_validation
        if not ev.evidence_supports_prediction:
            parts.append(f"Evidence may not fully support detections (coherence: {ev.coherence_score:.0%})")
            if ev.alternative_interpretation:
                parts.append(f"Alternative interpretation: {ev.alternative_interpretation}")

        adv = verification.adversarial_check
        if adv.likely_adversarial:
            parts.append(f"Warning: Input flagged as potentially {adv.adversarial_type}")
        elif adv.authenticity_score < 0.6:
            parts.append(f"Input authenticity score is low ({adv.authenticity_score:.0%})")

        conf = verification.confidence_analysis
        if conf.should_trust_prediction != "high":
            parts.append(f"Confidence assessment: {conf.should_trust_prediction}. {conf.reasoning}")
        if conf.potential_confounders:
            parts.append(f"Potential confounders: {', '.join(conf.potential_confounders)}")

        if not parts:
            return "All verification checks passed. Screening appears reliable."

        return " | ".join(parts)
