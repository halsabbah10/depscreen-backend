"""
LLM service for generating screening explanations using OpenAI-compatible API.

Generates per-symptom clinical explanations enriched with RAG context.
Uses OpenAI-compatible API format.
"""

import json
import logging
import re

from openai import AsyncOpenAI

from app.core import localization
from app.core.config import Settings
from app.middleware.llm_resilience import llm_retry
from app.schemas.analysis import ExplanationReport, PostSymptomSummary

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response that may contain <think> tags, markdown fences, or preamble."""
    if not text:
        raise ValueError("Empty response from LLM")
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*\n?", "", cleaned).strip()
    cleaned = cleaned.rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")


# Localized safety content (Bahrain)
SAFETY_DISCLAIMER = localization.SAFETY_DISCLAIMER

# Rendered resource list used by fallbacks and response defaults
DEFAULT_RESOURCES = [
    f"{r['name_short']}: {r['phone_display']}" + (" (24/7)" if r.get("available_247") else "")
    for r in localization.CRISIS_RESOURCES_BY_PRIORITY[:5]
]


class LLMService:
    """Service for LLM-powered explanation generation.

    Supports per-task model selection via the model parameter.
    Default: settings.llm_model_pro (explanation is patient-facing, needs quality).
    Override at init to use a different tier.
    """

    def __init__(self, settings: Settings, model: str | None = None):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "DepScreen Explanation",
            },
        )
        # Explanation is patient-facing → Pro by default
        self.model = model or settings.llm_model_pro

    async def generate_explanation(
        self,
        text: str,
        symptom_analysis: PostSymptomSummary,
        verification_summary: str | None = None,
        rag_context: str | None = None,
    ) -> ExplanationReport:
        """Generate a human-readable explanation of the screening results.

        Args:
            text: Original input text
            symptom_analysis: Detected symptoms with sentence-level evidence
            verification_summary: Summary from LLM verification layer
            rag_context: Clinical context retrieved via RAG (optional)
        """
        prompt = self._build_explanation_prompt(text, symptom_analysis, verification_summary, rag_context)

        try:

            @llm_retry
            async def _call():
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=1200,
                    timeout=60,
                )

            response = await _call()

            content = response.choices[0].message.content or ""
            data = extract_json(content)

            # Ensure safety fields are always present
            data["safety_disclaimer"] = SAFETY_DISCLAIMER
            data["resources"] = data.get("resources", DEFAULT_RESOURCES)
            data.setdefault("symptom_explanations", {})

            # Safety guard on narrative fields — patient-visible text that could
            # contain unsafe clinical claims from the LLM.
            try:
                from app.services.safety_guard import scan_text as _sg_scan

                for key in ("summary", "why_model_thinks_this", "uncertainty_notes"):
                    if isinstance(data.get(key), str):
                        result = _sg_scan(data[key], context="explanation")
                        if result.violations:
                            logger.warning(
                                f"[llm explanation] {key} violations: "
                                f"{[(v.category, v.severity) for v in result.violations]}"
                            )
                        data[key] = result.redacted
                # Symptom explanations dict — scan each value
                symp_exp = data.get("symptom_explanations") or {}
                if isinstance(symp_exp, dict):
                    for sym_key, sym_val in list(symp_exp.items()):
                        if isinstance(sym_val, str):
                            result = _sg_scan(sym_val, context="explanation")
                            if result.violations:
                                logger.warning(
                                    f"[llm explanation] symptom {sym_key} violations: "
                                    f"{[(v.category, v.severity) for v in result.violations]}"
                                )
                            symp_exp[sym_key] = result.redacted
                    data["symptom_explanations"] = symp_exp
            except Exception as _sg_err:
                logger.warning(f"Safety guard error on explanation (non-fatal): {_sg_err}")

            return ExplanationReport(**data)

        except Exception as e:
            logger.error(f"LLM explanation generation failed: {e}")
            return self._fallback_explanation(symptom_analysis)

    def _get_system_prompt(self) -> str:
        return """You are an empathetic AI assistant that explains depression screening results
to patients. You are NOT a doctor. You do NOT diagnose.

Audience: a patient who may be anxious or tired, reading this alone after a
vulnerable disclosure. The explanation is ALSO seen by their clinician.

Dual bar you must meet:

1. CLINICAL SUBSTANCE — accurate, specific, evidence-based. Use correct
   clinical terminology where precision matters (DSM-5 criteria, symptom
   names, severity categories). Explain the WHY of each detection in a way
   a clinician reading over the patient's shoulder would endorse.

2. GENTLE DELIVERY — warm, grounded, non-alarmist. The patient may already
   fear the result. Lead with validation, not verdicts.

How to strike the balance in each field:

- `summary`: Open with something grounding, THEN name the clinical picture
  clearly. Not "The screening detected 5 DSM-5 symptoms indicating moderate
  severity" (cold). Not "Everything will be okay" (vague, useless). Rather:
  "Your words suggest several patterns consistent with moderate depressive
  symptoms — specifically around [X, Y, Z]. This is information to work
  with, not a verdict."

- `symptom_explanations`: Be CLINICALLY USEFUL. Name the symptom correctly,
  explain what it is in plain language, and reference the patient's own
  evidence. Don't soften past the point of substance.

- `why_model_thinks_this`: Explain the model's reasoning honestly — which
  sentences triggered which classifications. The patient and clinician both
  deserve to see the logic.

- `uncertainty_notes`: Real limitations, plainly stated. Screening windows
  (2 weeks of symptoms needed for DSM-5), the fact that a classifier reads
  text not context, etc. Frame as humility, not hedging.

Tone vocabulary:
- Prefer: "we noticed", "the screening points to", "often shows up as",
  "worth exploring with a clinician", "patterns in your words suggest"
- Avoid: "elevated risk", "concerning", "abnormal", "flagged as" (too
  alarmist for patient-facing framing — but still use precise terms like
  "moderate" or "severe" when that's the clinical finding)

Respond in valid JSON matching this schema:
{
    "summary": "Grounding opener + clear clinical picture (2-3 sentences)",
    "risk_level": "none" | "mild" | "moderate" | "severe",
    "symptom_explanations": {
        "SYMPTOM_NAME": "Clinically accurate explanation of this symptom, in patient-friendly language, referencing the patient's own evidence."
    },
    "why_model_thinks_this": "Honest explanation of the model's reasoning — which sentences contributed to which classifications.",
    "key_evidence_quotes": ["exact sentence from input", ...],
    "uncertainty_notes": "Real limitations, plainly stated.",
    "safety_disclaimer": "...",
    "resources": ["resource1", ...]
}

CRITICAL: Screening aid, not diagnosis. Never claim to diagnose. But also:
don't be so gentle that you're vague. Both the patient and their clinician
benefit from a clear, warm, substantive explanation."""

    def _build_explanation_prompt(
        self,
        text: str,
        symptom_analysis: PostSymptomSummary,
        verification_summary: str | None,
        rag_context: str | None,
    ) -> str:
        display_text = text[:500] + "..." if len(text) > 500 else text

        # Build symptom evidence list
        symptom_lines = []
        for d in symptom_analysis.symptoms_detected:
            symptom_lines.append(f'- **{d.symptom_label}** ({d.confidence:.0%}): "{d.sentence_text}"')
        symptoms_block = "\n".join(symptom_lines) if symptom_lines else "No symptoms detected."

        prompt = f"""Explain these depression screening results to the patient:

## Input Text
"{display_text}"

## Detected DSM-5 Symptoms ({symptom_analysis.unique_symptom_count} of 9 criteria)
{symptoms_block}

## Severity Level: {symptom_analysis.severity_level.upper()}
{symptom_analysis.severity_explanation}
"""

        if rag_context:
            prompt += f"""
## Clinical Context (from knowledge base)
{rag_context}
"""

        if verification_summary:
            prompt += f"""
## Verification Notes
{verification_summary}
"""

        prompt += """
Generate a JSON explanation. Be empathetic, clear, and honest about limitations.
For each detected symptom, explain what it means in plain language."""

        return prompt

    def _fallback_explanation(self, symptom_analysis: PostSymptomSummary) -> ExplanationReport:
        """Generate a fallback explanation when LLM fails."""
        severity = symptom_analysis.severity_level
        count = symptom_analysis.unique_symptom_count
        criteria = symptom_analysis.dsm5_criteria_met

        symptom_explanations = {}
        for d in symptom_analysis.symptoms_detected:
            if d.symptom not in symptom_explanations:
                symptom_explanations[d.symptom] = (
                    f"The screening detected language consistent with {d.symptom_label.lower()} "
                    f'in the sentence: "{d.sentence_text[:100]}"'
                )

        evidence_quotes = [d.sentence_text for d in symptom_analysis.symptoms_detected[:5]]

        return ExplanationReport(
            summary=(
                f"The screening detected {count} DSM-5 symptom(s) "
                f"({', '.join(criteria) if criteria else 'none'}), "
                f"indicating {severity} severity."
            ),
            risk_level=severity,
            symptom_explanations=symptom_explanations,
            why_model_thinks_this=(
                f"The model identified language patterns in {count} sentence(s) matching DSM-5 depression criteria."
            ),
            key_evidence_quotes=evidence_quotes,
            uncertainty_notes=(
                "This is an automated screening and may not capture all context. "
                "Sarcasm, metaphor, and cultural expressions may affect accuracy. "
                "Professional evaluation is recommended."
            ),
            safety_disclaimer=SAFETY_DISCLAIMER,
            resources=DEFAULT_RESOURCES,
        )
