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
    """Service for LLM-powered explanation generation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "DepScreen Explanation",
            },
        )
        self.model = settings.llm_model

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

            content = response.choices[0].message.content
            data = extract_json(content)

            # Ensure safety fields are always present
            data["safety_disclaimer"] = SAFETY_DISCLAIMER
            data["resources"] = data.get("resources", DEFAULT_RESOURCES)
            data.setdefault("symptom_explanations", {})

            return ExplanationReport(**data)

        except Exception as e:
            logger.error(f"LLM explanation generation failed: {e}")
            return self._fallback_explanation(symptom_analysis)

    def _get_system_prompt(self) -> str:
        return """You are an empathetic AI assistant that explains depression screening results
in clear, supportive language. You are NOT a doctor. You do NOT diagnose.

Your role:
1. Explain which DSM-5 symptoms were detected and what they mean
2. Provide clinical context for each detected symptom
3. Highlight the specific sentences that triggered each detection
4. Acknowledge uncertainty and limitations
5. Always include safety disclaimers
6. Recommend professional evaluation when appropriate

Respond in valid JSON matching this schema:
{
    "summary": "Brief 1-2 sentence summary of the screening",
    "risk_level": "none" | "mild" | "moderate" | "severe",
    "symptom_explanations": {
        "SYMPTOM_NAME": "What this symptom means and why it was detected..."
    },
    "why_model_thinks_this": "Overall explanation of what patterns were found",
    "key_evidence_quotes": ["exact sentence from input", ...],
    "uncertainty_notes": "Limitations and caveats",
    "safety_disclaimer": "...",
    "resources": ["resource1", ...]
}

CRITICAL: This is a screening aid, not a diagnostic tool. Never claim to diagnose."""

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
