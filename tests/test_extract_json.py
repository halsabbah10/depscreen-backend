"""Tests for the shared extract_json utility."""

import pytest

from app.utils.json_extract import extract_json


class TestCleanJSON:
    """Cases where the input is already valid JSON."""

    def test_simple_object(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_nested_object(self):
        result = extract_json('{"a": {"b": {"c": 1}}}')
        assert result == {"a": {"b": {"c": 1}}}

    def test_object_with_array(self):
        result = extract_json('{"items": [1, 2, 3]}')
        assert result == {"items": [1, 2, 3]}


class TestMarkdownFences:
    """Cases where JSON is wrapped in markdown code fences."""

    def test_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}

    def test_trailing_backticks(self):
        text = '{"key": "value"}```'
        assert extract_json(text) == {"key": "value"}

    def test_fence_with_whitespace(self):
        text = '```json  \n  {"key": "value"}  \n```'
        assert extract_json(text) == {"key": "value"}


class TestThinkTags:
    """Cases where response contains <think> tags."""

    def test_think_before_json(self):
        text = '<think>Let me analyze this...</think>\n{"key": "value"}'
        assert extract_json(text) == {"key": "value"}

    def test_think_with_json_inside(self):
        text = '<think>{"wrong": true}</think>\n{"key": "value"}'
        assert extract_json(text) == {"key": "value"}

    def test_think_and_fences_combined(self):
        text = '<think>reasoning</think>\n```json\n{"key": "value"}\n```'
        assert extract_json(text) == {"key": "value"}


class TestPreambleText:
    """Cases where JSON is preceded by free-form text."""

    def test_text_before_json(self):
        text = 'Here is the result:\n{"key": "value"}'
        assert extract_json(text) == {"key": "value"}

    def test_multiple_lines_before_json(self):
        text = 'I analyzed the input.\nThe findings are:\n{"score": 0.8}'
        assert extract_json(text) == {"score": 0.8}


class TestTruncatedJSON:
    """Cases where JSON is truncated (missing closing braces)."""

    def test_single_missing_brace(self):
        text = '{"key": "value", "score": 0.8'
        result = extract_json(text)
        assert result["key"] == "value"
        assert result["score"] == 0.8

    def test_nested_missing_braces(self):
        text = '{"outer": {"inner": "value"'
        result = extract_json(text)
        assert result["outer"]["inner"] == "value"

    def test_truncated_with_preamble(self):
        text = 'Result:\n{"evidence_supports_prediction": false, "coherence_score": 0.7'
        result = extract_json(text)
        assert result["evidence_supports_prediction"] is False
        assert result["coherence_score"] == 0.7

    def test_truncated_inside_fence(self):
        text = '```json\n{"should_trust_prediction": "low", "reasoning": "short text'
        result = extract_json(text)
        assert result["should_trust_prediction"] == "low"

    def test_truncated_with_complete_fields(self):
        """Realistic case: Gemini response truncated after some complete fields."""
        text = """{
    "likely_adversarial": false,
    "adversarial_type": null,
    "authenticity_score": 0.9,
    "warning": "The text appears genui"""
        result = extract_json(text)
        assert result["likely_adversarial"] is False
        assert result["authenticity_score"] == 0.9


class TestMultipleObjects:
    """When response contains multiple JSON objects."""

    def test_returns_first_valid(self):
        text = 'prefix {"first": 1} suffix {"second": 2}'
        result = extract_json(text)
        assert result == {"first": 1}


class TestErrorCases:
    """Cases that should raise ValueError."""

    def test_empty_string(self):
        with pytest.raises(ValueError, match="Empty response"):
            extract_json("")

    def test_complete_garbage(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("this is not json at all")

    def test_only_think_tags(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("<think>just thinking</think>")

    def test_only_fences(self):
        with pytest.raises(ValueError, match="No valid JSON"):
            extract_json("```json\nnot json\n```")


class TestRealisticLLMResponses:
    """End-to-end tests matching actual Gemini response patterns from logs."""

    def test_evidence_validation_response(self):
        text = """{
    "evidence_supports_prediction": false,
    "coherence_score": 0.6,
    "alternative_interpretation": "The text contains negated statements that the model may have misinterpreted",
    "flagged_for_review": true
}"""
        result = extract_json(text)
        assert result["evidence_supports_prediction"] is False
        assert result["flagged_for_review"] is True

    def test_confidence_calibration_response(self):
        text = """{
    "should_trust_prediction": "medium",
    "reasoning": "Mixed signals with some negated statements",
    "potential_confounders": ["negation", "sarcasm"],
    "recommended_threshold_adjustment": null
}"""
        result = extract_json(text)
        assert result["should_trust_prediction"] == "medium"
        assert "negation" in result["potential_confounders"]

    def test_explanation_with_markdown_wrap(self):
        """Matches the actual log error: response wrapped in ```json."""
        text = """```json
{
    "summary": "Your words suggest several patterns",
    "why_model_thinks_this": "The model detected indicators",
    "uncertainty_notes": "Some statements were ambiguous",
    "symptom_explanations": {"DEPRESSED_MOOD": "Mood fluctuation noted"},
    "safety_disclaimer": "This is not a diagnosis",
    "resources": ["988 Lifeline"]
}
```"""
        result = extract_json(text)
        assert "patterns" in result["summary"]
        assert "DEPRESSED_MOOD" in result["symptom_explanations"]
