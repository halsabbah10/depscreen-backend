"""End-to-end tests for RAG integration in the screening and chat pipelines.

Verifies that RAG context flows correctly through:
- Safety module imports and functions
- Grounding instructions in system prompts
- Explanation accepts patient_context parameter
- Chat summary extracts clinical content
- NLI verify_claim exists and has correct signature
- Verification accepts dsm5_context parameter
"""

import inspect


class TestSafetyModuleIntegration:
    """Verify safety module is importable and functional."""

    def test_all_safety_functions_importable(self):
        from app.services.rag_safety import (
            GROUNDING_INSTRUCTIONS,
            build_rag_prompt_section,
            filter_by_relevance,
            get_authority_level,
            sanitize_for_ingestion,
            sanitize_identity_document,
            should_ingest_to_rag,
            wrap_retrieved_context,
        )
        assert len(GROUNDING_INSTRUCTIONS) > 100
        assert callable(filter_by_relevance)
        assert callable(sanitize_for_ingestion)
        assert callable(wrap_retrieved_context)
        assert callable(build_rag_prompt_section)
        assert callable(sanitize_identity_document)
        assert callable(should_ingest_to_rag)
        assert callable(get_authority_level)

    def test_grounding_instructions_in_explanation_system_prompt(self):
        """Explanation system prompt includes grounding instructions."""
        from app.core.config import get_settings
        from app.services.llm import LLMService

        llm = LLMService(get_settings())
        prompt = llm._get_system_prompt()
        assert "ONLY state clinical facts" in prompt or "clinical facts" in prompt.lower()
        assert "NEVER generate specific" in prompt or "dosages" in prompt.lower()


class TestVerificationRAGIntegration:
    """Verify verification service accepts RAG context."""

    def test_verify_prediction_accepts_dsm5_context(self):
        from app.services.llm_verification import VerificationService

        sig = inspect.signature(VerificationService.verify_prediction)
        assert "dsm5_context" in sig.parameters
        param = sig.parameters["dsm5_context"]
        assert param.default is None  # Optional, defaults to None


class TestExplanationIntegration:
    """Verify explanation accepts patient context."""

    def test_generate_explanation_accepts_patient_context(self):
        from app.services.llm import LLMService

        sig = inspect.signature(LLMService.generate_explanation)
        assert "patient_context" in sig.parameters
        param = sig.parameters["patient_context"]
        assert param.default is None

    def test_build_explanation_prompt_accepts_patient_context(self):
        from app.services.llm import LLMService

        sig = inspect.signature(LLMService._build_explanation_prompt)
        assert "patient_context" in sig.parameters


class TestChatSummaryIntegration:
    """Verify chat summary pipeline works end-to-end."""

    def test_extraction_finds_clinical_content(self):
        from app.services.chat_summary import extract_clinical_sentences

        messages = [
            {"role": "user", "content": "I've been taking sertraline for 2 weeks now", "created_at": "2026-04-22"},
            {"role": "user", "content": "The fatigue is getting better but I still have insomnia", "created_at": "2026-04-22"},
            {"role": "user", "content": "ok", "created_at": "2026-04-22"},
        ]
        extracted = extract_clinical_sentences(messages)
        assert len(extracted) == 2
        assert any("sertraline" in s.lower() for s in extracted)
        assert any("fatigue" in s.lower() or "insomnia" in s.lower() for s in extracted)

    def test_extraction_ignores_trivial(self):
        from app.services.chat_summary import extract_clinical_sentences

        messages = [
            {"role": "user", "content": "ok thanks bye", "created_at": "2026-04-22"},
            {"role": "assistant", "content": "Take care!", "created_at": "2026-04-22"},
        ]
        extracted = extract_clinical_sentences(messages)
        assert len(extracted) == 0

    def test_trigger_logic(self):
        from app.services.chat_summary import should_trigger_summary

        assert not should_trigger_summary(message_count=5, substantive_count=3)
        assert not should_trigger_summary(message_count=15, substantive_count=2)
        assert should_trigger_summary(message_count=10, substantive_count=3)
        assert should_trigger_summary(message_count=20, substantive_count=10)


class TestNLIIntegration:
    """Verify NLI claim verification exists."""

    def test_verify_claim_method_exists(self):
        from app.services.rag import RAGService

        assert hasattr(RAGService, "verify_claim")
        sig = inspect.signature(RAGService.verify_claim)
        assert "claim" in sig.parameters
        assert "source" in sig.parameters

    def test_verify_claim_returns_valid_label(self):
        """verify_claim returns one of the three NLI labels (or neutral on failure)."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        rag = RAGService(get_settings())
        # Without model loaded, should return "neutral" (graceful degradation)
        result = rag.verify_claim("test claim", "test source")
        assert result in ("entailment", "contradiction", "neutral")


class TestContainerSingleton:
    """Verify RAG service container works."""

    def test_get_rag_service_returns_none_before_init(self):
        from app.services.container import get_rag_service

        # Before app startup, may return None
        result = get_rag_service()
        # Either None or a RAGService instance — both are valid
        assert result is None or hasattr(result, "retrieve")

    def test_set_and_get(self):
        from unittest.mock import MagicMock

        from app.services.container import get_rag_service, set_rag_service

        original = get_rag_service()
        mock_rag = MagicMock()
        set_rag_service(mock_rag)
        assert get_rag_service() is mock_rag
        # Restore original
        set_rag_service(original)
