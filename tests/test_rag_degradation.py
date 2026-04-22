"""Graceful degradation tests for RAG service.

Verifies: reranker fails → RRF fallback, embedding fails → None,
uninitialized → None, ingestion → silent no-op.
"""

from unittest.mock import MagicMock


class TestGracefulDegradation:

    def test_reranker_failure_returns_candidates(self):
        """If reranker predict() fails, results still returned with RRF ordering."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        settings = get_settings()
        rag = RAGService(settings)
        rag._initialized = True

        # The reranker is loaded lazily via _load_reranker(). Mock the private
        # attribute directly so _load_reranker() returns without hitting disk,
        # then make predict() raise to exercise the fallback branch.
        mock_reranker = MagicMock()
        mock_reranker.predict.side_effect = RuntimeError("Reranker OOM")
        rag._reranker = mock_reranker

        candidates = [
            {
                "id": "1",
                "text": "test doc",
                "metadata": {},
                "rrf_score": 0.5,
                "rank": 1,
                "method": "dense",
            },
            {
                "id": "2",
                "text": "test doc 2",
                "metadata": {},
                "rrf_score": 0.3,
                "rank": 2,
                "method": "dense",
            },
        ]
        result = rag._rerank("test query", candidates, top_k=5)
        assert result is not None
        assert len(result) >= 1

    def test_embedding_failure_returns_none(self):
        """If embedding model encode() raises, embed() returns None."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        settings = get_settings()
        rag = RAGService(settings)
        rag.embedder = MagicMock()
        rag.embedder.encode.side_effect = RuntimeError("OOM")

        result = rag.embed("test")
        assert result is None

    def test_retrieve_uninitialized_returns_none(self):
        """Uninitialized RAG returns None from retrieve(), doesn't crash."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        rag = RAGService(get_settings())
        result = rag.retrieve("depression symptoms")
        assert result is None

    def test_patient_retrieval_uninitialized_returns_none(self):
        """Uninitialized patient retrieval returns None."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        rag = RAGService(get_settings())
        result = rag.retrieve_patient_history("patient-1", "query")
        assert result is None

    def test_ingest_uninitialized_no_crash(self):
        """Ingestion on uninitialized service does nothing and doesn't crash."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        rag = RAGService(get_settings())
        rag.ingest_patient_screening(
            patient_id="test",
            screening_id="test",
            text="test text",
            symptoms_detected=[],
            severity_level="none",
        )
        # No exception = pass

    def test_retrieve_for_symptoms_uninitialized(self):
        """retrieve_for_symptoms returns empty dict when not initialized."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        rag = RAGService(get_settings())
        result = rag.retrieve_for_symptoms(["depressed_mood"])
        assert result == {} or result == {"depressed_mood": []}
