"""Integration tests for RAG hybrid search — requires PostgreSQL with pgvector."""

import os

import pytest

POSTGRES_URL = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL.startswith("postgresql"),
    reason="Integration tests require PostgreSQL with pgvector",
)


@pytest.fixture(scope="module")
def rag_service():
    import asyncio
    from pathlib import Path

    from app.core.config import get_settings

    from app.services.rag import RAGService

    settings = get_settings()
    settings.knowledge_base_dir = Path(__file__).parent / "fixtures" / "knowledge_base_mini"
    service = RAGService(settings)
    asyncio.get_event_loop().run_until_complete(service.initialize())
    return service


class TestHybridSearch:
    def test_retrieve_returns_results(self, rag_service):
        results = rag_service.retrieve("depression symptoms")
        assert results is not None
        assert len(results) > 0

    def test_retrieve_results_have_fields(self, rag_service):
        results = rag_service.retrieve("depression symptoms")
        for r in results:
            assert "text" in r
            assert "metadata" in r

    def test_retrieve_with_category_filter(self, rag_service):
        results = rag_service.retrieve("treatment", category="medications")
        if results:
            for r in results:
                assert r["metadata"].get("category") == "medications"

    def test_retrieve_respects_top_k(self, rag_service):
        results = rag_service.retrieve("depression", n_results=2)
        assert len(results) <= 2

    def test_retrieve_for_symptoms(self, rag_service):
        results = rag_service.retrieve_for_symptoms(["depressed_mood"])
        assert "depressed_mood" in results
        assert len(results["depressed_mood"]) > 0


class TestRRFFusion:
    """Unit-style tests for RRF fusion logic (no DB needed, but module-level skip applies)."""

    def test_rrf_fusion_deduplicates(self, rag_service):
        dense = [
            {"id": "a", "text": "doc a", "metadata": {}, "rank": 1, "method": "dense"},
            {"id": "b", "text": "doc b", "metadata": {}, "rank": 2, "method": "dense"},
        ]
        bm25 = [
            {"id": "b", "text": "doc b", "metadata": {}, "rank": 1, "method": "bm25"},
            {"id": "c", "text": "doc c", "metadata": {}, "rank": 2, "method": "bm25"},
        ]
        fused = rag_service._rrf_fusion(dense, bm25)
        ids = [r["id"] for r in fused]
        assert len(ids) == len(set(ids)), "RRF should deduplicate"

    def test_rrf_fusion_scores_present(self, rag_service):
        dense = [{"id": "a", "text": "t", "metadata": {}, "rank": 1, "method": "dense"}]
        bm25 = [{"id": "b", "text": "t", "metadata": {}, "rank": 1, "method": "bm25"}]
        fused = rag_service._rrf_fusion(dense, bm25)
        for r in fused:
            assert "rrf_score" in r
            assert r["ranking_method"] == "rrf"

    def test_rrf_boosted_by_both_rankers(self, rag_service):
        """A doc appearing in both lists should rank higher than one in only one."""
        dense = [
            {"id": "both", "text": "t", "metadata": {}, "rank": 1, "method": "dense"},
            {"id": "dense_only", "text": "t", "metadata": {}, "rank": 2, "method": "dense"},
        ]
        bm25 = [
            {"id": "both", "text": "t", "metadata": {}, "rank": 1, "method": "bm25"},
            {"id": "bm25_only", "text": "t", "metadata": {}, "rank": 2, "method": "bm25"},
        ]
        fused = rag_service._rrf_fusion(dense, bm25)
        assert fused[0]["id"] == "both"


class TestGetChatContext:
    def test_chat_context_string(self, rag_service):
        ctx = rag_service.get_chat_context("I feel sad", ["depressed_mood"])
        assert isinstance(ctx, str)

    def test_chat_context_empty_when_no_symptoms(self, rag_service):
        ctx = rag_service.get_chat_context("hello", [])
        # Should still return something from query-based retrieval
        assert isinstance(ctx, str)
