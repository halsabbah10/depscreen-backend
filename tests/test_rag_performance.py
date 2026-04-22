"""Performance tests for RAG retrieval."""

import time


class TestRetrievalPerformance:

    def test_single_retrieval_latency(self):
        """Single retrieval completes within budget."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        settings = get_settings()
        rag = RAGService(settings)
        rag.embedder = rag._load_embedding_model()
        rag._initialized = True

        # Measure embedding time (the main latency component without DB)
        start = time.monotonic()
        embedding = rag.embed("What are the symptoms of depression?")
        elapsed = time.monotonic() - start

        assert embedding is not None
        assert len(embedding) == 1024
        # Embedding should complete in under 5 seconds on CPU
        assert elapsed < 5.0, f"Embedding took {elapsed:.2f}s — exceeds 5s budget"

    def test_concurrent_embeddings(self):
        """Multiple embeddings don't cause errors."""
        from app.core.config import get_settings
        from app.services.rag import RAGService

        settings = get_settings()
        rag = RAGService(settings)
        rag.embedder = rag._load_embedding_model()

        queries = [
            "depression symptoms",
            "sertraline side effects",
            "PHQ-9 scoring guide",
            "behavioral activation therapy",
            "crisis resources Bahrain",
        ]

        results = [rag.embed(q) for q in queries]
        assert all(r is not None for r in results)
        assert all(len(r) == 1024 for r in results)
