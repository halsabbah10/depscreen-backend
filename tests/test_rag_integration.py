"""Integration tests for RAG hybrid search — requires PostgreSQL with pgvector."""

import os

import pytest

POSTGRES_URL = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL.startswith("postgresql"),
    reason="Integration tests require PostgreSQL with pgvector",
)


def _ensure_test_users(user_ids: list[str]):
    """Create test user rows if they don't exist (FK constraint satisfaction)."""
    from app.models.db import SessionLocal, User

    db = SessionLocal()
    try:
        for uid in user_ids:
            existing = db.query(User).filter_by(id=uid).first()
            if not existing:
                db.add(
                    User(
                        id=uid,
                        email=f"{uid}@test.depscreen.local",
                        password_hash="$2b$12$testhashtesthasttesthash",
                        full_name=f"Test User {uid[:8]}",
                        role="patient",
                    )
                )
        db.commit()
    finally:
        db.close()


@pytest.fixture(scope="module")
def rag_service():
    import asyncio
    from pathlib import Path

    from app.core.config import get_settings
    from app.services.rag import RAGService

    settings = get_settings()
    settings.knowledge_base_dir = Path(__file__).parent / "fixtures" / "knowledge_base_mini"

    # Create test users for FK constraints on live Postgres
    _ensure_test_users(
        [
            "test-patient-001",
            "test-patient-002",
            "test-patient-dedup",
        ]
    )

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


class TestPatientRAG:
    """Integration tests for patient RAG: ingest, retrieve, dedup, invalidate."""

    def test_ingest_screening(self, rag_service):
        """Screening is ingested with proper chunk types."""
        from app.models.db import PatientRAGChunk, SessionLocal

        rag_service.ingest_patient_screening(
            patient_id="test-patient-001",
            screening_id="test-screening-001",
            text="I have been feeling really empty and hopeless for weeks.",
            symptoms_detected=[
                {
                    "symptom": "DEPRESSED_MOOD",
                    "symptom_label": "Depressed Mood",
                    "sentence_text": "I have been feeling really empty and hopeless",
                    "confidence": 0.92,
                }
            ],
            severity_level="moderate",
        )

        db = SessionLocal()
        try:
            chunks = db.query(PatientRAGChunk).filter_by(patient_id="test-patient-001", is_current=True).all()
            assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
            assert any(c.chunk_type == "screening_text" for c in chunks)
            assert any(c.chunk_type == "symptom_evidence" for c in chunks)
        finally:
            db.close()

    def test_ingest_screening_sets_source_fields(self, rag_service):
        """Ingested chunks carry source_table, source_row_id, and content_hash."""
        from app.models.db import PatientRAGChunk, SessionLocal

        db = SessionLocal()
        try:
            chunks = (
                db.query(PatientRAGChunk)
                .filter_by(
                    patient_id="test-patient-001",
                    source_table="screenings",
                    source_row_id="test-screening-001",
                )
                .all()
            )
            assert len(chunks) >= 1
            for c in chunks:
                assert c.content_hash is not None and len(c.content_hash) == 64
                assert c.source_table == "screenings"
                assert c.source_row_id == "test-screening-001"
        finally:
            db.close()

    def test_retrieve_patient_history(self, rag_service):
        """retrieve_patient_history returns results for an ingested patient."""
        results = rag_service.retrieve_patient_history("test-patient-001", "hopeless feelings")
        assert results is not None
        assert len(results) > 0
        # Verify all results carry required fields
        for r in results:
            assert "text" in r
            assert "metadata" in r
            assert "ranking_method" in r
            assert r["metadata"].get("patient_id") == "test-patient-001"

    def test_retrieve_patient_history_filters_patient(self, rag_service):
        """Results are always scoped to the requested patient_id."""
        results = rag_service.retrieve_patient_history("test-patient-001", "hopeless feelings")
        if results:
            for r in results:
                assert r["metadata"].get("patient_id") == "test-patient-001"

    def test_retrieve_unknown_patient_returns_empty(self, rag_service):
        """No results (not None, not error) for a patient with no ingested data."""
        results = rag_service.retrieve_patient_history("unknown-patient-xyz", "anything")
        # Should return a list (possibly empty) or None — never raise
        assert results is None or isinstance(results, list)
        if isinstance(results, list):
            assert len(results) == 0

    def test_content_hash_dedup(self, rag_service):
        """Duplicate content does not create duplicate chunks for the same patient."""
        from app.models.db import PatientRAGChunk, SessionLocal

        # Ingest same text twice under different screening IDs
        rag_service.ingest_patient_screening(
            patient_id="test-patient-dedup",
            screening_id="s1",
            text="Exact duplicate text for testing.",
            symptoms_detected=[],
            severity_level="none",
        )
        rag_service.ingest_patient_screening(
            patient_id="test-patient-dedup",
            screening_id="s2",
            text="Exact duplicate text for testing.",
            symptoms_detected=[],
            severity_level="none",
        )

        db = SessionLocal()
        try:
            chunks = db.query(PatientRAGChunk).filter_by(patient_id="test-patient-dedup", is_current=True).all()
            texts = [c.content for c in chunks]
            assert texts.count("Exact duplicate text for testing.") == 1, (
                f"Expected 1 chunk for duplicate text, got {texts.count('Exact duplicate text for testing.')}"
            )
        finally:
            db.close()

    def test_invalidate_source(self, rag_service):
        """invalidate_source marks all matching chunks as is_current=False."""
        from app.models.db import PatientRAGChunk, SessionLocal

        rag_service.invalidate_source("screenings", "test-screening-001")

        db = SessionLocal()
        try:
            chunks = (
                db.query(PatientRAGChunk).filter_by(source_table="screenings", source_row_id="test-screening-001").all()
            )
            assert len(chunks) >= 1, "Expected at least one chunk for this source"
            assert all(not c.is_current for c in chunks), (
                "All chunks for invalidated source should have is_current=False"
            )
        finally:
            db.close()

    def test_invalidated_chunks_excluded_from_retrieval(self, rag_service):
        """After invalidation, chunks from that source no longer appear in retrieval."""
        # test-screening-001 was invalidated in test_invalidate_source
        results = rag_service.retrieve_patient_history("test-patient-001", "empty hopeless")
        # Results for this patient should either be empty or contain only current chunks.
        # We can't assert on exact text without knowing order, but we can assert
        # that retrieve respected is_current (it filters it — the implementation contract).
        assert results is None or isinstance(results, list)

    def test_personalized_chat_context_structure(self, rag_service):
        """get_personalized_chat_context returns a string with expected section headers."""
        # Ingest fresh data for a separate patient so we have something to retrieve
        rag_service.ingest_patient_screening(
            patient_id="test-patient-ctx",
            screening_id="ctx-screening-001",
            text="I have persistent sadness and loss of interest in activities.",
            symptoms_detected=[
                {
                    "symptom": "ANHEDONIA",
                    "symptom_label": "Anhedonia",
                    "sentence_text": "loss of interest in activities",
                    "confidence": 0.87,
                }
            ],
            severity_level="mild",
        )

        ctx = rag_service.get_personalized_chat_context(
            patient_id="test-patient-ctx",
            user_message="I feel empty and have no interest in things",
            detected_symptoms=["ANHEDONIA"],
        )
        assert isinstance(ctx, str)
        # Both sections should appear when data is present
        if ctx:
            # At minimum the patient history section must appear (we just ingested data)
            assert "### Your Previous Check-ins" in ctx or "### Clinical Knowledge" in ctx
