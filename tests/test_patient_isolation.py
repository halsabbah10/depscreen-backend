"""Safety-critical patient isolation tests.

These tests verify that Patient A's RAG data NEVER appears in
Patient B's retrieval results. A failure here is a data breach.
Must pass on every PR — runs in CI.
"""

import os
import pytest

POSTGRES_URL = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL.startswith("postgresql"),
    reason="Isolation tests require PostgreSQL with pgvector",
)


def _ensure_test_users(user_ids: list[str]):
    """Create test user rows if they don't exist (FK constraint satisfaction)."""
    from app.models.db import SessionLocal, User

    db = SessionLocal()
    try:
        for uid in user_ids:
            if not db.query(User).filter_by(id=uid).first():
                db.add(User(
                    id=uid,
                    email=f"{uid}@test.depscreen.local",
                    password_hash="$2b$12$testhashtesthasttesthash",
                    full_name=f"Test User {uid[:8]}",
                    role="patient",
                ))
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

    _ensure_test_users(["isolation-patient-A", "isolation-patient-B"])

    service = RAGService(settings)
    asyncio.get_event_loop().run_until_complete(service.initialize())

    service.ingest_patient_screening(
        patient_id="isolation-patient-A",
        screening_id="screening-A-001",
        text="Patient A reports severe insomnia and racing thoughts at night.",
        symptoms_detected=[
            {
                "symptom": "SLEEP_ISSUES",
                "symptom_label": "Sleep Issues",
                "sentence_text": "severe insomnia and racing thoughts at night",
                "confidence": 0.95,
            }
        ],
        severity_level="moderate",
    )

    service.ingest_patient_screening(
        patient_id="isolation-patient-B",
        screening_id="screening-B-001",
        text="Patient B describes loss of appetite and weight loss over two weeks.",
        symptoms_detected=[
            {
                "symptom": "APPETITE_CHANGE",
                "symptom_label": "Appetite Change",
                "sentence_text": "loss of appetite and weight loss over two weeks",
                "confidence": 0.88,
            }
        ],
        severity_level="mild",
    )

    return service


class TestPatientIsolation:
    def test_patient_a_cannot_see_patient_b_data(self, rag_service):
        results = rag_service.retrieve_patient_history(
            patient_id="isolation-patient-A",
            query="appetite and weight loss",
        )
        if results:
            for r in results:
                patient_id = r.get("metadata", {}).get("patient_id", "")
                assert patient_id != "isolation-patient-B", \
                    "ISOLATION BREACH: Patient A retrieved Patient B's data"

    def test_patient_b_cannot_see_patient_a_data(self, rag_service):
        results = rag_service.retrieve_patient_history(
            patient_id="isolation-patient-B",
            query="insomnia and racing thoughts",
        )
        if results:
            for r in results:
                patient_id = r.get("metadata", {}).get("patient_id", "")
                assert patient_id != "isolation-patient-A", \
                    "ISOLATION BREACH: Patient B retrieved Patient A's data"

    def test_nonexistent_patient_gets_empty(self, rag_service):
        results = rag_service.retrieve_patient_history(
            patient_id="nonexistent-patient-XYZ",
            query="any query",
        )
        assert results is None or len(results) == 0

    def test_patient_sees_own_data(self, rag_service):
        results = rag_service.retrieve_patient_history(
            patient_id="isolation-patient-A",
            query="insomnia",
        )
        assert results is not None
        assert len(results) > 0
