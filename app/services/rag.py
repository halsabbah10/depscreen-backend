"""
RAG (Retrieval-Augmented Generation) service.

Uses PostgreSQL pgvector + sentence-transformers to retrieve relevant clinical
knowledge for detected symptoms. Enriches screening explanations and
grounds chatbot responses in evidence-based content.

Storage: Supabase PostgreSQL with pgvector extension (replaces ChromaDB).
Embeddings: sentence-transformers all-MiniLM-L6-v2 (384 dimensions).
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.models.db import SessionLocal, KnowledgeChunk, PatientRAGChunk

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval service over the clinical knowledge base using pgvector."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder: Optional[SentenceTransformer] = None
        self._initialized = False

    # ── Initialization ───────────────────────────────────────────────────────

    async def initialize(self):
        """Load embedding model and populate knowledge_chunks from knowledge base if empty."""
        if self._initialized:
            return

        logger.info("Initializing RAG service...")

        # Load embedding model
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded (all-MiniLM-L6-v2)")

        # Populate knowledge_chunks table if empty
        db = SessionLocal()
        try:
            count = db.query(KnowledgeChunk).count()
            if count == 0:
                self._load_knowledge_base(db)
            else:
                logger.info(f"Knowledge base already loaded ({count} chunks)")
        finally:
            db.close()

        self._initialized = True

    def _load_knowledge_base(self, db):
        """Read markdown files from knowledge base and load into knowledge_chunks table."""
        kb_dir = self.settings.knowledge_base_dir
        if not kb_dir.exists():
            logger.warning(f"Knowledge base directory not found: {kb_dir}")
            return

        chunks_to_add = []

        for md_file in sorted(kb_dir.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8").strip()
            if not content:
                continue

            # Extract metadata from first line (e.g., "symptom: DEPRESSED_MOOD")
            symptom = ""
            category = md_file.parent.name  # dsm5_criteria, coping_strategies, etc.
            first_line = content.split("\n")[0]
            if first_line.lower().startswith("symptom:"):
                symptom = first_line.split(":", 1)[1].strip()
                content = "\n".join(content.split("\n")[1:]).strip()

            # Chunk by ## headers (each section becomes a separate chunk)
            text_chunks = self._chunk_by_headers(content)

            for i, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < 20:
                    continue

                chunk_text = chunk_text.strip()
                embedding = self.embedder.encode([chunk_text])[0].tolist()

                chunk = KnowledgeChunk(
                    id=f"{md_file.stem}_{i}",
                    content=chunk_text,
                    category=category,
                    symptom=symptom,
                    source_file=md_file.name,
                    embedding=embedding,
                )
                chunks_to_add.append(chunk)

        if chunks_to_add:
            db.add_all(chunks_to_add)
            db.commit()
            logger.info(
                f"Loaded {len(chunks_to_add)} chunks from "
                f"{len(list(kb_dir.rglob('*.md')))} files"
            )
        else:
            logger.warning("No knowledge base documents found")

    def _chunk_by_headers(self, text: str) -> list[str]:
        """Split markdown text by ## headers into chunks."""
        chunks = re.split(r"\n(?=## )", text)
        # If no headers, return the whole text as one chunk
        if len(chunks) == 1 and not chunks[0].startswith("## "):
            return [text]
        return [c for c in chunks if c.strip()]

    # ── Clinical Knowledge Retrieval ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
        symptom: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve relevant documents for a query using pgvector cosine distance.

        Returns list of dicts with 'text', 'metadata', and 'distance' keys.
        """
        if not self._initialized or self.embedder is None:
            logger.warning("RAG not initialized — returning empty results")
            return []

        query_embedding = self.embedder.encode([query])[0].tolist()

        db = SessionLocal()
        try:
            q = db.query(KnowledgeChunk)

            # Apply filters
            if category:
                q = q.filter(KnowledgeChunk.category == category)
            if symptom:
                q = q.filter(KnowledgeChunk.symptom == symptom)

            # Order by cosine distance (ascending = most similar first)
            results = (
                q.order_by(KnowledgeChunk.embedding.cosine_distance(query_embedding))
                .limit(n_results)
                .all()
            )

            output = []
            for chunk in results:
                output.append({
                    "text": chunk.content,
                    "metadata": {
                        "source_file": chunk.source_file or "",
                        "category": chunk.category or "",
                        "symptom": chunk.symptom or "",
                    },
                    "distance": 0,  # pgvector ordering handles relevance
                })
            return output
        finally:
            db.close()

    def retrieve_for_symptoms(self, symptoms: list[str]) -> dict[str, list[dict]]:
        """Retrieve clinical context for each detected symptom.

        Returns a dict mapping symptom names to lists of relevant documents.
        Retrieves both DSM-5 criteria descriptions and coping strategies.
        """
        if not self._initialized:
            return {}

        results = {}
        for symptom in symptoms:
            # Get DSM-5 criteria info
            criteria_docs = self.retrieve(
                query=f"DSM-5 criterion for {symptom}",
                n_results=2,
                symptom=symptom,
            )
            # Get coping strategies
            coping_docs = self.retrieve(
                query=f"coping strategies for {symptom}",
                n_results=2,
                category="coping_strategies",
            )
            results[symptom] = criteria_docs + coping_docs

        return results

    def get_chat_context(
        self,
        user_message: str,
        detected_symptoms: list[str],
        n_results: int = 5,
    ) -> str:
        """Build RAG context for chatbot response.

        Combines retrieved docs relevant to the user's question with
        context about their detected symptoms.
        """
        if not self._initialized:
            return ""

        # Retrieve docs for user's specific question
        docs = self.retrieve(user_message, n_results=n_results)

        # Also retrieve symptom-specific context
        for symptom in detected_symptoms[:3]:  # Limit to top 3 symptoms
            symptom_docs = self.retrieve(
                query=f"{symptom} information and guidance",
                n_results=2,
                symptom=symptom,
            )
            docs.extend(symptom_docs)

        # Deduplicate by text content
        seen = set()
        unique_docs = []
        for doc in docs:
            text = doc["text"][:100]  # Use first 100 chars as key
            if text not in seen:
                seen.add(text)
                unique_docs.append(doc)

        # Format as context string
        if not unique_docs:
            return ""

        context_parts = []
        for doc in unique_docs[:7]:  # Limit total context
            source = doc.get("metadata", {}).get("source_file", "knowledge base")
            context_parts.append(f"[{source}]\n{doc['text']}")

        return "\n\n---\n\n".join(context_parts)

    # ── Patient-Specific Document Ingestion ──────────────────────────────────

    def ingest_patient_screening(
        self,
        patient_id: str,
        screening_id: str,
        text: str,
        symptoms_detected: list[dict],
        severity_level: str,
    ):
        """Ingest a patient's screening into patient_rag_chunks.

        Stores the full screening text and each detected symptom sentence
        as separate chunks with embeddings for personalized retrieval.
        """
        if not self._initialized or self.embedder is None:
            return

        chunks_to_add = []

        # Store the full screening text
        screening_embedding = self.embedder.encode([text])[0].tolist()
        chunks_to_add.append(PatientRAGChunk(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            screening_id=screening_id,
            content=text,
            chunk_type="screening_text",
            metadata_json={
                "severity_level": severity_level,
                "symptom_count": len(symptoms_detected),
                "symptoms": ",".join(d.get("symptom", "") for d in symptoms_detected),
            },
            embedding=screening_embedding,
        ))

        # Store each detected symptom sentence as a separate chunk
        for det in symptoms_detected:
            sentence_text = det.get("sentence_text", "")
            if not sentence_text:
                continue

            symptom_embedding = self.embedder.encode([sentence_text])[0].tolist()
            chunks_to_add.append(PatientRAGChunk(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                screening_id=screening_id,
                content=sentence_text,
                chunk_type="symptom_evidence",
                metadata_json={
                    "symptom": det.get("symptom", ""),
                    "symptom_label": det.get("symptom_label", ""),
                    "confidence": str(det.get("confidence", 0)),
                },
                embedding=symptom_embedding,
            ))

        db = SessionLocal()
        try:
            db.add_all(chunks_to_add)
            db.commit()
            logger.info(
                f"Ingested screening {screening_id} for patient {patient_id[:8]} "
                f"({len(chunks_to_add)} chunks)"
            )
        finally:
            db.close()

    def ingest_patient_document(
        self,
        patient_id: str,
        doc_id: str,
        doc_type: str,
        title: str,
        content: str,
    ):
        """Ingest a clinician-uploaded patient document into patient_rag_chunks.

        Supported doc_types: intake_form, phq9, medication_list, session_notes,
        treatment_plan, medical_history, safety_plan, other.
        """
        if not self._initialized or self.embedder is None:
            return

        # Chunk the document by paragraphs (clinical docs tend to be structured)
        text_chunks = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        if not text_chunks:
            text_chunks = [content]

        chunks_to_add = []
        for chunk_text in text_chunks:
            embedding = self.embedder.encode([chunk_text])[0].tolist()
            chunks_to_add.append(PatientRAGChunk(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                doc_id=doc_id,
                content=chunk_text,
                chunk_type="patient_document",
                metadata_json={
                    "doc_type": doc_type,
                    "title": title,
                },
                embedding=embedding,
            ))

        db = SessionLocal()
        try:
            db.add_all(chunks_to_add)
            db.commit()
            logger.info(
                f"Ingested document '{title}' ({doc_type}) for patient "
                f"{patient_id[:8]} — {len(chunks_to_add)} chunks"
            )
        finally:
            db.close()

    def retrieve_patient_history(
        self,
        patient_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Retrieve relevant past screening/document content for a specific patient."""
        if not self._initialized or self.embedder is None:
            return []

        query_embedding = self.embedder.encode([query])[0].tolist()

        db = SessionLocal()
        try:
            results = (
                db.query(PatientRAGChunk)
                .filter(PatientRAGChunk.patient_id == patient_id)
                .order_by(PatientRAGChunk.embedding.cosine_distance(query_embedding))
                .limit(n_results)
                .all()
            )

            output = []
            for chunk in results:
                meta = chunk.metadata_json or {}
                meta["type"] = chunk.chunk_type
                output.append({
                    "text": chunk.content,
                    "metadata": meta,
                    "distance": 0,
                })
            return output
        finally:
            db.close()

    def get_personalized_chat_context(
        self,
        patient_id: str,
        user_message: str,
        detected_symptoms: list[str],
        n_results: int = 5,
    ) -> str:
        """Build personalized RAG context combining clinical knowledge + patient history.

        This is the main method the chatbot uses. It retrieves:
        1. Clinical knowledge relevant to the question
        2. Past screening content specific to this patient
        """
        # Clinical knowledge
        clinical_context = self.get_chat_context(
            user_message=user_message,
            detected_symptoms=detected_symptoms,
            n_results=n_results,
        )

        # Patient history
        patient_docs = self.retrieve_patient_history(
            patient_id=patient_id,
            query=user_message,
            n_results=3,
        )

        parts = []
        if clinical_context:
            parts.append("### Clinical Knowledge\n" + clinical_context)

        if patient_docs:
            history_parts = []
            for doc in patient_docs:
                meta = doc.get("metadata", {})
                doc_type = meta.get("type", "")
                if doc_type == "screening_text":
                    severity = meta.get("severity_level", "unknown")
                    history_parts.append(
                        f"[Previous check-in, severity={severity}]\n{doc['text'][:300]}"
                    )
                elif doc_type == "symptom_evidence":
                    symptom = meta.get("symptom_label", meta.get("symptom", ""))
                    history_parts.append(
                        f"[Previous detection: {symptom}]\n{doc['text']}"
                    )

            if history_parts:
                parts.append(
                    "### Your Previous Check-ins\n" + "\n\n".join(history_parts)
                )

        return "\n\n---\n\n".join(parts)

    @property
    def is_initialized(self) -> bool:
        return self._initialized
