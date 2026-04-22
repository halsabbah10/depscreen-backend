"""
RAG (Retrieval-Augmented Generation) service — V2.

Hybrid retrieval over PostgreSQL pgvector + BM25 full-text search, with
cross-encoder reranking and NLI-based hallucination filtering.

Models:
  Embedding : Alibaba-NLP/gte-large-en-v1.5  (1024-dim, eager load)
  Reranker  : BAAI/bge-reranker-v2-m3         (lazy load)
  NLI       : cross-encoder/nli-deberta-v3-base (lazy load)

Storage: Supabase PostgreSQL with pgvector extension.
"""

import hashlib
import json
import logging
import re
import uuid
from typing import Any

import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.db import KnowledgeChunk, PatientRAGChunk, SessionLocal
from app.services.chunking import Chunk, chunk_json_entry, chunk_text
from app.services.document_extractor import extract_text

logger = logging.getLogger(__name__)


class RAGService:
    """Hybrid retrieval service over the clinical knowledge base.

    Task 7: initialization, embedding, knowledge base loading, frontmatter helpers.
    Tasks 8-9 will add retrieval, reranking, patient history, and chat context methods.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None
        self._nli_model: CrossEncoder | None = None
        self._initialized = False
        self._knowledge_base_loaded = False

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model (eager, at startup)."""
        logger.info("Loading embedding model: %s", self.settings.rag_embedding_model)
        model = SentenceTransformer(
            self.settings.rag_embedding_model,
            trust_remote_code=True,
        )
        logger.info(
            "Embedding model loaded (%s, %d-dim)",
            self.settings.rag_embedding_model,
            self.settings.rag_embedding_dimensions,
        )
        return model

    def _load_reranker(self) -> CrossEncoder:
        """Load the reranker model (lazy, first use)."""
        if self._reranker is not None:
            return self._reranker
        logger.info("Loading reranker model: %s", self.settings.rag_reranker_model)
        self._reranker = CrossEncoder(self.settings.rag_reranker_model)
        logger.info("Reranker model loaded")
        return self._reranker

    def _load_nli_model(self) -> CrossEncoder:
        """Load the NLI model (lazy, first use)."""
        if self._nli_model is not None:
            return self._nli_model
        logger.info("Loading NLI model: %s", self.settings.rag_nli_model)
        self._nli_model = CrossEncoder(self.settings.rag_nli_model)
        logger.info("NLI model loaded")
        return self._nli_model

    # ── Initialization ───────────────────────────────────────────────────────

    async def initialize(self):
        """Load embedding model and populate knowledge_chunks from knowledge base if empty."""
        if self._initialized:
            return

        logger.info("Initializing RAG service...")

        # Eager-load embedding model
        self.embedder = self._load_embedding_model()

        # Populate knowledge_chunks table if empty
        db = SessionLocal()
        try:
            count = db.query(KnowledgeChunk).count()
            if count == 0:
                self._load_knowledge_base(db)
            else:
                logger.info("Knowledge base already loaded (%d chunks)", count)
                self._knowledge_base_loaded = True
        finally:
            db.close()

        self._initialized = True

    # ── Knowledge Base Loading ───────────────────────────────────────────────

    def _load_knowledge_base(self, db: Session) -> None:
        """Read knowledge base directory and load all documents into knowledge_chunks."""
        kb_dir = self.settings.knowledge_base_dir
        if not kb_dir.exists():
            logger.warning("Knowledge base directory not found: %s", kb_dir)
            return

        total_chunks = 0
        files_processed = 0

        # 1. Markdown files (top-level and subdirectories, but not sources/)
        for md_file in sorted(kb_dir.rglob("*.md")):
            # Skip files inside sources/ directory
            try:
                md_file.relative_to(kb_dir / "sources")
                continue
            except ValueError:
                pass  # Not inside sources/, process it

            chunks_added = self._process_markdown_file(db, md_file)
            total_chunks += chunks_added
            files_processed += 1

        # 2. JSON files in structured/
        structured_dir = kb_dir / "structured"
        if structured_dir.exists():
            for json_file in sorted(structured_dir.rglob("*.json")):
                chunks_added = self._process_json_file(db, json_file)
                total_chunks += chunks_added
                files_processed += 1

        # 3. PDF files in sources/
        sources_dir = kb_dir / "sources"
        if sources_dir.exists():
            for pdf_file in sorted(sources_dir.rglob("*.pdf")):
                chunks_added = self._process_pdf_file(db, pdf_file)
                total_chunks += chunks_added
                files_processed += 1

        if total_chunks > 0:
            db.commit()
            self._knowledge_base_loaded = True
            logger.info(
                "Knowledge base loaded: %d chunks from %d files",
                total_chunks,
                files_processed,
            )
        else:
            logger.warning("No knowledge base documents found in %s", kb_dir)

    def _process_markdown_file(self, db: Session, md_file: Any) -> int:
        """Process a single markdown file: extract frontmatter, chunk, embed, store."""
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            return 0

        # Extract and strip frontmatter
        frontmatter = self._extract_frontmatter(content)
        body = self._strip_frontmatter(content)

        if not body or len(body.strip()) < 20:
            return 0

        # Derive metadata from frontmatter
        category = frontmatter.get("category", md_file.parent.name)
        subcategory = frontmatter.get("subcategory", "")
        symptoms = self._normalize_symptoms(frontmatter)
        metadata = {k: v for k, v in frontmatter.items() if k not in ("category", "subcategory", "symptoms", "symptom")}

        # Chunk the body
        chunks = chunk_text(
            body,
            source_type="markdown",
            max_tokens=self.settings.rag_child_chunk_max,
            overlap_tokens=self.settings.rag_child_chunk_overlap,
            parent_max_tokens=self.settings.rag_parent_chunk_max,
        )

        chunks_added = 0
        for chunk in chunks:
            embedding = self.embed(chunk.content)
            if embedding is None:
                continue

            db_chunk = KnowledgeChunk(
                id=chunk.id,
                parent_chunk_id=chunk.parent_id,
                chunk_level=chunk.chunk_level,
                sequence_index=chunk.sequence_index,
                content=chunk.content,
                search_vector=func.to_tsvector("english", chunk.content),
                category=category,
                subcategory=subcategory,
                symptoms=symptoms,
                source_file=md_file.name,
                source_type="markdown",
                token_count=chunk.token_count,
                embedding=embedding,
                metadata_json=metadata if metadata else None,
            )
            db.add(db_chunk)
            chunks_added += 1

        return chunks_added

    def _process_json_file(self, db: Session, json_file: Any) -> int:
        """Process a structured JSON file: convert entries to natural language, embed, store."""
        try:
            raw = json_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse JSON file %s: %s", json_file.name, exc)
            return 0

        # Support both list-of-dicts and single-dict formats
        entries = data if isinstance(data, list) else [data]

        # Detect template from filename or directory
        template = self._detect_json_template(json_file)
        category = json_file.parent.name if json_file.parent.name != "structured" else json_file.stem

        chunks_added = 0
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            chunk = chunk_json_entry(entry, template=template)
            if not chunk.content or len(chunk.content.strip()) < 20:
                continue

            embedding = self.embed(chunk.content)
            if embedding is None:
                continue

            db_chunk = KnowledgeChunk(
                id=chunk.id,
                content=chunk.content,
                search_vector=func.to_tsvector("english", chunk.content),
                category=category,
                source_file=json_file.name,
                source_type="json",
                token_count=chunk.token_count,
                embedding=embedding,
                metadata_json=entry,
            )
            db.add(db_chunk)
            chunks_added += 1

        return chunks_added

    def _process_pdf_file(self, db: Session, pdf_file: Any) -> int:
        """Process a PDF file from sources/: extract text, chunk, embed, store."""
        try:
            raw_bytes = pdf_file.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read PDF %s: %s", pdf_file.name, exc)
            return 0

        result = extract_text(raw_bytes, pdf_file.name)
        if result is None or not result.text.strip():
            logger.warning("No text extracted from PDF %s", pdf_file.name)
            return 0

        chunks = chunk_text(
            result.text,
            source_type="text",
            max_tokens=self.settings.rag_child_chunk_max,
            overlap_tokens=self.settings.rag_child_chunk_overlap,
            parent_max_tokens=self.settings.rag_parent_chunk_max,
        )

        chunks_added = 0
        for chunk in chunks:
            embedding = self.embed(chunk.content)
            if embedding is None:
                continue

            db_chunk = KnowledgeChunk(
                id=chunk.id,
                parent_chunk_id=chunk.parent_id,
                chunk_level=chunk.chunk_level,
                sequence_index=chunk.sequence_index,
                content=chunk.content,
                search_vector=func.to_tsvector("english", chunk.content),
                category="sources",
                source_file=pdf_file.name,
                source_type="pdf",
                token_count=chunk.token_count,
                embedding=embedding,
                metadata_json={
                    "extraction_method": result.method,
                    "page_count": result.page_count,
                    "has_tables": result.has_tables,
                },
            )
            db.add(db_chunk)
            chunks_added += 1

        return chunks_added

    # ── Embedding ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float] | None:
        """Encode a single text string into a 1024-dim embedding vector.

        Returns None if the embedding model is not loaded.
        """
        if self.embedder is None:
            logger.warning("Embedding model not loaded — cannot embed text")
            return None
        try:
            return self.embedder.encode([text])[0].tolist()
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            return None

    # ── YAML Frontmatter Helpers ─────────────────────────────────────────────

    def _extract_frontmatter(self, content: str) -> dict:
        """Parse YAML frontmatter from markdown content.

        Supports two formats:
        1. Standard YAML frontmatter delimited by ``---``
        2. Legacy format: ``key: VALUE`` on the first line (no delimiters)
        """
        # Standard YAML frontmatter: ---\n...\n---
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
        if match:
            try:
                parsed = yaml.safe_load(match.group(1))
                return parsed if isinstance(parsed, dict) else {}
            except yaml.YAMLError as exc:
                logger.warning("Failed to parse YAML frontmatter: %s", exc)
                return {}

        # Legacy format: first line is "key: value" (no --- delimiters)
        first_line = content.split("\n", 1)[0].strip()
        legacy_match = re.match(r"^(\w+):\s+(.+)$", first_line)
        if legacy_match:
            key = legacy_match.group(1)
            value = legacy_match.group(2).strip()
            return {key: value}

        return {}

    def _strip_frontmatter(self, content: str) -> str:
        """Remove frontmatter from content, returning only the body."""
        # Standard YAML frontmatter
        match = re.match(r"^---\s*\n.*?\n---\s*\n?", content, re.DOTALL)
        if match:
            body = content[match.end():]
            return body.lstrip("\n")

        # Legacy format: strip first "key: value" line
        first_line = content.split("\n", 1)[0].strip()
        legacy_match = re.match(r"^(\w+):\s+(.+)$", first_line)
        if legacy_match:
            rest = content.split("\n", 1)[1] if "\n" in content else ""
            return rest.lstrip("\n")

        return content

    # ── Internal Helpers ─────────────────────────────────────────────────────

    def _normalize_symptoms(self, frontmatter: dict) -> list[str]:
        """Extract and normalize symptoms from frontmatter into a list of strings.

        Handles:
        - ``symptoms: ["depressed_mood", "anhedonia"]`` (YAML list)
        - ``symptoms: "depressed_mood, anhedonia"`` (comma-separated string)
        - ``symptom: DEPRESSED_MOOD`` (legacy single-symptom format)
        """
        symptoms = frontmatter.get("symptoms")
        if symptoms is not None:
            if isinstance(symptoms, list):
                return [str(s).strip() for s in symptoms if s]
            if isinstance(symptoms, str):
                return [s.strip() for s in symptoms.split(",") if s.strip()]

        # Legacy single symptom
        symptom = frontmatter.get("symptom")
        if symptom:
            return [str(symptom).strip()]

        return []

    @staticmethod
    def _detect_json_template(json_file: Any) -> str:
        """Infer the chunk_json_entry template from filename conventions."""
        name = json_file.stem.lower()
        if "medication" in name or "drug" in name:
            return "medication"
        if "scoring" in name or "scale" in name or "phq" in name:
            return "scoring"
        if "symptom" in name or "criterion" in name or "dsm" in name:
            return "symptom"
        return "generic"

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """Whether the service has completed initialization (model loaded)."""
        return self._initialized

    @property
    def knowledge_base_loaded(self) -> bool:
        """Whether the knowledge base has been loaded into the database."""
        return self._knowledge_base_loaded

    # ── Hybrid Retrieval (Task 8) ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        n_results: int | None = None,
        category: str | None = None,
        symptom: str | None = None,
    ) -> list[dict] | None:
        """Main hybrid retrieval: dense + BM25 + RRF fusion + cross-encoder reranking.

        Returns None if the service is not initialized (graceful degradation).
        Falls back to BM25-only if the embedding model fails.
        """
        if not self._initialized:
            return None

        top_k = n_results or self.settings.rag_rerank_top_k
        retrieval_k = self.settings.rag_retrieval_top_k  # candidates per ranker

        # Embed the query; fall back to BM25-only on failure
        query_embedding = self.embed(query)
        if query_embedding is None:
            logger.warning("Embedding failed for query — falling back to BM25-only")
            return self._retrieve_bm25_only(query, top_k, category, symptom)

        db = SessionLocal()
        try:
            # Tune HNSW ef_search for this session (higher = more accurate, slower)
            db.execute(text("SET LOCAL hnsw.ef_search = 150"))

            dense_results = self._dense_search(db, query_embedding, retrieval_k, category, symptom)
            bm25_results = self._bm25_search(db, query, retrieval_k, category, symptom)

            fused = self._rrf_fusion(dense_results, bm25_results)
            reranked = self._rerank(query, fused, top_k)

            return reranked
        except Exception as exc:
            logger.error("Hybrid retrieval failed: %s", exc, exc_info=True)
            return None
        finally:
            db.close()

    def _dense_search(
        self,
        db: Session,
        query_embedding: list[float],
        top_k: int,
        category: str | None,
        symptom: str | None,
    ) -> list[dict]:
        """Cosine-similarity search over pgvector embeddings."""
        q = db.query(KnowledgeChunk).filter(KnowledgeChunk.is_current.is_(True))

        if category:
            q = q.filter(KnowledgeChunk.category == category)
        if symptom:
            q = q.filter(KnowledgeChunk.symptoms.contains([symptom]))

        results = (
            q.order_by(KnowledgeChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )

        return [
            {
                "id": chunk.id,
                "text": chunk.content,
                "metadata": {
                    "source_file": chunk.source_file or "",
                    "category": chunk.category or "",
                    "subcategory": chunk.subcategory or "",
                    "symptoms": chunk.symptoms or [],
                    "chunk_level": chunk.chunk_level or "child",
                    "parent_chunk_id": chunk.parent_chunk_id,
                },
                "rank": rank + 1,
                "method": "dense",
            }
            for rank, chunk in enumerate(results)
        ]

    def _bm25_search(
        self,
        db: Session,
        query: str,
        top_k: int,
        category: str | None,
        symptom: str | None,
    ) -> list[dict]:
        """Full-text BM25 search using PostgreSQL tsvector/tsquery."""
        try:
            ts_query = func.websearch_to_tsquery("english", query)

            q = (
                db.query(KnowledgeChunk)
                .filter(KnowledgeChunk.is_current.is_(True))
                .filter(KnowledgeChunk.search_vector.op("@@")(ts_query))
            )

            if category:
                q = q.filter(KnowledgeChunk.category == category)
            if symptom:
                q = q.filter(KnowledgeChunk.symptoms.contains([symptom]))

            results = (
                q.order_by(func.ts_rank_cd(KnowledgeChunk.search_vector, ts_query).desc())
                .limit(top_k)
                .all()
            )

            return [
                {
                    "id": chunk.id,
                    "text": chunk.content,
                    "metadata": {
                        "source_file": chunk.source_file or "",
                        "category": chunk.category or "",
                        "subcategory": chunk.subcategory or "",
                        "symptoms": chunk.symptoms or [],
                        "chunk_level": chunk.chunk_level or "child",
                        "parent_chunk_id": chunk.parent_chunk_id,
                    },
                    "rank": rank + 1,
                    "method": "bm25",
                }
                for rank, chunk in enumerate(results)
            ]
        except Exception as exc:
            logger.warning("BM25 search failed (non-fatal): %s", exc)
            return []

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Reciprocal Rank Fusion: merge dense and BM25 result lists.

        score(d) = SUM over rankers: weight / (k + rank)
        """
        k = self.settings.rag_rrf_k
        semantic_w = self.settings.rag_semantic_weight
        fulltext_w = self.settings.rag_full_text_weight

        # Accumulate scores keyed by doc id
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for result in dense_results:
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + semantic_w / (k + result["rank"])
            doc_map[doc_id] = result

        for result in bm25_results:
            doc_id = result["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + fulltext_w / (k + result["rank"])
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

        fused: list[dict] = []
        for doc_id in sorted_ids:
            entry = dict(doc_map[doc_id])  # shallow copy
            entry["rrf_score"] = scores[doc_id]
            entry["ranking_method"] = "rrf"
            fused.append(entry)

        return fused

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Cross-encoder reranking for precision. Falls back to RRF order on failure."""
        if not candidates:
            return []

        try:
            reranker = self._load_reranker()
        except Exception as exc:
            logger.warning("Reranker failed to load — returning RRF results: %s", exc)
            return candidates[:top_k]

        try:
            pairs = [(query, c["text"]) for c in candidates]
            scores = reranker.predict(pairs)

            for i, candidate in enumerate(candidates):
                candidate["reranker_score"] = float(scores[i])
                candidate["ranking_method"] = "reranked"

            candidates.sort(key=lambda c: c["reranker_score"], reverse=True)
            return candidates[:top_k]
        except Exception as exc:
            logger.warning("Reranker prediction failed — returning RRF results: %s", exc)
            return candidates[:top_k]

    def _retrieve_bm25_only(
        self,
        query: str,
        n_results: int,
        category: str | None,
        symptom: str | None,
    ) -> list[dict] | None:
        """Fallback retrieval using only BM25 (when embedding model is unavailable)."""
        db = SessionLocal()
        try:
            results = self._bm25_search(db, query, n_results, category, symptom)
            if not results:
                return None
            for r in results:
                r["ranking_method"] = "bm25_only"
            return results
        except Exception as exc:
            logger.error("BM25-only fallback also failed: %s", exc)
            return None
        finally:
            db.close()

    def retrieve_for_symptoms(self, symptoms: list[str]) -> dict[str, list[dict]]:
        """Retrieve DSM-5 criteria and coping strategy docs for each symptom."""
        if not self._initialized:
            return {}

        results: dict[str, list[dict]] = {}
        for symptom in symptoms:
            criteria_docs = self.retrieve(
                query=f"DSM-5 criterion for {symptom}",
                n_results=2,
                symptom=symptom,
            ) or []
            coping_docs = self.retrieve(
                query=f"coping strategies for {symptom}",
                n_results=2,
                category="coping_strategies",
            ) or []
            results[symptom] = criteria_docs + coping_docs
        return results

    def get_chat_context(
        self,
        user_message: str,
        detected_symptoms: list[str],
        n_results: int = 5,
    ) -> str:
        """Build RAG context string for chatbot response with source attribution.

        Retrieves docs for the user's question plus symptom-specific context,
        deduplicates, and formats with source attribution. Limits to 7 docs max.
        """
        if not self._initialized:
            return ""

        docs = self.retrieve(user_message, n_results=n_results) or []

        for symptom in detected_symptoms[:3]:
            symptom_docs = self.retrieve(
                query=f"{symptom} information and guidance",
                n_results=2,
                symptom=symptom,
            ) or []
            docs.extend(symptom_docs)

        # Deduplicate by text prefix
        seen: set[str] = set()
        unique_docs: list[dict] = []
        for doc in docs:
            key = doc["text"][:100]
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        if not unique_docs:
            return ""

        # Format with source attribution, limit to 7
        parts = []
        for doc in unique_docs[:7]:
            source = doc.get("metadata", {}).get("source_file", "knowledge base")
            parts.append(f"[{source}]\n{doc['text']}")

        return "\n\n---\n\n".join(parts)

    def ingest_patient_screening(
        self,
        patient_id: str,
        screening_id: str,
        text: str,
        symptoms_detected: list[dict],
        severity_level: str,
    ) -> None:
        """Ingest a patient's screening into patient_rag_chunks. (Stub — Task 9.)"""
        if not self._initialized or self.embedder is None:
            return

        chunks_to_add = []

        screening_embedding = self.embed(text)
        if screening_embedding is not None:
            chunks_to_add.append(
                PatientRAGChunk(
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
                )
            )

        for det in symptoms_detected:
            sentence_text = det.get("sentence_text", "")
            if not sentence_text:
                continue
            symptom_embedding = self.embed(sentence_text)
            if symptom_embedding is None:
                continue
            chunks_to_add.append(
                PatientRAGChunk(
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
                )
            )

        if chunks_to_add:
            db = SessionLocal()
            try:
                db.add_all(chunks_to_add)
                db.commit()
                logger.info(
                    "Ingested screening %s for patient %s (%d chunks)",
                    screening_id,
                    patient_id[:8],
                    len(chunks_to_add),
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
    ) -> None:
        """Ingest a clinician-uploaded patient document. (Stub — Task 9.)"""
        if not self._initialized or self.embedder is None:
            return

        text_chunks = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        if not text_chunks:
            text_chunks = [content]

        chunks_to_add = []
        for chunk_text_str in text_chunks:
            embedding = self.embed(chunk_text_str)
            if embedding is None:
                continue
            chunks_to_add.append(
                PatientRAGChunk(
                    id=str(uuid.uuid4()),
                    patient_id=patient_id,
                    doc_id=doc_id,
                    content=chunk_text_str,
                    chunk_type="patient_document",
                    metadata_json={
                        "doc_type": doc_type,
                        "title": title,
                    },
                    embedding=embedding,
                )
            )

        if chunks_to_add:
            db = SessionLocal()
            try:
                db.add_all(chunks_to_add)
                db.commit()
                logger.info(
                    "Ingested document '%s' (%s) for patient %s — %d chunks",
                    title,
                    doc_type,
                    patient_id[:8],
                    len(chunks_to_add),
                )
            finally:
                db.close()

    def retrieve_patient_history(
        self,
        patient_id: str,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Retrieve relevant past content for a specific patient. (Stub — Task 9.)"""
        if not self._initialized or self.embedder is None:
            return []

        query_embedding = self.embed(query)
        if query_embedding is None:
            return []

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
                output.append(
                    {
                        "text": chunk.content,
                        "metadata": meta,
                        "distance": 0,
                    }
                )
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
        (Stub — Task 9.)
        """
        clinical_context = self.get_chat_context(
            user_message=user_message,
            detected_symptoms=detected_symptoms,
            n_results=n_results,
        )

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
                    history_parts.append(f"[Previous check-in, severity={severity}]\n{doc['text'][:300]}")
                elif doc_type == "symptom_evidence":
                    symptom = meta.get("symptom_label", meta.get("symptom", ""))
                    history_parts.append(f"[Previous detection: {symptom}]\n{doc['text']}")

            if history_parts:
                parts.append("### Your Previous Check-ins\n" + "\n\n".join(history_parts))

        return "\n\n---\n\n".join(parts)
