"""Unit tests for RAG components — no external dependencies required."""



class TestDocumentExtractor:
    """Tests for PDF/DOCX text extraction."""

    def test_extract_text_txt_success(self):
        """Plain text files return content directly."""
        from app.services.document_extractor import extract_text

        content = b"I have been feeling hopeless for weeks."
        result = extract_text(content, "journal.txt")

        assert result is not None
        assert result.text == "I have been feeling hopeless for weeks."
        assert result.method == "text"

    def test_extract_text_md_success(self):
        """Markdown files return content directly."""
        from app.services.document_extractor import extract_text

        content = b"## Depression Overview\n\nDepression is a common condition."
        result = extract_text(content, "overview.md")

        assert result is not None
        assert "Depression" in result.text
        assert result.method == "text"

    def test_extract_text_corrupted_pdf(self):
        """Corrupted PDF returns None (graceful failure)."""
        from app.services.document_extractor import extract_text

        result = extract_text(b"not a real pdf", "broken.pdf")
        assert result is None

    def test_extract_text_empty_content(self):
        """Empty content returns None."""
        from app.services.document_extractor import extract_text

        result = extract_text(b"", "empty.pdf")
        assert result is None

    def test_extract_text_unsupported_format(self):
        """Unsupported format returns None."""
        from app.services.document_extractor import extract_text

        result = extract_text(b"some content", "data.xlsx")
        assert result is None

    def test_extract_text_csv_success(self):
        """CSV files return content as text."""
        from app.services.document_extractor import extract_text

        content = b"date,mood,score\n2026-04-01,sad,7\n2026-04-02,better,4"
        result = extract_text(content, "mood_log.csv")

        assert result is not None
        assert "mood" in result.text
        assert result.method == "text"


class TestAdaptiveChunking:

    def test_chunk_markdown_by_headers(self):
        """Markdown with ## headers splits into sections."""
        from app.services.chunking import chunk_text
        text = "## Section One\n\nContent for section one is here.\n\n## Section Two\n\nContent for section two."
        chunks = chunk_text(text, source_type="markdown", max_tokens=500)
        assert len(chunks) >= 2
        assert all(c.content for c in chunks)

    def test_chunk_respects_max_tokens(self):
        """No child chunk exceeds max_tokens."""
        from app.services.chunking import chunk_text
        text = "This is a sentence about depression. " * 200
        chunks = chunk_text(text, source_type="text", max_tokens=100)
        child_chunks = [c for c in chunks if c.chunk_level == "child"]
        for chunk in child_chunks:
            assert chunk.token_count <= 150  # Allow buffer for estimation

    def test_chunk_creates_hierarchy(self):
        """Large sections create parent + child chunks."""
        from app.services.chunking import chunk_text
        text = "## Big Section\n\n" + "This is a long sentence about clinical aspects. " * 100
        chunks = chunk_text(text, source_type="markdown", max_tokens=50)
        parents = [c for c in chunks if c.chunk_level == "parent"]
        children = [c for c in chunks if c.chunk_level == "child"]
        assert len(parents) >= 1
        assert len(children) >= 2

    def test_chunk_json_medication(self):
        """Medication JSON converts to natural language."""
        from app.services.chunking import chunk_json_entry
        entry = {"drug": "sertraline", "class": "SSRI", "brand_names": ["Zoloft"], "typical_dose": "50-200mg daily", "common_side_effects": ["nausea", "insomnia"]}
        chunk = chunk_json_entry(entry, template="medication")
        assert "sertraline" in chunk.content.lower() or "Sertraline" in chunk.content
        assert "SSRI" in chunk.content
        assert chunk.chunk_level == "child"

    def test_chunk_json_scoring(self):
        """Scoring JSON converts to natural language."""
        from app.services.chunking import chunk_json_entry
        entry = {"tool": "PHQ-9", "range": "10-14", "severity": "moderate", "clinical_action": "Consider counseling"}
        chunk = chunk_json_entry(entry, template="scoring")
        assert "PHQ-9" in chunk.content
        assert "moderate" in chunk.content

    def test_chunk_minimum_size_filter(self):
        """Chunks below minimum size are filtered out."""
        from app.services.chunking import chunk_text
        text = "## Header\n\nOk.\n\n## Another\n\nThis section has real clinical content here."
        chunks = chunk_text(text, source_type="markdown", max_tokens=500, min_chars=20)
        child_chunks = [c for c in chunks if c.chunk_level == "child"]
        assert all(len(c.content) >= 20 for c in child_chunks)

    def test_chunk_content_hash_computed(self):
        """Every chunk has a SHA-256 content hash."""
        from app.services.chunking import chunk_text
        text = "## Test\n\nSome content for hashing verification."
        chunks = chunk_text(text, source_type="markdown", max_tokens=500)
        for chunk in chunks:
            assert len(chunk.content_hash) == 64  # SHA-256 hex digest

    def test_chunk_empty_text_returns_empty(self):
        """Empty text returns no chunks."""
        from app.services.chunking import chunk_text
        assert chunk_text("", source_type="text") == []
        assert chunk_text("   ", source_type="text") == []


class TestRAGServiceInit:
    """Tests for RAGService frontmatter helpers and properties."""

    def test_frontmatter_extraction(self):
        """YAML frontmatter is extracted correctly."""
        from app.core.config import get_settings
        from app.services.rag import RAGService
        rag = RAGService(get_settings())

        content = '---\ntitle: "Test"\ncategory: medications\nsymptoms: ["depressed_mood"]\n---\n\n## Body\n\nContent here.'
        meta = rag._extract_frontmatter(content)
        assert meta["title"] == "Test"
        assert meta["category"] == "medications"
        assert meta["symptoms"] == ["depressed_mood"]

    def test_frontmatter_legacy_format(self):
        """Legacy 'symptom: X' format on first line is handled."""
        from app.core.config import get_settings
        from app.services.rag import RAGService
        rag = RAGService(get_settings())

        content = "symptom: DEPRESSED_MOOD\n\n## Depressed Mood\n\nContent."
        meta = rag._extract_frontmatter(content)
        assert meta.get("symptom") == "DEPRESSED_MOOD"

    def test_strip_frontmatter(self):
        """Frontmatter is stripped, body remains."""
        from app.core.config import get_settings
        from app.services.rag import RAGService
        rag = RAGService(get_settings())

        content = '---\ntitle: "Test"\n---\n\n## Body\n\nContent here.'
        body = rag._strip_frontmatter(content)
        assert body.startswith("## Body")
        assert "---" not in body

    def test_strip_frontmatter_legacy(self):
        """Legacy format: first metadata line stripped."""
        from app.core.config import get_settings
        from app.services.rag import RAGService
        rag = RAGService(get_settings())

        content = "symptom: DEPRESSED_MOOD\n\n## Content"
        body = rag._strip_frontmatter(content)
        assert body.startswith("## Content")


class TestPatientContextService:
    """Tests for PatientContextService.build_context() filtering parameters."""

    def _make_user(self):
        from unittest.mock import MagicMock
        user = MagicMock()
        user.full_name = "Test Patient"
        user.date_of_birth = None
        user.gender = "male"
        user.nationality = "Bahraini"
        user.language_preference = "en"
        user.timezone = "Asia/Bahrain"
        user.cpr_number = "123456789"
        user.medical_record_number = "MRN001"
        user.blood_type = "O+"
        user.id = "test-patient"
        user.reddit_username = None
        user.twitter_username = None
        return user

    def _make_db(self):
        from unittest.mock import MagicMock
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.filter.return_value.first.return_value = None
        db.query.return_value.filter_by.return_value.all.return_value = []
        db.query.return_value.filter_by.return_value.first.return_value = None
        db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        db.query.return_value.filter_by.return_value.order_by.return_value.all.return_value = []
        db.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
        db.query.return_value.filter_by.return_value.order_by.return_value.first.return_value = None
        return db

    def test_build_context_sections_filter(self):
        """sections parameter limits output to specified sections only."""
        from app.services.patient_context import PatientContextService

        svc = PatientContextService()
        user = self._make_user()
        db = self._make_db()

        # Full context should include demographics
        full = svc.build_context(user, db)
        assert "Test Patient" in full or "male" in full.lower()

        # medications-only should NOT include demographics name or gender
        filtered = svc.build_context(user, db, sections=["medications"])
        assert "Test Patient" not in filtered
        assert "male" not in filtered.lower()

    def test_build_context_sections_filter_empty_returns_empty_state(self):
        """Requesting a section with no data returns the empty-state message."""
        from app.services.patient_context import PatientContextService

        svc = PatientContextService()
        user = self._make_user()
        db = self._make_db()

        # medications section only — mock returns no meds, so no content produced
        result = svc.build_context(user, db, sections=["medications"])
        assert result == "### Patient Medical Profile\nNo medical history recorded yet."

    def test_include_pii_false(self):
        """include_pii=False excludes CPR and MRN from output."""
        from app.services.patient_context import PatientContextService

        svc = PatientContextService()
        user = self._make_user()
        db = self._make_db()

        result = svc.build_context(user, db, include_pii=False)
        assert "123456789" not in result
        assert "MRN001" not in result

    def test_include_pii_true_includes_medical_identifiers(self):
        """include_pii=True (default) includes redacted CPR and MRN."""
        from app.services.patient_context import PatientContextService

        svc = PatientContextService()
        user = self._make_user()
        db = self._make_db()

        result = svc.build_context(user, db, include_pii=True)
        # CPR is redacted (last 4 shown), MRN shown in full per existing logic
        assert "MRN001" in result
        assert "Medical Identifiers" in result

    def test_default_sections_none_is_backwards_compatible(self):
        """sections=None includes all sections (backwards compatible)."""
        from app.services.patient_context import PatientContextService

        svc = PatientContextService()
        user = self._make_user()
        db = self._make_db()

        result = svc.build_context(user, db)
        # With no DB data except user fields, demographics should appear
        assert "Test Patient" in result or "Male" in result or "Bahraini" in result
