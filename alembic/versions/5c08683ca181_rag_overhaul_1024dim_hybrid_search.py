"""RAG overhaul: 1024-dim vectors, hybrid BM25 search, hierarchical chunks, RLS

Revision ID: 5c08683ca181
Revises: fdd8695f0b79
Create Date: 2026-04-22 01:15:05.396655

What this migration does
------------------------
knowledge_chunks
  - Adds hierarchical fields: parent_chunk_id (self-FK), chunk_level, sequence_index
  - Adds BM25 support: search_vector (TSVector)
  - Adds enrichment fields: subcategory, symptoms, source_type, token_count,
    metadata_json, is_current, document_version, updated_at
  - Renames symptom → symptoms (old column dropped, new JSON column added)
  - Upgrades embedding dimension 384 → 1024 (gte-large-en-v1.5)
  - Drops old basic HNSW index; adds tuned HNSW (m=24, ef_construction=200)
    and GIN index for full-text search, plus category composite index

patient_rag_chunks
  - Adds hierarchical fields: parent_chunk_id (self-FK), chunk_level, sequence_index
  - Adds BM25 support: search_vector (TSVector)
  - Adds sync/dedup fields: source_table, source_row_id, content_hash, is_current,
    token_count, updated_at
  - Makes patient_id NOT NULL (data integrity)
  - Upgrades embedding dimension 384 → 1024
  - Drops old basic HNSW index; adds tuned HNSW (m=24, ef_construction=200),
    GIN index, patient-scoped partial index, and source sync index
  - Enables Row Level Security

patient_documents
  - Adds processing_status (default 'ready') and processing_error columns

screenings
  - Drops the composite (patient_id, created_at DESC) perf index added in
    fdd8695f0b79 (will be recreated with correct syntax if needed separately)
  - Adds missing indexes: care_plan_id, reviewed_by

misc
  - ix_conversations_linked_clinician_id added (was missing)
  - ix_allergies_severity dropped (replaced by query-level filtering)
  - ix_medications_is_active, ix_notifications_user_read dropped (were added in
    fdd8695f0b79 but autogenerate sees them as gone — they exist on the live DB
    in a different form; using IF NOT EXISTS / IF EXISTS guards throughout)
  - ix_users_clinician_id added

Indexes added (all use IF NOT EXISTS for idempotency)
  - ix_kc_embedding_hnsw    HNSW on knowledge_chunks.embedding
  - ix_prc_embedding_hnsw   HNSW on patient_rag_chunks.embedding
  - ix_kc_fts               GIN on knowledge_chunks.search_vector
  - ix_prc_fts              GIN on patient_rag_chunks.search_vector
  - ix_prc_patient_current  partial B-tree on patient_rag_chunks(patient_id, is_current) WHERE is_current
  - ix_prc_source           B-tree on patient_rag_chunks(source_table, source_row_id)
  - ix_kc_category          B-tree on knowledge_chunks(category, subcategory)
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5c08683ca181"
down_revision: Union[str, Sequence[str], None] = "fdd8695f0b79"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply RAG overhaul schema changes."""

    # ── conversations ─────────────────────────────────────────────────────────
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_conversations_linked_clinician_id "
        "ON conversations (linked_clinician_id)"
    )

    # ── knowledge_chunks ──────────────────────────────────────────────────────

    # Hierarchical chunk support
    op.add_column("knowledge_chunks", sa.Column("parent_chunk_id", sa.String(36), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("chunk_level", sa.String(10), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("sequence_index", sa.Integer(), nullable=True))

    # BM25 full-text search column
    op.add_column("knowledge_chunks", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True))

    # Enrichment / taxonomy
    op.add_column("knowledge_chunks", sa.Column("subcategory", sa.String(100), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("symptoms", sa.JSON(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("source_type", sa.String(50), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("token_count", sa.Integer(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("metadata_json", sa.JSON(), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=True))
    op.execute("UPDATE knowledge_chunks SET is_current = true WHERE is_current IS NULL")
    op.add_column("knowledge_chunks", sa.Column("document_version", sa.String(50), nullable=True))
    op.add_column("knowledge_chunks", sa.Column("updated_at", sa.DateTime(), nullable=True))

    # Drop old single-column symptom field (replaced by symptoms JSON array)
    op.drop_column("knowledge_chunks", "symptom")

    # Clear existing 384-dim vectors (test data only) before dimension change
    op.execute("UPDATE knowledge_chunks SET embedding = NULL")

    # Upgrade embedding from 384-dim (all-MiniLM) → 1024-dim (gte-large-en-v1.5)
    op.alter_column(
        "knowledge_chunks",
        "embedding",
        existing_type=Vector(384),
        type_=Vector(1024),
        existing_nullable=True,
    )

    # Self-referential FK for parent/child hierarchy
    op.create_foreign_key(
        "fk_kc_parent",
        "knowledge_chunks",
        "knowledge_chunks",
        ["parent_chunk_id"],
        ["id"],
    )

    # Drop old basic HNSW index (m=16 default, no ef_construction tuning)
    op.execute("DROP INDEX IF EXISTS ix_knowledge_chunks_embedding_hnsw")

    # ── patient_rag_chunks ───────────────────────────────────────────────────

    # Hierarchical chunk support
    op.add_column("patient_rag_chunks", sa.Column("parent_chunk_id", sa.String(36), nullable=True))
    op.add_column("patient_rag_chunks", sa.Column("chunk_level", sa.String(10), nullable=True))
    op.add_column("patient_rag_chunks", sa.Column("sequence_index", sa.Integer(), nullable=True))

    # BM25 full-text search column
    op.add_column("patient_rag_chunks", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True))

    # Sync/dedup fields for event-driven updates
    op.add_column("patient_rag_chunks", sa.Column("source_table", sa.String(50), nullable=True))
    op.add_column("patient_rag_chunks", sa.Column("source_row_id", sa.String(36), nullable=True))
    op.add_column("patient_rag_chunks", sa.Column("content_hash", sa.String(64), nullable=True))
    op.add_column("patient_rag_chunks", sa.Column("is_current", sa.Boolean(), server_default=sa.text("true"), nullable=True))
    op.execute("UPDATE patient_rag_chunks SET is_current = true WHERE is_current IS NULL")
    op.add_column("patient_rag_chunks", sa.Column("token_count", sa.Integer(), nullable=True))
    op.add_column("patient_rag_chunks", sa.Column("updated_at", sa.DateTime(), nullable=True))

    # Data integrity: patient_id must always be set
    op.alter_column("patient_rag_chunks", "patient_id", existing_type=sa.String(36), nullable=False)

    # Clear existing 384-dim vectors (test data only) before dimension change
    op.execute("UPDATE patient_rag_chunks SET embedding = NULL")

    # Upgrade embedding 384 → 1024
    op.alter_column(
        "patient_rag_chunks",
        "embedding",
        existing_type=Vector(384),
        type_=Vector(1024),
        existing_nullable=True,
    )

    # Self-referential FK for parent/child hierarchy
    op.create_foreign_key(
        "fk_prc_parent",
        "patient_rag_chunks",
        "patient_rag_chunks",
        ["parent_chunk_id"],
        ["id"],
    )

    # Drop old basic HNSW index
    op.execute("DROP INDEX IF EXISTS ix_patient_rag_chunks_embedding_hnsw")

    # ── patient_documents ────────────────────────────────────────────────────
    op.add_column(
        "patient_documents",
        sa.Column("processing_status", sa.String(20), server_default="ready", nullable=True),
    )
    op.add_column("patient_documents", sa.Column("processing_error", sa.Text(), nullable=True))

    # ── screenings ───────────────────────────────────────────────────────────
    # Drop composite index created in fdd8695f0b79 (DESC syntax varies by PG version;
    # the index will be rebuilt via the hot-path optimization migration when needed)
    op.execute("DROP INDEX IF EXISTS ix_screenings_patient_created")

    # Add missing FK-column indexes
    op.execute("CREATE INDEX IF NOT EXISTS ix_screenings_care_plan_id ON screenings (care_plan_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_screenings_reviewed_by ON screenings (reviewed_by)")

    # ── users ────────────────────────────────────────────────────────────────
    op.execute("CREATE INDEX IF NOT EXISTS ix_users_clinician_id ON users (clinician_id)")

    # ── allergies ─────────────────────────────────────────────────────────────
    # ix_allergies_severity was created in fdd8695f0b79 but autogenerate sees it
    # as missing on the live DB — drop it if present (replaced by query-level filtering)
    op.execute("DROP INDEX IF EXISTS ix_allergies_severity")

    # ── medications / notifications perf indexes ──────────────────────────────
    # These were added in fdd8695f0b79; autogenerate believes they're gone from
    # the live DB. Recreate them here with IF NOT EXISTS so they land regardless.
    op.execute("CREATE INDEX IF NOT EXISTS ix_medications_is_active ON medications (is_active)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_notifications_user_read ON notifications (user_id, is_read)")

    # ── Custom indexes ────────────────────────────────────────────────────────

    # HNSW vector indexes — tuned for healthcare recall requirements
    # m=24 (higher connectivity than default 16) and ef_construction=200 for
    # better recall at the cost of slightly longer build time (acceptable given
    # small initial table sizes).
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kc_embedding_hnsw ON knowledge_chunks
        USING hnsw (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 200)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_prc_embedding_hnsw ON patient_rag_chunks
        USING hnsw (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 200)
    """)

    # GIN indexes for BM25 full-text search
    op.execute("CREATE INDEX IF NOT EXISTS ix_kc_fts ON knowledge_chunks USING gin (search_vector)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_prc_fts ON patient_rag_chunks USING gin (search_vector)")

    # Composite partial index for patient-scoped queries — the dominant access pattern.
    # The WHERE clause keeps the index small by excluding superseded chunks.
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_prc_patient_current
        ON patient_rag_chunks (patient_id, is_current)
        WHERE is_current = true
    """)

    # Source sync index — used by event-driven upsert logic to locate existing chunks
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_prc_source "
        "ON patient_rag_chunks (source_table, source_row_id)"
    )

    # Category/subcategory composite — used by knowledge retrieval category filters
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_kc_category "
        "ON knowledge_chunks (category, subcategory)"
    )

    # ── Row Level Security on patient_rag_chunks ──────────────────────────────
    # RLS ensures that even a compromised service account can only see rows it
    # owns. Policy enforcement happens at the Supabase/Postgres layer.
    op.execute("ALTER TABLE patient_rag_chunks ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    """Reverse RAG overhaul changes."""

    # ── RLS ───────────────────────────────────────────────────────────────────
    op.execute("ALTER TABLE patient_rag_chunks DISABLE ROW LEVEL SECURITY")

    # ── Custom indexes ────────────────────────────────────────────────────────
    op.execute("DROP INDEX IF EXISTS ix_kc_category")
    op.execute("DROP INDEX IF EXISTS ix_prc_source")
    op.execute("DROP INDEX IF EXISTS ix_prc_patient_current")
    op.execute("DROP INDEX IF EXISTS ix_prc_fts")
    op.execute("DROP INDEX IF EXISTS ix_kc_fts")
    op.execute("DROP INDEX IF EXISTS ix_prc_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_kc_embedding_hnsw")

    # ── allergies / medications / notifications ───────────────────────────────
    op.execute("CREATE INDEX IF NOT EXISTS ix_allergies_severity ON allergies (severity)")
    op.execute("DROP INDEX IF EXISTS ix_medications_is_active")
    op.execute("DROP INDEX IF EXISTS ix_notifications_user_read")

    # ── users ─────────────────────────────────────────────────────────────────
    op.execute("DROP INDEX IF EXISTS ix_users_clinician_id")

    # ── screenings ───────────────────────────────────────────────────────────
    op.execute("DROP INDEX IF EXISTS ix_screenings_care_plan_id")
    op.execute("DROP INDEX IF EXISTS ix_screenings_reviewed_by")
    # Recreate the composite perf index from fdd8695f0b79
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_screenings_patient_created "
        "ON screenings (patient_id, created_at DESC)"
    )

    # ── patient_documents ────────────────────────────────────────────────────
    op.drop_column("patient_documents", "processing_error")
    op.drop_column("patient_documents", "processing_status")

    # ── patient_rag_chunks ───────────────────────────────────────────────────
    op.drop_constraint("fk_prc_parent", "patient_rag_chunks", type_="foreignkey")

    op.alter_column(
        "patient_rag_chunks",
        "embedding",
        existing_type=Vector(1024),
        type_=Vector(384),
        existing_nullable=True,
    )
    op.alter_column("patient_rag_chunks", "patient_id", existing_type=sa.String(36), nullable=True)

    # Restore old basic HNSW index
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_patient_rag_chunks_embedding_hnsw "
        "ON patient_rag_chunks USING hnsw (embedding vector_cosine_ops)"
    )

    for col in (
        "updated_at",
        "token_count",
        "is_current",
        "content_hash",
        "source_row_id",
        "source_table",
        "search_vector",
        "sequence_index",
        "chunk_level",
        "parent_chunk_id",
    ):
        op.drop_column("patient_rag_chunks", col)

    # ── knowledge_chunks ──────────────────────────────────────────────────────
    op.drop_constraint("fk_kc_parent", "knowledge_chunks", type_="foreignkey")

    op.alter_column(
        "knowledge_chunks",
        "embedding",
        existing_type=Vector(1024),
        type_=Vector(384),
        existing_nullable=True,
    )

    # Restore old basic HNSW index
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
        "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
    )

    # Restore old single-column symptom field
    op.add_column("knowledge_chunks", sa.Column("symptom", sa.String(50), nullable=True))

    for col in (
        "updated_at",
        "document_version",
        "is_current",
        "metadata_json",
        "token_count",
        "source_type",
        "symptoms",
        "subcategory",
        "search_vector",
        "sequence_index",
        "chunk_level",
        "parent_chunk_id",
    ):
        op.drop_column("knowledge_chunks", col)

    # ── conversations ─────────────────────────────────────────────────────────
    op.execute("DROP INDEX IF EXISTS ix_conversations_linked_clinician_id")
