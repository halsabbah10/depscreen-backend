"""performance indexes: hot-path columns + HNSW pgvector

Revision ID: fdd8695f0b79
Revises: 72deba99af9e
Create Date: 2026-04-15 18:44:28.730280

Adds indexes purely for read-path performance. No schema changes, no
column additions. Safe to apply on live data: CREATE INDEX IF NOT
EXISTS is idempotent, and the query planner picks up new indexes
transparently without any code change.

What's added (and why each one)

  * medications(is_active) — `patient.py:592` filters active meds on
    every list call; full-scan today.
  * care_plans(status) — filtered in dashboard, patient, patient_context,
    and the scheduler's daily care-plan-review sweep.
  * notifications(user_id, is_read) — composite. list_notifications
    always filters by BOTH; a single-column index on user_id still
    forces a filter pass in memory.
  * allergies(severity) — patient_context ORDERs by severity on every
    LLM-driven chat. Today it's an in-memory sort of the whole patient
    list.
  * screenings(patient_id, created_at DESC) — composite. Every history
    / latest-screening query hits these two together. PG will happily
    use the existing patient_id index for filter, but the composite
    avoids the follow-up sort step.

pgvector HNSW indexes

  * knowledge_chunks(embedding, vector_cosine_ops) — retrieve() does
    a k-NN by cosine distance. Today it's a full-table scan computing
    distance per row. HNSW is O(log n) in practice.
  * patient_rag_chunks(embedding, vector_cosine_ops) — same.

HNSW index build is slow on large tables but our knowledge_chunks has
~23 rows and patient_rag_chunks grows slowly; build cost is negligible.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fdd8695f0b79"
down_revision: str | Sequence[str] | None = "72deba99af9e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add perf indexes. Idempotent — safe to re-run."""
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    # Hot-path B-tree indexes. IF NOT EXISTS so partial applies or reruns
    # don't blow up.
    op.execute("CREATE INDEX IF NOT EXISTS ix_medications_is_active ON medications(is_active)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_care_plans_status ON care_plans(status)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_notifications_user_read "
        "ON notifications(user_id, is_read)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS ix_allergies_severity ON allergies(severity)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_screenings_patient_created "
        "ON screenings(patient_id, created_at DESC)"
    )

    # pgvector HNSW indexes — Postgres-only. Skipping on SQLite dev DBs
    # (no pgvector there anyway).
    if is_postgres:
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
            "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_patient_rag_chunks_embedding_hnsw "
            "ON patient_rag_chunks USING hnsw (embedding vector_cosine_ops)"
        )


def downgrade() -> None:
    """Drop the perf indexes. Queries go back to pre-migration plan."""
    for name in (
        "ix_medications_is_active",
        "ix_care_plans_status",
        "ix_notifications_user_read",
        "ix_allergies_severity",
        "ix_screenings_patient_created",
        "ix_knowledge_chunks_embedding_hnsw",
        "ix_patient_rag_chunks_embedding_hnsw",
    ):
        op.execute(f"DROP INDEX IF EXISTS {name}")
