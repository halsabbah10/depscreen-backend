"""
Database models and session management.

Tables:
  User, Screening, ChatMessage, PatientDocument, EmergencyContact,
  Medication, Allergy, Diagnosis, ScreeningSchedule, Appointment,
  Notification, Conversation, CarePlan, AuditLog

Uses SQLAlchemy with PostgreSQL + pgvector. SQLite is not supported.
"""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import TSVECTOR as TSVector
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")

# SQLite (used in tests) does not support connection-pool tuning kwargs;
# only apply them for real PostgreSQL connections.
pool_config: dict = (
    {}
    if _is_sqlite
    else {
        # Bumped from 10 → 20 after dashboard load profiling: on a moderately
        # active clinician session (dashboard + patient detail + screenings)
        # we'd saturate the old pool and spend 50–200 ms per request in
        # queue. 20 + 40 overflow comfortably covers concurrent loads without
        # overwhelming Supabase's session pooler.
        "pool_size": 20,
        "max_overflow": 40,
        # 30 min is safer than the default 1 h for long-lived session-pooler
        # connections which can go stale without warning.
        "pool_recycle": 1800,
        "pool_pre_ping": True,  # keep — Supabase pooler needs it
    }
)

engine = create_engine(settings.database_url, **pool_config)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
class Base(DeclarativeBase):
    pass


# ── Users ─────────────────────────────────────────────────────────────────────


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, index=True)  # patient, clinician, admin
    is_active = Column(Boolean, default=True)

    # Patient ↔ Clinician relationship
    clinician_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    clinician_code = Column(String(10), unique=True, nullable=True)

    # Demographics (Phase 2)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String(20), nullable=True)  # male, female, prefer_not_to_say
    nationality = Column(String(50), nullable=True)
    cpr_number = Column(String(9), unique=True, nullable=True, index=True)  # Bahrain CPR
    medical_record_number = Column(String(50), nullable=True)  # Hospital MRN (clinician-assigned)
    blood_type = Column(String(5), nullable=True)  # A+, A-, B+, B-, AB+, AB-, O+, O-

    # Contact
    phone = Column(String(20), nullable=True)
    profile_picture_url = Column(String(500), nullable=True)

    # Social media (for screening analysis)
    reddit_username = Column(String(50), nullable=True)
    twitter_username = Column(String(50), nullable=True)

    # Clinician-specific
    specialization = Column(String(100), nullable=True)
    license_number = Column(String(50), nullable=True)

    # Preferences
    language_preference = Column(String(10), default="en")  # en, ar
    timezone = Column(String(50), default="Asia/Bahrain")
    email_notifications = Column(Boolean, default=True)
    notification_preferences = Column(JSON, nullable=True)  # {email: bool, sms: bool, in_app: bool}

    # Onboarding
    onboarding_completed = Column(Boolean, default=False)

    # Tracking
    last_login_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patients = relationship("User", backref="clinician", remote_side=[id])
    screenings = relationship("Screening", back_populates="patient", foreign_keys="Screening.patient_id")
    emergency_contacts = relationship("EmergencyContact", back_populates="patient")
    medications = relationship("Medication", back_populates="patient", cascade="all, delete-orphan")
    allergies = relationship("Allergy", back_populates="patient", cascade="all, delete-orphan")
    diagnoses = relationship("Diagnosis", back_populates="patient", cascade="all, delete-orphan")
    screening_schedules = relationship(
        "ScreeningSchedule", back_populates="patient", foreign_keys="ScreeningSchedule.patient_id"
    )
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")


# ── Screenings ────────────────────────────────────────────────────────────────


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Input
    text = Column(Text, nullable=False)
    source = Column(String(50), default="checkin", index=True)

    # Symptom analysis
    symptom_data = Column(JSON, nullable=True)
    symptom_count = Column(Integer, default=0)
    severity_level = Column(String(20), nullable=True, index=True)

    # Verification + Explanation
    verification_data = Column(JSON, nullable=True)
    explanation_data = Column(JSON, nullable=True)
    rag_context = Column(JSON, nullable=True)

    # Final output
    final_prediction = Column(String(50), nullable=False)
    final_confidence = Column(Float, nullable=False)
    confidence_adjusted = Column(Boolean, default=False)

    # Flags
    flagged_for_review = Column(Boolean, default=False, index=True)
    adversarial_warning = Column(Text, nullable=True)

    # Clinical workflow
    triage_status = Column(String(50), default="new", index=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    clinician_notes = Column(Text, nullable=True)
    next_action = Column(String(255), nullable=True)
    next_action_date = Column(DateTime, nullable=True)

    # Link to care plan (if active)
    care_plan_id = Column(String(36), ForeignKey("care_plans.id"), nullable=True, index=True)

    # Relationships
    patient = relationship("User", back_populates="screenings", foreign_keys=[patient_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    chat_messages = relationship("ChatMessage", back_populates="screening", cascade="all, delete-orphan")
    care_plan = relationship("CarePlan", back_populates="screenings")


# ── Chat Messages ─────────────────────────────────────────────────────────────


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String(36), primary_key=True, index=True)
    # Can belong to a screening (screening followup) or a conversation (standalone)
    screening_id = Column(String(36), ForeignKey("screenings.id"), nullable=True, index=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True, index=True)
    role = Column(String(20), nullable=False)  # "user", "assistant", "clinician"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    screening = relationship("Screening", back_populates="chat_messages")
    conversation = relationship("Conversation", back_populates="messages")


# ── Conversations (Standalone Chat) ──────────────────────────────────────────


class Conversation(Base):
    """Standalone conversations not tied to a specific screening."""

    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), default="New Conversation")
    context_type = Column(String(50), default="general")  # general, screening_followup, clinician_direct
    linked_screening_id = Column(String(36), nullable=True)  # Optional screening context
    linked_clinician_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)  # For clinician-direct
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")


# ── Patient Documents ─────────────────────────────────────────────────────────


class PatientDocument(Base):
    __tablename__ = "patient_documents"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    uploaded_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    doc_type = Column(String(50), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    file_url = Column(String(500), nullable=True)  # Supabase Storage URL (for PDFs/images)
    file_size = Column(Integer, nullable=True)  # bytes
    processing_status = Column(String(20), default="ready")  # processing, ready, failed
    processing_error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Emergency Contacts ────────────────────────────────────────────────────────


class EmergencyContact(Base):
    __tablename__ = "emergency_contacts"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    contact_name = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    relation = Column(String(50), nullable=False)
    is_primary = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("User", back_populates="emergency_contacts")


# ── Medications ───────────────────────────────────────────────────────────────


class Medication(Base):
    __tablename__ = "medications"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)  # Drug name (RxNorm-autocomplete supported)
    dosage = Column(String(100), nullable=True)  # e.g., "50mg"
    frequency = Column(String(50), nullable=True)  # daily, twice_daily, weekly, as_needed
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)  # null = ongoing
    prescribed_by = Column(String(255), nullable=True)  # clinician name or external
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", back_populates="medications")


# ── Allergies ─────────────────────────────────────────────────────────────────


class Allergy(Base):
    __tablename__ = "allergies"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    allergen = Column(String(255), nullable=False)
    allergy_type = Column(String(50), nullable=True)  # medication, food, environmental, other
    severity = Column(String(50), nullable=True)  # mild, moderate, severe, life_threatening
    reaction = Column(Text, nullable=True)  # description of allergic reaction
    diagnosed_date = Column(Date, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("User", back_populates="allergies")


# ── Diagnoses ─────────────────────────────────────────────────────────────────


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    condition = Column(String(255), nullable=False)
    icd10_code = Column(String(20), nullable=True)  # ICD-10 code (optional, API-assisted)
    diagnosed_date = Column(Date, nullable=True)
    status = Column(String(50), default="active")  # active, remission, resolved
    diagnosed_by = Column(String(255), nullable=True)  # clinician name or external
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", back_populates="diagnoses")


# ── Screening Schedule ────────────────────────────────────────────────────────


class ScreeningSchedule(Base):
    __tablename__ = "screening_schedules"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    frequency = Column(String(50), nullable=False)  # weekly, biweekly, monthly, custom
    custom_days = Column(Integer, nullable=True)  # for custom frequency
    day_of_week = Column(Integer, nullable=True)  # 0=Monday ... 6=Sunday (for weekly/biweekly)
    preferred_time = Column(String(10), nullable=True)  # HH:MM format
    next_due_at = Column(DateTime, nullable=True, index=True)
    last_completed_at = Column(DateTime, nullable=True)
    last_reminder_sent_at = Column(DateTime, nullable=True)  # de-dupe reminder emails
    assigned_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", back_populates="screening_schedules", foreign_keys=[patient_id])
    assigned_clinician = relationship("User", foreign_keys=[assigned_by])


# ── Appointments ──────────────────────────────────────────────────────────────


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    clinician_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    scheduled_at = Column(DateTime, nullable=False, index=True)
    duration_minutes = Column(Integer, default=60)
    appointment_type = Column(String(50), default="followup")  # intake, followup, crisis, review
    status = Column(String(50), default="scheduled", index=True)  # scheduled, confirmed, completed, cancelled, no_show
    notes = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    reminder_sent_at = Column(DateTime, nullable=True)  # de-dupe 24h reminder emails
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", foreign_keys=[patient_id])
    clinician = relationship("User", foreign_keys=[clinician_id])


# ── Notifications ─────────────────────────────────────────────────────────────


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, index=True, default=lambda: str(uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    notification_type = Column(String(50), nullable=False, index=True)
    # Types: screening_due, screening_missed, new_message, appointment_reminder,
    #        care_plan_updated, crisis_alert, document_uploaded
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    link = Column(String(500), nullable=True)  # Frontend route to navigate to
    is_read = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="notifications")


# ── Care Plans ────────────────────────────────────────────────────────────────


class CarePlan(Base):
    __tablename__ = "care_plans"

    id = Column(String(36), primary_key=True, index=True)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    clinician_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    template_name = Column(String(100), nullable=True)  # which template was used
    goals = Column(JSON, nullable=True)  # [{text, target_date, status}]
    interventions = Column(JSON, nullable=True)  # [{name, frequency, instructions, assigned_date}]
    review_date = Column(Date, nullable=True)
    status = Column(String(50), default="active", index=True)  # active, review_needed, completed, archived
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", foreign_keys=[patient_id])
    clinician = relationship("User", foreign_keys=[clinician_id])
    screenings = relationship("Screening", back_populates="care_plan")


# ── Audit Log ─────────────────────────────────────────────────────────────────


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), default="")
    resource_id = Column(String(36), nullable=True)
    ip_address = Column(String(45), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


# ── Email Deliveries (Resend webhook tracking) ──────────────────────────────
#
# Records every outbound transactional email. Status transitions are driven
# by Resend webhook events: queued → sent → delivered → opened / clicked,
# or sent → bounced / complained. The `events` column is an append-only
# audit trail of every hook we received for this email.


class EmailDelivery(Base):
    __tablename__ = "email_deliveries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    resend_email_id = Column(String(64), nullable=True, unique=True, index=True)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    recipient = Column(String(255), nullable=False, index=True)
    subject = Column(String(500), nullable=False)
    template_key = Column(String(64), nullable=False, index=True)  # e.g., 'welcome', 'appointment_reminder'
    status = Column(
        String(32),
        nullable=False,
        default="queued",
        index=True,
    )  # queued, sent, delivered, delivery_delayed, opened, clicked, bounced, complained, failed
    last_event_at = Column(DateTime, nullable=True)
    events = Column(JSON, nullable=True)  # list of {type, at, raw}
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# ── RAG Vector Chunks ───────────────────────────────────────────────────────


class KnowledgeChunk(Base):
    """Clinical knowledge base chunks with pgvector embeddings and BM25 search."""

    __tablename__ = "knowledge_chunks"

    id = Column(String(36), primary_key=True)
    parent_chunk_id = Column(String(36), ForeignKey("knowledge_chunks.id"), nullable=True)
    chunk_level = Column(String(10), default="child")  # 'parent' or 'child'
    sequence_index = Column(Integer, default=0)
    content = Column(Text, nullable=False)
    search_vector = Column(TSVector)  # BM25 full-text search
    category = Column(String(50))  # dsm5_criteria, medications, etc.
    subcategory = Column(String(100))
    symptoms = Column(JSON, default=list)  # Array of symptom tags
    source_file = Column(String(255))
    source_type = Column(String(50))  # markdown, json, pdf, admin_upload
    token_count = Column(Integer)
    embedding = Column(Vector(1024))  # gte-large-en-v1.5
    metadata_json = Column(JSON, nullable=True)
    is_current = Column(Boolean, default=True, server_default=sa_text("true"))
    document_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent = relationship("KnowledgeChunk", remote_side=[id], lazy="select")


class PatientRAGChunk(Base):
    """Per-patient RAG chunks with pgvector embeddings and BM25 search."""

    __tablename__ = "patient_rag_chunks"

    id = Column(String(36), primary_key=True)
    parent_chunk_id = Column(String(36), ForeignKey("patient_rag_chunks.id"), nullable=True)
    chunk_level = Column(String(10), default="child")
    sequence_index = Column(Integer, default=0)
    patient_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    screening_id = Column(String(36), nullable=True)
    doc_id = Column(String(36), nullable=True)
    content = Column(Text, nullable=False)
    search_vector = Column(TSVector)
    chunk_type = Column(String(50))  # screening_text, symptom_evidence, patient_document, chat_summary
    source_table = Column(String(50))  # screenings, patient_documents, chat_messages
    source_row_id = Column(String(36))  # FK to source row for sync
    content_hash = Column(String(64))  # SHA-256 for dedup
    is_current = Column(Boolean, default=True, server_default=sa_text("true"))
    token_count = Column(Integer)
    embedding = Column(Vector(1024))
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parent = relationship("PatientRAGChunk", remote_side=[id], lazy="select")
    patient = relationship("User", foreign_keys=[patient_id], lazy="select")


# ── Session Management ────────────────────────────────────────────────────────


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database: enable pgvector, then sync schema via Alembic.

    Two-path startup:
      1. If the database already carries an alembic_version row, run
         `alembic upgrade head` — every pending migration applies forward.
      2. If not (fresh environment, or an existing DB adopting Alembic for
         the first time), we Base.metadata.create_all() for raw bootstrap,
         then stamp the DB at head so subsequent boots use the upgrade path.

    This lets a freshly-provisioned Postgres stand up from empty AND lets
    the already-live Supabase database adopt Alembic cleanly without
    trying to re-create tables it already has.
    """
    import logging
    from pathlib import Path

    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import text

    from alembic import command

    log = logging.getLogger(__name__)

    # pgvector extension — required for embedding similarity search.
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Locate alembic.ini in the project root (one level up from this module)
    backend_root = Path(__file__).resolve().parents[2]
    alembic_ini = backend_root / "alembic.ini"

    if not alembic_ini.exists():
        # Alembic not configured on this checkout — fall back to bare create_all
        log.warning("alembic.ini missing; falling back to Base.metadata.create_all()")
        Base.metadata.create_all(bind=engine)
        return

    alembic_cfg = Config(str(alembic_ini))

    # alembic/env.py reads DATABASE_URL directly via get_settings() and
    # bypasses configparser — that's the right pattern for URLs containing
    # url-encoded secrets (configparser chokes on `%` interpolation).
    # So we do NOT call set_main_option("sqlalchemy.url", ...) here.

    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        current_rev = ctx.get_current_revision()

    if current_rev is None:
        # First-time adoption: create tables directly and stamp at head.
        log.info("alembic_version table absent — bootstrapping from Base.metadata and stamping at head")
        Base.metadata.create_all(bind=engine)
        command.stamp(alembic_cfg, "head")
        return

    # Check if there's anything to do before invoking `alembic upgrade`,
    # which otherwise spends 2–5s re-validating every revision even when
    # the DB is already at head.
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(alembic_cfg)
    head_rev = script.get_current_head()

    if current_rev == head_rev:
        log.info(f"alembic_version at {current_rev} (== head) — nothing to migrate")
        return

    log.info(f"alembic_version at {current_rev} (head={head_rev}) — running `alembic upgrade head`")
    command.upgrade(alembic_cfg, "head")
