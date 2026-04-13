"""
Database models and session management.

Tables:
  User, Screening, ChatMessage, PatientDocument, EmergencyContact,
  Medication, Allergy, Diagnosis, ScreeningSchedule, Appointment,
  Notification, Conversation, CarePlan, AuditLog

Uses SQLAlchemy with SQLite (development) / PostgreSQL (production).
"""

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Integer, Date, Text, JSON,
    ForeignKey, create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime

from app.core.config import get_settings

settings = get_settings()

# SQLite needs check_same_thread=False; PostgreSQL doesn't use it
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

pool_config = {}
if not settings.database_url.startswith("sqlite"):
    pool_config = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "pool_pre_ping": True,  # Verify connection is alive before using
    }

engine = create_engine(settings.database_url, connect_args=connect_args, **pool_config)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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
    clinician_id = Column(String(36), ForeignKey("users.id"), nullable=True)
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
    screening_schedules = relationship("ScreeningSchedule", back_populates="patient", foreign_keys="ScreeningSchedule.patient_id")
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
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    clinician_notes = Column(Text, nullable=True)
    next_action = Column(String(255), nullable=True)
    next_action_date = Column(DateTime, nullable=True)

    # Link to care plan (if active)
    care_plan_id = Column(String(36), ForeignKey("care_plans.id"), nullable=True)

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
    linked_clinician_id = Column(String(36), ForeignKey("users.id"), nullable=True)  # For clinician-direct
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("User", foreign_keys=[patient_id])
    clinician = relationship("User", foreign_keys=[clinician_id])


# ── Notifications ─────────────────────────────────────────────────────────────

class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, index=True)
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


# ── RAG Vector Chunks ───────────────────────────────────────────────────────

class KnowledgeChunk(Base):
    """Clinical knowledge base chunks with pgvector embeddings."""
    __tablename__ = "knowledge_chunks"

    id = Column(String(36), primary_key=True)
    content = Column(Text, nullable=False)
    category = Column(String(50))  # dsm5_criteria, coping_strategies, psychoeducation, crisis
    symptom = Column(String(50))   # DEPRESSED_MOOD, ANHEDONIA, etc. (or empty)
    source_file = Column(String(255))
    embedding = Column(Vector(384))  # all-MiniLM-L6-v2 outputs 384-dim vectors
    created_at = Column(DateTime, default=datetime.utcnow)


class PatientRAGChunk(Base):
    """Per-patient RAG chunks (screenings, documents) with pgvector embeddings."""
    __tablename__ = "patient_rag_chunks"

    id = Column(String(36), primary_key=True)
    patient_id = Column(String(36), ForeignKey("users.id"), index=True)
    screening_id = Column(String(36), nullable=True)
    doc_id = Column(String(36), nullable=True)
    content = Column(Text, nullable=False)
    chunk_type = Column(String(50))  # screening_text, symptom_evidence, patient_document
    metadata_json = Column(JSON, nullable=True)
    embedding = Column(Vector(384))
    created_at = Column(DateTime, default=datetime.utcnow)


# ── Session Management ────────────────────────────────────────────────────────

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables. Enables pgvector extension for PostgreSQL."""
    from sqlalchemy import text
    if not settings.database_url.startswith("sqlite"):
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    Base.metadata.create_all(bind=engine)
