"""
Patient self-service API routes.

Endpoints for patient profile management, document uploads,
symptom trends, data export, emergency contacts, medications,
allergies, diagnoses, screening schedules, and onboarding.
"""

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.localization import normalize_phone, validate_cpr, validate_dob
from app.middleware.rate_limiter import limiter
from app.models.db import (
    Allergy,
    Appointment,
    CarePlan,
    ChatMessage,
    Conversation,
    Diagnosis,
    EmergencyContact,
    Medication,
    Notification,
    PatientDocument,
    Screening,
    ScreeningSchedule,
    User,
    get_db,
)
from app.schemas.analysis import (
    AllergyCreate,
    AllergyResponse,
    DiagnosisResponse,
    DirectMessageCreate,
    DirectMessageResponse,
    DirectMessageThread,
    MedicationCreate,
    MedicationResponse,
    NotificationResponse,
    OnboardingProgress,
    ProfileUpdate,
    ScreeningScheduleCreate,
    ScreeningScheduleResponse,
)
from app.services.auth import get_current_user, hash_password, log_audit
from app.services.container import get_rag_service
from app.services.rag import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)

VALID_BLOOD_TYPES = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}


# ── Request Schemas ───────────────────────────────────────────────────────────


class UpdateProfileRequest(BaseModel):
    full_name: str | None = Field(None, min_length=1, max_length=255)
    phone: str | None = Field(None, max_length=20)
    new_password: str | None = Field(None, min_length=8, max_length=128)
    email_notifications: bool | None = None


class DocumentUploadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    doc_type: str = Field(description="phq9, gad7, medication_list, journal_entry, previous_diagnosis")
    content: str = Field(min_length=10, max_length=100000)


class EmergencyContactRequest(BaseModel):
    contact_name: str = Field(min_length=1, max_length=255)
    phone: str = Field(min_length=5, max_length=20)
    relation: str = Field(min_length=1, max_length=50)
    is_primary: bool = False


PATIENT_DOC_TYPES = [
    "phq9", "gad7", "medication_list", "journal_entry", "previous_diagnosis",
    "medical_report", "therapy_notes", "mood_diary", "sleep_log",
    "wellness_plan", "cpr_id", "passport", "insurance_card", "other",
]


def _get_rag() -> "RAGService | None":
    return get_rag_service()


async def _ingest_document_background(patient_id: str, doc_id: str, doc_type: str, title: str, content: str):
    """Background task: embed document into patient RAG and update status."""
    from app.models.db import PatientDocument, SessionLocal
    from app.services.container import get_rag_service

    rag = get_rag_service()
    db = SessionLocal()
    try:
        if rag and rag.is_initialized:
            rag.ingest_patient_document(
                patient_id=patient_id,
                doc_id=doc_id,
                doc_type=doc_type,
                title=title,
                content=content,
            )
        doc = db.query(PatientDocument).filter_by(id=doc_id).first()
        if doc:
            doc.processing_status = "ready"
            db.commit()
    except Exception as e:
        logger.error(f"Document ingestion failed for {doc_id}: {e}")
        doc = db.query(PatientDocument).filter_by(id=doc_id).first()
        if doc:
            doc.processing_status = "failed"
            doc.processing_error = str(e)[:500]
            db.commit()
    finally:
        db.close()


# ── Profile Management ────────────────────────────────────────────────────────


@router.put("/profile")
@limiter.limit("30/minute")
async def update_profile(
    request: Request,
    body: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update the patient's profile information (extended).

    Accepts all demographic fields, validates CPR, phone, DOB, and blood type.
    Automatically marks onboarding as complete when all required fields are filled.
    """
    changes = []

    # ── Simple text fields ────────────────────────────────────────────────
    if body.full_name is not None:
        current_user.full_name = body.full_name
        changes.append("full_name")

    if body.gender is not None:
        allowed_genders = {"male", "female", "prefer_not_to_say"}
        if body.gender not in allowed_genders:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid gender. Must be one of: {', '.join(sorted(allowed_genders))}",
            )
        current_user.gender = body.gender
        changes.append("gender")

    if body.nationality is not None:
        current_user.nationality = body.nationality
        changes.append("nationality")

    if body.language_preference is not None:
        if body.language_preference not in {"en", "ar"}:
            raise HTTPException(status_code=400, detail="language_preference must be 'en' or 'ar'")
        current_user.language_preference = body.language_preference
        changes.append("language_preference")

    if body.timezone is not None:
        current_user.timezone = body.timezone
        changes.append("timezone")

    if body.email_notifications is not None:
        current_user.email_notifications = body.email_notifications
        changes.append("email_notifications")

    if body.reddit_username is not None:
        # Strip leading 'u/' or '/u/' if present; keep alphanumeric + underscore/dash
        val = body.reddit_username.strip().lstrip("/").lstrip("u/") if body.reddit_username else ""
        current_user.reddit_username = val or None
        changes.append("reddit_username")

    if body.twitter_username is not None:
        val = body.twitter_username.strip().lstrip("@") if body.twitter_username else ""
        current_user.twitter_username = val or None
        changes.append("twitter_username")

    if body.onboarding_completed is not None:
        current_user.onboarding_completed = body.onboarding_completed
        changes.append("onboarding_completed")

    # ── Phone validation ──────────────────────────────────────────────────
    if body.phone is not None:
        try:
            normalized = normalize_phone(body.phone)
            current_user.phone = normalized
            changes.append("phone")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid phone number: {e}")

    # ── Date of birth validation ──────────────────────────────────────────
    if body.date_of_birth is not None:
        try:
            dob = date.fromisoformat(body.date_of_birth)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="Invalid date_of_birth format. Use ISO format (YYYY-MM-DD).")
        if not validate_dob(dob, min_age=13, max_age=120):
            raise HTTPException(
                status_code=400, detail="Date of birth must correspond to an age between 13 and 120 years."
            )
        current_user.date_of_birth = dob
        changes.append("date_of_birth")

    # ── CPR validation ────────────────────────────────────────────────────
    if body.cpr_number is not None:
        if not validate_cpr(body.cpr_number):
            raise HTTPException(status_code=400, detail="Invalid CPR number.")
        # Check uniqueness
        existing = (
            db.query(User)
            .filter(
                User.cpr_number == body.cpr_number,
                User.id != current_user.id,
            )
            .first()
        )
        if existing:
            raise HTTPException(status_code=409, detail="CPR number is already registered to another account.")
        current_user.cpr_number = body.cpr_number
        changes.append("cpr_number")

    # ── Blood type validation ─────────────────────────────────────────────
    if body.blood_type is not None:
        if body.blood_type not in VALID_BLOOD_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid blood_type. Must be one of: {', '.join(sorted(VALID_BLOOD_TYPES))}",
            )
        current_user.blood_type = body.blood_type
        changes.append("blood_type")

    # ── Password change ───────────────────────────────────────────────────
    if body.new_password is not None:
        current_user.password_hash = hash_password(body.new_password)
        changes.append("password")

    # ── Auto-complete onboarding ──────────────────────────────────────────
    has_emergency_contact = (
        db.query(EmergencyContact)
        .filter(
            EmergencyContact.patient_id == current_user.id,
        )
        .first()
        is not None
    )

    if current_user.date_of_birth and current_user.gender and current_user.phone and has_emergency_contact:
        if not current_user.onboarding_completed:
            current_user.onboarding_completed = True
            changes.append("onboarding_completed")

    db.commit()
    log_audit(db, current_user.id, "profile_updated", resource_type="user")
    logger.info(f"Profile updated for user {current_user.id}: {', '.join(changes)}")

    return {
        "status": "updated",
        "fields_changed": changes,
        "onboarding_completed": current_user.onboarding_completed,
    }


@router.delete("/account")
@limiter.limit("30/minute")
async def deactivate_account(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deactivate the patient's account (soft delete)."""
    current_user.is_active = False
    db.commit()
    log_audit(db, current_user.id, "account_deactivated", resource_type="user")

    # Clean up patient RAG data
    from app.models.db import PatientRAGChunk

    try:
        rag_cleanup = (
            db.query(PatientRAGChunk)
            .filter_by(patient_id=current_user.id)
            .update({"is_current": False})
        )
        db.commit()
        logger.info(f"Deactivated {rag_cleanup} RAG chunks for patient {current_user.id[:8]}")
    except Exception as e:
        logger.warning(f"RAG cleanup on deactivation failed: {e}")

    return {"status": "account_deactivated"}


@router.post("/unlink-clinician")
@limiter.limit("30/minute")
async def unlink_from_clinician(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove the link between patient and clinician."""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can unlink from a clinician")
    current_user.clinician_id = None
    db.commit()
    return {"status": "unlinked"}


# ── Patient Document Uploads ──────────────────────────────────────────────────


@router.post("/documents")
@limiter.limit("30/minute")
async def upload_my_document(
    request: Request,
    body: DocumentUploadRequest,
    current_user: User = Depends(get_current_user),
    rag: RAGService = Depends(_get_rag),
    db: Session = Depends(get_db),
):
    """Upload a patient's own document (PHQ-9, medication list, journal, etc.).

    Document is saved to DB and ingested into the patient's personal RAG
    collection for personalized chatbot responses.
    """
    from app.services.rag_safety import sanitize_identity_document, should_ingest_to_rag

    if body.doc_type not in PATIENT_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_type for patients. Must be one of: {', '.join(PATIENT_DOC_TYPES)}",
        )

    content = body.content

    # Sanitize identity documents
    if body.doc_type in ("cpr_id", "passport", "insurance_card"):
        content = sanitize_identity_document(content, body.doc_type)

    doc_id = str(uuid4())
    doc = PatientDocument(
        id=doc_id,
        patient_id=current_user.id,
        uploaded_by=current_user.id,
        doc_type=body.doc_type,
        title=body.title,
        content=content,
    )
    db.add(doc)
    db.commit()

    # Only ingest to RAG if doc type is not excluded
    if should_ingest_to_rag(body.doc_type) and rag and rag.is_initialized:
        rag.ingest_patient_document(
            patient_id=current_user.id,
            doc_id=doc_id,
            doc_type=body.doc_type,
            title=body.title,
            content=content,
        )

    log_audit(db, current_user.id, "document_uploaded", resource_type="document", resource_id=doc_id)

    return {"status": "uploaded", "document_id": doc_id}


@router.post("/documents/upload")
@limiter.limit("20/minute")
async def upload_my_document_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Query(..., min_length=1, max_length=255),
    doc_type: str = Query(...),
    current_user: User = Depends(get_current_user),
    rag: RAGService = Depends(_get_rag),
    db: Session = Depends(get_db),
):
    """Upload a document file (PDF, DOCX, TXT, MD, CSV).

    Accepts multipart/form-data. PDFs are extracted via Docling (with
    pdfplumber fallback); DOCX via Docling; text formats are decoded as
    UTF-8. Identity documents are PII-sanitized. RAG-eligible documents
    are ingested asynchronously in the background.
    """
    from app.services.document_extractor import extract_text
    from app.services.rag_safety import sanitize_identity_document, should_ingest_to_rag

    if doc_type not in PATIENT_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_type. Must be one of: {', '.join(PATIENT_DOC_TYPES)}",
        )

    raw = await file.read()
    max_upload_bytes = 10 * 1024 * 1024  # 10 MB
    if len(raw) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
    filename_lower = (file.filename or "").lower()
    file_ext = "." + filename_lower.rsplit(".", 1)[-1] if "." in filename_lower else ""

    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    result = extract_text(raw, filename_lower)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this document. The file may be damaged or in an unsupported format.",
        )
    content = result.text

    if len(content) < 10:
        raise HTTPException(status_code=400, detail="This file seems empty. Please add some content.")

    if len(content) > 100_000:
        content = content[:100_000]

    # Sanitize identity documents
    if doc_type in ("cpr_id", "passport", "insurance_card"):
        content = sanitize_identity_document(content, doc_type)

    doc_id = str(uuid4())
    doc = PatientDocument(
        id=doc_id,
        patient_id=current_user.id,
        uploaded_by=current_user.id,
        doc_type=doc_type,
        title=title,
        content=content,
        processing_status="processing" if should_ingest_to_rag(doc_type) else "ready",
    )
    db.add(doc)
    db.commit()

    # Background RAG ingestion
    if should_ingest_to_rag(doc_type) and rag and rag.is_initialized:
        background_tasks.add_task(
            _ingest_document_background,
            patient_id=current_user.id,
            doc_id=doc_id,
            doc_type=doc_type,
            title=title,
            content=content,
        )

    log_audit(
        db,
        current_user.id,
        "document_uploaded_file",
        resource_type="document",
        resource_id=doc_id,
    )

    return {
        "status": "processing" if should_ingest_to_rag(doc_type) else "ready",
        "document_id": doc_id,
        "extracted_chars": len(content),
    }


@router.get("/documents")
async def list_my_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all of the patient's own uploaded documents."""
    docs = (
        db.query(PatientDocument)
        .filter(PatientDocument.patient_id == current_user.id)
        .order_by(desc(PatientDocument.created_at))
        .all()
    )

    return [
        {
            "id": d.id,
            "doc_type": d.doc_type,
            "title": d.title,
            "created_at": d.created_at,
            "content_preview": d.content[:200] + "..." if len(d.content) > 200 else d.content,
            "processing_status": getattr(d, "processing_status", "ready") or "ready",
        }
        for d in docs
    ]


@router.delete("/documents/{doc_id}")
async def delete_my_document(
    doc_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete one of the patient's own uploaded documents."""
    doc = (
        db.query(PatientDocument)
        .filter(
            PatientDocument.id == doc_id,
            PatientDocument.patient_id == current_user.id,
        )
        .first()
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    db.delete(doc)
    db.commit()

    # Invalidate RAG chunks for this document
    rag = get_rag_service()
    if rag and rag.is_initialized:
        rag.invalidate_source("patient_documents", doc_id)

    log_audit(
        db,
        current_user.id,
        "document_deleted",
        resource_type="document",
        resource_id=doc_id,
    )

    return {"status": "deleted"}


# ── Symptom Trends ────────────────────────────────────────────────────────────


@router.get("/trends")
async def get_my_symptom_trends(
    days: int = 90,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the patient's symptom severity trends over time.

    Returns a timeseries of screenings with severity, symptom count,
    and which specific symptoms were detected at each point.
    """
    start_date = datetime.now(UTC) - timedelta(days=days)
    screenings = (
        db.query(Screening)
        .filter(
            Screening.patient_id == current_user.id,
            Screening.created_at >= start_date,
        )
        .order_by(Screening.created_at)
        .all()
    )

    timeline = []
    all_symptoms_seen = set()

    for s in screenings:
        symptoms = []
        if s.symptom_data:
            for d in s.symptom_data.get("symptoms_detected", []):
                symptoms.append(d.get("symptom", ""))
                all_symptoms_seen.add(d.get("symptom", ""))

        timeline.append(
            {
                "screening_id": s.id,
                "date": s.created_at.isoformat(),
                "source": s.source,
                "severity_level": s.severity_level,
                "symptom_count": s.symptom_count or 0,
                "symptoms_detected": sorted(set(symptoms)),
                "flagged_for_review": s.flagged_for_review,
            }
        )

    # Compute trend summary
    if len(timeline) >= 2:
        first = timeline[0]
        last = timeline[-1]
        severity_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        first_sev = severity_order.get(first["severity_level"], 0)
        last_sev = severity_order.get(last["severity_level"], 0)

        if last_sev > first_sev:
            trend = "worsening"
        elif last_sev < first_sev:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    return {
        "days_analyzed": days,
        "total_screenings": len(timeline),
        "trend": trend,
        "all_symptoms_observed": sorted(all_symptoms_seen),
        "timeline": timeline,
    }


# ── Emergency Contacts ────────────────────────────────────────────────────────


@router.post("/emergency-contacts")
@limiter.limit("30/minute")
async def add_emergency_contact(
    request: Request,
    body: EmergencyContactRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add an emergency contact for crisis escalation."""
    contact = EmergencyContact(
        id=str(uuid4()),
        patient_id=current_user.id,
        contact_name=body.contact_name,
        phone=body.phone,
        relation=body.relation,
        is_primary=body.is_primary,
    )
    db.add(contact)
    db.commit()
    return {"status": "added", "contact_id": contact.id}


@router.get("/emergency-contacts")
async def list_emergency_contacts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the patient's emergency contacts."""
    contacts = db.query(EmergencyContact).filter(EmergencyContact.patient_id == current_user.id).all()
    return [
        {
            "id": c.id,
            "contact_name": c.contact_name,
            "phone": c.phone,
            "relation": c.relation,
            "is_primary": c.is_primary,
        }
        for c in contacts
    ]


@router.delete("/emergency-contacts/{contact_id}")
@limiter.limit("30/minute")
async def remove_emergency_contact(
    contact_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove an emergency contact."""
    contact = (
        db.query(EmergencyContact)
        .filter(
            EmergencyContact.id == contact_id,
            EmergencyContact.patient_id == current_user.id,
        )
        .first()
    )
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    db.delete(contact)
    db.commit()
    return {"status": "deleted"}


# ── Medications CRUD ──────────────────────────────────────────────────────────


@router.get("/medications", response_model=list[MedicationResponse])
async def list_medications(
    all: bool = Query(False, description="Include inactive medications"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the patient's medications. Active only by default; pass ?all=true for all."""
    query = db.query(Medication).filter(Medication.patient_id == current_user.id)
    if not all:
        query = query.filter(Medication.is_active == True)
    meds = query.order_by(desc(Medication.created_at)).all()

    return [
        MedicationResponse(
            id=m.id,
            name=m.name,
            dosage=m.dosage,
            frequency=m.frequency,
            start_date=m.start_date.isoformat() if m.start_date else None,
            end_date=m.end_date.isoformat() if m.end_date else None,
            prescribed_by=m.prescribed_by,
            notes=m.notes,
            is_active=m.is_active,
            created_at=m.created_at,
        )
        for m in meds
    ]


@router.post("/medications", response_model=MedicationResponse, status_code=201)
@limiter.limit("30/minute")
async def add_medication(
    request: Request,
    body: MedicationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a new medication to the patient's record."""
    med_id = str(uuid4())

    start_dt = None
    if body.start_date:
        try:
            start_dt = date.fromisoformat(body.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")

    end_dt = None
    if body.end_date:
        try:
            end_dt = date.fromisoformat(body.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")

    if start_dt and end_dt and end_dt < start_dt:
        raise HTTPException(status_code=400, detail="end_date cannot be before start_date.")

    med = Medication(
        id=med_id,
        patient_id=current_user.id,
        name=body.name,
        dosage=body.dosage,
        frequency=body.frequency,
        start_date=start_dt,
        end_date=end_dt,
        prescribed_by=body.prescribed_by,
        notes=body.notes,
        is_active=True,
    )
    db.add(med)
    db.commit()

    log_audit(db, current_user.id, "medication_added", resource_type="medication", resource_id=med_id)

    return MedicationResponse(
        id=med.id,
        name=med.name,
        dosage=med.dosage,
        frequency=med.frequency,
        start_date=med.start_date.isoformat() if med.start_date else None,
        end_date=med.end_date.isoformat() if med.end_date else None,
        prescribed_by=med.prescribed_by,
        notes=med.notes,
        is_active=med.is_active,
        created_at=med.created_at,
    )


@router.put("/medications/{medication_id}", response_model=MedicationResponse)
@limiter.limit("30/minute")
async def update_medication(
    request: Request,
    medication_id: str,
    body: MedicationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an existing medication."""
    med = (
        db.query(Medication)
        .filter(
            Medication.id == medication_id,
            Medication.patient_id == current_user.id,
        )
        .first()
    )
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")

    med.name = body.name
    med.dosage = body.dosage
    med.frequency = body.frequency
    med.prescribed_by = body.prescribed_by
    med.notes = body.notes

    if body.start_date:
        try:
            med.start_date = date.fromisoformat(body.start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
    else:
        med.start_date = None

    if body.end_date:
        try:
            med.end_date = date.fromisoformat(body.end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")
    else:
        med.end_date = None

    if med.start_date and med.end_date and med.end_date < med.start_date:
        raise HTTPException(status_code=400, detail="end_date cannot be before start_date.")

    db.commit()
    log_audit(db, current_user.id, "medication_updated", resource_type="medication", resource_id=medication_id)

    return MedicationResponse(
        id=med.id,
        name=med.name,
        dosage=med.dosage,
        frequency=med.frequency,
        start_date=med.start_date.isoformat() if med.start_date else None,
        end_date=med.end_date.isoformat() if med.end_date else None,
        prescribed_by=med.prescribed_by,
        notes=med.notes,
        is_active=med.is_active,
        created_at=med.created_at,
    )


@router.delete("/medications/{medication_id}")
@limiter.limit("30/minute")
async def delete_medication(
    medication_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete a medication (set is_active=False)."""
    med = (
        db.query(Medication)
        .filter(
            Medication.id == medication_id,
            Medication.patient_id == current_user.id,
        )
        .first()
    )
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found")

    med.is_active = False
    db.commit()
    log_audit(db, current_user.id, "medication_deactivated", resource_type="medication", resource_id=medication_id)

    return {"status": "deactivated", "medication_id": medication_id}


# ── Allergies CRUD ────────────────────────────────────────────────────────────


@router.get("/allergies", response_model=list[AllergyResponse])
async def list_allergies(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the patient's allergies."""
    allergies = db.query(Allergy).filter(Allergy.patient_id == current_user.id).order_by(desc(Allergy.created_at)).all()

    return [
        AllergyResponse(
            id=a.id,
            allergen=a.allergen,
            allergy_type=a.allergy_type,
            severity=a.severity,
            reaction=a.reaction,
            diagnosed_date=a.diagnosed_date.isoformat() if a.diagnosed_date else None,
            notes=a.notes,
            created_at=a.created_at,
        )
        for a in allergies
    ]


@router.post("/allergies", response_model=AllergyResponse, status_code=201)
@limiter.limit("30/minute")
async def add_allergy(
    request: Request,
    body: AllergyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a new allergy to the patient's record."""
    allergy_id = str(uuid4())

    diagnosed_dt = None
    if body.diagnosed_date:
        try:
            diagnosed_dt = date.fromisoformat(body.diagnosed_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid diagnosed_date format. Use YYYY-MM-DD.")

    allergy = Allergy(
        id=allergy_id,
        patient_id=current_user.id,
        allergen=body.allergen,
        allergy_type=body.allergy_type,
        severity=body.severity,
        reaction=body.reaction,
        diagnosed_date=diagnosed_dt,
        notes=body.notes,
    )
    db.add(allergy)
    db.commit()

    log_audit(db, current_user.id, "allergy_added", resource_type="allergy", resource_id=allergy_id)

    return AllergyResponse(
        id=allergy.id,
        allergen=allergy.allergen,
        allergy_type=allergy.allergy_type,
        severity=allergy.severity,
        reaction=allergy.reaction,
        diagnosed_date=allergy.diagnosed_date.isoformat() if allergy.diagnosed_date else None,
        notes=allergy.notes,
        created_at=allergy.created_at,
    )


@router.put("/allergies/{allergy_id}", response_model=AllergyResponse)
@limiter.limit("30/minute")
async def update_allergy(
    allergy_id: str,
    request: Request,
    body: AllergyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an existing allergy on the patient's record."""
    allergy = db.query(Allergy).filter(Allergy.id == allergy_id, Allergy.patient_id == current_user.id).first()
    if not allergy:
        raise HTTPException(status_code=404, detail="Allergy not found")

    allergy.allergen = body.allergen
    if body.severity is not None:
        allergy.severity = body.severity
    if body.allergy_type is not None:
        allergy.allergy_type = body.allergy_type
    if body.reaction is not None:
        allergy.reaction = body.reaction
    if body.notes is not None:
        allergy.notes = body.notes
    db.commit()
    db.refresh(allergy)
    log_audit(db, current_user.id, "allergy_updated", "allergy", allergy.id)
    return AllergyResponse(
        id=allergy.id,
        allergen=allergy.allergen,
        severity=allergy.severity,
        allergy_type=allergy.allergy_type,
        reaction=allergy.reaction,
        diagnosed_date=allergy.diagnosed_date.isoformat() if allergy.diagnosed_date else None,
        notes=allergy.notes,
        created_at=allergy.created_at,
    )


@router.delete("/allergies/{allergy_id}")
@limiter.limit("30/minute")
async def delete_allergy(
    allergy_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an allergy from the patient's record."""
    allergy = (
        db.query(Allergy)
        .filter(
            Allergy.id == allergy_id,
            Allergy.patient_id == current_user.id,
        )
        .first()
    )
    if not allergy:
        raise HTTPException(status_code=404, detail="Allergy not found")

    db.delete(allergy)
    db.commit()
    log_audit(db, current_user.id, "allergy_deleted", resource_type="allergy", resource_id=allergy_id)

    return {"status": "deleted", "allergy_id": allergy_id}


# ── Diagnoses (read-only for patients) ────────────────────────────────────────


@router.get("/diagnoses", response_model=list[DiagnosisResponse])
async def list_diagnoses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the patient's diagnoses. Patients can view; clinicians add via dashboard."""
    diagnoses = (
        db.query(Diagnosis).filter(Diagnosis.patient_id == current_user.id).order_by(desc(Diagnosis.created_at)).all()
    )

    log_audit(db, current_user.id, "diagnoses_viewed", resource_type="diagnosis")

    return [
        DiagnosisResponse(
            id=d.id,
            condition=d.condition,
            icd10_code=d.icd10_code,
            diagnosed_date=d.diagnosed_date.isoformat() if d.diagnosed_date else None,
            status=d.status,
            diagnosed_by=d.diagnosed_by,
            notes=d.notes,
            created_at=d.created_at,
        )
        for d in diagnoses
    ]


# ── Screening Schedule ───────────────────────────────────────────────────────


@router.get("/screening-schedule", response_model=Optional[ScreeningScheduleResponse])
async def get_screening_schedule(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get the patient's active screening schedule."""
    schedule = (
        db.query(ScreeningSchedule)
        .filter(
            ScreeningSchedule.patient_id == current_user.id,
            ScreeningSchedule.is_active == True,
        )
        .first()
    )

    if not schedule:
        return None

    return ScreeningScheduleResponse(
        id=schedule.id,
        frequency=schedule.frequency,
        custom_days=schedule.custom_days,
        day_of_week=schedule.day_of_week,
        preferred_time=schedule.preferred_time,
        next_due_at=schedule.next_due_at,
        last_completed_at=schedule.last_completed_at,
        is_active=schedule.is_active,
        created_at=schedule.created_at,
    )


@router.post("/screening-schedule", response_model=ScreeningScheduleResponse, status_code=201)
@limiter.limit("30/minute")
async def create_or_update_screening_schedule(
    request: Request,
    body: ScreeningScheduleCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create or update the patient's screening schedule.

    If an active schedule already exists, it is deactivated and replaced.
    """
    valid_frequencies = {"weekly", "biweekly", "monthly", "custom"}
    if body.frequency not in valid_frequencies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frequency. Must be one of: {', '.join(sorted(valid_frequencies))}",
        )

    if body.frequency == "custom" and not body.custom_days:
        raise HTTPException(status_code=400, detail="custom_days is required when frequency is 'custom'.")

    if body.frequency in {"weekly", "biweekly"} and body.day_of_week is None:
        raise HTTPException(status_code=400, detail="day_of_week is required for weekly/biweekly schedules.")

    # Deactivate any existing active schedule
    existing = (
        db.query(ScreeningSchedule)
        .filter(
            ScreeningSchedule.patient_id == current_user.id,
            ScreeningSchedule.is_active == True,
        )
        .all()
    )
    for s in existing:
        s.is_active = False

    # Calculate next_due_at
    now = datetime.now(UTC)
    if body.frequency == "weekly":
        days_ahead = (body.day_of_week - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        next_due = now + timedelta(days=days_ahead)
    elif body.frequency == "biweekly":
        days_ahead = (body.day_of_week - now.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 14
        next_due = now + timedelta(days=days_ahead)
    elif body.frequency == "monthly":
        next_due = now + timedelta(days=30)
    elif body.frequency == "custom":
        next_due = now + timedelta(days=body.custom_days)
    else:
        next_due = now + timedelta(days=7)

    # Apply preferred_time if provided
    if body.preferred_time:
        try:
            hour, minute = map(int, body.preferred_time.split(":"))
            next_due = next_due.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="Invalid preferred_time format. Use HH:MM.")

    schedule_id = str(uuid4())
    schedule = ScreeningSchedule(
        id=schedule_id,
        patient_id=current_user.id,
        frequency=body.frequency,
        custom_days=body.custom_days,
        day_of_week=body.day_of_week,
        preferred_time=body.preferred_time,
        next_due_at=next_due,
        is_active=True,
    )
    db.add(schedule)
    db.commit()

    log_audit(
        db, current_user.id, "screening_schedule_created", resource_type="screening_schedule", resource_id=schedule_id
    )

    return ScreeningScheduleResponse(
        id=schedule.id,
        frequency=schedule.frequency,
        custom_days=schedule.custom_days,
        day_of_week=schedule.day_of_week,
        preferred_time=schedule.preferred_time,
        next_due_at=schedule.next_due_at,
        last_completed_at=schedule.last_completed_at,
        is_active=schedule.is_active,
        created_at=schedule.created_at,
    )


@router.delete("/screening-schedule/{schedule_id}")
@limiter.limit("30/minute")
async def deactivate_screening_schedule(
    schedule_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Deactivate a screening schedule."""
    schedule = (
        db.query(ScreeningSchedule)
        .filter(
            ScreeningSchedule.id == schedule_id,
            ScreeningSchedule.patient_id == current_user.id,
        )
        .first()
    )
    if not schedule:
        raise HTTPException(status_code=404, detail="Screening schedule not found")

    schedule.is_active = False
    db.commit()
    log_audit(
        db,
        current_user.id,
        "screening_schedule_deactivated",
        resource_type="screening_schedule",
        resource_id=schedule_id,
    )

    return {"status": "deactivated", "schedule_id": schedule_id}


# ── Onboarding Status ────────────────────────────────────────────────────────


@router.get("/onboarding-status", response_model=OnboardingProgress)
async def get_onboarding_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return which onboarding steps are complete by checking filled fields in DB."""
    # Demographics: DOB, gender, nationality
    demographics_complete = all(
        [
            current_user.date_of_birth,
            current_user.gender,
            current_user.nationality,
        ]
    )

    # Contact: phone number
    contact_complete = current_user.phone is not None and len(current_user.phone) > 0

    # Medical history: at least one diagnosis exists
    medical_history_complete = (
        db.query(Diagnosis)
        .filter(
            Diagnosis.patient_id == current_user.id,
        )
        .first()
        is not None
    )

    # Medications: at least one medication recorded (active or not)
    medications_complete = (
        db.query(Medication)
        .filter(
            Medication.patient_id == current_user.id,
        )
        .first()
        is not None
    )

    # Allergies: at least one allergy recorded
    allergies_complete = (
        db.query(Allergy)
        .filter(
            Allergy.patient_id == current_user.id,
        )
        .first()
        is not None
    )

    # Emergency contacts: at least one
    emergency_contacts_complete = (
        db.query(EmergencyContact)
        .filter(
            EmergencyContact.patient_id == current_user.id,
        )
        .first()
        is not None
    )

    # Consent: use onboarding_completed as a proxy (set after all required steps)
    consent_accepted = current_user.onboarding_completed or False

    return OnboardingProgress(
        demographics_complete=demographics_complete,
        contact_complete=contact_complete,
        medical_history_complete=medical_history_complete,
        medications_complete=medications_complete,
        allergies_complete=allergies_complete,
        emergency_contacts_complete=emergency_contacts_complete,
        consent_accepted=consent_accepted,
    )


@router.post("/onboarding-complete")
@limiter.limit("30/minute")
async def mark_onboarding_complete(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark onboarding as complete. Validates all required fields are filled.

    Required fields: date_of_birth, gender, phone, at least 1 emergency contact.
    """
    missing = []

    if not current_user.date_of_birth:
        missing.append("date_of_birth")
    if not current_user.gender:
        missing.append("gender")
    if not current_user.phone:
        missing.append("phone")

    has_emergency_contact = (
        db.query(EmergencyContact)
        .filter(
            EmergencyContact.patient_id == current_user.id,
        )
        .first()
        is not None
    )
    if not has_emergency_contact:
        missing.append("emergency_contact")

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Cannot complete onboarding. Missing required fields.",
                "missing_fields": missing,
            },
        )

    current_user.onboarding_completed = True
    db.commit()
    log_audit(db, current_user.id, "onboarding_completed", resource_type="user")

    return {"status": "onboarding_complete"}


# ── Data Export ───────────────────────────────────────────────────────────────


@router.get("/export")
async def export_my_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export all of the patient's data as JSON.

    Includes: profile, screenings, documents, chat messages, emergency contacts.
    Supports right to data portability (GDPR Article 20).
    """
    # Screenings
    screenings = (
        db.query(Screening).filter(Screening.patient_id == current_user.id).order_by(Screening.created_at).all()
    )

    # Documents
    documents = db.query(PatientDocument).filter(PatientDocument.patient_id == current_user.id).all()

    # Chat messages (across all screenings)
    screening_ids = [s.id for s in screenings]
    chat_messages = []
    if screening_ids:
        chat_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.screening_id.in_(screening_ids))
            .order_by(ChatMessage.created_at)
            .all()
        )

    # Emergency contacts
    contacts = db.query(EmergencyContact).filter(EmergencyContact.patient_id == current_user.id).all()

    log_audit(db, current_user.id, "data_exported", resource_type="user")

    return {
        "exported_at": datetime.now(UTC).isoformat(),
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "full_name": current_user.full_name,
            "role": current_user.role,
            "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        },
        "screenings": [
            {
                "id": s.id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "source": s.source,
                "severity_level": s.severity_level,
                "symptom_count": s.symptom_count,
                "final_prediction": s.final_prediction,
                "text_preview": s.text[:200],
            }
            for s in screenings
        ],
        "documents": [
            {
                "id": d.id,
                "doc_type": d.doc_type,
                "title": d.title,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in documents
        ],
        "chat_messages": [
            {
                "screening_id": m.screening_id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in chat_messages
        ],
        "emergency_contacts": [
            {
                "contact_name": c.contact_name,
                "phone": c.phone,
                "relation": c.relation,
            }
            for c in contacts
        ],
        "total_screenings": len(screenings),
        "total_documents": len(documents),
        "total_chat_messages": len(chat_messages),
    }


@router.get("/export/pdf")
async def export_my_data_pdf(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export the patient's full record as a printable PDF."""
    from fastapi.responses import StreamingResponse

    from app.services.reports import build_patient_export_pdf

    screenings = (
        db.query(Screening).filter(Screening.patient_id == current_user.id).order_by(desc(Screening.created_at)).all()
    )
    documents = db.query(PatientDocument).filter(PatientDocument.patient_id == current_user.id).all()
    contacts = db.query(EmergencyContact).filter(EmergencyContact.patient_id == current_user.id).all()
    medications = db.query(Medication).filter(Medication.patient_id == current_user.id).all()
    allergies = db.query(Allergy).filter(Allergy.patient_id == current_user.id).all()
    diagnoses = db.query(Diagnosis).filter(Diagnosis.patient_id == current_user.id).all()
    care_plans = db.query(CarePlan).filter(CarePlan.patient_id == current_user.id).all()

    patient_dict = {
        "full_name": current_user.full_name,
        "email": current_user.email,
        "phone": current_user.phone,
        "date_of_birth": current_user.date_of_birth,
        "gender": current_user.gender,
        "nationality": current_user.nationality,
        "cpr_number": current_user.cpr_number,
        "medical_record_number": current_user.medical_record_number,
        "blood_type": current_user.blood_type,
        "language_preference": current_user.language_preference,
        "timezone": current_user.timezone,
    }

    export_dict = {
        "screenings": [
            {
                "created_at": s.created_at,
                "severity_label": s.severity_level or "none",
                "severity_score": s.symptom_count,
                "flagged_for_review": s.flagged_for_review,
            }
            for s in screenings
        ],
        "medications": [
            {
                "name": m.name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "start_date": m.start_date,
                "prescribed_by": m.prescribed_by,
                "is_active": m.is_active,
            }
            for m in medications
        ],
        "allergies": [
            {
                "allergen": a.allergen,
                "allergy_type": a.allergy_type,
                "severity": a.severity,
                "reaction": a.reaction,
            }
            for a in allergies
        ],
        "diagnoses": [
            {
                "condition": d.condition,
                "icd10_code": d.icd10_code,
                "status": d.status,
                "diagnosed_date": d.diagnosed_date,
                "diagnosed_by": d.diagnosed_by,
            }
            for d in diagnoses
        ],
        "emergency_contacts": [
            {
                "contact_name": c.contact_name,
                "phone": c.phone,
                "relation": c.relation,
                "is_primary": c.is_primary,
            }
            for c in contacts
        ],
        "care_plans": [
            {
                "title": cp.title,
                "description": cp.description,
                "status": cp.status,
                "review_date": cp.review_date,
            }
            for cp in care_plans
        ],
        "documents": [{"title": d.title, "doc_type": d.doc_type, "created_at": d.created_at} for d in documents],
    }

    buf = build_patient_export_pdf(patient_dict, export_dict)
    log_audit(db, current_user.id, "data_exported_pdf", resource_type="user")

    filename = f"depscreen-my-record-{datetime.now(UTC).strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Patient Notifications ────────────────────────────────────────────────────


@router.get("/notifications", response_model=list[NotificationResponse])
async def list_notifications(
    all: bool = Query(False, description="Include read notifications"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the patient's notifications. Unread only by default; pass ?all=true for all."""
    query = db.query(Notification).filter(Notification.user_id == current_user.id)
    if not all:
        query = query.filter(Notification.is_read == False)
    notifications = query.order_by(desc(Notification.created_at)).all()

    return [
        NotificationResponse(
            id=n.id,
            notification_type=n.notification_type,
            title=n.title,
            message=n.message,
            link=n.link,
            is_read=n.is_read,
            created_at=n.created_at,
        )
        for n in notifications
    ]


@router.get("/notifications/unread-count")
async def get_unread_notification_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the count of unread notifications for the patient."""
    count = (
        db.query(Notification)
        .filter(
            Notification.user_id == current_user.id,
            Notification.is_read == False,
        )
        .count()
    )
    return {"unread_count": count}


@router.patch("/notifications/read-all")
@limiter.limit("30/minute")
async def mark_all_notifications_read(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark all of the patient's unread notifications as read."""
    updated = (
        db.query(Notification)
        .filter(
            Notification.user_id == current_user.id,
            Notification.is_read == False,
        )
        .update({"is_read": True})
    )
    db.commit()
    log_audit(db, current_user.id, "notifications_read_all", resource_type="notification")

    return {"status": "all_read", "count": updated}


@router.patch("/notifications/{notification_id}/read")
@limiter.limit("30/minute")
async def mark_notification_read(
    notification_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Mark a single notification as read."""
    notification = (
        db.query(Notification)
        .filter(
            Notification.id == notification_id,
            Notification.user_id == current_user.id,
        )
        .first()
    )
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    db.commit()
    log_audit(db, current_user.id, "notification_read", resource_type="notification", resource_id=notification_id)

    return {"status": "read", "notification_id": notification_id}


# ── Profile Picture Upload ───────────────────────────────────────────────────


@router.post("/profile/picture")
@limiter.limit("10/minute")
async def upload_profile_picture(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
):
    """Upload a profile picture.

    The image is normalised server-side (EXIF rotation applied, metadata
    stripped for privacy, centre-cropped square, resized to 512×512, and
    re-encoded as WebP) then pushed to Supabase Storage. The returned
    public URL is saved on the user record and echoed back to the client.
    """
    from app.services.avatar import AvatarError, get_avatar_service

    avatar_svc = get_avatar_service(settings)
    if not avatar_svc.enabled:
        raise HTTPException(
            status_code=503,
            detail="Profile pictures aren't available on this environment. Please try again later.",
        )

    raw = await file.read()
    try:
        public_url = avatar_svc.upload(current_user.id, raw, file.content_type)
    except AvatarError as e:
        # User-facing input problem: surface the message verbatim
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Profile picture upload failed for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    current_user.profile_picture_url = public_url
    db.commit()
    log_audit(db, current_user.id, "profile_picture_uploaded", resource_type="user")

    return {"status": "uploaded", "url": public_url}


@router.delete("/profile/picture")
@limiter.limit("10/minute")
async def delete_profile_picture(
    request: Request,
    current_user: User = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
):
    """Remove the user's profile picture.

    Clears the URL on the user row and deletes the object from storage.
    Idempotent — callable even if the user has no avatar set.
    """
    from app.services.avatar import get_avatar_service

    avatar_svc = get_avatar_service(settings)
    avatar_svc.delete(current_user.id)

    current_user.profile_picture_url = None
    db.commit()
    log_audit(db, current_user.id, "profile_picture_deleted", resource_type="user")

    return {"status": "deleted"}


# ── Patient-Facing Appointments ───────────────────────────────────────────────


@router.get("/appointments")
async def list_my_appointments(
    status: str = Query(None, description="Filter by status, or 'all' for every status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the patient's own appointments. Defaults to upcoming scheduled/confirmed."""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access their own appointments")

    query = db.query(Appointment).filter(Appointment.patient_id == current_user.id)

    if status and status != "all":
        query = query.filter(Appointment.status == status)
    elif not status:
        query = query.filter(
            Appointment.status.in_(["scheduled", "confirmed"]),
            Appointment.scheduled_at >= datetime.now(UTC),
        )

    results = []
    for appt in query.order_by(Appointment.scheduled_at).all():
        clinician = db.query(User).filter(User.id == appt.clinician_id).first()
        results.append(
            {
                "id": appt.id,
                "clinician_id": appt.clinician_id,
                "clinician_name": clinician.full_name if clinician else "Unknown",
                "scheduled_at": appt.scheduled_at.isoformat() if appt.scheduled_at else None,
                "duration_minutes": appt.duration_minutes,
                "appointment_type": appt.appointment_type,
                "status": appt.status,
                "notes": appt.notes,
                "location": appt.location,
            }
        )
    return results


# ── Patient-Facing Care Plan ──────────────────────────────────────────────────


@router.get("/care-plan")
async def get_my_care_plan(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the patient's active care plan (or null if none)."""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access their own care plan")

    cp = (
        db.query(CarePlan)
        .filter(
            CarePlan.patient_id == current_user.id,
            CarePlan.status.in_(["active", "review_needed"]),
        )
        .order_by(CarePlan.created_at.desc())
        .first()
    )
    if not cp:
        return None

    clinician = db.query(User).filter(User.id == cp.clinician_id).first()
    return {
        "id": cp.id,
        "clinician_id": cp.clinician_id,
        "clinician_name": clinician.full_name if clinician else "Unknown",
        "title": cp.title,
        "description": cp.description,
        "template_name": cp.template_name,
        "goals": cp.goals,
        "interventions": cp.interventions,
        "review_date": cp.review_date.isoformat() if cp.review_date else None,
        "status": cp.status,
        "created_at": cp.created_at.isoformat() if cp.created_at else None,
        "updated_at": cp.updated_at.isoformat() if cp.updated_at else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Clinician ↔ Patient Direct Messages
# ══════════════════════════════════════════════════════════════════════════════


def _get_or_create_clinician_thread(db: Session, patient: User) -> Conversation | None:
    """Return the active clinician-direct conversation for this patient,
    lazily creating it if the patient has an assigned clinician and one doesn't exist yet.
    Returns None if the patient has no clinician assigned."""
    if not patient.clinician_id:
        return None

    conv = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == patient.id,
            Conversation.context_type == "clinician_direct",
            Conversation.linked_clinician_id == patient.clinician_id,
            Conversation.is_active == True,
        )
        .first()
    )
    if conv:
        return conv

    conv = Conversation(
        id=str(uuid4()),
        user_id=patient.id,
        title="Messages with your clinician",
        context_type="clinician_direct",
        linked_clinician_id=patient.clinician_id,
        is_active=True,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def _serialize_dm(msg: ChatMessage, sender_lookup: dict) -> DirectMessageResponse:
    return DirectMessageResponse(
        id=msg.id,
        role=msg.role,
        sender_name=sender_lookup.get(msg.role),
        content=msg.content,
        created_at=msg.created_at,
    )


@router.get("/clinician-messages", response_model=DirectMessageThread | None)
async def get_clinician_thread(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the patient's message thread with their assigned clinician.
    Returns null if the patient has no assigned clinician."""
    conv = _get_or_create_clinician_thread(db, current_user)
    if not conv:
        return None

    clinician = db.query(User).filter(User.id == conv.linked_clinician_id).first()

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conv.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    sender_lookup = {
        "user": current_user.full_name,
        "clinician": clinician.full_name if clinician else "Your clinician",
    }

    # Mark patient's new_message notifications as read (conversation opened)
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.notification_type == "new_message",
        Notification.is_read == False,
    ).update({"is_read": True})
    db.commit()

    return DirectMessageThread(
        conversation_id=conv.id,
        patient_id=current_user.id,
        patient_name=current_user.full_name,
        clinician_id=clinician.id if clinician else None,
        clinician_name=clinician.full_name if clinician else None,
        messages=[_serialize_dm(m, sender_lookup) for m in messages],
        unread_count=0,
    )


@router.post("/clinician-messages", response_model=DirectMessageResponse, status_code=201)
@limiter.limit("30/minute")
async def post_clinician_message(
    request: Request,
    body: DirectMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Patient posts a message to their clinician."""
    conv = _get_or_create_clinician_thread(db, current_user)
    if not conv:
        raise HTTPException(status_code=400, detail="No clinician is assigned to your account yet.")

    msg = ChatMessage(
        id=str(uuid4()),
        conversation_id=conv.id,
        role="user",
        content=body.content.strip(),
    )
    db.add(msg)
    conv.updated_at = datetime.now(UTC)

    # Notify clinician
    db.add(
        Notification(
            user_id=conv.linked_clinician_id,
            notification_type="new_message",
            title=f"New message from {current_user.full_name}",
            message=body.content.strip()[:140],
            link=f"/patients/{current_user.id}",
        )
    )

    db.commit()
    db.refresh(msg)

    log_audit(db, current_user.id, "clinician_message_sent", resource_type="message", resource_id=msg.id)

    return _serialize_dm(msg, {"user": current_user.full_name})
