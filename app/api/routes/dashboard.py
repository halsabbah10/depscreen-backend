"""
Clinician Dashboard API routes.

Patient management, screening triage, clinical notes,
appointments, care plans, diagnoses, and notifications.
Requires clinician role.
"""

import logging
from datetime import date, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import (
    Appointment,
    CarePlan,
    ChatMessage,
    Diagnosis,
    Notification,
    PatientDocument,
    Screening,
    User,
    get_db,
)
from app.schemas.analysis import (
    AppointmentCreate,
    AppointmentResponse,
    AppointmentStatusUpdate,
    CarePlanCreate,
    CarePlanResponse,
    DashboardStats,
    DiagnosisCreate,
    DiagnosisResponse,
    NotificationResponse,
    PatientSummary,
    ScreeningHistoryResponse,
    ScreeningListItem,
)
from app.services.auth import log_audit, require_clinician
from app.services.rag import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get aggregate statistics for the clinician's dashboard."""
    # Get all patients assigned to this clinician
    patient_ids = [p.id for p in db.query(User).filter(User.clinician_id == current_user.id).all()]

    total_patients = len(patient_ids)

    if not patient_ids:
        return DashboardStats(
            total_patients=0,
            total_screenings=0,
            flagged_count=0,
            severity_distribution={"none": 0, "mild": 0, "moderate": 0, "severe": 0},
            screenings_this_week=0,
        )

    # Count screenings
    total_screenings = db.query(Screening).filter(Screening.patient_id.in_(patient_ids)).count()

    # Flagged count
    flagged_count = (
        db.query(Screening).filter(Screening.patient_id.in_(patient_ids), Screening.flagged_for_review == True).count()
    )

    # Severity distribution
    severity_rows = (
        db.query(Screening.severity_level, func.count())
        .filter(Screening.patient_id.in_(patient_ids))
        .group_by(Screening.severity_level)
        .all()
    )
    severity_dist = {"none": 0, "mild": 0, "moderate": 0, "severe": 0}
    for level, count in severity_rows:
        if level in severity_dist:
            severity_dist[level] = count

    # Screenings this week
    week_ago = datetime.utcnow() - timedelta(days=7)
    screenings_this_week = (
        db.query(Screening).filter(Screening.patient_id.in_(patient_ids), Screening.created_at >= week_ago).count()
    )

    return DashboardStats(
        total_patients=total_patients,
        total_screenings=total_screenings,
        flagged_count=flagged_count,
        severity_distribution=severity_dist,
        screenings_this_week=screenings_this_week,
    )


@router.get("/patients", response_model=list[PatientSummary])
async def get_patients(
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get all patients assigned to this clinician with their latest screening info."""
    patients = db.query(User).filter(User.clinician_id == current_user.id).all()

    summaries = []
    for patient in patients:
        # Get latest screening
        latest = (
            db.query(Screening).filter(Screening.patient_id == patient.id).order_by(desc(Screening.created_at)).first()
        )

        total = db.query(Screening).filter(Screening.patient_id == patient.id).count()

        summaries.append(
            PatientSummary(
                id=patient.id,
                full_name=patient.full_name,
                email=patient.email,
                last_screening_date=latest.created_at if latest else None,
                last_severity=latest.severity_level if latest else None,
                last_symptom_count=latest.symptom_count if latest else None,
                total_screenings=total,
            )
        )

    # Sort by severity (severe first, then by last screening date)
    severity_order = {"severe": 0, "moderate": 1, "mild": 2, "none": 3, None: 4}
    summaries.sort(
        key=lambda s: (severity_order.get(s.last_severity, 4), s.last_screening_date or datetime.min), reverse=False
    )

    return summaries


@router.get("/patients/{patient_id}/screenings", response_model=ScreeningHistoryResponse)
async def get_patient_screenings(
    patient_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get screening history for a specific patient."""
    # Verify this patient belongs to the clinician
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    total = db.query(Screening).filter(Screening.patient_id == patient_id).count()
    screenings = (
        db.query(Screening)
        .filter(Screening.patient_id == patient_id)
        .order_by(desc(Screening.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [
        ScreeningListItem(
            id=s.id,
            created_at=s.created_at,
            text_preview=s.text[:100] + "..." if len(s.text) > 100 else s.text,
            final_prediction=s.final_prediction,
            final_confidence=s.final_confidence,
            symptom_count=s.symptom_count or 0,
            severity_level=s.severity_level or "none",
            flagged_for_review=s.flagged_for_review,
        )
        for s in screenings
    ]

    return ScreeningHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/screenings", response_model=ScreeningHistoryResponse)
async def get_all_screenings(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    severity: str = Query(None, description="Filter by severity level"),
    flagged_only: bool = Query(False, description="Show only flagged screenings"),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get all screenings across patients, sorted by severity (urgent first)."""
    patient_ids = [p.id for p in db.query(User).filter(User.clinician_id == current_user.id).all()]

    query = db.query(Screening).filter(Screening.patient_id.in_(patient_ids))

    if severity:
        query = query.filter(Screening.severity_level == severity)
    if flagged_only:
        query = query.filter(Screening.flagged_for_review == True)

    total = query.count()

    # Sort: flagged first, then by severity, then by date
    screenings = (
        query.order_by(
            desc(Screening.flagged_for_review),
            desc(Screening.created_at),
        )
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = [
        ScreeningListItem(
            id=s.id,
            created_at=s.created_at,
            text_preview=s.text[:100] + "..." if len(s.text) > 100 else s.text,
            final_prediction=s.final_prediction,
            final_confidence=s.final_confidence,
            symptom_count=s.symptom_count or 0,
            severity_level=s.severity_level or "none",
            flagged_for_review=s.flagged_for_review,
        )
        for s in screenings
    ]

    return ScreeningHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/screenings/{screening_id}")
async def get_screening_detail(
    screening_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get full screening detail including chat log for clinician review."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Verify access
    if screening.patient_id:
        patient = db.query(User).filter(User.id == screening.patient_id).first()
        if patient and patient.clinician_id != current_user.id:
            raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    # Get chat messages
    messages = (
        db.query(ChatMessage).filter(ChatMessage.screening_id == screening_id).order_by(ChatMessage.created_at).all()
    )

    # Get patient info
    patient_name = None
    if screening.patient_id:
        patient = db.query(User).filter(User.id == screening.patient_id).first()
        patient_name = patient.full_name if patient else None

    return {
        "screening": {
            "id": screening.id,
            "created_at": screening.created_at,
            "text": screening.text,
            "symptom_data": screening.symptom_data,
            "symptom_count": screening.symptom_count,
            "severity_level": screening.severity_level,
            "verification_data": screening.verification_data,
            "explanation_data": screening.explanation_data,
            "final_prediction": screening.final_prediction,
            "final_confidence": screening.final_confidence,
            "flagged_for_review": screening.flagged_for_review,
            "adversarial_warning": screening.adversarial_warning,
            "clinician_notes": screening.clinician_notes,
        },
        "patient_name": patient_name,
        "chat_messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    }


@router.put("/screenings/{screening_id}/notes")
@limiter.limit("30/minute")
async def update_clinician_notes(
    screening_id: str,
    request: Request,
    notes: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Add or update clinician notes on a screening."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    screening.clinician_notes = notes
    db.commit()

    return {"status": "updated", "screening_id": screening_id}


# ── Triage Status ─────────────────────────────────────────────────────────


@router.patch("/screenings/{screening_id}/triage")
@limiter.limit("30/minute")
async def update_triage_status(
    screening_id: str,
    request: Request,
    status: str,
    next_action: str = None,
    next_action_date: str = None,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Update screening triage status (new → in_review → reviewed → escalated → closed)."""
    valid_statuses = ["new", "in_review", "reviewed", "escalated", "closed"]
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of: {', '.join(valid_statuses)}")

    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    screening.triage_status = status
    screening.reviewed_by = current_user.id
    screening.reviewed_at = datetime.utcnow()

    if next_action:
        screening.next_action = next_action
    if next_action_date:
        screening.next_action_date = datetime.fromisoformat(next_action_date)

    db.commit()
    log_audit(db, current_user.id, f"triage_{status}", resource_type="screening", resource_id=screening_id)

    return {"status": status, "screening_id": screening_id}


# ── Patient Symptom Trends (Clinician View) ───────────────────────────────


@router.get("/patients/{patient_id}/trends")
async def get_patient_trends(
    patient_id: str,
    days: int = 90,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get symptom severity trends over time for a specific patient."""
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    start_date = datetime.utcnow() - timedelta(days=days)
    screenings = (
        db.query(Screening)
        .filter(Screening.patient_id == patient_id, Screening.created_at >= start_date)
        .order_by(Screening.created_at)
        .all()
    )

    timeline = []
    all_symptoms_seen = set()

    for s in screenings:
        symptoms = []
        if s.symptom_data:
            for d in s.symptom_data.get("symptoms_detected", []):
                sym = d.get("symptom", "")
                symptoms.append(sym)
                all_symptoms_seen.add(sym)

        timeline.append(
            {
                "screening_id": s.id,
                "date": s.created_at.isoformat(),
                "source": s.source,
                "severity_level": s.severity_level,
                "symptom_count": s.symptom_count or 0,
                "symptoms_detected": sorted(set(symptoms)),
                "triage_status": s.triage_status,
                "flagged_for_review": s.flagged_for_review,
            }
        )

    return {
        "patient_id": patient_id,
        "patient_name": patient.full_name,
        "days_analyzed": days,
        "total_screenings": len(timeline),
        "all_symptoms_observed": sorted(all_symptoms_seen),
        "timeline": timeline,
    }


# ── Patient Document Upload ───────────────────────────────────────────────

# Document types and who can upload them
CLINICIAN_DOC_TYPES = [
    "intake_form",  # Clinician completes during intake
    "session_notes",  # Clinician's clinical observations
    "treatment_plan",  # Goals, interventions, timeline
    "medical_history",  # Reviewed medical history
    "safety_plan",  # Crisis safety plan
    "referral_notes",  # Notes from referring provider
]

PATIENT_DOC_TYPES = [
    "phq9",  # PHQ-9 questionnaire responses
    "gad7",  # GAD-7 anxiety questionnaire
    "medication_list",  # Current medications patient is on
    "journal_entry",  # Personal journal / feelings
    "previous_diagnosis",  # Diagnoses patient is aware of
]

VALID_DOC_TYPES = CLINICIAN_DOC_TYPES + PATIENT_DOC_TYPES + ["other"]

_rag_service = None


async def _get_rag(settings: Settings = Depends(get_settings)):
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(settings)
        await _rag_service.initialize()
    return _rag_service


@router.post("/patients/{patient_id}/documents")
@limiter.limit("30/minute")
async def upload_patient_document(
    patient_id: str,
    request: Request,
    title: str,
    doc_type: str,
    content: str,
    current_user: User = Depends(require_clinician()),
    rag: RAGService = Depends(_get_rag),
    db: Session = Depends(get_db),
):
    """Upload a patient document for RAG ingestion.

    Clinicians can upload intake forms, PHQ-9 scores, medication lists,
    session notes, treatment plans, medical history, or safety plans.
    The content is ingested into the patient's personal RAG collection
    for personalized chatbot responses and screening context.
    """
    if doc_type not in VALID_DOC_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_type. Must be one of: {', '.join(VALID_DOC_TYPES)}",
        )

    # Verify patient exists and belongs to this clinician
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    # Save to DB
    doc_id = str(uuid4())
    doc = PatientDocument(
        id=doc_id,
        patient_id=patient_id,
        uploaded_by=current_user.id,
        doc_type=doc_type,
        title=title,
        content=content,
    )
    db.add(doc)
    db.commit()

    # Ingest into RAG
    rag.ingest_patient_document(
        patient_id=patient_id,
        doc_id=doc_id,
        doc_type=doc_type,
        title=title,
        content=content,
    )

    log_audit(db, current_user.id, "document_uploaded", resource_type="document", resource_id=doc_id)

    return {
        "status": "uploaded",
        "document_id": doc_id,
        "chunks_ingested": len([p for p in content.split("\n\n") if len(p.strip()) > 20]) or 1,
    }


@router.get("/patients/{patient_id}/documents")
async def list_patient_documents(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List all documents uploaded for a patient."""
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    docs = (
        db.query(PatientDocument)
        .filter(PatientDocument.patient_id == patient_id)
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
        }
        for d in docs
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Appointments
# ═══════════════════════════════════════════════════════════════════════════════

# Valid status transitions for appointments
_APPOINTMENT_TRANSITIONS: dict[str, set[str]] = {
    "scheduled": {"confirmed", "cancelled"},
    "confirmed": {"completed", "cancelled", "no_show"},
    # terminal states — no outgoing edges
    "completed": set(),
    "cancelled": set(),
    "no_show": set(),
}


def _verify_patient_access(db: Session, patient_id: str, clinician_id: str) -> User:
    """Return the patient or raise 404/403."""
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    if patient.clinician_id != clinician_id:
        raise HTTPException(status_code=403, detail="This patient is not assigned to you")
    return patient


def _appointment_to_response(appt: Appointment) -> AppointmentResponse:
    return AppointmentResponse(
        id=appt.id,
        patient_id=appt.patient_id,
        clinician_id=appt.clinician_id,
        scheduled_at=appt.scheduled_at,
        duration_minutes=appt.duration_minutes,
        appointment_type=appt.appointment_type,
        status=appt.status,
        notes=appt.notes,
        location=appt.location,
        created_at=appt.created_at,
    )


@router.get("/appointments", response_model=list[AppointmentResponse])
async def list_appointments(
    status: str = Query(None, description="Filter by status, or 'all' for every status"),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List the clinician's appointments. By default returns upcoming scheduled/confirmed only."""
    query = db.query(Appointment).filter(Appointment.clinician_id == current_user.id)

    if status and status != "all":
        query = query.filter(Appointment.status == status)
    elif not status:
        # Default: upcoming non-terminal appointments
        query = query.filter(
            Appointment.status.in_(["scheduled", "confirmed"]),
            Appointment.scheduled_at >= datetime.utcnow(),
        )

    appointments = query.order_by(Appointment.scheduled_at).all()
    return [_appointment_to_response(a) for a in appointments]


@router.post("/appointments", response_model=AppointmentResponse, status_code=201)
@limiter.limit("30/minute")
async def create_appointment(
    request: Request,
    payload: AppointmentCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Create a new appointment for a patient."""
    _verify_patient_access(db, payload.patient_id, current_user.id)

    appt_id = str(uuid4())
    appt = Appointment(
        id=appt_id,
        patient_id=payload.patient_id,
        clinician_id=current_user.id,
        scheduled_at=datetime.fromisoformat(payload.scheduled_at),
        duration_minutes=payload.duration_minutes,
        appointment_type=payload.appointment_type,
        status="scheduled",
        notes=payload.notes,
        location=payload.location,
    )
    db.add(appt)

    # Also create an in-app notification for the patient
    patient = db.query(User).filter(User.id == payload.patient_id).first()
    formatted = appt.scheduled_at.strftime("%d/%m/%Y at %H:%M")
    db.add(
        Notification(
            user_id=payload.patient_id,
            notification_type="appointment_scheduled",
            title="New appointment scheduled",
            message=f"Your clinician scheduled an appointment on {formatted}.",
            link="/appointments",
        )
    )

    db.commit()
    db.refresh(appt)

    # Email the patient immediately (confirmation). The 24h reminder fires separately via scheduler.
    try:
        from app.services.email import get_email_service

        if patient and patient.email and patient.email_notifications:
            get_email_service(settings).send_appointment_reminder(
                patient_name=patient.full_name,
                patient_email=patient.email,
                appointment_at=formatted,
                clinician_name=current_user.full_name,
            )
    except Exception as e:
        import logging as _logging

        _logging.getLogger(__name__).warning(f"Appointment email failed: {e}")

    log_audit(db, current_user.id, "appointment_created", resource_type="appointment", resource_id=appt_id)
    return _appointment_to_response(appt)


@router.get("/appointments/{appointment_id}", response_model=AppointmentResponse)
async def get_appointment(
    appointment_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get a single appointment's details."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This appointment is not assigned to you")
    return _appointment_to_response(appt)


@router.patch("/appointments/{appointment_id}/status", response_model=AppointmentResponse)
@limiter.limit("30/minute")
async def update_appointment_status(
    appointment_id: str,
    request: Request,
    payload: AppointmentStatusUpdate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Transition an appointment's status with validation."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This appointment is not assigned to you")

    allowed = _APPOINTMENT_TRANSITIONS.get(appt.status, set())
    if not allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Appointment is in terminal state '{appt.status}' and cannot be changed",
        )
    if payload.status not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot transition from '{appt.status}' to '{payload.status}'. Allowed: {', '.join(sorted(allowed))}",
        )

    appt.status = payload.status
    appt.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(appt)

    log_audit(
        db, current_user.id, f"appointment_{payload.status}", resource_type="appointment", resource_id=appointment_id
    )
    return _appointment_to_response(appt)


@router.delete("/appointments/{appointment_id}", response_model=AppointmentResponse)
@limiter.limit("30/minute")
async def cancel_appointment(
    appointment_id: str,
    request: Request,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Cancel an appointment (shorthand for setting status to 'cancelled')."""
    appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if appt.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This appointment is not assigned to you")

    if appt.status in ("completed", "cancelled", "no_show"):
        raise HTTPException(
            status_code=400,
            detail=f"Appointment is already in terminal state '{appt.status}'",
        )

    appt.status = "cancelled"
    appt.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(appt)

    log_audit(db, current_user.id, "appointment_cancelled", resource_type="appointment", resource_id=appointment_id)
    return _appointment_to_response(appt)


# ═══════════════════════════════════════════════════════════════════════════════
# Care Plans
# ═══════════════════════════════════════════════════════════════════════════════


def _care_plan_to_response(cp: CarePlan) -> CarePlanResponse:
    return CarePlanResponse(
        id=cp.id,
        patient_id=cp.patient_id,
        clinician_id=cp.clinician_id,
        title=cp.title,
        description=cp.description,
        template_name=cp.template_name,
        goals=cp.goals or [],
        interventions=cp.interventions or [],
        review_date=cp.review_date.strftime("%d/%m/%Y") if cp.review_date else None,
        status=cp.status,
        created_at=cp.created_at,
        updated_at=cp.updated_at,
    )


@router.get("/patients/{patient_id}/care-plans", response_model=list[CarePlanResponse])
async def list_care_plans(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """List all care plans for a patient."""
    _verify_patient_access(db, patient_id, current_user.id)

    plans = db.query(CarePlan).filter(CarePlan.patient_id == patient_id).order_by(desc(CarePlan.created_at)).all()
    return [_care_plan_to_response(cp) for cp in plans]


@router.post("/care-plans", response_model=CarePlanResponse, status_code=201)
@limiter.limit("30/minute")
async def create_care_plan(
    request: Request,
    payload: CarePlanCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Create a new care plan."""
    _verify_patient_access(db, payload.patient_id, current_user.id)

    cp_id = str(uuid4())
    review_dt = None
    if payload.review_date:
        review_dt = date.fromisoformat(payload.review_date)

    cp = CarePlan(
        id=cp_id,
        patient_id=payload.patient_id,
        clinician_id=current_user.id,
        title=payload.title,
        description=payload.description,
        template_name=payload.template_name,
        goals=[g.model_dump() for g in payload.goals],
        interventions=[i.model_dump() for i in payload.interventions],
        review_date=review_dt,
        status="active",
    )
    db.add(cp)
    db.commit()
    db.refresh(cp)

    log_audit(db, current_user.id, "care_plan_created", resource_type="care_plan", resource_id=cp_id)
    return _care_plan_to_response(cp)


@router.get("/care-plans/{care_plan_id}", response_model=CarePlanResponse)
async def get_care_plan(
    care_plan_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get a single care plan's details."""
    cp = db.query(CarePlan).filter(CarePlan.id == care_plan_id).first()
    if not cp:
        raise HTTPException(status_code=404, detail="Care plan not found")
    if cp.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This care plan is not assigned to you")
    return _care_plan_to_response(cp)


@router.put("/care-plans/{care_plan_id}", response_model=CarePlanResponse)
@limiter.limit("30/minute")
async def update_care_plan(
    care_plan_id: str,
    request: Request,
    goals: list[dict] = Body(None),
    interventions: list[dict] = Body(None),
    status: str = Body(None),
    review_date: str = Body(None),
    description: str = Body(None),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """Update goals, interventions, status, or review date of a care plan."""
    cp = db.query(CarePlan).filter(CarePlan.id == care_plan_id).first()
    if not cp:
        raise HTTPException(status_code=404, detail="Care plan not found")
    if cp.clinician_id != current_user.id:
        raise HTTPException(status_code=403, detail="This care plan is not assigned to you")

    valid_statuses = {"active", "review_needed", "completed", "archived"}
    if status is not None:
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Status must be one of: {', '.join(sorted(valid_statuses))}",
            )
        cp.status = status

    if goals is not None:
        cp.goals = goals
    if interventions is not None:
        cp.interventions = interventions
    if review_date is not None:
        cp.review_date = date.fromisoformat(review_date)
    if description is not None:
        cp.description = description

    cp.updated_at = datetime.utcnow()

    # In-app notification for the patient
    db.add(
        Notification(
            user_id=cp.patient_id,
            notification_type="care_plan_updated",
            title="Your care plan has been updated",
            message=f"Your clinician updated: {cp.title}",
            link="/care-plan",
        )
    )

    db.commit()
    db.refresh(cp)

    # Email the patient
    try:
        from app.services.email import get_email_service

        patient = db.query(User).filter(User.id == cp.patient_id).first()
        if patient and patient.email and patient.email_notifications:
            get_email_service(settings).send_care_plan_update(
                patient_name=patient.full_name,
                patient_email=patient.email,
                plan_title=cp.title,
            )
    except Exception as e:
        import logging as _logging

        _logging.getLogger(__name__).warning(f"Care plan update email failed: {e}")

    log_audit(db, current_user.id, "care_plan_updated", resource_type="care_plan", resource_id=care_plan_id)
    return _care_plan_to_response(cp)


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnoses
# ═══════════════════════════════════════════════════════════════════════════════


def _diagnosis_to_response(dx: Diagnosis) -> DiagnosisResponse:
    return DiagnosisResponse(
        id=dx.id,
        condition=dx.condition,
        icd10_code=dx.icd10_code,
        diagnosed_date=dx.diagnosed_date.strftime("%d/%m/%Y") if dx.diagnosed_date else None,
        status=dx.status,
        diagnosed_by=dx.diagnosed_by,
        notes=dx.notes,
        created_at=dx.created_at,
    )


@router.post("/patients/{patient_id}/diagnoses", response_model=DiagnosisResponse, status_code=201)
@limiter.limit("30/minute")
async def add_diagnosis(
    patient_id: str,
    request: Request,
    payload: DiagnosisCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Add a diagnosis to a patient."""
    _verify_patient_access(db, patient_id, current_user.id)

    dx_id = str(uuid4())
    diagnosed_dt = None
    if payload.diagnosed_date:
        diagnosed_dt = date.fromisoformat(payload.diagnosed_date)

    dx = Diagnosis(
        id=dx_id,
        patient_id=patient_id,
        condition=payload.condition,
        icd10_code=payload.icd10_code,
        diagnosed_date=diagnosed_dt,
        status=payload.status,
        diagnosed_by=payload.diagnosed_by or current_user.full_name,
        notes=payload.notes,
    )
    db.add(dx)
    db.commit()
    db.refresh(dx)

    log_audit(db, current_user.id, "diagnosis_added", resource_type="diagnosis", resource_id=dx_id)
    return _diagnosis_to_response(dx)


@router.put("/diagnoses/{diagnosis_id}", response_model=DiagnosisResponse)
@limiter.limit("30/minute")
async def update_diagnosis(
    diagnosis_id: str,
    request: Request,
    status: str = Body(..., embed=True),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Update a diagnosis status (active, remission, resolved)."""
    dx = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not dx:
        raise HTTPException(status_code=404, detail="Diagnosis not found")

    # Verify clinician has access to this patient
    _verify_patient_access(db, dx.patient_id, current_user.id)

    valid_statuses = {"active", "remission", "resolved"}
    if status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status must be one of: {', '.join(sorted(valid_statuses))}",
        )

    dx.status = status
    dx.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(dx)

    log_audit(db, current_user.id, "diagnosis_updated", resource_type="diagnosis", resource_id=diagnosis_id)
    return _diagnosis_to_response(dx)


# ═══════════════════════════════════════════════════════════════════════════════
# Care Plan Templates (hardcoded fixtures)
# ═══════════════════════════════════════════════════════════════════════════════

_CARE_PLAN_TEMPLATES = [
    {
        "name": "Behavioral Activation — Mild Depression",
        "goals": [
            {"text": "Schedule 3 pleasurable activities per week"},
            {"text": "Track daily mood on 1-10 scale"},
        ],
        "interventions": [
            {"name": "Activity scheduling", "frequency": "daily"},
            {"name": "Mood journaling", "frequency": "daily"},
        ],
    },
    {
        "name": "CBT-Based — Moderate Depression",
        "goals": [
            {"text": "Identify and challenge 2 negative thought patterns per week"},
            {"text": "Complete CBT thought records"},
        ],
        "interventions": [
            {"name": "Cognitive restructuring worksheet", "frequency": "twice_weekly"},
            {"name": "Behavioral experiments", "frequency": "weekly"},
        ],
    },
    {
        "name": "Crisis Stabilization — High Risk",
        "goals": [
            {"text": "Complete safety plan with clinician"},
            {"text": "Establish daily check-in routine"},
            {"text": "Connect with support network"},
        ],
        "interventions": [
            {"name": "Safety planning", "frequency": "immediate"},
            {"name": "Daily check-in screening", "frequency": "daily"},
            {"name": "Emergency contact verification", "frequency": "immediate"},
        ],
    },
    {
        "name": "Medication Management + Therapy",
        "goals": [
            {"text": "Medication adherence for 30 consecutive days"},
            {"text": "Attend weekly therapy sessions"},
        ],
        "interventions": [
            {"name": "Medication tracking", "frequency": "daily"},
            {"name": "Therapy session attendance", "frequency": "weekly"},
            {"name": "Side effect monitoring", "frequency": "weekly"},
        ],
    },
]


@router.get("/care-plan-templates")
async def get_care_plan_templates(
    current_user: User = Depends(require_clinician()),
):
    """Return predefined care plan templates."""
    return _CARE_PLAN_TEMPLATES


# ═══════════════════════════════════════════════════════════════════════════════
# Notification Sending (clinician → patient)
# ═══════════════════════════════════════════════════════════════════════════════


@router.post("/patients/{patient_id}/notify", response_model=NotificationResponse, status_code=201)
@limiter.limit("30/minute")
async def send_patient_notification(
    patient_id: str,
    request: Request,
    title: str = Body(...),
    message: str = Body(...),
    notification_type: str = Body("new_message"),
    link: str = Body(None),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Send a notification to a patient."""
    _verify_patient_access(db, patient_id, current_user.id)

    notif_id = str(uuid4())
    notif = Notification(
        id=notif_id,
        user_id=patient_id,
        notification_type=notification_type,
        title=title,
        message=message,
        link=link,
    )
    db.add(notif)
    db.commit()
    db.refresh(notif)

    log_audit(db, current_user.id, "notification_sent", resource_type="notification", resource_id=notif_id)

    return NotificationResponse(
        id=notif.id,
        notification_type=notif.notification_type,
        title=notif.title,
        message=notif.message,
        link=notif.link,
        is_read=notif.is_read,
        created_at=notif.created_at,
    )
