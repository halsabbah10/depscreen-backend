"""
Screening list, detail, triage, notes, trends, and patient document endpoints.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.middleware.rate_limiter import limiter
from app.models.db import (
    ChatMessage,
    PatientDocument,
    Screening,
    User,
    get_db,
)
from app.schemas.analysis import (
    ScreeningHistoryResponse,
    ScreeningListItem,
)
from app.services.auth import log_audit, require_clinician
from app.services.rag import RAGService

from ._shared import _get_rag, _verify_patient_access

router = APIRouter()

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


class _ClinicianNotesBody(BaseModel):
    notes: str = Field(default="", max_length=10000)


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
    body: _ClinicianNotesBody,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Add or update clinician notes on a screening."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Verify the screening belongs to one of this clinician's patients
    _verify_patient_access(db, screening.patient_id, current_user.id)

    screening.clinician_notes = body.notes
    db.commit()

    return {"status": "updated", "screening_id": screening_id}


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

    # Verify the screening belongs to one of this clinician's patients
    _verify_patient_access(db, screening.patient_id, current_user.id)

    screening.triage_status = status
    screening.reviewed_by = current_user.id
    screening.reviewed_at = datetime.now(UTC)

    if next_action:
        screening.next_action = next_action
    if next_action_date:
        screening.next_action_date = datetime.fromisoformat(next_action_date)

    db.commit()
    log_audit(db, current_user.id, f"triage_{status}", resource_type="screening", resource_id=screening_id)

    return {"status": status, "screening_id": screening_id}


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

    start_date = datetime.now(UTC) - timedelta(days=days)
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

    # Compute overall trend (same logic as patient self-service endpoint)
    if len(timeline) >= 2:
        severity_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        first_sev = severity_order.get(timeline[0]["severity_level"], 0)
        last_sev = severity_order.get(timeline[-1]["severity_level"], 0)
        trend = "worsening" if last_sev > first_sev else "improving" if last_sev < first_sev else "stable"
    else:
        trend = "insufficient_data"

    return {
        "patient_id": patient_id,
        "patient_name": patient.full_name,
        "days_analyzed": days,
        "total_screenings": len(timeline),
        "trend": trend,
        "all_symptoms_observed": sorted(all_symptoms_seen),
        "timeline": timeline,
    }


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
    if rag and rag.is_initialized:
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


@router.post("/patients/{patient_id}/documents/upload")
@limiter.limit("30/minute")
async def upload_patient_document_file(
    patient_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Query(..., min_length=1, max_length=255),
    doc_type: str = Query(...),
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Upload a file (PDF, DOCX, TXT) as a patient document. Clinician access."""
    from app.services.document_extractor import extract_text
    from app.services.rag_safety import should_ingest_to_rag

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

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    filename = (file.filename or "document.txt").lower()
    result = extract_text(raw, filename)
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this document. The file may be damaged or in an unsupported format.",
        )
    content = result.text

    if len(content) > 100_000:
        content = content[:100_000]

    doc_id = str(uuid4())
    doc = PatientDocument(
        id=doc_id,
        patient_id=patient_id,
        uploaded_by=current_user.id,
        doc_type=doc_type,
        title=title,
        content=content,
        processing_status="processing" if should_ingest_to_rag(doc_type) else "ready",
    )
    db.add(doc)
    db.commit()

    if should_ingest_to_rag(doc_type):
        from app.services.container import get_rag_service

        rag = get_rag_service()
        if rag and rag.is_initialized:
            background_tasks.add_task(
                _ingest_clinician_doc_background,
                patient_id=patient_id,
                doc_id=doc_id,
                doc_type=doc_type,
                title=title,
                content=content,
            )

    log_audit(db, current_user.id, "document_uploaded_file", resource_type="document", resource_id=doc_id)

    return {
        "status": "processing" if should_ingest_to_rag(doc_type) else "ready",
        "document_id": doc_id,
        "extracted_chars": len(content),
    }


async def _ingest_clinician_doc_background(patient_id: str, doc_id: str, doc_type: str, title: str, content: str):
    """Background task: embed clinician-uploaded document into patient RAG and update status."""
    import logging

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
        logging.getLogger(__name__).error(f"Clinician doc ingestion failed for {doc_id}: {e}")
        doc = db.query(PatientDocument).filter_by(id=doc_id).first()
        if doc:
            doc.processing_status = "failed"
            doc.processing_error = str(e)[:500]
            db.commit()
    finally:
        db.close()
