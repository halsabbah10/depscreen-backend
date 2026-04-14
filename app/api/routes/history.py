"""
Patient screening history API routes.

Provides authenticated access to a patient's own screening history.
Clinicians access patient history via /api/dashboard endpoints instead.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.middleware.rate_limiter import limiter
from app.models.db import Screening, User, get_db
from app.services.reports import build_screening_pdf
from app.schemas.analysis import (
    Evidence,
    ExplanationReport,
    PostSymptomSummary,
    ScreeningHistoryResponse,
    ScreeningListItem,
    ScreeningResponse,
    VerificationReport,
)
from app.services.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)


def _screening_to_response(s: Screening) -> ScreeningResponse:
    """Convert a Screening DB model to a ScreeningResponse schema."""
    symptom_analysis = PostSymptomSummary(
        **(
            s.symptom_data
            or {
                "symptoms_detected": [],
                "unique_symptom_count": 0,
                "total_sentences_analyzed": 0,
                "severity_level": "none",
                "severity_explanation": "",
                "dsm5_criteria_met": [],
            }
        )
    )

    evidence = Evidence(
        sentence_evidence=symptom_analysis.symptoms_detected,
        top_evidence_sentences=[
            d.sentence_text
            for d in sorted(
                symptom_analysis.symptoms_detected,
                key=lambda d: d.confidence,
                reverse=True,
            )[:5]
        ],
    )

    verification = VerificationReport(
        **(
            s.verification_data
            or {
                "evidence_validation": {
                    "evidence_supports_prediction": True,
                    "coherence_score": 0.7,
                    "flagged_for_review": False,
                },
                "confidence_analysis": {
                    "should_trust_prediction": "medium",
                    "reasoning": "No verification data",
                    "potential_confounders": [],
                },
                "adversarial_check": {"likely_adversarial": False, "authenticity_score": 0.8},
            }
        )
    )

    explanation = ExplanationReport(
        **(
            s.explanation_data
            or {
                "summary": "No explanation available",
                "risk_level": s.severity_level or "none",
                "why_model_thinks_this": "",
                "key_evidence_quotes": [],
                "uncertainty_notes": "",
                "safety_disclaimer": "",
                "resources": [],
            }
        )
    )

    return ScreeningResponse(
        id=s.id,
        created_at=s.created_at,
        text=s.text,
        symptom_analysis=symptom_analysis,
        evidence=evidence,
        verification=verification,
        final_prediction=s.final_prediction,
        final_confidence=s.final_confidence,
        confidence_adjusted=s.confidence_adjusted,
        explanation_report=explanation,
        flagged_for_review=s.flagged_for_review,
        adversarial_warning=s.adversarial_warning,
    )


@router.get("", response_model=ScreeningHistoryResponse)
async def list_my_screenings(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List the current patient's screening history with pagination."""
    query = db.query(Screening).filter(Screening.patient_id == current_user.id)

    total = query.count()
    screenings = query.order_by(desc(Screening.created_at)).offset((page - 1) * page_size).limit(page_size).all()

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


@router.get("/{screening_id}", response_model=ScreeningResponse)
async def get_screening_by_id(
    screening_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a specific screening by ID. Patients can only access their own."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Authorization: patients see own, clinicians see their patients'
    if current_user.role == "patient" and screening.patient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot access this screening")
    elif current_user.role == "clinician":
        if screening.patient and screening.patient.clinician_id != current_user.id:
            raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    return _screening_to_response(screening)


@router.get("/{screening_id}/pdf")
async def download_screening_pdf(
    screening_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download a single screening as a printable PDF report."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    # Same authorization as the JSON endpoint
    if current_user.role == "patient" and screening.patient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot access this screening")
    if current_user.role == "clinician":
        if screening.patient and screening.patient.clinician_id != current_user.id:
            raise HTTPException(status_code=403, detail="This patient is not assigned to you")

    patient = screening.patient
    symptom_data = screening.symptom_data or {}
    explanation_data = screening.explanation_data or {}

    screening_dict = {
        "id": screening.id,
        "created_at": screening.created_at,
        "severity_label": screening.severity_level or "none",
        "severity_score": symptom_data.get("total_sentences_analyzed"),
        "symptoms": [
            {"criterion": sym.get("dsm5_criterion", sym.get("criterion", "")),
             "confidence": sym.get("confidence", 0)}
            for sym in symptom_data.get("symptoms_detected", [])
        ],
        "detected_sentences": [
            sym.get("sentence_text", "")
            for sym in symptom_data.get("symptoms_detected", [])
            if sym.get("sentence_text")
        ],
        "llm_explanation": explanation_data.get("why_model_thinks_this") or explanation_data.get("summary") or "",
        "flagged_for_review": bool(screening.flagged_for_review),
    }

    patient_dict = {
        "full_name": patient.full_name if patient else "—",
        "email": patient.email if patient else None,
        "date_of_birth": patient.date_of_birth if patient else None,
        "cpr_number": patient.cpr_number if patient else None,
        "medical_record_number": patient.medical_record_number if patient else None,
    }

    buf = build_screening_pdf(screening_dict, patient_dict)
    filename = f"depscreen-screening-{screening_id[:8]}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{screening_id}")
@limiter.limit("30/minute")
async def delete_screening(
    screening_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a screening. Patients can only delete their own."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    if current_user.role == "patient" and screening.patient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot delete this screening")

    db.delete(screening)
    db.commit()
    return {"status": "deleted", "id": screening_id}
