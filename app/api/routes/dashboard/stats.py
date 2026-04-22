"""
Dashboard stats, patient list, full patient profile, and PDF summary endpoints.
"""

from datetime import date as _date
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.models.db import (
    Allergy,
    Appointment,
    CarePlan,
    Diagnosis,
    EmergencyContact,
    Medication,
    PatientDocument,
    Screening,
    ScreeningSchedule,
    User,
    get_db,
)
from app.schemas.analysis import DashboardStats, PatientSummary
from app.services.auth import log_audit, require_clinician

from ._shared import _verify_patient_access

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get aggregate statistics for the clinician's dashboard.

    Single aggregation query over the screenings table instead of four
    separate counts + a group-by. Drops this endpoint from 5 queries to
    2 on every dashboard load.
    """
    from sqlalchemy import case

    # 1. Patient IDs assigned to this clinician
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

    # 2. All five metrics from one aggregation row. `func.sum(case(...))`
    # counts rows matching a predicate — same semantics as `COUNT(*) WHERE
    # predicate` but in a single pass over the screenings table.
    week_ago = datetime.utcnow() - timedelta(days=7)
    agg = (
        db.query(
            func.count(Screening.id).label("total"),
            func.coalesce(func.sum(case((Screening.flagged_for_review == True, 1), else_=0)), 0).label("flagged"),
            func.coalesce(func.sum(case((Screening.created_at >= week_ago, 1), else_=0)), 0).label("this_week"),
            func.coalesce(func.sum(case((Screening.severity_level == "severe", 1), else_=0)), 0).label("sev_severe"),
            func.coalesce(func.sum(case((Screening.severity_level == "moderate", 1), else_=0)), 0).label(
                "sev_moderate"
            ),
            func.coalesce(func.sum(case((Screening.severity_level == "mild", 1), else_=0)), 0).label("sev_mild"),
            func.coalesce(func.sum(case((Screening.severity_level == "none", 1), else_=0)), 0).label("sev_none"),
        )
        .filter(Screening.patient_id.in_(patient_ids))
        .one()
    )

    return DashboardStats(
        total_patients=total_patients,
        total_screenings=int(agg.total or 0),
        flagged_count=int(agg.flagged or 0),
        severity_distribution={
            "none": int(agg.sev_none or 0),
            "mild": int(agg.sev_mild or 0),
            "moderate": int(agg.sev_moderate or 0),
            "severe": int(agg.sev_severe or 0),
        },
        screenings_this_week=int(agg.this_week or 0),
    )


@router.get("/patients", response_model=list[PatientSummary])
async def get_patients(
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get all patients assigned to this clinician with their latest screening info.

    Was O(1 + 2N) queries (one per patient for latest + count). Now O(3):
    one query for patients, one group-by for counts, one subquery-joined
    query for each patient's latest screening row. Scales flat regardless
    of how many patients the clinician has.
    """
    from sqlalchemy import and_

    patients = db.query(User).filter(User.clinician_id == current_user.id).all()
    if not patients:
        return []

    patient_ids = [p.id for p in patients]

    # Screening counts per patient — single GROUP BY.
    count_rows = (
        db.query(Screening.patient_id, func.count(Screening.id))
        .filter(Screening.patient_id.in_(patient_ids))
        .group_by(Screening.patient_id)
        .all()
    )
    counts_by_patient = dict(count_rows)

    # Latest screening per patient — cross-dialect pattern via a
    # max(created_at) subquery joined back to the base table. Avoids
    # fetching every historical screening just to filter in memory.
    # (DISTINCT ON would be faster on Postgres but isn't portable.)
    latest_subq = (
        db.query(
            Screening.patient_id.label("pid"),
            func.max(Screening.created_at).label("latest_at"),
        )
        .filter(Screening.patient_id.in_(patient_ids))
        .group_by(Screening.patient_id)
        .subquery()
    )
    latest_rows = (
        db.query(Screening)
        .join(
            latest_subq,
            and_(
                Screening.patient_id == latest_subq.c.pid,
                Screening.created_at == latest_subq.c.latest_at,
            ),
        )
        .all()
    )
    latest_by_patient = {s.patient_id: s for s in latest_rows}

    summaries = []
    for patient in patients:
        latest = latest_by_patient.get(patient.id)
        summaries.append(
            PatientSummary(
                id=patient.id,
                full_name=patient.full_name,
                email=patient.email,
                last_screening_date=latest.created_at if latest else None,
                last_severity=latest.severity_level if latest else None,
                last_symptom_count=latest.symptom_count if latest else None,
                total_screenings=counts_by_patient.get(patient.id, 0),
            )
        )

    # Sort by severity (severe first, then by last screening date)
    severity_order = {"severe": 0, "moderate": 1, "mild": 2, "none": 3, None: 4}
    summaries.sort(
        key=lambda s: (severity_order.get(s.last_severity, 4), s.last_screening_date or datetime.min), reverse=False
    )

    return summaries


@router.get("/patients/{patient_id}/full-profile")
async def get_patient_full_profile(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Return ALL profile + clinical data for a patient in a single response.

    Used by the clinician's patient-detail view. Combining 8+ queries into one
    HTTP round-trip eliminates the waterfall that dominated page load time.

    Returns: demographics, medical identifiers, medications, allergies, active
    diagnoses, emergency contacts, social media handles, screening schedule,
    and aggregate counts (screenings, documents, appointments, care plans).
    """
    patient = _verify_patient_access(db, patient_id, current_user.id)

    # Demographics
    age = None
    if patient.date_of_birth:
        today = _date.today()
        age = (
            today.year
            - patient.date_of_birth.year
            - ((today.month, today.day) < (patient.date_of_birth.month, patient.date_of_birth.day))
        )

    meds = (
        db.query(Medication)
        .filter(Medication.patient_id == patient_id)
        .order_by(Medication.start_date.desc().nullslast())
        .all()
    )
    allergies = db.query(Allergy).filter(Allergy.patient_id == patient_id).all()
    diagnoses = (
        db.query(Diagnosis)
        .filter(Diagnosis.patient_id == patient_id)
        .order_by(Diagnosis.diagnosed_date.desc().nullslast())
        .all()
    )
    contacts = db.query(EmergencyContact).filter(EmergencyContact.patient_id == patient_id).all()
    schedule = (
        db.query(ScreeningSchedule)
        .filter(ScreeningSchedule.patient_id == patient_id, ScreeningSchedule.is_active == True)
        .first()
    )

    # Aggregate counts — single DB round trip via scalar subqueries
    screening_sq = db.query(func.count(Screening.id)).filter(Screening.patient_id == patient_id).scalar_subquery()
    doc_sq = db.query(func.count(PatientDocument.id)).filter(PatientDocument.patient_id == patient_id).scalar_subquery()
    appt_sq = (
        db.query(func.count(Appointment.id))
        .filter(
            Appointment.patient_id == patient_id,
            Appointment.status.in_(["scheduled", "confirmed"]),
            Appointment.scheduled_at >= datetime.utcnow(),
        )
        .scalar_subquery()
    )
    cp_sq = (
        db.query(func.count(CarePlan.id))
        .filter(CarePlan.patient_id == patient_id, CarePlan.status.in_(["active", "review_needed"]))
        .scalar_subquery()
    )
    screening_count, doc_count, appt_count, cp_count = db.query(screening_sq, doc_sq, appt_sq, cp_sq).one()
    latest_screening = (
        db.query(Screening).filter(Screening.patient_id == patient_id).order_by(desc(Screening.created_at)).first()
    )

    return {
        "id": patient.id,
        "email": patient.email,
        "full_name": patient.full_name,
        "demographics": {
            "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else None,
            "age": age,
            "gender": patient.gender,
            "nationality": patient.nationality,
            "language_preference": patient.language_preference,
            "timezone": patient.timezone,
        },
        "medical_identifiers": {
            "cpr_number": patient.cpr_number,
            "medical_record_number": patient.medical_record_number,
            "blood_type": patient.blood_type,
        },
        "contact": {
            "phone": patient.phone,
            "reddit_username": patient.reddit_username,
            "twitter_username": patient.twitter_username,
        },
        "onboarding_completed": bool(patient.onboarding_completed),
        "profile_picture_url": patient.profile_picture_url,
        "created_at": patient.created_at.isoformat() if patient.created_at else None,
        "last_login_at": patient.last_login_at.isoformat() if patient.last_login_at else None,
        "medications": [
            {
                "id": m.id,
                "name": m.name,
                "dosage": m.dosage,
                "frequency": m.frequency,
                "start_date": m.start_date.isoformat() if m.start_date else None,
                "end_date": m.end_date.isoformat() if m.end_date else None,
                "is_active": m.is_active,
                "prescribed_by": m.prescribed_by,
                "notes": m.notes,
            }
            for m in meds
        ],
        "allergies": [
            {
                "id": a.id,
                "allergen": a.allergen,
                "severity": a.severity,
                "allergy_type": a.allergy_type,
                "reaction": a.reaction,
                "diagnosed_date": a.diagnosed_date.isoformat() if a.diagnosed_date else None,
                "notes": a.notes,
            }
            for a in allergies
        ],
        "diagnoses": [
            {
                "id": d.id,
                "condition": d.condition,
                "icd10_code": d.icd10_code,
                "status": d.status,
                "diagnosed_date": d.diagnosed_date.isoformat() if d.diagnosed_date else None,
                "diagnosed_by": d.diagnosed_by,
                "notes": d.notes,
            }
            for d in diagnoses
        ],
        "emergency_contacts": [
            {
                "id": c.id,
                "contact_name": c.contact_name,
                "phone": c.phone,
                "relation": c.relation,
                "is_primary": c.is_primary,
            }
            for c in contacts
        ],
        "screening_schedule": (
            {
                "id": schedule.id,
                "frequency": schedule.frequency,
                "day_of_week": schedule.day_of_week,
                "preferred_time": schedule.preferred_time,
                "next_due_at": schedule.next_due_at.isoformat() if schedule.next_due_at else None,
                "last_completed_at": schedule.last_completed_at.isoformat() if schedule.last_completed_at else None,
            }
            if schedule
            else None
        ),
        "stats": {
            "total_screenings": screening_count,
            "total_documents": doc_count,
            "upcoming_appointments": appt_count,
            "active_care_plans": cp_count,
            "last_severity": latest_screening.severity_level if latest_screening else None,
            "last_screening_date": latest_screening.created_at.isoformat() if latest_screening else None,
        },
    }


@router.get("/patients/{patient_id}/summary.pdf")
async def download_patient_summary_pdf(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician downloads a PDF clinical summary for a patient."""
    from fastapi.responses import StreamingResponse

    from app.services.reports import build_patient_summary_pdf

    patient = _verify_patient_access(db, patient_id, current_user.id)

    medications = db.query(Medication).filter(Medication.patient_id == patient_id).all()
    allergies = db.query(Allergy).filter(Allergy.patient_id == patient_id).all()
    diagnoses = db.query(Diagnosis).filter(Diagnosis.patient_id == patient_id).all()
    contacts = db.query(EmergencyContact).filter(EmergencyContact.patient_id == patient_id).all()
    care_plans = db.query(CarePlan).filter(CarePlan.patient_id == patient_id).order_by(desc(CarePlan.updated_at)).all()
    screenings = (
        db.query(Screening)
        .filter(Screening.patient_id == patient_id)
        .order_by(desc(Screening.created_at))
        .limit(10)
        .all()
    )

    patient_dict = {
        "full_name": patient.full_name,
        "email": patient.email,
        "phone": patient.phone,
        "date_of_birth": patient.date_of_birth,
        "gender": patient.gender,
        "nationality": patient.nationality,
        "cpr_number": patient.cpr_number,
        "medical_record_number": patient.medical_record_number,
        "blood_type": patient.blood_type,
    }

    export_dict = {
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
                "notes": a.notes,
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
        "screenings": [
            {
                "created_at": s.created_at,
                "severity_label": s.severity_level or "none",
                "severity_score": s.symptom_count,
                "flagged_for_review": s.flagged_for_review,
                "clinician_notes": s.clinician_notes,
            }
            for s in screenings
        ],
    }

    buf = build_patient_summary_pdf(patient_dict, export_dict, clinician_name=current_user.full_name)
    log_audit(db, current_user.id, "patient_summary_pdf", resource_type="patient", resource_id=patient_id)

    filename = f"depscreen-summary-{patient.full_name.replace(' ', '-')}-{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
