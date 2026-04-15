"""Route tests for /api/dashboard — clinician-only views and aggregations.

The Phase A perf refactor collapsed `/stats` and `/patients` from 5 / (1+2N)
queries to 2 / 3 queries. These tests both exercise that refactored SQL
path AND guard the behavior so a future "simplification" can't regress it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from app.models.db import Notification, Screening, User


def _seed_screening(
    db,
    *,
    patient_id: str,
    severity: str,
    flagged: bool = False,
    created_at: datetime | None = None,
    symptom_count: int = 3,
) -> Screening:
    s = Screening(
        id=str(uuid4()),
        patient_id=patient_id,
        text="seed",
        source="api",
        severity_level=severity,
        symptom_count=symptom_count,
        final_prediction="some_indicators",
        final_confidence=0.7,
        flagged_for_review=flagged,
        created_at=created_at or datetime.utcnow(),
    )
    db.add(s)
    db.commit()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Auth + RBAC
# ─────────────────────────────────────────────────────────────────────────────


def test_dashboard_stats_requires_auth(client):
    resp = client.get("/api/dashboard/stats")
    assert resp.status_code == 401


def test_dashboard_stats_rejects_patient(client, patient_headers):
    resp = client.get("/api/dashboard/stats", headers=patient_headers)
    assert resp.status_code == 403


def test_dashboard_patients_rejects_patient(client, patient_headers):
    resp = client.get("/api/dashboard/patients", headers=patient_headers)
    assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# /stats — the refactored aggregation query
# ─────────────────────────────────────────────────────────────────────────────


def test_dashboard_stats_empty_when_no_patients(client, clinician_headers):
    resp = client.get("/api/dashboard/stats", headers=clinician_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_patients"] == 0
    assert data["total_screenings"] == 0
    assert data["flagged_count"] == 0
    assert data["severity_distribution"] == {"none": 0, "mild": 0, "moderate": 0, "severe": 0}


def test_dashboard_stats_aggregates_severity_distribution(
    client, clinician_user, linked_patient, clinician_headers, db
):
    """Two mild, one moderate, one severe (flagged), one this-week —
    the aggregation must return each count correctly."""
    patient_id = linked_patient.id
    now = datetime.utcnow()

    _seed_screening(db, patient_id=patient_id, severity="mild", created_at=now - timedelta(days=20))
    _seed_screening(db, patient_id=patient_id, severity="mild", created_at=now - timedelta(days=15))
    _seed_screening(db, patient_id=patient_id, severity="moderate", created_at=now - timedelta(days=10))
    _seed_screening(db, patient_id=patient_id, severity="severe", flagged=True, created_at=now - timedelta(days=2))
    _seed_screening(db, patient_id=patient_id, severity="none", created_at=now - timedelta(hours=3))

    resp = client.get("/api/dashboard/stats", headers=clinician_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_patients"] == 1
    assert data["total_screenings"] == 5
    assert data["flagged_count"] == 1
    assert data["severity_distribution"] == {
        "none": 1,
        "mild": 2,
        "moderate": 1,
        "severe": 1,
    }
    # last seed was within the hour — counts as "this week"
    assert data["screenings_this_week"] >= 1


def test_dashboard_stats_scopes_to_own_patients(client, clinician_user, linked_patient, clinician_headers, db):
    """Screenings from unrelated patients must NOT be counted."""
    # A patient belonging to a DIFFERENT clinician
    other_clinician = User(
        id=str(uuid4()),
        email="other-doc@test.local",
        password_hash="x",
        full_name="Dr Other",
        role="clinician",
        is_active=True,
    )
    other_patient = User(
        id=str(uuid4()),
        email="other-patient@test.local",
        password_hash="x",
        full_name="Other Patient",
        role="patient",
        clinician_id=other_clinician.id,
        is_active=True,
    )
    db.add_all([other_clinician, other_patient])
    db.commit()

    _seed_screening(db, patient_id=linked_patient.id, severity="mild")
    _seed_screening(db, patient_id=other_patient.id, severity="severe", flagged=True)

    resp = client.get("/api/dashboard/stats", headers=clinician_headers)
    assert resp.status_code == 200
    data = resp.json()
    # Only our clinician's one patient + their one mild screening
    assert data["total_patients"] == 1
    assert data["total_screenings"] == 1
    assert data["flagged_count"] == 0
    assert data["severity_distribution"]["severe"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# /patients — latest-screening subquery JOIN
# ─────────────────────────────────────────────────────────────────────────────


def test_dashboard_patients_returns_linked_patients_with_latest_screening(
    client, clinician_user, linked_patient, clinician_headers, db
):
    """Multiple screenings per patient — must return ONLY the latest row's
    severity / symptom_count in the summary."""
    now = datetime.utcnow()
    _seed_screening(
        db,
        patient_id=linked_patient.id,
        severity="mild",
        symptom_count=2,
        created_at=now - timedelta(days=10),
    )
    # This one is most recent → what the endpoint should surface
    _seed_screening(
        db,
        patient_id=linked_patient.id,
        severity="severe",
        symptom_count=7,
        created_at=now - timedelta(hours=1),
    )

    resp = client.get("/api/dashboard/patients", headers=clinician_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    p = data[0]
    assert p["id"] == linked_patient.id
    assert p["last_severity"] == "severe"
    assert p["last_symptom_count"] == 7
    assert p["total_screenings"] == 2


def test_dashboard_patients_empty_when_none_linked(client, clinician_headers):
    resp = client.get("/api/dashboard/patients", headers=clinician_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_dashboard_patients_never_returns_unassigned(client, clinician_user, clinician_headers, db):
    """A patient with no clinician_id must not surface in ANY clinician's view."""
    orphan = User(
        id=str(uuid4()),
        email="orphan@test.local",
        password_hash="x",
        full_name="Orphan Patient",
        role="patient",
        clinician_id=None,
        is_active=True,
    )
    db.add(orphan)
    db.commit()

    resp = client.get("/api/dashboard/patients", headers=clinician_headers)
    assert resp.status_code == 200
    assert resp.json() == []


# ─────────────────────────────────────────────────────────────────────────────
# /patients/{id}/notify — closes B.7 orphan endpoint → UI wiring
# ─────────────────────────────────────────────────────────────────────────────


def test_notify_patient_creates_notification_row(client, linked_patient, clinician_headers, db):
    resp = client.post(
        f"/api/dashboard/patients/{linked_patient.id}/notify",
        headers=clinician_headers,
        json={
            "title": "Appointment confirmed",
            "message": "See you on 20/04 at 10:30.",
            "notification_type": "appointment_reminder",
            "link": "/appointments",
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["title"] == "Appointment confirmed"
    assert data["notification_type"] == "appointment_reminder"

    # Verify row actually exists on the patient's side
    rows = db.query(Notification).filter(Notification.user_id == linked_patient.id).all()
    assert len(rows) == 1
    assert rows[0].title == "Appointment confirmed"


def test_notify_patient_refuses_cross_clinician_access(client, clinician_headers, db):
    """Clinician A must not be able to notify a patient assigned to Clinician B."""
    other_clin = User(
        id=str(uuid4()),
        email="otherclin@test.local",
        password_hash="x",
        full_name="Dr Other",
        role="clinician",
        is_active=True,
    )
    other_patient = User(
        id=str(uuid4()),
        email="otherpt@test.local",
        password_hash="x",
        full_name="Other Patient",
        role="patient",
        clinician_id=other_clin.id,
        is_active=True,
    )
    db.add_all([other_clin, other_patient])
    db.commit()

    resp = client.post(
        f"/api/dashboard/patients/{other_patient.id}/notify",
        headers=clinician_headers,  # OUR clinician, not their clinician
        json={"title": "Nope", "message": "Shouldn't work"},
    )
    assert resp.status_code in (403, 404)
