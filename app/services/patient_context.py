"""
Patient Context Service — builds a comprehensive context string for the LLM.

Pulls ALL relevant patient data (demographics, screenings, medications, allergies,
diagnoses, care plans, appointments, emergency contacts, patient documents, chat
history) and assembles a structured prompt section the LLM can use to give
personalized, clinically-grounded responses.

Privacy: sensitive fields like passwords are never included. CPR numbers and
medical record numbers are redacted (last 4 digits only) since the LLM doesn't
need full identifiers to personalize responses.
"""

import logging
from datetime import date, datetime

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core import localization
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
)

logger = logging.getLogger(__name__)


def _compute_age(dob: date | None) -> int | None:
    if not dob:
        return None
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return age


def _redact(value: str | None, keep_last: int = 4) -> str:
    if not value:
        return "—"
    if len(value) <= keep_last:
        return "*" * len(value)
    return "*" * (len(value) - keep_last) + value[-keep_last:]


def _fmt_date(d: date | datetime | None) -> str:
    if not d:
        return "—"
    if isinstance(d, datetime):
        d = d.date()
    return d.strftime("%d/%m/%Y")


class PatientContextService:
    """Assembles comprehensive patient context for LLM prompts."""

    def build_context(
        self,
        user: User,
        db: Session,
        max_screenings: int = 5,
        sections: list[str] | None = None,
        include_pii: bool = True,
    ) -> str:
        """Return a structured markdown-style context string about this patient.

        Args:
            user: The patient User record.
            db: SQLAlchemy session.
            max_screenings: How many recent screenings to include.
            sections: Optional allowlist of section names to include. When None
                all sections are included (backwards-compatible default).
                Valid names: demographics, medical_identifiers, social_media,
                medications, allergies, diagnoses, screenings, care_plan,
                appointments, screening_schedule, emergency_contacts, documents.
            include_pii: When False, CPR and MRN are omitted entirely (the
                medical_identifiers section is skipped). Default True.

        Sections (each only added if data exists):
        - Demographics
        - Medical Identifiers (redacted)
        - Active Medications
        - Allergies
        - Diagnoses
        - Recent Screenings
        - Active Care Plan
        - Upcoming Appointments
        - Screening Schedule
        - Emergency Contacts
        - Patient Documents
        """
        all_section_names = {
            "demographics", "medical_identifiers", "social_media",
            "medications", "allergies", "diagnoses", "screenings",
            "care_plan", "appointments", "screening_schedule",
            "emergency_contacts", "documents",
        }
        active_sections = set(sections) if sections else all_section_names

        sections_list: list[str] = []

        # 1. Demographics
        if "demographics" in active_sections:
            demo_parts = []
            age = _compute_age(user.date_of_birth)
            if age is not None:
                demo_parts.append(f"Age {age}")
            if user.gender:
                demo_parts.append(user.gender.capitalize())
            if user.nationality:
                demo_parts.append(user.nationality)
            if user.language_preference:
                demo_parts.append(f"Language: {user.language_preference}")
            if user.timezone:
                demo_parts.append(f"Timezone: {user.timezone}")
            if demo_parts:
                sections_list.append(f"**Demographics**\n{user.full_name}. " + ", ".join(demo_parts) + ".")

        # Social media handles (used for ingestion + context)
        if "social_media" in active_sections:
            social_parts = []
            if user.reddit_username:
                social_parts.append(f"Reddit: u/{user.reddit_username}")
            if user.twitter_username:
                social_parts.append(f"X/Twitter: @{user.twitter_username}")
            if social_parts:
                sections_list.append("**Social Media Handles**\n" + " · ".join(social_parts))

        # 2. Medical identifiers (redacted for LLM)
        if "medical_identifiers" in active_sections and include_pii:
            if user.cpr_number or user.medical_record_number or user.blood_type:
                id_parts = []
                if user.cpr_number:
                    id_parts.append(f"CPR ending {_redact(user.cpr_number)}")
                if user.medical_record_number:
                    id_parts.append(f"MRN: {user.medical_record_number}")
                if user.blood_type:
                    id_parts.append(f"Blood type: {user.blood_type}")
                sections_list.append("**Medical Identifiers**\n" + ", ".join(id_parts) + ".")

        # 3. Active medications
        if "medications" in active_sections:
            meds = (
                db.query(Medication)
                .filter(Medication.patient_id == user.id, Medication.is_active == True)
                .order_by(Medication.start_date.desc().nullslast())
                .all()
            )
            if meds:
                med_lines = []
                for m in meds:
                    parts = [f"- **{m.name}**"]
                    if m.dosage:
                        parts.append(m.dosage)
                    if m.frequency:
                        parts.append(m.frequency)
                    if m.prescribed_by:
                        parts.append(f"prescribed by {m.prescribed_by}")
                    if m.notes:
                        parts.append(f"({m.notes[:80]})")
                    med_lines.append(", ".join(parts))
                sections_list.append("**Active Medications**\n" + "\n".join(med_lines))

        # 4. Allergies (especially severe ones)
        if "allergies" in active_sections:
            allergies = (
                db.query(Allergy)
                .filter(Allergy.patient_id == user.id)
                .order_by(
                    # Severity: life-threatening > severe > moderate > mild
                    Allergy.severity.desc()
                )
                .all()
            )
            if allergies:
                allergy_lines = []
                for a in allergies:
                    parts = [f"- **{a.allergen}**"]
                    if a.severity:
                        parts.append(f"severity: {a.severity}")
                    if a.allergy_type:
                        parts.append(a.allergy_type)
                    if a.reaction:
                        parts.append(f"reaction: {a.reaction[:80]}")
                    allergy_lines.append(", ".join(parts))
                sections_list.append("**Allergies**\n" + "\n".join(allergy_lines))

        # 5. Active diagnoses
        if "diagnoses" in active_sections:
            diagnoses = (
                db.query(Diagnosis)
                .filter(
                    Diagnosis.patient_id == user.id,
                    Diagnosis.status.in_(["active", "in_remission"]),
                )
                .order_by(Diagnosis.diagnosed_date.desc().nullslast())
                .all()
            )
            if diagnoses:
                dx_lines = []
                for d in diagnoses:
                    parts = [f"- **{d.condition}**"]
                    if d.icd10_code:
                        parts.append(f"ICD-10: {d.icd10_code}")
                    if d.status:
                        parts.append(f"status: {d.status}")
                    if d.diagnosed_date:
                        parts.append(f"diagnosed: {_fmt_date(d.diagnosed_date)}")
                    if d.diagnosed_by:
                        parts.append(f"by {d.diagnosed_by}")
                    dx_lines.append(", ".join(parts))
                sections_list.append("**Known Diagnoses**\n" + "\n".join(dx_lines))

        # 6. Recent screenings (trend + symptoms)
        if "screenings" in active_sections:
            screenings = (
                db.query(Screening)
                .filter(Screening.patient_id == user.id)
                .order_by(desc(Screening.created_at))
                .limit(max_screenings)
                .all()
            )
            if screenings:
                scr_lines = []
                latest = screenings[0]
                # Latest screening detailed
                sym_labels = []
                if latest.symptom_data:
                    for s in latest.symptom_data.get("symptoms_detected", []):
                        label = s.get("symptom_label") or s.get("symptom") or ""
                        if label:
                            sym_labels.append(label)
                latest_parts = [f"Latest screening ({_fmt_date(latest.created_at)}):"]
                latest_parts.append(f"severity = {latest.severity_level or 'unknown'}")
                latest_parts.append(f"{latest.symptom_count or 0} DSM-5 symptoms detected")
                if sym_labels:
                    latest_parts.append(f"specifically: {', '.join(sym_labels[:5])}")
                scr_lines.append(" | ".join(latest_parts))

                # Historical trend
                if len(screenings) > 1:
                    trend = [
                        f"{_fmt_date(s.created_at)}: {s.severity_level or '?'} ({s.symptom_count or 0} sx)"
                        for s in screenings[1:]
                    ]
                    scr_lines.append(f"Earlier screenings: {' → '.join(trend)}")

                sections_list.append("**Screening History**\n" + "\n".join(scr_lines))

        # 7. Active care plan
        if "care_plan" in active_sections:
            care_plan = (
                db.query(CarePlan)
                .filter(CarePlan.patient_id == user.id, CarePlan.status == "active")
                .order_by(desc(CarePlan.created_at))
                .first()
            )
            if care_plan:
                cp_parts = [f"- Title: **{care_plan.title}**"]
                if care_plan.description:
                    cp_parts.append(f"- Description: {care_plan.description[:200]}")
                if care_plan.goals:
                    goal_texts = []
                    goals_data = care_plan.goals if isinstance(care_plan.goals, list) else []
                    for g in goals_data[:5]:
                        if isinstance(g, dict):
                            goal_texts.append(g.get("text", ""))
                    if goal_texts:
                        cp_parts.append("- Goals: " + "; ".join(filter(None, goal_texts)))
                if care_plan.interventions:
                    int_texts = []
                    int_data = care_plan.interventions if isinstance(care_plan.interventions, list) else []
                    for i in int_data[:5]:
                        if isinstance(i, dict):
                            name = i.get("name", "")
                            freq = i.get("frequency", "")
                            int_texts.append(f"{name} ({freq})" if freq else name)
                    if int_texts:
                        cp_parts.append("- Interventions: " + "; ".join(filter(None, int_texts)))
                if care_plan.review_date:
                    cp_parts.append(f"- Next review: {_fmt_date(care_plan.review_date)}")
                sections_list.append("**Active Care Plan**\n" + "\n".join(cp_parts))

        # 8. Upcoming appointments
        if "appointments" in active_sections:
            upcoming = (
                db.query(Appointment)
                .filter(
                    Appointment.patient_id == user.id,
                    Appointment.status.in_(["scheduled", "confirmed"]),
                    Appointment.scheduled_at >= datetime.utcnow(),
                )
                .order_by(Appointment.scheduled_at)
                .limit(3)
                .all()
            )
            if upcoming:
                appt_lines = []
                for a in upcoming:
                    parts = [f"- {_fmt_date(a.scheduled_at)} {a.scheduled_at.strftime('%H:%M')}"]
                    if a.appointment_type:
                        parts.append(a.appointment_type)
                    if a.location:
                        parts.append(f"at {a.location}")
                    if a.status:
                        parts.append(f"[{a.status}]")
                    appt_lines.append(", ".join(parts))
                sections_list.append("**Upcoming Appointments**\n" + "\n".join(appt_lines))

        # 9. Screening schedule (recurring check-ins)
        if "screening_schedule" in active_sections:
            schedule = (
                db.query(ScreeningSchedule)
                .filter(ScreeningSchedule.patient_id == user.id, ScreeningSchedule.is_active == True)
                .first()
            )
            if schedule:
                sched_parts = [f"Frequency: {schedule.frequency}"]
                if schedule.next_due_at:
                    sched_parts.append(f"next due: {_fmt_date(schedule.next_due_at)}")
                if schedule.last_completed_at:
                    sched_parts.append(f"last completed: {_fmt_date(schedule.last_completed_at)}")
                sections_list.append("**Check-in Schedule**\n" + ", ".join(sched_parts))

        # 10. Emergency contacts (primary only — for crisis reference)
        if "emergency_contacts" in active_sections:
            primary_contact = (
                db.query(EmergencyContact)
                .filter(EmergencyContact.patient_id == user.id, EmergencyContact.is_primary == True)
                .first()
            )
            if primary_contact:
                parts = [primary_contact.contact_name]
                if primary_contact.relation:
                    parts.append(f"({primary_contact.relation})")
                if primary_contact.phone:
                    parts.append(primary_contact.phone)
                sections_list.append("**Primary Emergency Contact**\n" + " ".join(parts))

        # 11. Patient documents (just titles — contents already embedded in RAG)
        if "documents" in active_sections:
            docs = (
                db.query(PatientDocument)
                .filter(PatientDocument.patient_id == user.id)
                .order_by(desc(PatientDocument.created_at))
                .limit(5)
                .all()
            )
            if docs:
                doc_lines = []
                for d in docs:
                    parts = [f"- {d.title}"]
                    if d.doc_type:
                        parts.append(f"({d.doc_type})")
                    if d.created_at:
                        parts.append(f"uploaded {_fmt_date(d.created_at)}")
                    doc_lines.append(" ".join(parts))
                sections_list.append(
                    "**Uploaded Documents**\n"
                    + "\n".join(doc_lines)
                    + "\n(Contents searchable via RAG — refer by title when relevant.)"
                )

        if not sections_list:
            return "### Patient Medical Profile\nNo medical history recorded yet."

        # Header with Bahrain context
        header = "# Patient Profile\n\n_All information below is about the patient you are currently chatting with. Use it to personalize your response. Never disclose CPR/MRN numbers back to the patient. Respect the patient's language preference._\n"
        crisis_reminder = (
            f"\n---\n**Bahrain crisis resources (always surface if relevant):** Emergency {localization.EMERGENCY_NUMBER}, "
            f"Shamsaha 17651421 (24/7 confidential)."
        )

        return header + "\n\n".join(sections_list) + crisis_reminder
