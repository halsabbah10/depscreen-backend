"""
Background scheduler for time-based email notifications.

Jobs:
- screening_reminders_job (hourly): scan ScreeningSchedule, email patients
  whose next_due_at is within the next hour OR who are overdue by N days
- appointment_reminders_job (hourly): scan Appointment, email patients whose
  scheduled_at is ~24h away (plus a 1-hour fuzzy window to tolerate missed ticks)
- care_plan_review_job (daily): scan CarePlan for plans whose review_date is
  today or past — mark status = 'review_needed', create Notification for clinician

Implementation notes:
- Uses APScheduler BackgroundScheduler (in-process)
- Runs inside the FastAPI app lifespan, started at startup, stopped at shutdown
- Works on HF Spaces when the container is active; sleeps when the Space sleeps
  (free-tier limitation — acceptable for demo)
- Deduplicates via a "last_email_sent_at" field pattern: every reminder job
  updates a timestamp to prevent sending the same reminder twice in one window
- All timezone math uses UTC; display formatting is done in the email templates
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import get_settings
from app.models.db import (
    Appointment,
    CarePlan,
    Notification,
    ScreeningSchedule,
    SessionLocal,
    User,
)
from app.services.email import get_email_service

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


# ── Individual job functions ─────────────────────────────────────────────────


def screening_reminders_job() -> None:
    """Email patients whose scheduled check-in is due soon or overdue.

    Triggers:
    - Due today (within next 24h): 'Your check-in is due' email
    - Overdue by 3+ days: 'Your check-in is overdue' email (once per week after that)
    """
    settings = get_settings()
    email_svc = get_email_service(settings)
    if not email_svc.enabled:
        logger.debug("Screening reminders: email disabled, skipping job")
        return

    now = datetime.now(UTC).replace(tzinfo=None)
    db = SessionLocal()
    sent_count = 0
    try:
        # Due within next 24 hours, not yet reminded today
        due_soon = (
            db.query(ScreeningSchedule)
            .join(User, User.id == ScreeningSchedule.patient_id)
            .filter(
                ScreeningSchedule.is_active == True,
                ScreeningSchedule.next_due_at >= now,
                ScreeningSchedule.next_due_at <= now + timedelta(hours=24),
                User.email_notifications == True,
            )
            .all()
        )
        for sched in due_soon:
            # Skip if already reminded in the last 20 hours (de-dupe)
            if sched.last_reminder_sent_at and (now - sched.last_reminder_sent_at) < timedelta(hours=20):
                continue
            patient = db.query(User).filter(User.id == sched.patient_id).first()
            if not patient or not patient.email:
                continue
            if email_svc.send_screening_reminder(patient.full_name, patient.email, days_overdue=0):
                sched.last_reminder_sent_at = now
                sent_count += 1
                # Also create an in-app notification
                db.add(
                    Notification(
                        user_id=patient.id,
                        notification_type="screening_due",
                        title="Your check-in is due",
                        message="A scheduled check-in is due today.",
                        link="/screening",
                    )
                )

        # Overdue by 3+ days — remind once a week
        overdue = (
            db.query(ScreeningSchedule)
            .join(User, User.id == ScreeningSchedule.patient_id)
            .filter(
                ScreeningSchedule.is_active == True,
                ScreeningSchedule.next_due_at < now - timedelta(days=3),
                User.email_notifications == True,
            )
            .all()
        )
        for sched in overdue:
            if sched.last_reminder_sent_at and (now - sched.last_reminder_sent_at) < timedelta(days=7):
                continue
            patient = db.query(User).filter(User.id == sched.patient_id).first()
            if not patient or not patient.email:
                continue
            days_overdue = (now - sched.next_due_at).days
            if email_svc.send_screening_reminder(patient.full_name, patient.email, days_overdue=days_overdue):
                sched.last_reminder_sent_at = now
                sent_count += 1
                db.add(
                    Notification(
                        user_id=patient.id,
                        notification_type="screening_overdue",
                        title="Your check-in is overdue",
                        message=f"It's been {days_overdue} day(s) since your scheduled check-in.",
                        link="/screening",
                    )
                )

        db.commit()
        if sent_count:
            logger.info(f"Screening reminders: sent {sent_count} emails")
    except Exception as e:
        logger.error(f"Screening reminders job failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def appointment_reminders_job() -> None:
    """Email patients about appointments scheduled ~24 hours from now."""
    settings = get_settings()
    email_svc = get_email_service(settings)
    if not email_svc.enabled:
        return

    now = datetime.now(UTC).replace(tzinfo=None)
    # Reminder window: 23-25h away, so we catch every appointment exactly once
    window_start = now + timedelta(hours=23)
    window_end = now + timedelta(hours=25)

    db = SessionLocal()
    sent_count = 0
    try:
        upcoming = (
            db.query(Appointment)
            .filter(
                Appointment.status.in_(["scheduled", "confirmed"]),
                Appointment.scheduled_at >= window_start,
                Appointment.scheduled_at <= window_end,
                Appointment.reminder_sent_at.is_(None),
            )
            .all()
        )
        for appt in upcoming:
            patient = db.query(User).filter(User.id == appt.patient_id).first()
            clinician = db.query(User).filter(User.id == appt.clinician_id).first()
            if not patient or not patient.email or not patient.email_notifications:
                continue

            formatted_time = appt.scheduled_at.strftime("%d/%m/%Y at %H:%M")
            clinician_name = clinician.full_name if clinician else None

            if email_svc.send_appointment_reminder(
                patient_name=patient.full_name,
                patient_email=patient.email,
                appointment_at=formatted_time,
                clinician_name=clinician_name,
            ):
                appt.reminder_sent_at = now
                sent_count += 1
                db.add(
                    Notification(
                        user_id=patient.id,
                        notification_type="appointment_reminder",
                        title="Appointment tomorrow",
                        message=f"You have an appointment on {formatted_time}.",
                        link="/appointments",
                    )
                )

        db.commit()
        if sent_count:
            logger.info(f"Appointment reminders: sent {sent_count} emails")
    except Exception as e:
        logger.error(f"Appointment reminders job failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def care_plan_review_job() -> None:
    """Flag care plans whose review_date has passed, notify their clinicians."""
    now = datetime.now(UTC).replace(tzinfo=None)
    today = now.date()

    db = SessionLocal()
    flagged_count = 0
    try:
        plans = (
            db.query(CarePlan)
            .filter(
                CarePlan.status == "active",
                CarePlan.review_date.isnot(None),
                CarePlan.review_date <= today,
            )
            .all()
        )
        for plan in plans:
            plan.status = "review_needed"
            flagged_count += 1
            clinician = db.query(User).filter(User.id == plan.clinician_id).first()
            patient = db.query(User).filter(User.id == plan.patient_id).first()

            if clinician:
                db.add(
                    Notification(
                        user_id=clinician.id,
                        notification_type="care_plan_review",
                        title="Care plan review needed",
                        message=f"Review date reached for {patient.full_name if patient else 'a patient'}'s care plan: {plan.title}",
                        link=f"/patients/{plan.patient_id}",
                    )
                )

        db.commit()
        if flagged_count:
            logger.info(f"Care plan review: flagged {flagged_count} plans")
    except Exception as e:
        logger.error(f"Care plan review job failed: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


# ── Lifecycle ────────────────────────────────────────────────────────────────


def start_scheduler() -> BackgroundScheduler | None:
    """Start the background scheduler. Idempotent."""
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    try:
        _scheduler = BackgroundScheduler(timezone="UTC")

        # Every hour at the top of the hour
        _scheduler.add_job(
            screening_reminders_job,
            trigger=CronTrigger(minute=0),
            id="screening_reminders",
            name="Screening reminders",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
        )
        # Every hour at :30 (staggered so jobs don't all fire at once)
        _scheduler.add_job(
            appointment_reminders_job,
            trigger=CronTrigger(minute=30),
            id="appointment_reminders",
            name="Appointment reminders",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
        )
        # Daily at 02:00 UTC (05:00 Bahrain time, low traffic)
        _scheduler.add_job(
            care_plan_review_job,
            trigger=CronTrigger(hour=2, minute=0),
            id="care_plan_review",
            name="Care plan review sweeps",
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )

        _scheduler.start()
        logger.info("Background scheduler started (3 jobs registered)")
        return _scheduler
    except Exception as e:
        logger.error(f"Scheduler startup failed: {e}", exc_info=True)
        return None


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Background scheduler stopped")


def get_scheduler() -> BackgroundScheduler | None:
    return _scheduler
