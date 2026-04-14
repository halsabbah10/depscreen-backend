"""
Email notification service using Resend.

Sends transactional emails for:
- Welcome on registration
- Crisis alert to clinician (patient flagged severe)
- Screening reminder
- Care plan updates
- Appointment reminders

Design principles:
- Clinical Sanctuary aesthetic in HTML templates (cream, teal, Cormorant Garamond)
- Always graceful: failures log but don't break the user flow
- Respect user preferences: if user.email_notifications=False, skip
- Never include sensitive info (CPR, full medical history) in emails — security layer
- Every email links back to the app for full context
"""

import logging

import resend

from app.core.config import Settings

logger = logging.getLogger(__name__)


# ── Email templates ──────────────────────────────────────────────────────────
# Minimal shared wrapper so every email looks consistent.

_BASE_STYLES = """
  body { margin: 0; padding: 0; background: hsl(35, 25%, 97%); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: hsl(220, 15%, 15%); }
  .wrap { max-width: 560px; margin: 40px auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.06); }
  .header { background: linear-gradient(135deg, hsl(175, 45%, 24%), hsl(175, 40%, 20%)); padding: 24px 32px; }
  .header h1 { margin: 0; color: white; font-family: 'Cormorant Garamond', Georgia, serif; font-size: 22px; font-weight: 400; letter-spacing: 0.05em; }
  .body { padding: 32px; }
  .body h2 { font-family: 'Cormorant Garamond', Georgia, serif; font-weight: 400; font-size: 24px; color: hsl(220, 15%, 15%); margin: 0 0 16px; }
  .body p { font-size: 15px; line-height: 1.6; color: hsl(220, 10%, 35%); margin: 0 0 12px; }
  .cta { display: inline-block; background: hsl(175, 45%, 32%); color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-size: 14px; font-weight: 500; margin-top: 8px; }
  .cta:hover { background: hsl(175, 45%, 28%); }
  .card { background: hsl(35, 20%, 94%); border-radius: 8px; padding: 16px 20px; margin: 16px 0; border-left: 3px solid hsl(175, 45%, 32%); }
  .card.warning { border-left-color: hsl(10, 50%, 45%); }
  .footer { padding: 20px 32px; font-size: 12px; color: hsl(220, 10%, 55%); border-top: 1px solid hsl(35, 15%, 85%); }
  .footer p { margin: 4px 0; }
  a { color: hsl(175, 45%, 32%); }
"""


def _wrap(title: str, body_html: str, cta_label: str | None = None, cta_url: str | None = None) -> str:
    cta_block = f'<p><a href="{cta_url}" class="cta">{cta_label}</a></p>' if cta_label and cta_url else ""
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{_BASE_STYLES}</style>
</head>
<body>
  <div class="wrap">
    <div class="header"><h1>DepScreen</h1></div>
    <div class="body">
      {body_html}
      {cta_block}
    </div>
    <div class="footer">
      <p>DepScreen — depression screening platform localized for Bahrain.</p>
      <p>This is a research prototype. Not for clinical use without professional supervision.</p>
      <p>If you're in crisis, call <strong>999</strong> (Bahrain emergency) or Shamsaha <strong>17651421</strong> (24/7).</p>
    </div>
  </div>
</body>
</html>"""


# ── Email service ────────────────────────────────────────────────────────────


class EmailService:
    """Wraps Resend API. Fails gracefully if not configured."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.resend_api_key) and settings.email_enabled
        self.from_address = settings.email_from
        if self.enabled:
            resend.api_key = settings.resend_api_key
            logger.info("Email service initialized (Resend)")
        else:
            logger.info("Email service disabled (no RESEND_API_KEY configured)")

    def send(self, to: str, subject: str, html: str) -> bool:
        """Low-level send. Returns True on success, False on failure (never raises)."""
        if not self.enabled:
            logger.debug(f"Email skipped (disabled): to={to}, subject={subject}")
            return False
        if not to or "@" not in to:
            logger.warning(f"Invalid recipient address: {to}")
            return False
        try:
            resend.Emails.send(
                {
                    "from": self.from_address,
                    "to": [to],
                    "subject": subject,
                    "html": html,
                }
            )
            logger.info(f"Email sent: to={to}, subject={subject}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {type(e).__name__}: {e}")
            return False

    # ── High-level templates ──

    def send_welcome(self, patient_name: str, patient_email: str) -> bool:
        subject = "Welcome to DepScreen"
        body = f"""
            <h2>Welcome, {patient_name}</h2>
            <p>Thank you for creating your DepScreen account. You've taken a meaningful step towards understanding your mental wellbeing.</p>
            <div class="card">
              <p style="margin:0"><strong>What's next</strong></p>
              <p style="margin:8px 0 0">Complete your profile and take your first screening whenever you feel ready. There's no rush.</p>
            </div>
            <p>Everything you share is confidential and stored securely. You control your data.</p>
        """
        return self.send(
            patient_email, subject, _wrap(subject, body, "Go to DepScreen", "https://depscreen.vercel.app")
        )

    def send_crisis_alert_to_clinician(
        self,
        clinician_name: str,
        clinician_email: str,
        patient_name: str,
        severity: str,
        symptom_count: int,
        screening_id: str,
    ) -> bool:
        subject = f"⚠ Urgent: {patient_name} flagged for review"
        body = f"""
            <h2>Clinical alert</h2>
            <p>Dr. {clinician_name}, a patient under your care has completed a screening that requires your attention.</p>
            <div class="card warning">
              <p style="margin:0"><strong>{patient_name}</strong></p>
              <p style="margin:8px 0 0">Severity: <strong>{severity}</strong> — {symptom_count} DSM-5 symptoms detected</p>
            </div>
            <p>Please review the screening and consider reaching out to the patient. DepScreen is a screening aid, not a diagnostic tool — your clinical judgment is essential.</p>
        """
        url = f"https://depscreen.vercel.app/screening/{screening_id}"
        return self.send(clinician_email, subject, _wrap(subject, body, "Review screening", url))

    def send_screening_reminder(self, patient_name: str, patient_email: str, days_overdue: int = 0) -> bool:
        if days_overdue > 0:
            subject = "Your check-in is overdue"
            headline = "We noticed you missed your check-in"
            detail = f"It's been {days_overdue} day{'s' if days_overdue > 1 else ''} since your scheduled check-in. No pressure — life happens. Whenever you're ready, we're here."
        else:
            subject = "Your check-in is due"
            headline = "A gentle reminder"
            detail = "Your scheduled check-in is due today. Taking a few minutes to reflect can make a real difference."
        body = f"""
            <h2>{headline}</h2>
            <p>Hello {patient_name},</p>
            <p>{detail}</p>
            <div class="card">
              <p style="margin:0">A check-in takes about 2-3 minutes and helps track how you've been feeling over time.</p>
            </div>
        """
        return self.send(
            patient_email, subject, _wrap(subject, body, "Start check-in", "https://depscreen.vercel.app/screening")
        )

    def send_appointment_reminder(
        self,
        patient_name: str,
        patient_email: str,
        appointment_at: str,
        clinician_name: str | None = None,
    ) -> bool:
        subject = "Appointment reminder"
        clinician_line = f" with Dr. {clinician_name}" if clinician_name else ""
        body = f"""
            <h2>Your appointment is tomorrow</h2>
            <p>Hello {patient_name},</p>
            <p>This is a reminder that you have an appointment{clinician_line} on <strong>{appointment_at}</strong>.</p>
            <div class="card">
              <p style="margin:0">If you need to reschedule, please let your clinician know in advance.</p>
            </div>
        """
        return self.send(
            patient_email,
            subject,
            _wrap(subject, body, "View appointments", "https://depscreen.vercel.app/appointments"),
        )

    def send_care_plan_update(self, patient_name: str, patient_email: str, plan_title: str) -> bool:
        subject = "Your care plan has been updated"
        body = f"""
            <h2>Care plan update</h2>
            <p>Hello {patient_name},</p>
            <p>Your clinician has updated your care plan: <strong>{plan_title}</strong></p>
            <p>Take a moment to review the updated goals and interventions when you have a chance.</p>
        """
        return self.send(
            patient_email, subject, _wrap(subject, body, "View care plan", "https://depscreen.vercel.app/care-plan")
        )


# Singleton (lazy initialized)
_email_service: EmailService | None = None


def get_email_service(settings: Settings) -> EmailService:
    global _email_service
    if _email_service is None:
        _email_service = EmailService(settings)
    return _email_service
