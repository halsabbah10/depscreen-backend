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
from datetime import UTC, datetime
from uuid import uuid4

import resend

from app.core.config import Settings

logger = logging.getLogger(__name__)


# ── Email templates ──────────────────────────────────────────────────────────
# Minimal shared wrapper so every email looks consistent.

# Logo: teal rounded square with lighter inner tile and a white Brain icon —
# matches the app header. Base64-inlined so it renders in every email client
# (Gmail, Outlook, Apple Mail) with no remote-image-loading prompts.
_LOGO_DATA_URI = "data:image/png;base64,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"


# Emails must inline styles for max client compatibility (Gmail, Outlook strip <style> in some views).
# Kept as reusable block but every email body also uses key inline styles as a fallback.
_BASE_STYLES = """
  body { margin: 0; padding: 0; background: #f5f1ea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; color: #1f2937; }
  table { border-collapse: collapse; }
  .wrap { max-width: 560px; margin: 40px auto; background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 32px rgba(20, 40, 40, 0.08); }
  .header { background: linear-gradient(135deg, #2c6360 0%, #1e4e4b 100%); padding: 28px 36px; }
  .logo-row { display: flex; align-items: center; gap: 12px; }
  .logo-mark { width: 36px; height: 36px; background: rgba(255,255,255,0.14); border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; color: #ffffff; }
  .brand { color: #ffffff; font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif; font-size: 20px; font-weight: 500; letter-spacing: 0.14em; margin: 0; text-transform: uppercase; }
  .body-section { padding: 36px 36px 8px; }
  .body-section h2 { font-family: 'Cormorant Garamond', Georgia, 'Times New Roman', serif; font-weight: 400; font-size: 26px; color: #1f2937; margin: 0 0 14px; line-height: 1.25; letter-spacing: 0.01em; }
  .body-section p { font-size: 15px; line-height: 1.65; color: #4a5560; margin: 0 0 14px; }
  .body-section p.lead { font-size: 16px; color: #1f2937; }
  .cta-wrap { padding: 14px 36px 32px; }
  .cta { display: inline-block; background: #2c6360; color: #ffffff !important; padding: 13px 28px; border-radius: 10px; text-decoration: none; font-size: 14px; font-weight: 500; letter-spacing: 0.02em; }
  .card { background: #f5f1ea; border-radius: 10px; padding: 18px 22px; margin: 18px 0; border-left: 3px solid #2c6360; }
  .card.warning { border-left-color: #b7472c; background: #faf0ed; }
  .card strong { color: #1f2937; font-weight: 600; }
  .divider { height: 1px; background: #ece6db; margin: 0 36px; }
  .footer { padding: 22px 36px 28px; font-size: 12px; color: #6b7280; line-height: 1.55; }
  .footer p { margin: 6px 0; }
  .footer .safety { color: #4a5560; }
  a { color: #2c6360; }
  @media only screen and (max-width: 480px) {
    .wrap { margin: 0; border-radius: 0; }
    .header, .body-section, .cta-wrap, .footer { padding-left: 24px; padding-right: 24px; }
  }
"""


def _wrap(title: str, body_html: str, cta_label: str | None = None, cta_url: str | None = None) -> str:
    cta_block = (
        f'<div class="cta-wrap"><a href="{cta_url}" class="cta" style="display:inline-block;background:#2c6360;color:#ffffff;padding:13px 28px;border-radius:10px;text-decoration:none;font-size:14px;font-weight:500;">{cta_label} →</a></div>'
        if cta_label and cta_url
        else ""
    )
    return f"""<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="x-apple-disable-message-reformatting" />
  <title>{title}</title>
  <style>{_BASE_STYLES}</style>
</head>
<body style="margin:0;padding:0;background:#f5f1ea;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f5f1ea;padding:32px 16px;">
    <tr>
      <td align="center">
        <div class="wrap">
          <div class="header" style="background:linear-gradient(135deg,#2c6360 0%,#1e4e4b 100%);padding:24px 36px;">
            <table role="presentation" cellpadding="0" cellspacing="0"><tr>
              <td style="padding-right:14px;vertical-align:middle;">
                <img src="https://depscreen.vercel.app/logo.png" alt="DepScreen" width="40" height="40" style="display:block;width:40px;height:40px;border-radius:10px;" />
              </td>
              <td style="vertical-align:middle;">
                <h1 class="brand" style="margin:0;color:#ffffff;font-family:'Cormorant Garamond',Georgia,serif;font-size:20px;font-weight:500;letter-spacing:0.14em;text-transform:uppercase;">DepScreen</h1>
              </td>
            </tr></table>
          </div>
          <div class="body-section">
            {body_html}
          </div>
          {cta_block}
          <div class="divider"></div>
          <div class="footer">
            <p>DepScreen is a research prototype and a screening aid, not a diagnostic tool.</p>
            <p class="safety">If you're in crisis, call <strong>999</strong> for emergencies or Shamsaha <strong>17651421</strong> (24/7 confidential).</p>
            <p>You are receiving this email because you have a DepScreen account. Manage your preferences in the app.</p>
          </div>
        </div>
      </td>
    </tr>
  </table>
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

    def send(
        self,
        to: str,
        subject: str,
        html: str,
        *,
        template_key: str = "generic",
        user_id: str | None = None,
    ) -> bool:
        """Low-level send. Returns True on success, False on failure (never raises).

        Also records an EmailDelivery row so Resend webhooks can correlate
        downstream delivery/open/bounce events back to this send. The DB
        write is wrapped so a database hiccup never blocks an outgoing email.
        """
        if not self.enabled:
            logger.debug(f"Email skipped (disabled): to={to}, subject={subject}")
            return False
        if not to or "@" not in to:
            logger.warning(f"Invalid recipient address: {to}")
            return False

        # Create a tracking row up-front (best-effort)
        delivery_id = str(uuid4())
        self._record_delivery(
            delivery_id=delivery_id,
            user_id=user_id,
            recipient=to,
            subject=subject,
            template_key=template_key,
            status="queued",
        )

        try:
            response = resend.Emails.send(
                {
                    "from": self.from_address,
                    "to": [to],
                    "subject": subject,
                    "html": html,
                }
            )
            resend_id = None
            if isinstance(response, dict):
                resend_id = response.get("id")
            self._update_delivery(
                delivery_id=delivery_id,
                resend_email_id=resend_id,
                status="sent",
            )
            logger.info(f"Email sent: to={to}, subject={subject}, resend_id={resend_id}")
            return True
        except Exception as e:
            logger.error(f"Email send failed: {type(e).__name__}: {e}")
            self._update_delivery(
                delivery_id=delivery_id,
                status="failed",
                error_message=f"{type(e).__name__}: {e}",
            )
            return False

    # ── Delivery tracking helpers ──

    def _record_delivery(
        self,
        *,
        delivery_id: str,
        user_id: str | None,
        recipient: str,
        subject: str,
        template_key: str,
        status: str,
    ) -> None:
        """Insert a row into email_deliveries. Best-effort — never raises."""
        try:
            from app.models.db import EmailDelivery, SessionLocal

            db = SessionLocal()
            try:
                row = EmailDelivery(
                    id=delivery_id,
                    user_id=user_id,
                    recipient=recipient,
                    subject=subject,
                    template_key=template_key,
                    status=status,
                    events=[],
                )
                db.add(row)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Could not record email delivery row ({delivery_id}): {e}")

    def _update_delivery(
        self,
        *,
        delivery_id: str,
        resend_email_id: str | None = None,
        status: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update the delivery row after the Resend call returns. Best-effort."""
        try:
            from app.models.db import EmailDelivery, SessionLocal

            db = SessionLocal()
            try:
                row = db.query(EmailDelivery).filter(EmailDelivery.id == delivery_id).first()
                if not row:
                    return
                if resend_email_id:
                    row.resend_email_id = resend_email_id
                if status:
                    row.status = status
                if error_message:
                    row.error_message = error_message
                row.updated_at = datetime.now(UTC)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Could not update email delivery row ({delivery_id}): {e}")

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
            patient_email,
            subject,
            _wrap(subject, body, "Go to DepScreen", "https://depscreen.vercel.app"),
            template_key="welcome",
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
        # /screening is patient-only, and /screening/{id} is not a route.
        # Both roles can read a screening via the shared /results/{id} page.
        url = f"https://depscreen.vercel.app/results/{screening_id}"
        return self.send(
            clinician_email,
            subject,
            _wrap(subject, body, "Review screening", url),
            template_key="crisis_alert_clinician",
        )

    def send_screening_reminder(self, patient_name: str, patient_email: str, days_overdue: int = 0) -> bool:
        # Intentionally avoid "missed" / "overdue" language — it reads as
        # shaming to a vulnerable audience. Reframe as waiting, not a failure.
        if days_overdue > 0:
            subject = "A gentle check-in, whenever you're ready"
            headline = "Thinking of you"
            detail = (
                f"It's been {days_overdue} day{'s' if days_overdue > 1 else ''} "
                "since your last check-in. No pressure at all — whenever feels right, we're here."
            )
        else:
            subject = "Your check-in is ready"
            headline = "A quiet reminder"
            detail = (
                "A check-in is waiting for you today. Just a few minutes of reflection can be surprisingly helpful."
            )
        body = f"""
            <h2>{headline}</h2>
            <p>Hello {patient_name},</p>
            <p>{detail}</p>
            <div class="card">
              <p style="margin:0">It usually takes 2–3 minutes. You can skip any question that doesn't feel right.</p>
            </div>
        """
        return self.send(
            patient_email,
            subject,
            _wrap(subject, body, "Start check-in", "https://depscreen.vercel.app/screening"),
            template_key="screening_reminder",
        )

    def send_appointment_reminder(
        self,
        patient_name: str,
        patient_email: str,
        appointment_at: str,
        clinician_name: str | None = None,
    ) -> bool:
        subject = "A quiet reminder about tomorrow"
        clinician_line = f" with Dr. {clinician_name}" if clinician_name else ""
        body = f"""
            <h2>A quiet reminder</h2>
            <p>Hello {patient_name},</p>
            <p>Just a gentle note that you have an appointment{clinician_line} on <strong>{appointment_at}</strong>.</p>
            <div class="card">
              <p style="margin:0">If the time no longer works, letting your clinician know ahead of time is always okay.</p>
            </div>
        """
        return self.send(
            patient_email,
            subject,
            _wrap(subject, body, "View appointments", "https://depscreen.vercel.app/appointments"),
            template_key="appointment_reminder",
        )

    def send_care_plan_update(self, patient_name: str, patient_email: str, plan_title: str) -> bool:
        subject = "Your care plan has been updated"
        body = f"""
            <h2>Care plan update</h2>
            <p>Hello {patient_name},</p>
            <p>Your clinician has updated your care plan: <strong>{plan_title}</strong></p>
            <p>No action needed from you right now — just know your clinician is thinking about your care.
            Take a look at the updated goals and interventions whenever you have a moment.</p>
        """
        return self.send(
            patient_email,
            subject,
            _wrap(subject, body, "View care plan", "https://depscreen.vercel.app/care-plan"),
            template_key="care_plan_update",
        )


# Singleton (lazy initialized)
_email_service: EmailService | None = None


def get_email_service(settings: Settings) -> EmailService:
    global _email_service
    if _email_service is None:
        _email_service = EmailService(settings)
    return _email_service
