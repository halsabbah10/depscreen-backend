"""
Public webhook receivers.

Endpoints here are NOT behind JWT auth — they're called by third-party
services (Resend) that don't have a user token. Integrity comes from
signature verification using a shared secret instead.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from svix.webhooks import Webhook, WebhookVerificationError

from app.core.config import Settings, get_settings
from app.models.db import EmailDelivery, get_db

router = APIRouter()
logger = logging.getLogger(__name__)


# Resend event types → our tracked statuses. We track the *furthest* state
# the email reached; e.g. a second 'opened' after 'clicked' shouldn't walk
# the status backward.
_EVENT_TO_STATUS: dict[str, str] = {
    "email.sent": "sent",
    "email.delivered": "delivered",
    "email.delivery_delayed": "delivery_delayed",
    "email.bounced": "bounced",
    "email.complained": "complained",
    "email.opened": "opened",
    "email.clicked": "clicked",
    "email.failed": "failed",
}

# Precedence: higher number = more advanced state. Prevents status from walking
# backwards when events arrive out of order (common in webhook systems).
_STATUS_RANK: dict[str, int] = {
    "queued": 0,
    "sent": 1,
    "delivery_delayed": 2,
    "delivered": 3,
    "opened": 4,
    "clicked": 5,
    # Terminal failure states — high rank so they aren't overwritten by later opens
    "bounced": 9,
    "complained": 9,
    "failed": 9,
}


@router.post("/resend")
async def resend_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
):
    """Receive a Resend delivery event and update the matching EmailDelivery row.

    Verified via Svix headers (svix-id, svix-timestamp, svix-signature)
    against the RESEND_WEBHOOK_SECRET configured in Resend's dashboard.

    We always return 2xx when the signature is valid, even if we don't
    recognize the email — otherwise Resend will retry indefinitely.
    """
    if not settings.resend_webhook_secret:
        logger.warning("Resend webhook received but RESEND_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=503, detail="Webhooks are not configured on this server.")

    raw_body = await request.body()

    # Svix requires the three standard headers; they're forwarded untouched
    # through any proxy (Vercel rewrite, HF Spaces router).
    headers = {
        "svix-id": request.headers.get("svix-id", ""),
        "svix-timestamp": request.headers.get("svix-timestamp", ""),
        "svix-signature": request.headers.get("svix-signature", ""),
    }
    if not all(headers.values()):
        raise HTTPException(status_code=400, detail="Missing Svix signature headers.")

    try:
        Webhook(settings.resend_webhook_secret).verify(raw_body, headers)
    except WebhookVerificationError:
        logger.warning("Resend webhook signature verification failed")
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    # Parse after verification (safer — never trust bytes until signed)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(status_code=400, detail="Webhook body is not valid JSON.")

    event_type = payload.get("type", "unknown")
    data = payload.get("data") or {}
    resend_email_id = data.get("email_id") or data.get("id")

    if not resend_email_id:
        logger.info(f"Resend webhook {event_type} has no email_id, ignoring")
        return {"status": "ignored", "reason": "missing_email_id"}

    row = db.query(EmailDelivery).filter(EmailDelivery.resend_email_id == resend_email_id).first()
    if not row:
        # Not a crime — could be an email sent outside this app against the same Resend account.
        logger.info(f"Resend webhook for unknown email {resend_email_id} ({event_type})")
        return {"status": "ignored", "reason": "unknown_email"}

    # Append to event trail (JSON column)
    events = list(row.events or [])
    events.append(
        {
            "type": event_type,
            "at": datetime.now(UTC).isoformat(),
            "raw": {
                # Keep only the fields we actually want to retain — avoid storing PII
                # we don't need. The `to` and subject are already on the row.
                "bounce_type": (data.get("bounce") or {}).get("type"),
                "bounce_subtype": (data.get("bounce") or {}).get("subType"),
                "click_link": (data.get("click") or {}).get("link"),
                "open_user_agent": (data.get("open") or {}).get("userAgent"),
            },
        }
    )
    row.events = events
    row.last_event_at = datetime.now(UTC)

    # Monotonic status update
    new_status = _EVENT_TO_STATUS.get(event_type)
    if new_status and _STATUS_RANK.get(new_status, -1) >= _STATUS_RANK.get(row.status or "queued", -1):
        row.status = new_status

    row.updated_at = datetime.now(UTC)
    db.commit()

    logger.info(f"Resend webhook processed: email_id={resend_email_id} event={event_type} -> status={row.status}")
    return {"status": "ok"}
