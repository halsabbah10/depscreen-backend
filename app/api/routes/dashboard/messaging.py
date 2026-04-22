"""
Clinician-to-patient notifications and direct messages endpoints.
"""

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, Request
from sqlalchemy.orm import Session

from app.middleware.rate_limiter import limiter
from app.models.db import (
    ChatMessage,
    Conversation,
    Notification,
    User,
    get_db,
)
from app.schemas.analysis import (
    DirectMessageCreate,
    DirectMessageResponse,
    DirectMessageThread,
    NotificationResponse,
)
from app.services.auth import log_audit, require_clinician

from ._shared import _verify_patient_access

router = APIRouter()


def _get_or_create_patient_thread(db: Session, patient: User, clinician: User) -> Conversation:
    """Get or create the clinician-direct conversation for this patient-clinician pair."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == patient.id,
            Conversation.context_type == "clinician_direct",
            Conversation.linked_clinician_id == clinician.id,
            Conversation.is_active == True,
        )
        .first()
    )
    if conv:
        return conv

    conv = Conversation(
        id=str(uuid4()),
        user_id=patient.id,
        title="Messages with your clinician",
        context_type="clinician_direct",
        linked_clinician_id=clinician.id,
        is_active=True,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


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


@router.get("/patients/{patient_id}/messages", response_model=DirectMessageThread)
async def get_patient_messages(
    patient_id: str,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Get the direct-message thread with a specific patient."""
    patient = _verify_patient_access(db, patient_id, current_user.id)
    conv = _get_or_create_patient_thread(db, patient, current_user)

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conv.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )

    sender_lookup = {
        "user": patient.full_name,
        "clinician": current_user.full_name,
    }

    # Mark clinician's new_message notifications linked to this patient as read
    link = f"/patients/{patient_id}"
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.notification_type == "new_message",
        Notification.link == link,
        Notification.is_read == False,
    ).update({"is_read": True})
    db.commit()

    return DirectMessageThread(
        conversation_id=conv.id,
        patient_id=patient.id,
        patient_name=patient.full_name,
        clinician_id=current_user.id,
        clinician_name=current_user.full_name,
        messages=[
            DirectMessageResponse(
                id=m.id,
                role=m.role,
                sender_name=sender_lookup.get(m.role),
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
        unread_count=0,
    )


@router.post("/patients/{patient_id}/messages", response_model=DirectMessageResponse, status_code=201)
@limiter.limit("30/minute")
async def post_patient_message(
    patient_id: str,
    request: Request,
    body: DirectMessageCreate,
    current_user: User = Depends(require_clinician()),
    db: Session = Depends(get_db),
):
    """Clinician posts a message to a patient."""
    patient = _verify_patient_access(db, patient_id, current_user.id)
    conv = _get_or_create_patient_thread(db, patient, current_user)

    msg = ChatMessage(
        id=str(uuid4()),
        conversation_id=conv.id,
        role="clinician",
        content=body.content.strip(),
    )
    db.add(msg)
    conv.updated_at = datetime.now(UTC)

    # Notify patient
    db.add(
        Notification(
            user_id=patient.id,
            notification_type="new_message",
            title=f"Message from {current_user.full_name}",
            message=body.content.strip()[:140],
            link="/messages",
        )
    )

    db.commit()
    db.refresh(msg)

    log_audit(db, current_user.id, "clinician_message_sent", resource_type="message", resource_id=msg.id)

    return DirectMessageResponse(
        id=msg.id,
        role=msg.role,
        sender_name=current_user.full_name,
        content=msg.content,
        created_at=msg.created_at,
    )
