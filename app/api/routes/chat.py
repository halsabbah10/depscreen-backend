"""
Chat API routes.

Supports two modes:
1. Screening-linked chat — discuss results of a specific screening
2. Standalone chat — general mental health psychoeducation, always accessible

The chatbot is RAG-grounded and crisis-aware. It NEVER diagnoses.
Tone: warm, empathetic, patient, non-judgmental — the user may be in crisis.
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.rate_limiter import limiter
from app.models.db import ChatMessage, Conversation, Screening, User, get_db
from app.schemas.analysis import (
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ConversationCreate,
    ConversationResponse,
)
from app.services.auth import get_current_user, log_audit
from app.services.chat import ChatService
from app.services.llm import LLMService
from app.services.rag import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)

_chat_service = None
_rag_service = None


async def get_chat_service(settings: Settings = Depends(get_settings)):
    global _chat_service, _rag_service
    if _chat_service is None:
        _rag_service = RAGService(settings)
        await _rag_service.initialize()
        llm_service = LLMService(settings)
        _chat_service = ChatService(llm_service, _rag_service)
    return _chat_service


def _verify_screening_access(screening: Screening, user: User):
    if user.role == "patient" and screening.patient_id != user.id:
        raise HTTPException(status_code=403, detail="You can only access your own screenings")
    if user.role == "clinician":
        if screening.patient_id and screening.patient:
            if screening.patient.clinician_id != user.id:
                raise HTTPException(status_code=403, detail="This patient is not assigned to you")


# ── Conversations (Standalone Chat) ──────────────────────────────────────────


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all conversations for the current user."""
    convos = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id, Conversation.is_active == True)
        .order_by(desc(Conversation.updated_at))
        .all()
    )

    results = []
    for c in convos:
        msg_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == c.id).count()
        results.append(
            ConversationResponse(
                id=c.id,
                title=c.title,
                context_type=c.context_type,
                linked_screening_id=c.linked_screening_id,
                is_active=c.is_active,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=msg_count,
            )
        )

    return results


@router.post("/conversations", response_model=ConversationResponse)
@limiter.limit("30/minute")
async def create_conversation(
    request: Request,
    body: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new standalone conversation.

    Patients can always create conversations — chat is accessible at all times,
    not just after a screening.
    """
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can create conversations")

    conv = Conversation(
        id=str(uuid4()),
        user_id=current_user.id,
        title=body.title or "New Conversation",
        context_type=body.context_type,
        linked_screening_id=body.linked_screening_id,
        linked_clinician_id=body.linked_clinician_id,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)

    log_audit(db, current_user.id, "conversation_created", "conversation", conv.id)

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        context_type=conv.context_type,
        linked_screening_id=conv.linked_screening_id,
        is_active=conv.is_active,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        message_count=0,
    )


@router.post("/conversations/{conversation_id}/message", response_model=ChatMessageResponse)
@limiter.limit("30/minute")
async def send_conversation_message(
    request: Request,
    conversation_id: str,
    body: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
):
    """Send a message in a standalone conversation.

    The chatbot responds with RAG-grounded psychoeducation.
    If the conversation is linked to a screening, that screening's
    context is included automatically.
    """
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can send messages")

    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get linked screening context if available
    screening = None
    if conv.linked_screening_id:
        screening = db.query(Screening).filter(Screening.id == conv.linked_screening_id).first()

    # Get conversation history
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    # Use screening context if available, otherwise create a minimal screening-like object
    if screening:
        assistant_msg = await chat_service.respond(
            screening=screening,
            message=body.message,
            history=history,
            db=db,
        )
        # Fix the conversation_id (chat service sets screening_id)
        assistant_msg.conversation_id = conversation_id
        # Also fix the user message that was saved by chat_service
        user_msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.screening_id == screening.id, ChatMessage.conversation_id == None)
            .order_by(desc(ChatMessage.created_at))
            .limit(2)
            .all()
        )
        for msg in user_msgs:
            msg.conversation_id = conversation_id
        db.commit()
    else:
        # Standalone chat without screening context
        import re

        from app.services.chat import CRISIS_KEYWORDS, CRISIS_RESPONSE

        # Save user message
        user_msg = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=body.message,
        )
        db.add(user_msg)

        # Crisis check
        message_lower = body.message.lower()
        if any(kw in message_lower for kw in CRISIS_KEYWORDS):
            logger.warning(f"Crisis keywords in standalone chat for user {current_user.id[:8]}")
            response_text = CRISIS_RESPONSE
        else:
            # RAG context based on user's detected symptoms from recent screenings
            detected_symptoms = []
            recent_screening = (
                db.query(Screening)
                .filter(Screening.patient_id == current_user.id)
                .order_by(desc(Screening.created_at))
                .first()
            )
            if recent_screening and recent_screening.symptom_data:
                detected_symptoms = [
                    d.get("symptom", "") for d in recent_screening.symptom_data.get("symptoms_detected", [])
                ]

            rag_context = ""
            if _rag_service and _rag_service.is_initialized:
                rag_context = _rag_service.get_personalized_chat_context(
                    patient_id=current_user.id,
                    user_message=body.message,
                    detected_symptoms=detected_symptoms,
                )

            # Build prompt
            from app.services.chat import CHAT_SYSTEM_PROMPT

            prompt_parts = [f"## Patient's Message\n{body.message}"]
            if rag_context:
                prompt_parts.insert(0, f"## Clinical Context\n{rag_context}")
            if recent_screening:
                severity = recent_screening.severity_level or "unknown"
                prompt_parts.insert(
                    0, f"## Recent Screening\nSeverity: {severity}, Symptoms: {recent_screening.symptom_count or 0}"
                )
            if history:
                hist_text = "\n".join(
                    f"{'Patient' if m.role == 'user' else 'Assistant'}: {m.content[:200]}" for m in history[-8:]
                )
                prompt_parts.insert(0, f"## Recent Conversation\n{hist_text}")

            prompt = "\n\n".join(prompt_parts)
            prompt += "\n\nRespond helpfully, empathetically, and concisely. Do not diagnose."

            try:
                llm_svc = chat_service.llm
                response = await llm_svc.client.chat.completions.create(
                    model=llm_svc.model,
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=600,
                )
                response_text = response.choices[0].message.content
                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
            except Exception as e:
                logger.error(f"Standalone chat LLM failed: {e}")
                response_text = (
                    "I'm sorry, I'm having trouble responding right now. "
                    "If you need immediate support, please call 999 for "
                    "emergency services in Bahrain, or contact the "
                    "Psychiatric Hospital at Salmaniya (+973 1728 8888)."
                )

        assistant_msg = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
        )
        db.add(assistant_msg)
        db.commit()

    # Update conversation timestamp
    conv.updated_at = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(desc(ChatMessage.created_at))
        .first()
        .created_at
    )
    db.commit()

    return ChatMessageResponse(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        created_at=assistant_msg.created_at,
    )


@router.get("/conversations/{conversation_id}/messages", response_model=ChatHistoryResponse)
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all messages in a conversation."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    return ChatHistoryResponse(
        screening_id=conv.linked_screening_id or conversation_id,
        messages=[
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@router.delete("/conversations/{conversation_id}")
@limiter.limit("30/minute")
async def archive_conversation(
    conversation_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Archive a conversation (soft delete)."""
    conv = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv.is_active = False
    db.commit()
    return {"status": "archived"}


# ── Screening-Linked Chat (existing functionality, preserved) ────────────────


@router.post("/screening/{screening_id}", response_model=ChatMessageResponse)
@limiter.limit("30/minute")
async def send_screening_message(
    request: Request,
    screening_id: str,
    body: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
):
    """Send a message about a specific screening's results."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    _verify_screening_access(screening, current_user)

    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can send chat messages")

    history = (
        db.query(ChatMessage).filter(ChatMessage.screening_id == screening_id).order_by(ChatMessage.created_at).all()
    )

    assistant_msg = await chat_service.respond(
        screening=screening,
        message=body.message,
        history=history,
        db=db,
    )

    return ChatMessageResponse(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        created_at=assistant_msg.created_at,
    )


@router.get("/screening/{screening_id}", response_model=ChatHistoryResponse)
async def get_screening_chat_history(
    screening_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get chat history for a specific screening."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    _verify_screening_access(screening, current_user)

    messages = (
        db.query(ChatMessage).filter(ChatMessage.screening_id == screening_id).order_by(ChatMessage.created_at).all()
    )

    return ChatHistoryResponse(
        screening_id=screening_id,
        messages=[
            ChatMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )
