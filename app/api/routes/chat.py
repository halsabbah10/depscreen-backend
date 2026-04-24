"""
Chat API routes.

Supports two modes:
1. Screening-linked chat — discuss results of a specific screening
2. Standalone chat — general mental health psychoeducation, always accessible

The chatbot is RAG-grounded and crisis-aware. It NEVER diagnoses.
Tone: warm, empathetic, patient, non-judgmental — the user may be in crisis.
"""

import asyncio
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.middleware.llm_resilience import llm_retry
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
from app.services.container import get_rag_service
from app.services.llm import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)

_chat_service = None


async def get_chat_service(settings: Settings = Depends(get_settings)):
    global _chat_service
    if _chat_service is None:
        # Chat uses Flash tier — latency matters for conversational UX
        llm_service = LLMService(settings, model=settings.llm_model_flash)
        _chat_service = ChatService(llm_service, get_rag_service())
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

    # Single grouped query for all message counts (avoids N+1)
    conv_ids = [c.id for c in convos]
    count_map: dict[str, int] = {}
    if conv_ids:
        counts = (
            db.query(ChatMessage.conversation_id, func.count(ChatMessage.id))
            .filter(ChatMessage.conversation_id.in_(conv_ids))
            .group_by(ChatMessage.conversation_id)
            .all()
        )
        count_map = dict(counts)

    results = []
    for c in convos:
        results.append(
            ConversationResponse(
                id=c.id,
                title=c.title,
                context_type=c.context_type,
                linked_screening_id=c.linked_screening_id,
                is_active=c.is_active,
                created_at=c.created_at,
                updated_at=c.updated_at,
                message_count=count_map.get(c.id, 0),
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


@router.patch("/conversations/{conversation_id}")
@limiter.limit("30/minute")
async def rename_conversation(
    request: Request,
    conversation_id: str,
    body: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename a conversation. Accepts { title: str }."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    new_title = (body.get("title") or "").strip()
    if not new_title or len(new_title) > 120:
        raise HTTPException(status_code=400, detail="Title must be 1-120 characters")

    conv.title = new_title
    db.commit()
    log_audit(db, current_user.id, "conversation_renamed", "conversation", conv.id)
    return {"status": "renamed", "title": new_title}


@router.post("/conversations/{conversation_id}/auto-title")
@limiter.limit("20/minute")
async def auto_title_conversation(
    request: Request,
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
):
    """Generate a concise title from the conversation's first exchange using the LLM."""
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msgs = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
        .limit(4)
        .all()
    )
    if not msgs:
        raise HTTPException(status_code=400, detail="Conversation has no messages yet")

    exchange = "\n".join(f"{'Patient' if m.role == 'user' else 'Assistant'}: {m.content[:200]}" for m in msgs)

    try:
        response = await chat_service.llm.client.chat.completions.create(
            model=chat_service.llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a concise 3-6 word title for this conversation that captures its topic. "
                        "Return ONLY the title, no quotes, no punctuation at the end, no prefix like 'Title:'. "
                        "Keep it gentle and non-clinical — something the patient would be okay seeing "
                        "in a sidebar list weeks later. Avoid clinical terms (depression, suicidal, "
                        "crisis, disorder, diagnosis). Prefer soft topical phrasing like 'Rough week' "
                        "or 'Thoughts about sleep' or 'A quieter night'. Never name a diagnosis."
                    ),
                },
                {"role": "user", "content": exchange},
            ],
            temperature=0.3,
            max_tokens=30,
        )
        title = (response.choices[0].message.content or "").strip().strip("\"'").rstrip(".")
        if len(title) < 2 or len(title) > 120:
            title = msgs[0].content[:50]
    except Exception as e:
        logger.warning(f"Auto-title LLM call failed: {e}")
        title = msgs[0].content[:50]

    # Safety guard on LLM-generated title
    from app.services.safety_guard import scan_text as _sg_scan_title

    _sg_title = _sg_scan_title(title, context="title")
    title = _sg_title.redacted

    conv.title = title
    db.commit()
    log_audit(db, current_user.id, "conversation_auto_titled", "conversation", conv.id)
    return {"status": "titled", "title": title}


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

        from app.services.chat import CRISIS_KEYWORDS

        # Save user message
        user_msg = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=body.message,
        )
        db.add(user_msg)

        # Crisis check — warm LLM response (not a canned dump).
        # Patient history retrieval for crisis personalization happens inside
        # ChatService.generate_warm_crisis_response(), not here — the standalone
        # path delegates fully so the crisis response starts immediately.
        message_lower = body.message.lower()
        if any(kw in message_lower for kw in CRISIS_KEYWORDS):
            logger.warning(f"Crisis keywords in standalone chat for user {current_user.id[:8]}")
            response_text = await chat_service.generate_warm_crisis_response(body.message)

            # Notify clinician (same pattern as streaming path)
            try:
                if current_user.clinician_id:
                    from app.core.config import get_settings as _get_settings
                    from app.models.db import Notification
                    from app.models.db import User as UserModel
                    from app.services.email import get_email_service

                    clinician = db.query(UserModel).filter(UserModel.id == current_user.clinician_id).first()
                    if clinician and clinician.email:
                        get_email_service(_get_settings()).send_crisis_alert_to_clinician(
                            clinician_name=clinician.full_name,
                            clinician_email=clinician.email,
                            patient_name=current_user.full_name,
                            severity="crisis_chat",
                            symptom_count=0,
                            screening_id=conversation_id,
                        )
                    db.add(
                        Notification(
                            id=str(uuid4()),
                            user_id=current_user.clinician_id,
                            notification_type="crisis_alert",
                            title="Crisis keywords detected in chat",
                            message=f"{current_user.full_name} used crisis-related language in a chat session.",
                            is_read=False,
                        )
                    )
                    db.commit()
            except Exception as e:
                logger.warning(f"Chat crisis notification failed (non-fatal): {e}")
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
            _rag = get_rag_service()
            if _rag and _rag.is_initialized:
                rag_context = _rag.get_personalized_chat_context(
                    patient_id=current_user.id,
                    user_message=body.message,
                    detected_symptoms=detected_symptoms,
                )

            # Full patient profile — demographics, meds, allergies, diagnoses, etc.
            patient_profile = ""
            try:
                patient_profile = chat_service.patient_context.build_context(
                    current_user,
                    db,
                    sections=["demographics", "medications", "allergies", "diagnoses", "care_plan", "screenings"],
                    include_pii=False,
                )
            except Exception as e:
                logger.warning(f"Patient context build failed: {e}")

            from app.services.chat import CHAT_SYSTEM_PROMPT
            from app.services.rag_safety import GROUNDING_INSTRUCTIONS

            prompt_parts: list[str] = []
            if patient_profile:
                prompt_parts.append(patient_profile)
            if recent_screening:
                severity = recent_screening.severity_level or "unknown"
                prompt_parts.append(
                    f"## Most Recent Screening\nSeverity: {severity}, Symptoms detected: {recent_screening.symptom_count or 0}"
                )
            if rag_context:
                prompt_parts.append(f"## Relevant Clinical Knowledge\n{rag_context}")
            if history:
                hist_text = "\n".join(
                    f"{'Patient' if m.role == 'user' else 'Assistant'}: {m.content[:300]}" for m in history[-10:]
                )
                prompt_parts.append(f"## Recent Conversation\n{hist_text}")
            prompt_parts.append(f"## Patient's Current Message\n{body.message}")

            prompt = "\n\n".join(prompt_parts)
            prompt += (
                "\n\nRespond helpfully, empathetically, and concisely. Personalize using the patient profile above. "
                "Reference medications, diagnoses, or care plan goals when clinically relevant. "
                "Never disclose CPR/MRN numbers back to the patient. Do not diagnose."
            )

            try:
                llm_svc = chat_service.llm

                @llm_retry
                async def _call_standalone():
                    return await llm_svc.client.chat.completions.create(
                        model=llm_svc.model,
                        messages=[
                            {"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n\n" + GROUNDING_INSTRUCTIONS},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=600,
                        timeout=60,
                        reasoning_effort="none",
                    )

                response = await _call_standalone()
                response_text = response.choices[0].message.content or ""
                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
                # Safety guard
                from app.services.safety_guard import scan_text as _sg_scan

                _sg = _sg_scan(response_text, context="chat")
                if _sg.violations:
                    logger.warning(
                        f"Standalone chat violations for user {current_user.id[:8]}: "
                        f"{[(v.category, v.severity) for v in _sg.violations]}"
                    )
                response_text = _sg.redacted
            except Exception as e:
                logger.error(f"Standalone chat LLM failed: {e}")
                response_text = (
                    "I'm having trouble putting something together for you right "
                    "now — that's on my side, not yours. Whatever you wrote still "
                    "matters. Try sending it again in a moment, or take a breath "
                    "and come back when you feel like it.\n\n"
                    "If things are heavy right now and you'd rather talk to a "
                    "person, Shamsaha (17651421) answers 24/7 — it's free and "
                    "confidential. For an emergency, 999."
                )

        assistant_msg = ChatMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
        )
        db.add(assistant_msg)
        db.commit()

        # Check if we should generate a chat summary
        try:
            from sqlalchemy import func as sa_func  # noqa: I001

            from app.services.chat_summary import should_trigger_summary, generate_and_ingest_summary

            if conversation_id:  # Only for conversations, not one-off chats
                total_messages = db.query(ChatMessage).filter_by(conversation_id=conversation_id).count()
                substantive_messages = (
                    db.query(ChatMessage)
                    .filter(
                        ChatMessage.conversation_id == conversation_id,
                        sa_func.length(ChatMessage.content) > 20,
                    )
                    .count()
                )

                if should_trigger_summary(total_messages, substantive_messages):
                    recent = (
                        db.query(ChatMessage)
                        .filter_by(conversation_id=conversation_id)
                        .order_by(ChatMessage.created_at.desc())
                        .limit(20)
                        .all()
                    )

                    messages_for_summary = [
                        {"role": m.role, "content": m.content, "created_at": str(m.created_at)}
                        for m in reversed(recent)
                    ]

                    _rag = get_rag_service()
                    await generate_and_ingest_summary(
                        patient_id=current_user.id,
                        conversation_id=conversation_id,
                        messages=messages_for_summary,
                        rag_service=_rag,
                    )
        except Exception as e:
            logger.debug(f"Chat summary check skipped: {e}")

    # Update conversation timestamp
    latest_msg = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(desc(ChatMessage.created_at))
        .first()
    )
    if latest_msg:
        conv.updated_at = latest_msg.created_at
    db.commit()

    return ChatMessageResponse(
        id=assistant_msg.id,
        role=assistant_msg.role,
        content=assistant_msg.content,
        created_at=assistant_msg.created_at,
    )


@router.post("/conversations/{conversation_id}/message/stream")
@limiter.limit("30/minute")
async def send_conversation_message_stream(
    request: Request,
    conversation_id: str,
    body: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
):
    """Streaming version of standalone conversation message — returns SSE."""
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

    # If linked to a screening, use the existing streaming respond method
    if conv.linked_screening_id:
        screening = db.query(Screening).filter(Screening.id == conv.linked_screening_id).first()
        if screening:
            history = (
                db.query(ChatMessage)
                .filter(ChatMessage.conversation_id == conversation_id)
                .order_by(ChatMessage.created_at)
                .all()
            )

            async def event_generator_linked():
                try:
                    async for chunk in chat_service.respond_stream(
                        screening=screening,
                        message=body.message,
                        history=history,
                        db=db,
                    ):
                        escaped = chunk.replace("\n", "\\n")
                        yield f"data: {escaped}\n\n"
                    yield "data: [DONE]\n\n"
                except (asyncio.CancelledError, GeneratorExit):
                    logger.info(f"Linked stream disconnected for conversation {conversation_id}")
                finally:
                    # Ensure conversation links are fixed even on disconnect
                    try:
                        recent_msgs = (
                            db.query(ChatMessage)
                            .filter(
                                ChatMessage.screening_id == screening.id,
                                ChatMessage.conversation_id == None,
                            )
                            .order_by(desc(ChatMessage.created_at))
                            .limit(2)
                            .all()
                        )
                        for m in recent_msgs:
                            m.conversation_id = conversation_id
                        db.commit()
                    except Exception as e:
                        logger.error(f"Failed to fix conversation links on disconnect: {e}")

            return StreamingResponse(
                event_generator_linked(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

    # Standalone conversation (no screening linked)
    import re as re_module

    from app.services.chat import CHAT_SYSTEM_PROMPT, CRISIS_KEYWORDS
    from app.services.rag_safety import GROUNDING_INSTRUCTIONS

    # Save user message
    user_msg = ChatMessage(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role="user",
        content=body.message,
    )
    db.add(user_msg)
    db.commit()

    # Build history
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    # Crisis check
    message_lower = body.message.lower()
    is_crisis = any(kw in message_lower for kw in CRISIS_KEYWORDS)

    # Build prompt
    detected_symptoms = []
    recent_screening = (
        db.query(Screening).filter(Screening.patient_id == current_user.id).order_by(desc(Screening.created_at)).first()
    )
    if recent_screening and recent_screening.symptom_data:
        detected_symptoms = [d.get("symptom", "") for d in recent_screening.symptom_data.get("symptoms_detected", [])]

    rag_context = ""
    try:
        _rag = get_rag_service()
        if _rag and _rag.is_initialized:
            rag_context = _rag.get_personalized_chat_context(
                patient_id=current_user.id,
                user_message=body.message,
                detected_symptoms=detected_symptoms,
            )
    except Exception as e:
        logger.warning(f"RAG retrieval failed in stream: {e}")

    # Full patient profile — demographics, meds, allergies, diagnoses, care plan, etc.
    patient_profile = ""
    try:
        patient_profile = chat_service.patient_context.build_context(
            current_user,
            db,
            sections=["demographics", "medications", "allergies", "diagnoses", "care_plan", "screenings"],
            include_pii=False,
        )
    except Exception as e:
        logger.warning(f"Patient context build failed: {e}")

    # Build prompt in order: patient profile → screening → RAG → conversation → current message
    prompt_parts: list[str] = []
    if patient_profile:
        prompt_parts.append(patient_profile)
    if recent_screening:
        severity = recent_screening.severity_level or "unknown"
        prompt_parts.append(
            f"## Most Recent Screening\nSeverity: {severity}, Symptoms detected: {recent_screening.symptom_count or 0}"
        )
    if rag_context:
        prompt_parts.append(f"## Relevant Clinical Knowledge\n{rag_context}")
    if history:
        hist_text = "\n".join(
            f"{'Patient' if m.role == 'user' else 'Assistant'}: {m.content[:300]}" for m in history[-10:]
        )
        prompt_parts.append(f"## Recent Conversation\n{hist_text}")
    prompt_parts.append(f"## Patient's Current Message\n{body.message}")

    prompt = "\n\n".join(prompt_parts)
    prompt += (
        "\n\nRespond helpfully, empathetically, and concisely. Personalize using the patient profile above. "
        "Reference medications, diagnoses, or care plan goals when clinically relevant. "
        "Never disclose CPR/MRN numbers back to the patient. Do not diagnose."
    )

    async def event_generator_standalone():
        full_response = ""

        try:
            if is_crisis:
                logger.warning(f"Crisis keywords in standalone stream for user {current_user.id[:8]}")
                warm_response = await chat_service.generate_warm_crisis_response(body.message)
                # Stream in small chunks so the UI shows a gentle progressive
                # appearance (matches the calm tone — no sudden wall of text).
                for i in range(0, len(warm_response), 40):
                    chunk_text = warm_response[i : i + 40]
                    escaped = chunk_text.replace("\n", "\\n")
                    yield f"data: {escaped}\n\n"
                full_response = warm_response

                # Notify clinician if patient has one linked
                try:
                    if current_user.clinician_id:
                        from app.core.config import get_settings as _get_settings
                        from app.models.db import Notification
                        from app.models.db import User as UserModel
                        from app.services.email import get_email_service

                        clinician = db.query(UserModel).filter(UserModel.id == current_user.clinician_id).first()
                        if clinician and clinician.email:
                            get_email_service(_get_settings()).send_crisis_alert_to_clinician(
                                clinician_name=clinician.full_name,
                                clinician_email=clinician.email,
                                patient_name=current_user.full_name,
                                severity="crisis_chat",
                                symptom_count=0,
                                screening_id=conversation_id,
                            )
                        db.add(
                            Notification(
                                id=str(uuid4()),
                                user_id=current_user.clinician_id,
                                notification_type="crisis_alert",
                                title="Crisis keywords detected in chat",
                                message=f"{current_user.full_name} used crisis-related language in a chat session.",
                                is_read=False,
                            )
                        )
                        db.commit()
                except Exception as e:
                    logger.warning(f"Chat crisis notification failed (non-fatal): {e}")
            else:
                try:
                    @llm_retry
                    async def _create_standalone_stream():
                        return await chat_service.llm.client.chat.completions.create(
                            model=chat_service.llm.model,
                            messages=[
                                {"role": "system", "content": CHAT_SYSTEM_PROMPT + "\n\n" + GROUNDING_INSTRUCTIONS},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.4,
                            max_tokens=600,
                            stream=True,
                            timeout=60,
                            reasoning_effort="none",
                        )

                    stream = await _create_standalone_stream()
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            delta = chunk.choices[0].delta.content
                            full_response += delta
                            escaped = delta.replace("\n", "\\n")
                            yield f"data: {escaped}\n\n"

                    full_response = re_module.sub(
                        r"<think>.*?</think>", "", full_response, flags=re_module.DOTALL
                    ).strip()
                    # Safety guard on stream output
                    try:
                        from app.services.safety_guard import scan_text as _sg_scan

                        _sg = _sg_scan(full_response, context="chat")
                        if _sg.violations:
                            logger.warning(
                                f"Streaming standalone chat violations for user {current_user.id[:8]}: "
                                f"{[(v.category, v.severity) for v in _sg.violations]}"
                            )
                        full_response = _sg.redacted
                    except Exception as _sg_err:
                        logger.warning(f"Safety guard error (non-fatal): {_sg_err}")
                except Exception as e:
                    logger.error(f"Standalone chat streaming failed: {type(e).__name__}: {e}", exc_info=True)
                    fallback = (
                        "I'm having trouble putting something together for you right "
                        "now — that's on my side, not yours. Whatever you wrote still "
                        "matters. Try sending it again in a moment, or take a breath "
                        "and come back when you feel like it.\n\n"
                        "If things are heavy right now and you'd rather talk to a "
                        "person, Shamsaha (17651421) answers 24/7 — it's free and "
                        "confidential. For an emergency, 999."
                    )
                    yield f"data: {fallback}\n\n"
                    full_response = fallback

            yield "data: [DONE]\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            logger.info(f"Stream disconnected for conversation {conversation_id}")
        finally:
            # Save assistant message even on disconnect (partial response)
            if full_response.strip():
                try:
                    assistant_msg = ChatMessage(
                        id=str(uuid4()),
                        conversation_id=conversation_id,
                        role="assistant",
                        content=full_response,
                    )
                    db.add(assistant_msg)
                    latest_msg = (
                        db.query(ChatMessage)
                        .filter(ChatMessage.conversation_id == conversation_id)
                        .order_by(desc(ChatMessage.created_at))
                        .first()
                    )
                    if latest_msg:
                        conv.updated_at = latest_msg.created_at
                    db.commit()
                except Exception as save_err:
                    logger.error(f"Failed to save assistant response: {save_err}")

    return StreamingResponse(
        event_generator_standalone(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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


@router.post("/screening/{screening_id}/stream")
@limiter.limit("30/minute")
async def send_screening_message_stream(
    request: Request,
    screening_id: str,
    body: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db),
):
    """Streaming version — returns SSE stream of response chunks."""
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")

    _verify_screening_access(screening, current_user)

    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can send chat messages")

    history = (
        db.query(ChatMessage).filter(ChatMessage.screening_id == screening_id).order_by(ChatMessage.created_at).all()
    )

    async def event_generator():
        async for chunk in chat_service.respond_stream(
            screening=screening,
            message=body.message,
            history=history,
            db=db,
        ):
            # SSE format: "data: <text>\n\n" — escape newlines in the chunk
            escaped = chunk.replace("\n", "\\n")
            yield f"data: {escaped}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
