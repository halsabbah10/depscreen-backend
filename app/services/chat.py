"""
Chat service for patient psychoeducation.

Provides a conversational interface grounded in screening results + RAG
clinical context. NEVER diagnoses — only provides psychoeducation,
coping strategies, and crisis resources.
"""

import logging
import re
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core import localization
from app.models.db import ChatMessage, Screening
from app.services.llm import LLMService
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

# Keywords that trigger an immediate crisis response (pre-LLM)
# English — Arabic keywords will be added in Phase 4 (i18n)
CRISIS_KEYWORDS = [
    # Direct suicidal ideation
    "kill myself",
    "killing myself",
    "suicide",
    "suicidal",
    "want to die",
    "wanna die",
    "wish i was dead",
    "wish i were dead",
    "end my life",
    "end it all",
    "end things",
    "ending it",
    "take my life",
    "take my own life",
    # Self-harm
    "self harm",
    "self-harm",
    "selfharm",
    "hurting myself",
    "hurt myself",
    "cutting myself",
    "cut myself",
    # Indirect / resigned
    "no reason to live",
    "nothing to live for",
    "better off dead",
    "can't go on",
    "cant go on",
    "cannot go on",
    "not worth living",
    "life isn't worth",
    "life isnt worth",
    "don't want to be here anymore",
    "dont want to be here anymore",
    "tired of being alive",
    "tired of living",
    # Plan-related
    "have a plan",
    "made a plan",
    "have the means",
    "pills",
    "overdose",
    "jump off",
    "hang myself",
    "hanging myself",
]

# Localized crisis response (Bahrain)
CRISIS_RESPONSE = localization.CRISIS_RESPONSE

CHAT_SYSTEM_PROMPT = """You are a supportive mental health psychoeducation assistant for the DepScreen platform.
You have access to this patient's depression screening results and evidence-based clinical information.

Your role:
- Explain detected symptoms in plain, empathetic language
- Share evidence-based coping strategies from the provided clinical context
- Help the patient understand their screening results
- Encourage professional help when appropriate
- Be warm, supportive, and non-judgmental

You must NEVER:
- Provide diagnoses ("You have depression")
- Prescribe medication or specific treatments
- Replace professional therapy or counseling
- Minimize the patient's feelings ("It's not that bad")
- Make promises about outcomes ("You'll feel better if...")
- Share information about other patients

Always remind users that this is a screening tool, not a diagnostic instrument, and that
professional evaluation is important for accurate assessment.

Keep responses concise (2-4 paragraphs) and focused on the patient's question."""


class ChatService:
    """Service for patient-facing psychoeducation chat."""

    def __init__(self, llm_service: LLMService, rag_service: RAGService):
        from app.services.patient_context import PatientContextService

        self.llm = llm_service
        self.rag = rag_service
        self.patient_context = PatientContextService()

    async def respond(
        self,
        screening: Screening,
        message: str,
        history: list[ChatMessage],
        db: Session,
    ) -> ChatMessage:
        """Generate a response grounded in screening results + RAG context.

        1. Check for crisis keywords → immediate crisis response
        2. Retrieve RAG context relevant to user's question
        3. Build prompt with screening context + RAG + chat history
        4. Call LLM
        5. Save both user message and response to DB
        """
        # Save user message
        user_msg = ChatMessage(
            id=str(uuid4()),
            screening_id=screening.id,
            role="user",
            content=message,
        )
        db.add(user_msg)

        # 1. Crisis check
        message_lower = message.lower()
        if any(kw in message_lower for kw in CRISIS_KEYWORDS):
            logger.warning(f"Crisis keywords detected in chat for screening {screening.id}")
            response_text = CRISIS_RESPONSE
        else:
            # 2. Retrieve RAG context
            detected_symptoms = []
            if screening.symptom_data:
                detected_symptoms = [d.get("symptom", "") for d in screening.symptom_data.get("symptoms_detected", [])]

            # Use personalized RAG (patient history + clinical knowledge)
            patient_id = screening.patient_id or ""
            try:
                rag_context = self.rag.get_personalized_chat_context(
                    patient_id=patient_id,
                    user_message=message,
                    detected_symptoms=detected_symptoms,
                )
            except Exception as e:
                logger.warning(f"RAG context retrieval failed: {e}")
                rag_context = ""

            # Full patient profile (demographics, meds, diagnoses, care plan, etc.)
            patient_profile = ""
            try:
                if screening.patient_id and screening.patient:
                    patient_profile = self.patient_context.build_context(screening.patient, db)
            except Exception as e:
                logger.warning(f"Patient context build failed: {e}")

            # 3. Build prompt
            prompt = self._build_prompt(screening, message, history, rag_context, patient_profile)

            # 4. Call LLM
            try:
                response = await self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=600,
                )
                response_text = response.choices[0].message.content

                # Strip any <think> tags from reasoning models (e.g. DeepSeek R1)
                response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

                # Safety guard: scan LLM output for prescription advice,
                # diagnostic claims, self-harm, undermining professional care.
                from app.services.safety_guard import scan_text

                safety = scan_text(response_text, context="chat")
                if safety.violations:
                    logger.warning(
                        f"Chat safety violations for screening {screening.id}: "
                        f"{[(v.category, v.severity) for v in safety.violations]}"
                    )
                response_text = safety.redacted

            except Exception as e:
                logger.error(f"Chat LLM call failed: {type(e).__name__}: {e}", exc_info=True)
                response_text = (
                    "I'm sorry, I'm having trouble responding right now. "
                    "If you need immediate support, please call 999 for "
                    "emergency services in Bahrain, or contact the "
                    "Psychiatric Hospital at Salmaniya (+973 1728 8888)."
                )

        # 5. Save assistant response
        assistant_msg = ChatMessage(
            id=str(uuid4()),
            screening_id=screening.id,
            role="assistant",
            content=response_text,
        )
        db.add(assistant_msg)
        db.commit()

        return assistant_msg

    async def respond_stream(
        self,
        screening: Screening,
        message: str,
        history: list[ChatMessage],
        db: Session,
    ):
        """Streaming version of respond() — yields text chunks as they arrive from the LLM.

        Saves the user message first, then yields SSE-style chunks as the LLM
        streams its response. Saves the final assistant message to DB when done.
        """
        # Save user message
        user_msg = ChatMessage(
            id=str(uuid4()),
            screening_id=screening.id,
            role="user",
            content=message,
        )
        db.add(user_msg)
        db.commit()

        # Crisis check — don't stream, send the full crisis response
        message_lower = message.lower()
        if any(kw in message_lower for kw in CRISIS_KEYWORDS):
            logger.warning(f"Crisis keywords detected in chat for screening {screening.id}")
            yield CRISIS_RESPONSE
            assistant_msg = ChatMessage(
                id=str(uuid4()),
                screening_id=screening.id,
                role="assistant",
                content=CRISIS_RESPONSE,
            )
            db.add(assistant_msg)
            db.commit()
            return

        # Retrieve RAG context
        detected_symptoms = []
        if screening.symptom_data:
            detected_symptoms = [d.get("symptom", "") for d in screening.symptom_data.get("symptoms_detected", [])]

        patient_id = screening.patient_id or ""
        try:
            rag_context = self.rag.get_personalized_chat_context(
                patient_id=patient_id,
                user_message=message,
                detected_symptoms=detected_symptoms,
            )
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            rag_context = ""

        # Full patient profile
        patient_profile = ""
        try:
            if screening.patient_id and screening.patient:
                patient_profile = self.patient_context.build_context(screening.patient, db)
        except Exception as e:
            logger.warning(f"Patient context build failed: {e}")

        prompt = self._build_prompt(screening, message, history, rag_context, patient_profile)

        # Stream from LLM
        full_response = ""
        try:
            stream = await self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=600,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_response += delta
                    yield delta

            # Clean up <think> tags for reasoning models
            full_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()

        except Exception as e:
            logger.error(f"Chat streaming failed: {type(e).__name__}: {e}", exc_info=True)
            fallback = (
                "I'm sorry, I'm having trouble responding right now. "
                "If you need immediate support, please call 999 for "
                "emergency services in Bahrain, or contact the "
                "Psychiatric Hospital at Salmaniya (+973 1728 8888)."
            )
            yield fallback
            full_response = fallback

        # Safety guard: scan the assembled stream output before persisting.
        # If violations were found, we persist the REDACTED version (what the
        # user already saw in their stream may include unsafe text — we can't
        # un-ring that bell, but we can prevent it from being re-delivered on
        # history reload). A post-scan audit log is written for review.
        try:
            from app.services.safety_guard import scan_text

            safety = scan_text(full_response, context="chat")
            if safety.violations:
                logger.warning(
                    f"Streaming chat violations for screening {screening.id}: "
                    f"{[(v.category, v.severity) for v in safety.violations]}"
                )
            full_response = safety.redacted
        except Exception as e:
            logger.warning(f"Safety guard error (non-fatal): {e}")

        # Save the complete assistant message
        assistant_msg = ChatMessage(
            id=str(uuid4()),
            screening_id=screening.id,
            role="assistant",
            content=full_response,
        )
        db.add(assistant_msg)
        db.commit()

    def _build_prompt(
        self,
        screening: Screening,
        message: str,
        history: list[ChatMessage],
        rag_context: str,
        patient_context: str = "",
    ) -> str:
        """Build the full prompt with patient profile, screening context, RAG docs, and chat history."""
        parts: list[str] = []

        # 1. Full patient profile (demographics, medications, allergies, diagnoses, care plan, etc.)
        if patient_context:
            parts.append(patient_context)

        # 2. Screening-specific context
        severity = screening.severity_level or "unknown"
        symptom_count = screening.symptom_count or 0
        symptoms_list = ""
        if screening.symptom_data:
            for d in screening.symptom_data.get("symptoms_detected", []):
                symptoms_list += (
                    f'\n- {d.get("symptom_label", d.get("symptom", ""))}: "{d.get("sentence_text", "")[:80]}"'
                )
        parts.append(
            f"## Context: Focused Screening\nSeverity: {severity} ({symptom_count} DSM-5 symptoms detected)"
            f"\nDetected symptoms:{symptoms_list if symptoms_list else ' None'}"
        )

        # 3. RAG clinical knowledge
        if rag_context:
            parts.append(f"## Relevant Clinical Information\n{rag_context}")

        # 4. Recent conversation history
        if history:
            hist = "\n".join(
                f"{'Patient' if m.role == 'user' else 'Assistant'}: {m.content[:300]}" for m in history[-10:]
            )
            parts.append(f"## Recent Conversation\n{hist}")

        # 5. Current message
        parts.append(
            f"## Patient's Current Message\n{message}\n\n"
            "Respond helpfully, empathetically, and concisely. Personalize using the patient profile. "
            "Reference medications, diagnoses, or care plan goals when clinically relevant. "
            "Do not diagnose. Recommend professional help when appropriate."
        )

        return "\n\n".join(parts)
