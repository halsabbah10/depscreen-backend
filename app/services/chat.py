"""
Chat service for patient psychoeducation.

Provides a conversational interface grounded in screening results + RAG
clinical context. NEVER diagnoses — only provides psychoeducation,
coping strategies, and crisis resources.
"""

import logging
import re
from typing import Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core import localization
from app.middleware.llm_resilience import llm_retry
from app.models.db import Screening, ChatMessage
from app.services.llm import LLMService, extract_json, SAFETY_DISCLAIMER, DEFAULT_RESOURCES
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

# Keywords that trigger an immediate crisis response (pre-LLM)
# English — Arabic keywords will be added in Phase 4 (i18n)
CRISIS_KEYWORDS = [
    # Direct suicidal ideation
    "kill myself", "killing myself", "suicide", "suicidal",
    "want to die", "wanna die", "wish i was dead", "wish i were dead",
    "end my life", "end it all", "end things", "ending it",
    "take my life", "take my own life",
    # Self-harm
    "self harm", "self-harm", "selfharm", "hurting myself", "hurt myself",
    "cutting myself", "cut myself",
    # Indirect / resigned
    "no reason to live", "nothing to live for", "better off dead",
    "can't go on", "cant go on", "cannot go on",
    "not worth living", "life isn't worth", "life isnt worth",
    "don't want to be here anymore", "dont want to be here anymore",
    "tired of being alive", "tired of living",
    # Plan-related
    "have a plan", "made a plan", "have the means",
    "pills", "overdose", "jump off", "hang myself", "hanging myself",
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
        self.llm = llm_service
        self.rag = rag_service

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
                detected_symptoms = [
                    d.get("symptom", "")
                    for d in screening.symptom_data.get("symptoms_detected", [])
                ]

            # Use personalized RAG (patient history + clinical knowledge)
            patient_id = screening.patient_id or ""
            rag_context = self.rag.get_personalized_chat_context(
                patient_id=patient_id,
                user_message=message,
                detected_symptoms=detected_symptoms,
            )

            # 3. Build prompt
            prompt = self._build_prompt(screening, message, history, rag_context)

            # 4. Call LLM
            try:
                @llm_retry
                async def _chat_call():
                    return await self.llm.client.chat.completions.create(
                        model=self.llm.model,
                        messages=[
                            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=600,
                        timeout=60,
                    )

                response = await _chat_call()
                response_text = response.choices[0].message.content

                # Strip any <think> tags from reasoning models (e.g. DeepSeek R1)
                response_text = re.sub(
                    r"<think>.*?</think>", "", response_text, flags=re.DOTALL
                ).strip()

            except Exception as e:
                logger.error(f"Chat LLM call failed: {e}")
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

    def _build_prompt(
        self,
        screening: Screening,
        message: str,
        history: list[ChatMessage],
        rag_context: str,
    ) -> str:
        """Build the full prompt with screening context, RAG docs, and chat history."""
        # Screening context
        severity = screening.severity_level or "unknown"
        symptom_count = screening.symptom_count or 0
        symptoms_list = ""
        if screening.symptom_data:
            for d in screening.symptom_data.get("symptoms_detected", []):
                symptoms_list += f"\n- {d.get('symptom_label', d.get('symptom', ''))}: \"{d.get('sentence_text', '')[:80]}\""

        prompt = f"""## Patient's Screening Results
Severity: {severity} ({symptom_count} DSM-5 symptoms detected)
Detected symptoms:{symptoms_list if symptoms_list else ' None'}
"""

        if rag_context:
            prompt += f"""
## Relevant Clinical Information
{rag_context}
"""

        # Chat history (last 10 messages for context window management)
        if history:
            prompt += "\n## Recent Conversation\n"
            for msg in history[-10:]:
                role_label = "Patient" if msg.role == "user" else "Assistant"
                prompt += f"{role_label}: {msg.content[:300]}\n"

        prompt += f"""
## Patient's Current Message
{message}

Respond helpfully, empathetically, and concisely. Use the clinical information above to ground your response.
Do not diagnose. Recommend professional help when appropriate."""

        return prompt
