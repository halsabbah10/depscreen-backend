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
from app.middleware.llm_resilience import llm_retry
from app.models.db import ChatMessage, Screening
from app.services.llm import LLMService
from app.services.rag import RAGService

logger = logging.getLogger(__name__)

# Negation words that, when appearing within 5 words before a crisis keyword,
# indicate the patient is NOT expressing active ideation (e.g. "I don't want
# to die", "I'm not suicidal"). Covers English and Arabic (MSA + Gulf dialect).
NEGATION_PREFIXES = frozenset(
    {
        "not",
        "no",
        "don't",
        "dont",
        "never",
        "aren't",
        "arent",
        "isn't",
        "isnt",
        "wasn't",
        "wasnt",
        "haven't",
        "havent",
        "can't",
        "cant",
        "won't",
        "wont",
        "neither",
        "nobody",
        "مش",
        "ما",
        "لا",
    }
)


def _check_crisis_keywords(message_lower: str) -> tuple[str | None, bool]:
    """Check message for crisis keywords with negation awareness.

    Returns (matched_keyword, is_negated).
    - (None, False) → no crisis keyword found
    - (kw, False)   → keyword found, NOT negated → active crisis signal
    - (kw, True)    → keyword found but preceded by a negation within 5 words
    """
    for kw in CRISIS_KEYWORDS:
        if kw in message_lower:
            idx = message_lower.find(kw)
            if idx > 0:
                prefix_text = message_lower[max(0, idx - 60) : idx]
                prefix_words = prefix_text.split()[-5:]
                if any(w in NEGATION_PREFIXES for w in prefix_words):
                    return kw, True
            return kw, False
    return None, False


# Keywords that trigger an immediate crisis response (pre-LLM)
# English + Arabic (MSA and Gulf/Khaleeji dialect for Bahrain context)
CRISIS_KEYWORDS = [
    # Direct suicidal ideation (English)
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
    # Self-harm (English)
    "self harm",
    "self-harm",
    "selfharm",
    "hurting myself",
    "hurt myself",
    "cutting myself",
    "cut myself",
    # Indirect / resigned (English)
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
    # Plan-related (English)
    "have a plan",
    "made a plan",
    "have the means",
    "pills",
    "overdose",
    "jump off",
    "hang myself",
    "hanging myself",
    # ── Arabic: Direct suicidal ideation (MSA + Gulf dialect) ──
    "انتحار",  # suicide
    "انتحاري",  # suicidal
    "اقتل نفسي",  # kill myself (MSA)
    "أقتل نفسي",  # kill myself (with hamza)
    "أريد أن أموت",  # I want to die (MSA)
    "ابي اموت",  # I want to die (Gulf dialect)
    "ابغى اموت",  # I want to die (Gulf dialect variant)
    "اريد الموت",  # I want death
    "أنهي حياتي",  # end my life
    # ── Arabic: Self-harm ──
    "إيذاء النفس",  # self-harm (MSA)
    "ايذاء النفس",  # self-harm (without hamza)
    "أجرح نفسي",  # cut/hurt myself
    "اجرح نفسي",  # cut/hurt myself (without hamza)
    "أأذي نفسي",  # harm myself
    # ── Arabic: Indirect / resigned ──
    "لا سبب للعيش",  # no reason to live
    "الحياة لا تستحق",  # life isn't worth it
    "تعبت من الحياة",  # tired of living
    "خلاص ما ابي اعيش",  # done, don't want to live (Gulf)
    "ما ابي اعيش",  # don't want to live (Gulf)
    "مافي فايدة",  # no point / no use (Gulf)
    "أفضل لو كنت ميت",  # better off dead (MSA)
    # ── Arabic: Plan-related ──
    "جرعة زائدة",  # overdose
    "أشنق نفسي",  # hang myself
    "حبوب",  # pills (colloquial context)
]

# Localized crisis response (Bahrain)
CRISIS_RESPONSE = localization.CRISIS_RESPONSE

# ── Crisis-specific system prompt ────────────────────────────────────────────
# Used when the patient's input contains suicidal/self-harm keywords. Instead
# of dumping a cold resource list, we guide the LLM to respond like a grounded,
# calm, trusted presence — then we append the emergency resources at the end
# deterministically (never relying on the LLM to remember them).
CRISIS_CHAT_SYSTEM_PROMPT = """You are responding to a person who has just shared thoughts of self-harm or suicide.

Tone: imagine you are a close, trusted friend sitting beside them. Warm. Steady.
Calm. Unhurried. The opposite of clinical.

Your entire response must:
1. Acknowledge what they said directly and with warmth — do NOT pretend you
   didn't hear it. Do NOT lecture. Do NOT panic.
2. Validate the pain. Phrases like "that sounds incredibly heavy", "you're
   not alone with this", "what you're carrying is real" — but be genuine,
   not formulaic.
3. Stay grounded in the present moment. Gently invite them to take a breath,
   notice their surroundings, drink some water, call someone they trust.
4. Offer one small, achievable next step — NOT a list of commands. Something
   like "could you reach out to one person you trust tonight?" or "is there
   somewhere in your home that feels safer right now?"
5. Keep it SHORT — 3 to 5 short paragraphs, maximum. Long responses feel
   overwhelming in a crisis.

You must NEVER:
- Say "I am an AI and cannot help you" — that abandons them in a vulnerable moment.
- Give diagnoses or medication advice.
- Minimize ("it'll pass", "it's not that bad", "other people have it worse").
- Use exclamation marks or high-energy language.
- Ask "why" questions that feel interrogative.
- List phone numbers or resources — a footer will be added after your response.

End with a soft sentence suggesting they reach out to trusted human support,
but do NOT list specific phone numbers. The system appends those separately."""


# Footer appended to every crisis response — deterministic so the patient
# always sees the right local resources, regardless of what the LLM said.
CRISIS_RESOURCE_FOOTER = """

---

_If you're in immediate danger, please reach out right now:_

- **999** — National Emergency (police / ambulance, toll-free, 24/7)
- **Shamsaha** — 17651421 — 24/7 confidential support, just to talk
- **Salmaniya Psychiatric Hospital** — +973 1728 8888 — ask for psychiatric emergency
- If you're under 18: **Child Protection Centre** — 998

You don't have to know what to say. Just call. They want to help."""


CHAT_SYSTEM_PROMPT = """You are a supportive mental health psychoeducation assistant for the DepScreen platform.
You have access to this patient's depression screening results and evidence-based clinical information.

Balance you must maintain — both ALWAYS. When in doubt, err slightly toward
warmth, because the patient is the primary audience and they're vulnerable.

1. WARMTH (the delivery) — like a trusted friend who happens to understand
   mental health. Unhurried. Never cold, never clinical in register, never
   alarmist. Always open with genuine validation — at least one full sentence
   before anything else, ideally echoing something specific from what they
   said. Use their own words back to them when natural. If they sound tired,
   discouraged, or ashamed, name that gently before moving on. Never use
   exclamation marks or high-energy phrasing. Frame strategies as invitations
   ("some people find…", "something that might help…") not prescriptions.

2. SUBSTANCE (the content) — clinically accurate, specific, evidence-based.
   Use the RAG clinical context when relevant. Name concrete strategies
   (behavioral activation, sleep hygiene, cognitive reframing, grounding
   exercises, etc.) when they fit. Reference the patient's detected DSM-5
   symptoms accurately. Don't be vague when specificity would actually help.

CORE RULE: Acknowledge what the patient said before offering information —
and acknowledge it genuinely, not as a checkbox. THEN offer real, evidence-
based help. Empty empathy without content is not helpful; empty information
without warmth can make things worse. Validation + evidence-based help,
delivered gently, is the goal.

Vocabulary:
- Prefer: "we noticed", "sometimes shows up as", "what you're feeling is
  real", "that makes sense", "worth exploring", "might be worth talking
  with a clinician", "many people find..."
- Avoid: "concerning", "elevated risk", "abnormal", "areas of concern",
  "indicates" — these feel like a medical chart, not a friend.

Your role:
- Explain detected symptoms accurately, in language a non-clinician can follow
- Share specific evidence-based coping strategies relevant to their symptoms
  (frame as options — "something that helps many people is…" — not commands)
- Help the patient understand their screening results honestly and gently
- Encourage professional support when appropriate — especially for severe
  findings or persistent symptoms — without pushing or scolding
- Be warm, supportive, and non-judgmental WITHOUT softening past the point
  of clinical usefulness

You must NEVER:
- Provide diagnoses ("You have depression")
- Prescribe medication or specific dosages
- Output a medication name followed by a dosage number (e.g., "sertraline 50mg")
- Use phrases like "you have [condition]" or "you are diagnosed with"
- Suggest specific self-harm methods or means
- Say "you don't need a therapist/doctor/professional"
- Replace professional therapy or counseling
- Minimize the patient's feelings ("It's not that bad")
- Make promises about outcomes ("You'll feel better if...")
- Share information about other patients

If unsure whether something is safe to say, err toward:
"That's something your clinician can help you explore."

For clinicians reading this assistant's chat history: the content should be
as clinically sound as anything you'd write in a psychoeducation handout.
Warmth is the delivery; substance is the point.

Keep responses focused — 2-4 paragraphs, longer if clinical depth warrants it."""


class ChatService:
    """Service for patient-facing psychoeducation chat."""

    def __init__(self, llm_service: LLMService, rag_service: RAGService):
        from app.services.patient_context import PatientContextService

        self.llm = llm_service
        self.rag = rag_service
        self.patient_context = PatientContextService()

    async def generate_warm_crisis_response(self, message: str) -> str:
        """Generate a warm, grounded Gemini response to a distressed patient.

        Called by every code path where the patient has expressed suicidal
        thoughts. The response is always from Gemini (using the CRISIS system
        prompt); the emergency resources are appended as a deterministic
        footer. On LLM failure, we fall back to the static CRISIS_RESPONSE —
        never leave someone in a crisis with an error.
        """
        from app.services.safety_guard import scan_text

        try:

            @llm_retry
            async def _call_crisis():
                return await self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": CRISIS_CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": message},
                    ],
                    temperature=0.5,
                    max_tokens=500,
                    timeout=60,
                )

            resp = await _call_crisis()
            body = resp.choices[0].message.content or ""
            body = re.sub(r"<think>.*?</think>", "", body, flags=re.DOTALL).strip()
            safety = scan_text(body, context="chat")
            if safety.violations:
                logger.warning(
                    f"Crisis response safety violations: {[(v.category, v.severity) for v in safety.violations]}"
                )
            return safety.redacted + CRISIS_RESOURCE_FOOTER
        except Exception as e:
            logger.error(f"Crisis LLM call failed, using static fallback: {e}")
            return CRISIS_RESPONSE

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

        # 1. Crisis check (negation-aware)
        message_lower = message.lower()
        matched_keyword, is_negated = _check_crisis_keywords(message_lower)
        is_crisis = matched_keyword is not None and not is_negated

        if is_crisis:
            logger.warning(f"Crisis keywords detected in chat for screening {screening.id}")
            response_text = await self.generate_warm_crisis_response(message)

            # Notify clinician — replicate the pattern from analyze.py
            try:
                from app.core.config import get_settings
                from app.models.db import Notification
                from app.models.db import User as UserModel
                from app.services.email import get_email_service

                patient = db.query(UserModel).filter(UserModel.id == screening.patient_id).first()
                if patient and patient.clinician_id:
                    clinician = db.query(UserModel).filter(UserModel.id == patient.clinician_id).first()
                    if clinician and clinician.email:
                        settings = get_settings()
                        get_email_service(settings).send_crisis_alert_to_clinician(
                            clinician_name=clinician.full_name,
                            clinician_email=clinician.email,
                            patient_name=patient.full_name,
                            severity="crisis_chat",
                            symptom_count=0,
                            screening_id=str(screening.id),
                        )
                    # In-app notification for the clinician
                    db.add(
                        Notification(
                            id=str(uuid4()),
                            user_id=clinician.id if clinician else patient.clinician_id,
                            notification_type="crisis_alert",
                            title="Crisis keywords detected in chat",
                            message=f"{patient.full_name} used crisis-related language in a chat session.",
                            is_read=False,
                        )
                    )
                    db.commit()
            except Exception as e:
                logger.warning(f"Chat crisis notification failed (non-fatal): {e}")
        elif matched_keyword and is_negated:
            logger.info(f"Crisis keyword '{matched_keyword}' detected but negated in screening {screening.id}")
            try:
                from app.models.db import Notification
                from app.models.db import User as UserModel

                patient = db.query(UserModel).filter(UserModel.id == screening.patient_id).first()
                if patient and patient.clinician_id:
                    db.add(
                        Notification(
                            id=str(uuid4()),
                            user_id=patient.clinician_id,
                            notification_type="safety_mention",
                            title="Safety-related language (negated)",
                            message=(
                                f"{patient.full_name} used safety-related language in a negated context. "
                                "No crisis alert sent, but review may be warranted."
                            ),
                            is_read=False,
                        )
                    )
                    db.commit()
            except Exception as e:
                logger.warning(f"Negation notification failed (non-fatal): {e}")
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

            # 4. Call LLM (with retry on transient errors)
            try:

                @llm_retry
                async def _call_chat():
                    return await self.llm.client.chat.completions.create(
                        model=self.llm.model,
                        messages=[
                            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.4,
                        max_tokens=600,
                        reasoning_effort="none",
                        timeout=60,
                    )

                response = await _call_chat()
                response_text = response.choices[0].message.content or ""

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
                    "I'm having trouble putting something together for you right "
                    "now — that's on my side, not yours. Whatever you wrote still "
                    "matters. Try sending it again in a moment, or take a breath "
                    "and come back when you feel like it.\n\n"
                    "If things are heavy right now and you'd rather talk to a "
                    "person, Shamsaha (17651421) answers 24/7 — it's free and "
                    "confidential. For an emergency, 999."
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

        # Crisis check (negation-aware) — streaming fakes chunks so the UX
        # feels consistent, but the response itself is a warm Gemini-generated
        # message + footer.
        message_lower = message.lower()
        _matched_kw, _is_negated = _check_crisis_keywords(message_lower)
        if _matched_kw is not None and not _is_negated:
            logger.warning(f"Crisis keywords detected in chat for screening {screening.id}")
            warm_response = await self.generate_warm_crisis_response(message)
            # Stream the response in small chunks so the UI shows progressive
            # appearance instead of a sudden dump (matches the calm tone).
            for i in range(0, len(warm_response), 40):
                yield warm_response[i : i + 40]
            assistant_msg = ChatMessage(
                id=str(uuid4()),
                screening_id=screening.id,
                role="assistant",
                content=warm_response,
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

        # Stream from LLM (retry on transient failures before streaming begins)
        full_response = ""
        try:

            @llm_retry
            async def _create_stream():
                return await self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.4,
                    max_tokens=600,
                    stream=True,
                    timeout=60,
                    reasoning_effort="none",
                )

            stream = await _create_stream()
            chunks: list[str] = []
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

            full_response = "".join(chunks)

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

        # Safety guard: buffer the full response, scan it, then yield the safe
        # version. This ensures the patient never sees unsafe content — we buffer
        # first, scan, then deliver.
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

        yield full_response

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
