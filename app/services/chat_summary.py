"""Chat conversation summary pipeline.

Hybrid: extractive sentence selection (keyword-based) for ingestion into
patient RAG. Extracts clinically relevant sentences from chat history.
"""

import logging
import re

logger = logging.getLogger(__name__)

CLINICAL_KEYWORDS = [
    "sad", "empty", "hopeless", "sleep", "insomnia", "tired", "fatigue",
    "appetite", "weight", "concentration", "worthless", "guilty", "suicid",
    "anxious", "anxiety", "panic", "worry", "numb", "crying",
    "sertraline", "fluoxetine", "escitalopram", "paroxetine", "citalopram",
    "venlafaxine", "duloxetine", "bupropion", "mirtazapine", "medication",
    "dosage", "dose", "side effect", "nauseous", "nausea",
    "exercise", "walk", "therapy", "therapist", "meditation", "breathing",
    "journal", "routine", "schedule",
    "better", "worse", "improving", "struggling", "difficult", "progress",
    "relapse", "setback", "breakthrough",
]


def extract_clinical_sentences(messages: list[dict], min_length: int = 20) -> list[str]:
    """Extract clinically relevant sentences from chat messages.

    Uses keyword matching. Returns formatted strings: '[role, date] sentence'
    """
    extracted = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        date = msg.get("created_at", "")
        if len(content.strip()) < min_length:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < min_length:
                continue
            if any(kw in sentence.lower() for kw in CLINICAL_KEYWORDS):
                prefix = f"[{role}, {str(date)[:10]}]" if date else f"[{role}]"
                extracted.append(f"{prefix} {sentence}")
    return extracted


def should_trigger_summary(
    message_count: int,
    substantive_count: int,
    min_messages: int = 10,
    min_substantive: int = 3,
) -> bool:
    """Check if chat summary should be generated."""
    return message_count >= min_messages and substantive_count >= min_substantive


async def generate_and_ingest_summary(
    patient_id: str,
    conversation_id: str,
    messages: list[dict],
    rag_service,
) -> bool:
    """Extract clinical sentences and ingest into patient RAG.

    Returns True if summary ingested, False otherwise.
    """
    extracted = extract_clinical_sentences(messages)
    if not extracted:
        logger.debug(f"No clinical sentences in conversation {conversation_id}")
        return False

    summary_content = "Chat conversation summary:\n" + "\n".join(extracted)

    import uuid
    if rag_service and rag_service.is_initialized:
        rag_service.ingest_patient_document(
            patient_id=patient_id,
            doc_id=f"chat-summary-{conversation_id}-{uuid.uuid4().hex[:8]}",
            doc_type="chat_summary",
            title=f"Chat summary ({len(extracted)} clinical excerpts)",
            content=summary_content,
        )
        logger.info(
            f"Ingested chat summary for patient {patient_id[:8]}, "
            f"conversation {conversation_id}: {len(extracted)} excerpts"
        )
        return True
    return False
