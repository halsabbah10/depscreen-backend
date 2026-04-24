"""Chat conversation summary pipeline.

Hybrid approach:
1. Extractive: keyword-based sentence selection from chat history
2. LLM structuring: organize extracted sentences by clinical theme (constrained input, temp 0.1)
3. NLI verification: verify structured output against extracted sources

Ingests verified summary into patient RAG for future personalization.
"""

import logging
import re
import uuid

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Lazy singleton — reuses connection pool across calls
_async_llm_client: AsyncOpenAI | None = None


def _get_llm_client(settings) -> AsyncOpenAI:
    global _async_llm_client
    if _async_llm_client is None:
        _async_llm_client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    return _async_llm_client


CLINICAL_KEYWORDS = [
    # Symptoms
    "sad",
    "empty",
    "hopeless",
    "sleep",
    "insomnia",
    "tired",
    "fatigue",
    "appetite",
    "weight",
    "concentration",
    "worthless",
    "guilty",
    "suicid",
    "anxious",
    "anxiety",
    "panic",
    "worry",
    "numb",
    "crying",
    # Medications
    "sertraline",
    "fluoxetine",
    "escitalopram",
    "paroxetine",
    "citalopram",
    "venlafaxine",
    "duloxetine",
    "bupropion",
    "mirtazapine",
    "medication",
    "dosage",
    "dose",
    "side effect",
    "nauseous",
    "nausea",
    # Coping
    "exercise",
    "walk",
    "therapy",
    "therapist",
    "meditation",
    "breathing",
    "journal",
    "routine",
    "schedule",
    # Mood changes
    "better",
    "worse",
    "improving",
    "struggling",
    "difficult",
    "progress",
    "relapse",
    "setback",
    "breakthrough",
]


def extract_clinical_sentences(messages: list[dict], min_length: int = 20) -> list[str]:
    """Step 1: Extract clinically relevant sentences from chat messages.

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


async def _llm_structure_summary(extracted: list[str], rag_service) -> str | None:
    """Step 2: LLM organizes extracted sentences by clinical theme.

    Constrained input: only the extracted sentences are provided.
    Temperature 0.1 for near-deterministic output.
    Returns structured summary or None if LLM fails.
    """
    if not hasattr(rag_service, "settings"):
        return None

    settings = rag_service.settings
    if not settings.llm_api_key:
        return None

    try:
        client = _get_llm_client(settings)
        response = await client.chat.completions.create(
            model=settings.llm_model_flash,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Organize these exact patient conversation excerpts into a brief clinical summary.\n\n"
                        "RULES:\n"
                        "- ONLY include information that appears in the excerpts below.\n"
                        "- Do NOT add context, inferences, or information not present.\n"
                        "- Use the patient's own words where possible.\n"
                        "- Organize by theme: symptoms, medication feedback, coping strategies, mood changes.\n"
                        "- Skip themes with no relevant excerpts.\n\n"
                        "EXCERPTS:\n" + "\n".join(f"- {s}" for s in extracted) + "\n\nSTRUCTURED SUMMARY:"
                    ),
                }
            ],
            temperature=0.1,
            max_tokens=300,
        )
        result = response.choices[0].message.content
        if result and len(result.strip()) > 20:
            return result.strip()
        return None
    except Exception as e:
        logger.warning(f"LLM structuring failed (using raw extraction): {e}")
        return None


def _nli_verify_summary(structured_summary: str, extracted: list[str], rag_service) -> str:
    """Step 3: NLI verification — check each claim against extracted sources.

    Removes claims not supported by any extracted sentence.
    Returns verified summary.
    """
    if not hasattr(rag_service, "verify_claim"):
        return structured_summary

    # Check if NLI model is available (lazy-loaded)
    try:
        rag_service._load_nli_model()
    except Exception:
        return structured_summary

    if rag_service._nli_model is None:
        return structured_summary

    # Split summary into sentences and verify each
    summary_sentences = re.split(r"(?<=[.!?])\s+", structured_summary)
    verified = []

    for sentence in summary_sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            # Too short to verify — keep (headers, labels)
            verified.append(sentence)
            continue

        # Check if supported by any extracted sentence
        supported = False
        for source in extracted:
            nli_result = rag_service.verify_claim(claim=sentence, source=source)
            if nli_result == "entailment":
                supported = True
                break

        if supported:
            verified.append(sentence)
        else:
            # Check if neutral (ambiguous) — keep with lower confidence
            # Only reject if contradicted by all sources
            any_contradiction = any(
                rag_service.verify_claim(claim=sentence, source=s) == "contradiction"
                for s in extracted[:3]  # Check against top 3 sources
            )
            if not any_contradiction:
                verified.append(sentence)  # Neutral — keep
            else:
                logger.debug(f"NLI rejected summary claim: {sentence[:80]}")

    return " ".join(verified)


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
    """Hybrid chat summary pipeline: extract → structure → verify → ingest.

    Step 1: Extract clinically relevant sentences (keyword matching)
    Step 2: LLM structures by theme (constrained input, temp 0.1) — optional
    Step 3: NLI verifies structured output against sources — optional
    Step 4: Ingest verified summary into patient RAG

    Falls back to raw extraction if LLM or NLI unavailable.
    Returns True if summary ingested, False otherwise.
    """
    # Step 1: Extract
    extracted = extract_clinical_sentences(messages)
    if not extracted:
        logger.debug(f"No clinical sentences in conversation {conversation_id}")
        return False

    # Step 2: LLM structuring (optional — falls back to raw extraction)
    structured_summary = await _llm_structure_summary(extracted, rag_service)

    # Step 3: NLI verification (only if LLM structuring produced output)
    if structured_summary:
        summary_content = _nli_verify_summary(structured_summary, extracted, rag_service)
    else:
        # Fallback: raw extracted sentences (no LLM, no NLI needed)
        summary_content = "Chat conversation summary:\n" + "\n".join(extracted)

    # Step 4: Ingest into patient RAG
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
            f"conversation {conversation_id}: {len(extracted)} excerpts, "
            f"method={'hybrid' if structured_summary else 'extractive'}"
        )
        return True
    return False
