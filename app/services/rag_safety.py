"""RAG safety layers for anti-hallucination defense-in-depth.

Layers implemented here:
1. Relevance threshold gate — filter low-quality retrieval results
2. Prompt injection defense — sanitize + structurally isolate retrieved content
Plus: grounding instructions for LLM prompts, authority level mapping.

Layers implemented elsewhere:
3. NLI claim verification — in rag.py (verify_claim method)
4. Context conflict resolution — in decision logic
5. Graceful degradation — in RAGService
6. Stale data prevention — is_current + live SQL
"""

import logging
import re

logger = logging.getLogger(__name__)


# ── Layer 1: Relevance Threshold Gate ────────────────────────────────────


def filter_by_relevance(results: list[dict], threshold: float = 0.35) -> list[dict]:
    """Filter retrieval results by reranker score threshold.

    Results without a reranker_score (e.g., BM25-only fallback) pass through.
    Returns empty list if nothing passes — caller decides whether to
    proceed without RAG context (no context is safer than wrong context).
    """
    if not results:
        return []

    filtered = []
    for r in results:
        score = r.get("reranker_score")
        if score is None or score >= threshold:
            filtered.append(r)
        else:
            logger.debug(f"Filtered chunk {r.get('id', '?')}: score {score:.3f} < {threshold}")

    return filtered


# ── Layer 2: Prompt Injection Defense ────────────────────────────────────


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|context)",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"you\s+are\s+now\s+a\s+",
    r"forget\s+(everything|all|your\s+instructions)",
    r"new\s+instructions?\s*:",
    r"disregard\s+(all|any|the)\s+(above|previous)",
    r"override\s+(your|all|the)\s+(instructions|rules|guidelines)",
]


def sanitize_for_ingestion(text: str) -> str:
    """Strip known prompt injection patterns before embedding into RAG.

    Preserves clinical content. Only removes text matching injection patterns.
    """
    sanitized = text
    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[content filtered]", sanitized, flags=re.IGNORECASE)
    return sanitized


# ── Authority Levels ─────────────────────────────────────────────────────


AUTHORITY_LEVELS = {
    # Clinical knowledge base (highest trust)
    "dsm5_criteria": "HIGH",
    "clinical_guidelines": "HIGH",
    "assessment_tools": "HIGH",
    "medications": "HIGH",
    "structured": "HIGH",
    "crisis": "HIGH",
    # Moderate trust
    "comorbidity": "MODERATE",
    "psychoeducation": "MODERATE",
    "coping_strategies": "MODERATE",
    "cultural": "MODERATE",
    # Patient-level content (lower trust)
    "screening_text": "PATIENT_TEXT",
    "symptom_evidence": "PATIENT_TEXT",
    "patient_document": "PATIENT_TEXT",
    "chat_summary": "EXTRACTED_CHAT",
}


def get_authority_level(category: str) -> str:
    """Map a category or chunk_type to an authority level."""
    return AUTHORITY_LEVELS.get(category, "MODERATE")


def wrap_retrieved_context(chunk: dict) -> str:
    """Wrap a retrieved chunk with structural isolation and authority tags.

    The XML-like delimiters tell the LLM this is reference material,
    not instructions to follow.
    """
    text = chunk.get("text", "")
    metadata = chunk.get("metadata", {})
    category = metadata.get("category", metadata.get("chunk_type", "unknown"))
    source = metadata.get("source_file", "knowledge base")
    authority = get_authority_level(category)

    return (
        f'<retrieved_context type="{category}" authority="{authority}" source="{source}">\n'
        f"{text}\n"
        f"</retrieved_context>"
    )


# ── PII Sanitization ────────────────────────────────────────────────────


RAG_EXCLUDED_DOC_TYPES = {"cpr_id", "passport", "insurance_card"}


def sanitize_identity_document(content: str, doc_type: str) -> str:
    """Strip PII from identity document text before storage.

    Removes: CPR numbers (9 digits), passport numbers, card numbers.
    Preserves: name, DOB, nationality, gender (clinically relevant).
    """
    sanitized = content
    sanitized = re.sub(r"\b\d{9}\b", "[CPR REDACTED]", sanitized)
    sanitized = re.sub(r"\b[A-Z]{1,2}\d{6,9}\b", "[PASSPORT# REDACTED]", sanitized)
    sanitized = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD REDACTED]", sanitized)
    return sanitized


def should_ingest_to_rag(doc_type: str) -> bool:
    """Check if a document type should be ingested into RAG.

    Identity documents (CPR, passport, insurance) are stored in DB
    but NOT embedded — zero clinical value, high PII risk.
    """
    return doc_type not in RAG_EXCLUDED_DOC_TYPES


# ── Grounding Instructions ───────────────────────────────────────────────


GROUNDING_INSTRUCTIONS = """
CRITICAL RULES FOR USING RETRIEVED CONTEXT:

1. ONLY state clinical facts that appear in the retrieved context below.
   If the context doesn't mention something, do NOT add it from your own knowledge.
   Say "the available clinical resources don't cover this specific topic" instead.

2. When citing retrieved context, use the source tag: [Source: filename].
   Never invent a source name. Only cite sources that appear in the context.

3. If retrieved context contradicts what you know, follow the retrieved context
   (it's from authoritative clinical sources) and note the specific source.

4. NEVER generate specific medication dosages, treatment durations, or
   diagnostic conclusions unless they appear verbatim in the retrieved context.

5. If no relevant context was retrieved, say so honestly: "I don't have specific
   clinical guidance on that topic in my current reference materials. Please
   discuss this with your clinician."
"""


def build_rag_prompt_section(
    results: list[dict],
    section_title: str = "Clinical Reference Material",
) -> str:
    """Build the RAG context section for an LLM prompt.

    Wraps each chunk with structural isolation and authority tags.
    Prepends grounding instructions.
    """
    if not results:
        return ""

    parts = [f"## {section_title}\n"]
    parts.append(GROUNDING_INSTRUCTIONS)

    for r in results:
        parts.append(wrap_retrieved_context(r))

    return "\n\n".join(parts)
