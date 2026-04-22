"""Adaptive chunking service for RAG document preprocessing.

Splits documents into hierarchical parent/child chunks with overlap,
supporting markdown (header-based), plain text (paragraph-based), and
structured JSON (template-based natural language conversion).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    content: str = ""
    chunk_level: str = "child"  # "parent" or "child"
    sequence_index: int = 0
    token_count: int = 0
    content_hash: str = ""  # SHA-256 hex digest

    def __post_init__(self) -> None:
        if not self.token_count:
            self.token_count = estimate_tokens(self.content)
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count: words / 0.75 approximates BPE tokenisation."""
    return int(len(text.split()) / 0.75)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_type: str = "text",
    max_tokens: int = 512,
    min_chars: int = 20,
    overlap_tokens: int = 64,
    parent_max_tokens: int = 1500,
) -> list[Chunk]:
    """Split *text* into hierarchical parent/child chunks.

    Args:
        text: Raw document text.
        source_type: ``"markdown"`` (split on ``## `` headers) or ``"text"``
            (split on blank lines).
        max_tokens: Maximum token budget for a child chunk.
        min_chars: Discard chunks shorter than this character count.
        overlap_tokens: Tokens of context to carry forward into the next child.
        parent_max_tokens: Sections larger than this get a parent wrapper even
            if they don't need sub-chunking (kept for future use; currently
            every section that is sub-chunked gets a parent).

    Returns:
        Flat list of :class:`Chunk` objects ordered parent-before-children.
    """
    if not text or not text.strip():
        return []

    # 1. Split into top-level sections
    if source_type == "markdown":
        sections = _split_markdown_sections(text)
    else:
        sections = _split_paragraphs(text)

    chunks: list[Chunk] = []
    seq = 0

    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_tokens = estimate_tokens(section)

        if section_tokens <= max_tokens:
            # Small section → single child chunk, no parent wrapper
            if len(section) < min_chars:
                continue
            chunk = Chunk(
                content=section,
                chunk_level="child",
                sequence_index=seq,
            )
            chunks.append(chunk)
            seq += 1
        else:
            # Large section → parent + child sub-chunks with overlap
            parent = Chunk(
                content=section,
                chunk_level="parent",
                sequence_index=seq,
            )
            chunks.append(parent)
            seq += 1

            children = _split_with_overlap(
                section,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            for child_text in children:
                child_text = child_text.strip()
                if len(child_text) < min_chars:
                    continue
                child = Chunk(
                    parent_id=parent.id,
                    content=child_text,
                    chunk_level="child",
                    sequence_index=seq,
                )
                chunks.append(child)
                seq += 1

    return chunks


def chunk_json_entry(entry: dict, template: str = "generic") -> Chunk:
    """Convert a structured dict into a natural-language :class:`Chunk`.

    Templates:
        ``"medication"``  — drug info sentence
        ``"scoring"``     — assessment scale score sentence
        ``"symptom"``     — DSM-5 criterion sentence
        ``"generic"``     — key: value pairs
    """
    content = _render_template(entry, template)
    return Chunk(content=content, chunk_level="child")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_markdown_sections(text: str) -> list[str]:
    """Split markdown on ``## `` headers, keeping header text with its body."""
    lines = text.splitlines(keepends=True)
    sections: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("## ") and current:
            sections.append("".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("".join(current))

    # If no ## headers were found, fall back to paragraph splitting
    if len(sections) == 1 and not sections[0].startswith("## "):
        return _split_paragraphs(text)

    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (double newlines)."""
    import re
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_with_overlap(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Greedily build word-based windows of *max_tokens* with *overlap_tokens*
    carried forward from the previous window.

    Splitting at word boundaries keeps chunks coherent and avoids mid-word
    cuts. Token estimates use the same :func:`estimate_tokens` heuristic
    (words / 0.75) so max_tokens comparisons are consistent.
    """
    words = text.split()
    # Convert token budgets to approximate word counts
    max_words = int(max_tokens * 0.75)
    overlap_words = int(overlap_tokens * 0.75)

    if max_words <= 0:
        max_words = 1

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        window = words[start:end]
        chunks.append(" ".join(window))
        if end == len(words):
            break
        # Advance by (max_words - overlap_words); at least 1 word to avoid infinite loop
        advance = max(max_words - overlap_words, 1)
        start += advance

    return chunks


def _render_template(entry: dict, template: str) -> str:
    """Convert *entry* dict to a natural language sentence using *template*."""
    if template == "medication":
        drug = entry.get("drug", entry.get("name", "Unknown drug"))
        drug_class = entry.get("class", entry.get("drug_class", ""))
        brands = entry.get("brand_names", entry.get("brand", []))
        if isinstance(brands, str):
            brands = [brands]
        dose = entry.get("typical_dose", entry.get("dose", ""))
        side_effects = entry.get("common_side_effects", entry.get("side_effects", []))
        if isinstance(side_effects, str):
            side_effects = [side_effects]

        parts = [f"{drug.capitalize()}"]
        if brands:
            parts[0] += f" ({', '.join(brands)})"
        if drug_class:
            parts[0] += f" is a {drug_class} antidepressant"
        if dose:
            parts.append(f"typical dose {dose}")
        if side_effects:
            parts.append(f"common side effects include {', '.join(side_effects)}")
        return ". ".join(parts) + "."

    if template == "scoring":
        tool = entry.get("tool", entry.get("scale", "Assessment tool"))
        score_range = entry.get("range", entry.get("score", ""))
        severity = entry.get("severity", "")
        action = entry.get("clinical_action", entry.get("action", ""))

        parts = [f"{tool}"]
        if score_range:
            parts[0] += f" score {score_range}"
        if severity:
            parts[0] += f" indicates {severity} severity"
        if action:
            parts.append(f"Clinical action: {action}")
        return ". ".join(parts) + "."

    if template == "symptom":
        source = entry.get("source", "DSM-5")
        criterion = entry.get("criterion", entry.get("symptom", "Symptom"))
        description = entry.get("description", entry.get("detail", ""))

        parts = [f"{source} criterion: {criterion}"]
        if description:
            parts.append(description)
        return ". ".join(parts) + "."

    # generic: key=value pairs
    parts = [f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in entry.items()]
    return ". ".join(parts) + "." if parts else ""
