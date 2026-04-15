"""
Data ingestion API routes.

Four ingestion methods, all running the full screening pipeline:
1. Reddit profile analysis — fetch public posts, screen each, aggregate
2. X/Twitter profile analysis — fetch public tweets, screen, aggregate
3. Guided clinical check-in — structured DSM-5-aligned prompts
4. Bulk text upload — social media data exports

Every method runs: DL classification → LLM verification → Decision → RAG enrichment → DB persistence.
"""

import logging
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.routes.analyze import get_services
from app.middleware.rate_limiter import limiter
from app.models.db import Screening, User, get_db
from app.schemas.analysis import (
    ScreeningListItem,
)
from app.services.auth import get_current_user, log_audit
from app.services.decision import DecisionService
from app.services.inference import ModelService
from app.services.ingestion import (
    combine_checkin_responses,
    fetch_reddit_posts,
    fetch_x_posts,
    get_checkin_prompts,
    parse_reddit_export,
)
from app.services.llm import LLMService
from app.services.llm_verification import VerificationService
from app.services.rag import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Request Schemas ───────────────────────────────────────────────────────────


class RedditIngestionRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    mental_health_only: bool = Field(default=True)
    max_posts: int = Field(default=50, ge=1, le=100)


class XIngestionRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    mental_health_filter: bool = Field(default=True)
    max_posts: int = Field(default=50, ge=1, le=100)


class CheckInResponse(BaseModel):
    responses: dict[str, str] = Field(description="Map of prompt_id to response text")


class BulkUploadRequest(BaseModel):
    content: str = Field(min_length=20, max_length=500000)
    format: str = Field(default="auto")


# ── Shared Pipeline Helper ────────────────────────────────────────────────────


async def _run_full_pipeline(
    text: str,
    source: str,
    patient_id: str,
    services: dict,
    db: Session,
) -> tuple[str, Screening]:
    """Run the complete screening pipeline and persist results.

    Shared by all ingestion methods to ensure consistency.
    Returns (screening_id, screening_record).
    """
    model_service: ModelService = services["model"]
    llm_service: LLMService = services["llm"]
    verification_service: VerificationService = services["verification"]
    decision_service: DecisionService = services["decision"]
    rag_service: RAGService = services["rag"]

    screening_id = str(uuid4())
    created_at = datetime.utcnow()

    # Step 1: DL symptom detection
    symptom_analysis = await model_service.predict_symptoms(text)

    # Step 2: LLM verification
    verification = await verification_service.verify_prediction(
        text=text,
        symptom_analysis=symptom_analysis,
    )

    # Step 3: Decision
    final_prediction, final_confidence, confidence_adjusted, flagged = decision_service.compute_final_prediction(
        symptom_analysis, verification
    )

    # Step 4: RAG retrieval
    rag_context_data = None
    rag_context_str = None
    if symptom_analysis.dsm5_criteria_met and rag_service.is_initialized:
        rag_context_data = rag_service.retrieve_for_symptoms(symptom_analysis.dsm5_criteria_met)
        rag_parts = []
        for symptom, docs in rag_context_data.items():
            for doc in docs:
                rag_parts.append(f"[{symptom}] {doc['text'][:300]}")
        rag_context_str = "\n\n".join(rag_parts[:10]) if rag_parts else None

    # Step 5: LLM explanation
    verification_summary = decision_service.get_verification_summary(verification)
    explanation = await llm_service.generate_explanation(
        text=text,
        symptom_analysis=symptom_analysis,
        verification_summary=verification_summary,
        rag_context=rag_context_str,
    )

    # Step 6: Adversarial warning
    adversarial_warning = None
    if verification.adversarial_check.likely_adversarial:
        adversarial_warning = (
            verification.adversarial_check.warning
            or f"Input flagged as potentially {verification.adversarial_check.adversarial_type}"
        )

    # Step 7: Persist to DB
    screening_record = Screening(
        id=screening_id,
        patient_id=patient_id,
        created_at=created_at,
        text=text[:50000],  # Cap at 50k chars
        source=source,
        symptom_data=symptom_analysis.model_dump(),
        symptom_count=symptom_analysis.unique_symptom_count,
        severity_level=symptom_analysis.severity_level,
        verification_data=verification.model_dump(),
        explanation_data=explanation.model_dump(),
        rag_context=rag_context_data,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        confidence_adjusted=confidence_adjusted,
        flagged_for_review=flagged,
        adversarial_warning=adversarial_warning,
    )
    db.add(screening_record)
    db.commit()

    # Step 8: Ingest into patient RAG
    if rag_service.is_initialized:
        rag_service.ingest_patient_screening(
            patient_id=patient_id,
            screening_id=screening_id,
            text=text,
            symptoms_detected=[d.model_dump() for d in symptom_analysis.symptoms_detected],
            severity_level=symptom_analysis.severity_level,
        )

    return screening_id, screening_record


def _screening_to_list_item(s: Screening) -> ScreeningListItem:
    """Convert a Screening DB record to a list item."""
    return ScreeningListItem(
        id=s.id,
        created_at=s.created_at,
        text_preview=s.text[:100] + "..." if len(s.text) > 100 else s.text,
        final_prediction=s.final_prediction,
        final_confidence=s.final_confidence,
        symptom_count=s.symptom_count or 0,
        severity_level=s.severity_level or "none",
        flagged_for_review=s.flagged_for_review,
    )


# ── Guided Check-in Prompts ───────────────────────────────────────────────────


@router.get("/checkin/prompts")
async def get_prompts(current_user: User = Depends(get_current_user)):
    """Get the 9 structured clinical check-in prompts (DSM-5 aligned)."""
    return {"prompts": get_checkin_prompts()}


@router.post("/checkin")
@limiter.limit("10/minute")
async def submit_checkin(
    request: Request,
    body: CheckInResponse,
    current_user: User = Depends(get_current_user),
    services: dict = Depends(get_services),
    db: Session = Depends(get_db),
):
    """Submit guided check-in responses. Runs the full screening pipeline."""
    if not body.responses:
        raise HTTPException(status_code=400, detail="At least one response is required")

    combined_text = combine_checkin_responses(body.responses)
    if len(combined_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Responses are too short for screening")

    screening_id, _ = await _run_full_pipeline(
        text=combined_text,
        source="checkin",
        patient_id=current_user.id,
        services=services,
        db=db,
    )

    log_audit(db, current_user.id, "screening_created", resource_type="screening", resource_id=screening_id)

    # Fetch and return full result
    from app.api.routes.history import _screening_to_response

    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    return _screening_to_response(screening)


# ── Reddit Profile Analysis ──────────────────────────────────────────────────


@router.post("/reddit")
@limiter.limit("10/minute")
async def analyze_reddit_profile(
    request: Request,
    body: RedditIngestionRequest,
    current_user: User = Depends(get_current_user),
    services: dict = Depends(get_services),
    db: Session = Depends(get_db),
):
    """Fetch Reddit public posts and screen each through the full pipeline.

    Aggregates symptom detections across all posts for a comprehensive
    longitudinal profile. Uses public Reddit JSON API — no API key needed.
    """
    model_service: ModelService = services["model"]

    # Fetch posts
    try:
        posts = await fetch_reddit_posts(
            username=body.username,
            limit=body.max_posts,
            mental_health_only=body.mental_health_only,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not posts:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant posts found for u/{body.username}.",
        )

    # Screen each post individually (DL only — verification on aggregate)
    per_post_results = []
    all_detections = []

    for post in posts:
        result = await model_service.predict_symptoms(post.text)
        per_post_results.append(
            {
                "subreddit": post.subreddit,
                "title": post.title[:100],
                "date": datetime.utcfromtimestamp(post.created_utc).isoformat() if post.created_utc else None,
                "symptoms": [d.model_dump() for d in result.symptoms_detected],
                "symptom_count": result.unique_symptom_count,
                "severity": result.severity_level,
            }
        )
        all_detections.extend(result.symptoms_detected)

    # Combined text for the aggregate screening record
    combined_text = "\n\n---\n\n".join(f"[r/{p.subreddit}] {p.text}" for p in posts)

    # Run full pipeline on the combined text
    screening_id, screening = await _run_full_pipeline(
        text=combined_text,
        source="reddit",
        patient_id=current_user.id,
        services=services,
        db=db,
    )

    log_audit(db, current_user.id, "screening_created", resource_type="screening", resource_id=screening_id)

    return {
        "screening_id": screening_id,
        "platform": "reddit",
        "username": request.username,
        "posts_fetched": len(posts),
        "posts_screened": len(per_post_results),
        "aggregate_severity": screening.severity_level,
        "aggregate_symptom_count": screening.symptom_count,
        "flagged_for_review": screening.flagged_for_review,
        "per_post_results": per_post_results,
        "subreddits_analyzed": sorted(set(p.subreddit for p in posts)),
    }


# ── X/Twitter Profile Analysis ───────────────────────────────────────────────


@router.post("/x")
@limiter.limit("10/minute")
async def analyze_x_profile(
    request: Request,
    body: XIngestionRequest,
    current_user: User = Depends(get_current_user),
    services: dict = Depends(get_services),
    db: Session = Depends(get_db),
):
    """Fetch X/Twitter public posts and screen through the full pipeline."""
    try:
        tweets = await fetch_x_posts(
            username=body.username,
            limit=body.max_posts,
            mental_health_filter=body.mental_health_filter,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not tweets:
        raise HTTPException(
            status_code=404,
            detail=f"No relevant posts found for @{body.username}.",
        )

    # Screen each tweet (DL only)
    per_post_results = []
    model_service: ModelService = services["model"]

    for tweet in tweets:
        result = await model_service.predict_symptoms(tweet.text)
        per_post_results.append(
            {
                "platform": "x",
                "text_preview": tweet.text[:100],
                "symptoms": [d.model_dump() for d in result.symptoms_detected],
                "symptom_count": result.unique_symptom_count,
                "severity": result.severity_level,
            }
        )

    # Combined text
    combined_text = "\n\n".join(f"[@{request.username}] {t.text}" for t in tweets)

    # Run full pipeline
    screening_id, screening = await _run_full_pipeline(
        text=combined_text,
        source="x",
        patient_id=current_user.id,
        services=services,
        db=db,
    )

    log_audit(db, current_user.id, "screening_created", resource_type="screening", resource_id=screening_id)

    return {
        "screening_id": screening_id,
        "platform": "x",
        "username": request.username,
        "posts_fetched": len(tweets),
        "posts_screened": len(per_post_results),
        "aggregate_severity": screening.severity_level,
        "aggregate_symptom_count": screening.symptom_count,
        "flagged_for_review": screening.flagged_for_review,
        "per_post_results": per_post_results,
    }


# ── Bulk Text Upload ──────────────────────────────────────────────────────────


@router.post("/bulk")
@limiter.limit("10/minute")
async def upload_bulk_text(
    request: Request,
    body: BulkUploadRequest,
    current_user: User = Depends(get_current_user),
    services: dict = Depends(get_services),
    db: Session = Depends(get_db),
):
    """Upload and screen multiple text entries from a data export.

    Supports Reddit GDPR exports (JSON/CSV) and plain text.
    Runs the full screening pipeline on the combined content.
    """
    entries = parse_reddit_export(body.content)
    if not entries:
        raise HTTPException(status_code=400, detail="No screenable text entries found in upload")

    # Cap entries
    entries = entries[:100]

    # Screen each entry (DL only for per-entry breakdown)
    model_service: ModelService = services["model"]
    per_entry_results = []

    for entry in entries:
        result = await model_service.predict_symptoms(entry["text"])
        per_entry_results.append(
            {
                "source": entry.get("source", "upload"),
                "text_preview": entry["text"][:100],
                "symptoms": [d.model_dump() for d in result.symptoms_detected],
                "severity": result.severity_level,
            }
        )

    # Combined text
    combined_text = "\n\n".join(e["text"] for e in entries)

    # Run full pipeline
    screening_id, screening = await _run_full_pipeline(
        text=combined_text,
        source="bulk",
        patient_id=current_user.id,
        services=services,
        db=db,
    )

    log_audit(db, current_user.id, "screening_created", resource_type="screening", resource_id=screening_id)

    return {
        "screening_id": screening_id,
        "entries_parsed": len(entries),
        "entries_screened": len(per_entry_results),
        "aggregate_severity": screening.severity_level,
        "aggregate_symptom_count": screening.symptom_count,
        "flagged_for_review": screening.flagged_for_review,
        "per_entry_results": per_entry_results,
    }
