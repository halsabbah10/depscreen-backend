"""
Core screening API route.

Accepts text input, runs the full DSM-5 symptom detection pipeline
(DL classifier → LLM verification → RAG enrichment → LLM explanation),
persists results to DB, and ingests into patient RAG.

Requires authentication — screenings are always attributed to a patient.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from datetime import datetime
from uuid import uuid4
import logging

from app.core.config import get_settings, Settings
from app.models.db import User, Screening, get_db
from app.schemas.analysis import (
    ScreeningRequest,
    ScreeningResponse,
    Evidence,
)
from app.services.inference import ModelService
from app.services.llm import LLMService
from app.services.llm_verification import VerificationService
from app.services.decision import DecisionService
from app.services.rag import RAGService
from app.services.auth import get_current_user, log_audit
from app.middleware.rate_limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)

# Service instances (initialized on first request)
_model_service = None
_llm_service = None
_verification_service = None
_decision_service = None
_rag_service = None


async def get_services(settings: Settings = Depends(get_settings)):
    """Initialize and return all service instances."""
    global _model_service, _llm_service, _verification_service, _decision_service, _rag_service

    if _model_service is None:
        _model_service = ModelService(settings)
        await _model_service.load_models()
        _llm_service = LLMService(settings)
        _verification_service = VerificationService(settings)
        _decision_service = DecisionService()
        _rag_service = RAGService(settings)
        await _rag_service.initialize()

    return {
        "model": _model_service,
        "llm": _llm_service,
        "verification": _verification_service,
        "decision": _decision_service,
        "rag": _rag_service,
        "settings": settings,
    }


@router.post("", response_model=ScreeningResponse)
@limiter.limit("20/minute")
async def screen_text(
    request: Request,
    body: ScreeningRequest,
    current_user: User = Depends(get_current_user),
    services: dict = Depends(get_services),
    db: Session = Depends(get_db),
):
    """Screen text for DSM-5 depression symptoms.

    Full pipeline:
    1. Sentence-level DL classification (DistilBERT → 11 classes)
    2. LLM verification (evidence validation, adversarial check, confidence calibration)
    3. Decision service (severity mapping, confidence adjustment, flagging)
    4. RAG retrieval (clinical context for detected symptoms)
    5. LLM explanation (per-symptom clinical narrative)
    6. Persist to DB + ingest into patient RAG

    Returns comprehensive screening result with sentence-level evidence.
    """
    model_service: ModelService = services["model"]
    llm_service: LLMService = services["llm"]
    verification_service: VerificationService = services["verification"]
    decision_service: DecisionService = services["decision"]
    rag_service: RAGService = services["rag"]
    settings: Settings = services["settings"]

    text = body.text
    screening_id = str(uuid4())
    created_at = datetime.utcnow()

    logger.info(f"Screening {screening_id} started for patient {current_user.id[:8]} ({len(text)} chars)")

    if len(text) > settings.max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum length of {settings.max_text_length} characters",
        )

    # Step 1: DL symptom detection
    symptom_analysis = await model_service.predict_symptoms(text)
    logger.info(
        f"  Step 1 — Detected {symptom_analysis.unique_symptom_count} symptoms, "
        f"severity={symptom_analysis.severity_level}"
    )

    # Step 2: LLM verification
    verification = await verification_service.verify_prediction(
        text=text,
        symptom_analysis=symptom_analysis,
    )
    logger.info(
        f"  Step 2 — Verification: evidence_supports={verification.evidence_validation.evidence_supports_prediction}, "
        f"adversarial={verification.adversarial_check.likely_adversarial}, "
        f"trust={verification.confidence_analysis.should_trust_prediction}"
    )

    # Step 3: Decision (severity mapping + confidence adjustment)
    final_prediction, final_confidence, confidence_adjusted, flagged = (
        decision_service.compute_final_prediction(symptom_analysis, verification)
    )
    logger.info(f"  Step 3 — Decision: {final_prediction} ({final_confidence:.2%}), flagged={flagged}")

    # Step 4: RAG retrieval for detected symptoms
    rag_context_data = None
    rag_context_str = None
    if symptom_analysis.dsm5_criteria_met:
        rag_context_data = rag_service.retrieve_for_symptoms(symptom_analysis.dsm5_criteria_met)
        # Flatten for LLM prompt
        rag_parts = []
        for symptom, docs in rag_context_data.items():
            for doc in docs:
                rag_parts.append(f"[{symptom}] {doc['text'][:300]}")
        rag_context_str = "\n\n".join(rag_parts[:10]) if rag_parts else None
        logger.info(f"  Step 4 — RAG: retrieved context for {len(rag_context_data)} symptoms")

    # Step 5: LLM explanation
    verification_summary = decision_service.get_verification_summary(verification)
    explanation = await llm_service.generate_explanation(
        text=text,
        symptom_analysis=symptom_analysis,
        verification_summary=verification_summary,
        rag_context=rag_context_str,
    )
    logger.info(f"  Step 5 — Explanation generated")

    # Step 6a: Persist to DB
    adversarial_warning = None
    if verification.adversarial_check.likely_adversarial:
        adversarial_warning = (
            verification.adversarial_check.warning
            or f"Input flagged as potentially {verification.adversarial_check.adversarial_type}"
        )

    screening_record = Screening(
        id=screening_id,
        patient_id=current_user.id,
        created_at=created_at,
        text=text,
        source="api",
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

    # Step 6b: Ingest into patient RAG for future chat context
    rag_service.ingest_patient_screening(
        patient_id=current_user.id,
        screening_id=screening_id,
        text=text,
        symptoms_detected=[d.model_dump() for d in symptom_analysis.symptoms_detected],
        severity_level=symptom_analysis.severity_level,
    )

    # Audit log
    log_audit(db, current_user.id, "screening_created", resource_type="screening", resource_id=screening_id)

    logger.info(f"Screening {screening_id} complete — saved to DB + RAG")

    # Build response
    evidence = Evidence(
        sentence_evidence=symptom_analysis.symptoms_detected,
        top_evidence_sentences=[
            d.sentence_text
            for d in sorted(
                symptom_analysis.symptoms_detected,
                key=lambda d: d.confidence,
                reverse=True,
            )[:5]
        ],
    )

    return ScreeningResponse(
        id=screening_id,
        created_at=created_at,
        text=text,
        symptom_analysis=symptom_analysis,
        evidence=evidence,
        verification=verification,
        final_prediction=final_prediction,
        final_confidence=final_confidence,
        confidence_adjusted=confidence_adjusted,
        explanation_report=explanation,
        flagged_for_review=flagged,
        adversarial_warning=adversarial_warning,
    )
