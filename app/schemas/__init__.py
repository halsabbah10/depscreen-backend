"""Pydantic schemas for the DepScreen API."""

from .analysis import (
    # Symptom detection
    SymptomDetection, PostSymptomSummary,
    # Verification
    EvidenceValidation, ConfidenceAnalysis, AdversarialCheck, VerificationReport,
    # Explanation
    ExplanationReport, Evidence,
    # Screening
    ScreeningRequest, ScreeningResponse, ScreeningListItem, ScreeningHistoryResponse,
    # Chat
    ChatMessageRequest, ChatMessageResponse, ChatHistoryResponse,
    # Auth
    RegisterRequest, LoginRequest, TokenResponse, UserProfile, RefreshRequest,
    # Dashboard
    DashboardStats, PatientSummary,
    # Medications
    MedicationCreate, MedicationResponse,
    # Allergies
    AllergyCreate, AllergyResponse,
    # Diagnoses
    DiagnosisCreate, DiagnosisResponse,
    # Screening Schedule
    ScreeningScheduleCreate, ScreeningScheduleResponse,
    # Appointments
    AppointmentCreate, AppointmentResponse, AppointmentStatusUpdate,
    # Notifications
    NotificationResponse,
    # Care Plans
    CarePlanCreate, CarePlanResponse, CarePlanGoal, CarePlanIntervention,
    # Conversations
    ConversationCreate, ConversationResponse,
    # Profile
    ProfileUpdate, OnboardingProgress,
)

__all__ = [
    "SymptomDetection", "PostSymptomSummary",
    "EvidenceValidation", "ConfidenceAnalysis", "AdversarialCheck", "VerificationReport",
    "ExplanationReport", "Evidence",
    "ScreeningRequest", "ScreeningResponse", "ScreeningListItem", "ScreeningHistoryResponse",
    "ChatMessageRequest", "ChatMessageResponse", "ChatHistoryResponse",
    "RegisterRequest", "LoginRequest", "TokenResponse", "UserProfile", "RefreshRequest",
    "DashboardStats", "PatientSummary",
    "MedicationCreate", "MedicationResponse",
    "AllergyCreate", "AllergyResponse",
    "DiagnosisCreate", "DiagnosisResponse",
    "ScreeningScheduleCreate", "ScreeningScheduleResponse",
    "AppointmentCreate", "AppointmentResponse", "AppointmentStatusUpdate",
    "NotificationResponse",
    "CarePlanCreate", "CarePlanResponse", "CarePlanGoal", "CarePlanIntervention",
    "ConversationCreate", "ConversationResponse",
    "ProfileUpdate", "OnboardingProgress",
]
