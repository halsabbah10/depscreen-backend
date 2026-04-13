"""Pydantic schemas for the DepScreen API."""

from .analysis import (
    AdversarialCheck,
    # Allergies
    AllergyCreate,
    AllergyResponse,
    # Appointments
    AppointmentCreate,
    AppointmentResponse,
    AppointmentStatusUpdate,
    # Care Plans
    CarePlanCreate,
    CarePlanGoal,
    CarePlanIntervention,
    CarePlanResponse,
    ChatHistoryResponse,
    # Chat
    ChatMessageRequest,
    ChatMessageResponse,
    ConfidenceAnalysis,
    # Conversations
    ConversationCreate,
    ConversationResponse,
    # Dashboard
    DashboardStats,
    # Diagnoses
    DiagnosisCreate,
    DiagnosisResponse,
    Evidence,
    # Verification
    EvidenceValidation,
    # Explanation
    ExplanationReport,
    LoginRequest,
    # Medications
    MedicationCreate,
    MedicationResponse,
    # Notifications
    NotificationResponse,
    OnboardingProgress,
    PatientSummary,
    PostSymptomSummary,
    # Profile
    ProfileUpdate,
    RefreshRequest,
    # Auth
    RegisterRequest,
    ScreeningHistoryResponse,
    ScreeningListItem,
    # Screening
    ScreeningRequest,
    ScreeningResponse,
    # Screening Schedule
    ScreeningScheduleCreate,
    ScreeningScheduleResponse,
    # Symptom detection
    SymptomDetection,
    TokenResponse,
    UserProfile,
    VerificationReport,
)

__all__ = [
    "SymptomDetection",
    "PostSymptomSummary",
    "EvidenceValidation",
    "ConfidenceAnalysis",
    "AdversarialCheck",
    "VerificationReport",
    "ExplanationReport",
    "Evidence",
    "ScreeningRequest",
    "ScreeningResponse",
    "ScreeningListItem",
    "ScreeningHistoryResponse",
    "ChatMessageRequest",
    "ChatMessageResponse",
    "ChatHistoryResponse",
    "RegisterRequest",
    "LoginRequest",
    "TokenResponse",
    "UserProfile",
    "RefreshRequest",
    "DashboardStats",
    "PatientSummary",
    "MedicationCreate",
    "MedicationResponse",
    "AllergyCreate",
    "AllergyResponse",
    "DiagnosisCreate",
    "DiagnosisResponse",
    "ScreeningScheduleCreate",
    "ScreeningScheduleResponse",
    "AppointmentCreate",
    "AppointmentResponse",
    "AppointmentStatusUpdate",
    "NotificationResponse",
    "CarePlanCreate",
    "CarePlanResponse",
    "CarePlanGoal",
    "CarePlanIntervention",
    "ConversationCreate",
    "ConversationResponse",
    "ProfileUpdate",
    "OnboardingProgress",
]
