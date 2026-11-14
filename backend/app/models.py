"""Domain models for Why Absent (PRD 1)."""
from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Barrier(str, Enum):
    ILLNESS = "illness"
    TRANSPORTATION = "transportation"
    HOUSING_FOOD = "housing or food insecurity"
    CAREGIVING = "caregiving or family obligation"
    SAFETY = "safety or bullying"
    DISENGAGEMENT = "disengagement or academic struggle"
    APPOINTMENT = "appointment or travel"
    UNKNOWN = "unknown"


class AbsenceStatus(str, Enum):
    REASON_CAPTURED = "reason captured"
    AWAITING_REPLY = "awaiting reply"
    UNKNOWN = "unknown"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    SENT = "sent"
    COMPLETED = "completed"


class Student(BaseModel):
    id: str
    first_name: str
    last_name: str
    grade: int
    classroom: str
    teacher: str
    bus_route: Optional[str] = None
    household_id: str
    preferred_language: str = "en"  # ISO 639-1
    guardian_name: str
    guardian_phone: str


class FamilyReply(BaseModel):
    channel: str = "text"  # "text" | "voice"
    text: str  # original text, or the voice transcript
    language: str = "en"
    received_at: str


class Classification(BaseModel):
    barrier: Barrier
    confidence: float
    evidence: str  # quote from the reply supporting the label
    safety_critical: bool = False


class Absence(BaseModel):
    id: str
    student_id: str
    date: date
    sis_code: str = "unverified"  # excused | unexcused | unverified, as exported by the SIS
    status: AbsenceStatus = AbsenceStatus.AWAITING_REPLY
    reply: Optional[FamilyReply] = None
    classification: Optional[Classification] = None


class Ablation(BaseModel):
    """How many students each signal alone would have grouped, versus both together."""

    text_only: int
    network_only: int
    combined: int


class Cluster(BaseModel):
    id: str
    barrier: Barrier
    link_type: str  # bus_route | classroom | household
    link_value: str
    student_ids: List[str]
    absence_ids: List[str]
    first_date: date
    last_date: date
    dollars_at_risk_week: float
    ablation: Ablation
    summary: str


class ActionPlan(BaseModel):
    id: str
    scope: str  # "cluster" | "case"
    cluster_id: Optional[str] = None
    absence_id: Optional[str] = None
    barrier: Barrier
    school_action: str
    owner: str
    target_date: date
    family_request: str
    message_language: str
    message: str
    recipient_student_ids: List[str]
    status: PlanStatus = PlanStatus.DRAFT
    approved_by: Optional[str] = None


class LearningPlanStep(BaseModel):
    order: int
    topic: str
    missed_date: date
    activity: str
    minutes: int
    source: str  # pacing-guide reference supporting this step


class LearningPlan(BaseModel):
    student_id: str
    course: str
    barrier: Barrier
    missed_days: int
    load_note: str
    steps: List[LearningPlanStep]
    teacher_checkins: List[date]
    total_minutes: int


class Escalation(BaseModel):
    absence_id: str
    student_id: str
    reason: str
    evidence: str
    assigned_to: str = "Counselor"
    status: str = "open"


class RevenueSettings(BaseModel):
    state: str = "CA"
    daily_rate: float = Field(65.0, gt=0, description="Funding dollars per student per day of attendance")
    recovery_enabled: bool = True  # CA Attendance Recovery lets make-up instruction recover funding


class RootCauseCost(BaseModel):
    barrier: Barrier
    absences: int
    dollars: float


class RevenueSummary(BaseModel):
    settings: RevenueSettings
    week_start: date
    week_end: date
    absences_this_week: int
    dollars_at_risk_week: float
    days_recovered_to_date: int
    dollars_recovered_to_date: float
    by_root_cause: List[RootCauseCost]
