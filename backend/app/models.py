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
    # gemini | keywords (Gemini not configured, or seed data) | keywords_fallback (Gemini failed or timed out)
    # | safety_keywords (the keyword safety check fired; it always wins over the model)
    source: str = "keywords"
    language: Optional[str] = None  # ISO 639-1, detected by Gemini; None from the keyword rules


class CaseReport(BaseModel):
    """One saved version of a counselor's internal case report for an escalation."""

    escalation_id: str
    version: int
    text: str
    saved_by: str
    saved_at: str


class Message(BaseModel):
    """One text between the school and a family (PRD 3 section 1). An absence can have many."""

    message_id: str
    absence_id: Optional[str] = None  # None when the text could not be tied to an absence
    student_id: Optional[str] = None
    direction: str  # "inbound" | "outbound"
    channel: Optional[str] = None  # imessage | sms | simulated | seed; None if an outbound message never left
    language: str = "en"
    raw_text_or_transcript: str
    timestamp: str  # inbound: when the family's message arrived; outbound: when we tried to send (UTC)
    # inbound: arrival -> classified and stored. outbound: time the send took, or for an
    # acknowledgement, family's message arrival -> reply accepted. None when not measured (seed data).
    latency_ms: Optional[int] = None
    kind: str = "reply"  # inbound: reply. outbound: check_in | plan | ack
    status: str = "received"  # inbound: received. outbound: sent | queued | blocked | failed
    reason: Optional[str] = None  # outbound: why it was queued, blocked or failed. inbound: opt-out or opt-in note
    # inbound: cluster | case | unknown | escalated | opted_out | opted_in | unlinked | no_absence
    route: Optional[str] = None
    classification: Optional[Classification] = None  # the label this inbound message received
    phone: Optional[str] = None  # the family's number
    plan_id: Optional[str] = None
    external_id: Optional[str] = None  # the channel's own message id (Photon)


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
    """A safety case handed to the counselor (PRD 3 sections 1 and 3).

    While it is not closed, nothing automated is sent to the student's household, and the
    absence keeps its safety label: only the counselor can downgrade it.
    """

    escalation_id: str
    absence_id: str
    student_id: str
    household_id: str
    reason: str
    evidence: str
    assigned_to: str = "Counselor"
    status: str = "open"  # open -> acknowledged -> closed
    opened_at: str
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    # Mandated-reporting decision (CANRA). The system is not the reporter; staff record what they did.
    report_status: Optional[str] = None  # report_filed | not_required
    report_by: Optional[str] = None
    report_at: Optional[str] = None
    closed_at: Optional[str] = None
    closed_by: Optional[str] = None
    outcome: Optional[str] = None  # resolved (real concern, handled) | not_safety (false alarm, relabeled)
    relabeled_to: Optional[Barrier] = None
    note: Optional[str] = None


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
