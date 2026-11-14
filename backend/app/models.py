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
    channel: str = "text"  # "text" (iMessage / SMS)
    text: str  # original reply text
    language: str = "en"
    received_at: str
    message_id: Optional[str] = None  # PRD 2 Message.message_id; set by the store ("M-<absence id>")


class Classification(BaseModel):
    barrier: Barrier
    confidence: float
    evidence: str  # quote from the reply supporting the label
    safety_critical: bool = False
    # Where the label came from: keyword | gemini | correction_override | staff_correction
    source: str = "keyword"
    # F8: the past staff correction that decided (or was shown as an example for) this label
    correction_ref: Optional[str] = None
    correction_similarity: Optional[float] = None


class Correction(BaseModel):
    """PRD 2 Correction entity: a staff-confirmed barrier label (F8 learning from corrections)."""

    absence_id: str
    old_label: Optional[Barrier] = None
    new_label: Barrier
    corrected_by: str
    timestamp: str
    school_id: str = "demo"
    text: str = ""  # the reply the correction was made on (what later replies are compared with)
    language: str = "en"


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
    link_type: str  # bus_route | classroom | grade | neighborhood | contact (household is a unit, not a link)
    link_value: str
    student_ids: List[str]
    absence_ids: List[str]
    first_date: date
    last_date: date
    dollars_at_risk_week: float
    ablation: Ablation
    summary: str
    # ---- PRD 2 section 2 Cluster fields (all optional, so older clients keep working) ----
    layers_involved: List[str] = Field(default_factory=list)  # e.g. ["route", "labels"]
    zone: Optional[str] = None  # "<layer>:<zone id>", e.g. "route:Route 7"
    member_ids: List[str] = Field(default_factory=list)
    likely_cause: Optional[Barrier] = None
    cause_provisional: bool = False  # True when fewer than 2 replies (or under half the members) back the cause
    observed: Optional[float] = None
    expected: Optional[float] = None
    p_value: Optional[float] = None
    start_date: Optional[date] = None
    detected_on: Optional[date] = None
    weekly_revenue_at_risk: Optional[float] = None
    evidence_message_ids: List[str] = Field(default_factory=list)
    # Which method produced this cluster: "scan:<detector>", "scan:<detector>+synthetic_baseline" or
    # "rule_based" (fallback, PRD 2 section 3.2). method_note says what the statistic means.
    method: str = "rule_based"
    method_note: Optional[str] = None
    dataset: str = "demo"  # demo | sim


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


class ClaimFlag(BaseModel):
    """One unsupported or wrong statement found by the claim check (F5)."""

    where: str  # e.g. "steps[0].topic", "load_note", "reteach_suggestion"
    claim: str
    reason: str
    rule: str  # topic_not_in_guide | topic_date_mismatch | source_mismatch | date_not_missed | ... | nli


class LearningPlan(BaseModel):
    student_id: str
    course: str
    barrier: Barrier
    missed_days: int
    load_note: str
    steps: List[LearningPlanStep]
    teacher_checkins: List[date]
    total_minutes: int
    # ---- F5 additions (optional for older clients) ----
    eligible: bool = True  # PRD 1 F5: catch-up plans are for students with 2+ missed days
    eligibility_note: str = ""
    reteach_suggestion: str = ""  # one small-group reteach suggestion (teacher-facing)
    take_home: str = ""  # paper take-home in the home language
    attendance_recovery_offer: str = ""
    home_language: str = "en"
    verification_status: str = "unverified"  # verified | flagged | unverified
    verification_method: str = ""
    flags: List[ClaimFlag] = Field(default_factory=list)


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
