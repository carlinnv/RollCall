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
    # Where the label came from: gemini | keywords (Gemini off, or seed data) | keywords_fallback (Gemini failed
    # or timed out) | safety_keywords (the keyword safety check fired; it always wins) | correction_override
    # (a similar past staff correction decided it, F8) | staff_correction (a person set it)
    source: str = "keywords"
    language: Optional[str] = None  # ISO 639-1, detected by Gemini; None from the keyword rules
    # F8: the past staff correction that decided (or was shown as an example for) this label
    correction_ref: Optional[str] = None
    correction_similarity: Optional[float] = None
    # Kinds of safety concern found: self_harm | imminent_danger | abuse_or_neglect | bullying | supervision | other
    safety_types: List[str] = []


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
    kind: str = "reply"  # inbound: reply. outbound: check_in | plan | ack | crisis
    status: str = "received"  # inbound: received. outbound: sent | queued | blocked | failed
    reason: Optional[str] = None  # outbound: why it was queued, blocked or failed. inbound: opt-out or opt-in note
    # inbound: cluster | case | unknown | escalated | opted_out | opted_in | unlinked | no_absence
    route: Optional[str] = None
    classification: Optional[Classification] = None  # the label this inbound message received
    phone: Optional[str] = None  # the family's number
    plan_id: Optional[str] = None
    external_id: Optional[str] = None  # the channel's own message id (Photon)
    # Held for quiet hours: the earliest time (UTC) it may go. Cleared once it is sent or blocked.
    deliver_after: Optional[str] = None


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


class Staff(BaseModel):
    """A staff member on the on-call roster."""

    staff_id: str
    name: str
    role: str  # counselor | administrator | principal
    phone: str
    on_call: bool = True


class Page(BaseModel):
    """One page sent (or attempted) to a staff member about an escalation."""

    staff_id: Optional[str] = None  # None when no one was on call
    name: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    level: str = "first"  # first | escalated (unacknowledged after 15 minutes)
    at: str
    status: str  # sent | queued | blocked | failed
    reason: Optional[str] = None


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
    safety_types: List[str] = []  # every kind of concern seen in this case's messages
    # The fixed crisis-resource message (PRD 3 section 3): sent at most once, only for self-harm or
    # imminent danger, never when abuse is disclosed. None means it was not sent and is not due.
    crisis_message_status: Optional[str] = None  # sent | queued | blocked | failed | withheld
    crisis_message_reason: Optional[str] = None
    crisis_message_at: Optional[str] = None
    pages: List[Page] = []  # staff paged about this case, oldest first


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
