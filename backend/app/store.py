"""In-memory store and the end-to-end flow from PRD 1 section 5.

Absence logged -> family replies -> safety gate -> classify & price ->
cluster or case plan drafted -> clerk approves & sends -> learning plan.

State lives in memory for the hackathon demo; `reset()` reloads the seed.
"""
import csv
import io
import statistics
import threading
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

from .data import seed
from .models import (
    Absence, AbsenceStatus, ActionPlan, Barrier, Classification, Cluster, Escalation, FamilyReply,
    LearningPlan, Message, PlanStatus, RevenueSettings, RevenueSummary, Student,
)
from .services import action_plan, classifier, clustering, learning_plan, outreach, revenue

TODAY = date(2026, 11, 14)
PRIOR_DAYS_RECOVERED = 14  # recovered through Attendance Recovery before this week (synthetic)
LATENCY_TARGET_MS = 10_000  # PRD 3 F2: a reply is labeled within 10 seconds


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms(start: datetime) -> int:
    """Milliseconds since `start`. Never negative, in case the channel's clock runs ahead of ours."""
    return max(0, round((utc_now() - start).total_seconds() * 1000))


def _no_channel(student: Student, phone: str, text: str, key: str) -> dict:
    """Default delivery when no messaging channel is running: the message waits in the outbox."""
    return {"status": "queued", "reason": "messaging channel is off", "channel": None}


class Store:
    def __init__(self) -> None:
        # Delivery function, set by the Photon channel when it starts. Survives reset().
        # (student, phone, text, idempotency_key) -> {"status": sent | queued | failed, "reason", "channel"}
        self.sender: Callable[[Student, str, str, str], dict] = _no_channel
        self._log_lock = threading.Lock()  # the Photon thread and API requests both append messages
        self.reset()

    # ---------- setup ----------
    def reset(self) -> None:
        self.students: Dict[str, Student] = {s["id"]: Student(**s) for s in seed.students()}
        self.absences: Dict[str, Absence] = {}
        self.escalations: Dict[str, Escalation] = {}
        self.plans: Dict[str, ActionPlan] = {}
        self.messages: List[Message] = []  # every inbound and outbound text, oldest first
        self.completed_learning: Dict[str, int] = {}  # student_id -> days recovered
        self.settings: RevenueSettings = revenue.STATE_DEFAULTS["CA"].model_copy()
        for rec in seed.absences():
            reply = rec.pop("reply")
            self.absences[rec["id"]] = Absence(**rec)
            if reply:  # precomputed history: logged, but with no latency measurement
                self.record_reply(rec["id"], FamilyReply(**reply), channel="seed", measure=False)

    # ---------- F1: absence tracking & cause capture ----------
    def import_csv(self, text: str) -> Tuple[int, List[str]]:
        """Ingest a daily SIS export with columns: student_id,date,sis_code."""
        added, errors = 0, []
        reader = csv.DictReader(io.StringIO(text.strip()))
        missing = {"student_id", "date"} - set(reader.fieldnames or [])
        if missing:
            return 0, [f"missing column(s): {', '.join(sorted(missing))}"]
        existing = {(a.student_id, a.date) for a in self.absences.values()}
        for line, row in enumerate(reader, start=2):
            sid = (row.get("student_id") or "").strip()
            if sid not in self.students:
                errors.append(f"line {line}: unknown student_id '{sid}'")
                continue
            try:
                d = date.fromisoformat((row.get("date") or "").strip())
            except ValueError:
                errors.append(f"line {line}: bad date '{row.get('date')}'")
                continue
            if (sid, d) in existing:
                continue
            aid = f"A{len(self.absences) + 1:03d}"
            self.absences[aid] = Absence(id=aid, student_id=sid, date=d,
                                         sis_code=(row.get("sis_code") or "unverified").strip())
            existing.add((sid, d))
            added += 1
        return added, errors

    def record_reply(self, absence_id: str, reply: FamilyReply, *, channel: str = "simulated",
                     received: Optional[datetime] = None, external_id: Optional[str] = None,
                     phone: Optional[str] = None, reason: Optional[str] = None, measure: bool = True) -> str:
        """Classify a family reply, log it, and return the route: escalated | cluster | case | unknown.

        `received` is when the family's message arrived (defaults to now). Latency is measured
        from then until the label is stored. The absence keeps its latest reply; the message
        log keeps every one.
        """
        start = received or utc_now()
        route = self._route_reply(absence_id, reply)
        absence = self.absences[absence_id]
        self.log_message(
            absence_id=absence_id, student_id=absence.student_id, direction="inbound", channel=channel,
            language=reply.language, raw_text_or_transcript=reply.text,
            timestamp=iso(start) if measure else reply.received_at,
            latency_ms=elapsed_ms(start) if measure else None,
            route=route, classification=absence.classification, phone=phone, external_id=external_id,
            reason=reason,
        )
        return route

    def _route_reply(self, absence_id: str, reply: FamilyReply) -> str:
        """Run the safety gate and classify."""
        absence = self.absences[absence_id]
        absence.reply = reply
        result: Classification = classifier.classify(reply.text)
        absence.classification = result

        if result.safety_critical:
            # Counselor takes over immediately; no automated reply is ever sent.
            absence.status = AbsenceStatus.REASON_CAPTURED
            self.escalations[absence_id] = Escalation(
                absence_id=absence_id, student_id=absence.student_id,
                reason=result.barrier.value, evidence=result.evidence,
            )
            return "escalated"

        if result.barrier == Barrier.UNKNOWN:
            absence.status = AbsenceStatus.UNKNOWN
            return "unknown"

        absence.status = AbsenceStatus.REASON_CAPTURED
        in_cluster = any(absence_id in c.absence_ids for c in self.clusters())
        return "cluster" if in_cluster else "case"

    def absence_count(self, student_id: str) -> int:
        return sum(1 for a in self.absences.values() if a.student_id == student_id)

    # ---------- message log ----------
    def log_message(self, **fields) -> Message:
        with self._log_lock:
            message = Message(message_id=f"M{len(self.messages) + 1:04d}", **fields)
            self.messages.append(message)
        return message

    def thread(self, absence_id: str) -> List[Message]:
        """Every message tied to one absence, oldest first."""
        return [m for m in self.messages if m.absence_id == absence_id]

    def latency_summary(self, channel: Optional[str] = None) -> dict:
        """How fast inbound replies were labeled, against the 10-second target."""
        timed = [m.latency_ms for m in self.messages
                 if m.direction == "inbound" and m.latency_ms is not None and channel in (None, m.channel)]
        return {
            "target_ms": LATENCY_TARGET_MS,
            "measured": len(timed),
            "within_target": sum(1 for ms in timed if ms <= LATENCY_TARGET_MS),
            "median_ms": round(statistics.median(timed)) if timed else None,
            "max_ms": max(timed) if timed else None,
        }

    @property
    def outbox(self) -> List[dict]:
        """Outbound messages in the shape the Plans page reads."""
        return [self._outbox_entry(m) for m in self.messages if m.direction == "outbound"]

    @staticmethod
    def _outbox_entry(m: Message) -> dict:
        entry = {"message_id": m.message_id, "kind": m.kind, "to": m.phone, "student_id": m.student_id,
                 "text": m.raw_text_or_transcript, "sent_at": m.timestamp, "status": m.status,
                 "reason": m.reason, "channel": m.channel, "latency_ms": m.latency_ms}
        if m.plan_id:
            entry["plan_id"] = m.plan_id
        if m.absence_id:
            entry["absence_id"] = m.absence_id
        return entry

    def check_in_message(self, absence: Absence) -> str:
        return outreach.check_in_text(self.students[absence.student_id], absence.date)

    # ---------- F2: outbound messages ----------
    def household_suppressed(self, student_id: str) -> bool:
        """True while any student in this household has an open safety escalation."""
        household = self.students[student_id].household_id
        return any(e.status == "open" and self.students[e.student_id].household_id == household
                   for e in self.escalations.values())

    def send_to_family(self, student_id: str, text: str, kind: str, key: str,
                       absence_id: Optional[str] = None, plan_id: Optional[str] = None) -> dict:
        """Apply the outreach rules, deliver if allowed, and record the outcome in the message log.

        status: sent (accepted by the channel) | queued (allowed, not deliverable yet) |
                blocked (a rule forbids it) | failed (the channel rejected it).
        """
        student = self.students[student_id]
        phone = outreach.phone_for(student)
        reason = outreach.blocked_reason(phone, text, self.household_suppressed(student_id))
        start = utc_now()
        if reason:
            outcome = {"status": "blocked", "reason": reason, "channel": None}
        else:
            outcome = self.sender(student, phone, text, key)
        message = self.log_message(
            absence_id=absence_id, student_id=student_id, direction="outbound", kind=kind,
            language=student.preferred_language, raw_text_or_transcript=text, timestamp=iso(start),
            latency_ms=elapsed_ms(start) if outcome["status"] == "sent" else None,
            phone=phone, plan_id=plan_id, **outcome,
        )
        return self._outbox_entry(message)

    def send_check_in(self, absence_id: str) -> dict:
        """Send the supportive first check-in for an absence that has no reason on file."""
        absence = self.absences[absence_id]
        if absence.reply is not None:
            return {"status": "blocked", "reason": "a reason is already on file for this absence",
                    "absence_id": absence_id}
        return self.send_to_family(absence.student_id, self.check_in_message(absence), "check_in",
                                   key=f"rollcall-checkin:{absence_id}", absence_id=absence_id)

    def queue(self) -> List[dict]:
        """Case queue. Ordered by risk of harm, never by dollar value."""
        rank = {Barrier.SAFETY: 0}
        status_rank = {AbsenceStatus.UNKNOWN: 1, AbsenceStatus.AWAITING_REPLY: 2, AbsenceStatus.REASON_CAPTURED: 3}

        def key(a: Absence):
            barrier = a.classification.barrier if a.classification else None
            return (rank.get(barrier, 1), status_rank[a.status], -a.date.toordinal(), a.id)

        out = []
        for a in sorted(self.absences.values(), key=key):
            s = self.students[a.student_id]
            out.append({**a.model_dump(), "student": s.model_dump(), "absence_count": self.absence_count(s.id),
                        "escalated": a.id in self.escalations})
        return out

    # ---------- clustering ----------
    def clusters(self) -> List[Cluster]:
        return clustering.detect_clusters(self.students, list(self.absences.values()), self.settings,
                                          seed.WEEK_START, seed.WEEK_END)

    # ---------- F4: school action plans ----------
    def draft_plan(self, cluster_id: Optional[str] = None, absence_id: Optional[str] = None) -> ActionPlan:
        if cluster_id:
            cluster = next((c for c in self.clusters() if c.id == cluster_id), None)
            if cluster is None:
                raise KeyError(f"cluster {cluster_id} not found")
            plan = action_plan.for_cluster(cluster, self.students, TODAY)
        elif absence_id:
            absence = self.absences[absence_id]
            if absence_id in self.escalations:
                raise ValueError("safety cases are handled by the counselor, not automated plans")
            barrier = absence.classification.barrier if absence.classification else Barrier.UNKNOWN
            plan = action_plan.for_case(absence_id, barrier, self.students[absence.student_id], TODAY)
        else:
            raise ValueError("cluster_id or absence_id is required")
        existing = self.plans.get(plan.id)
        if existing and existing.status != PlanStatus.DRAFT:
            return existing
        self.plans[plan.id] = plan
        return plan

    def update_plan(self, plan_id: str, fields: dict) -> ActionPlan:
        plan = self.plans[plan_id]
        if plan.status != PlanStatus.DRAFT:
            raise ValueError("only draft plans can be edited")
        editable = {"school_action", "owner", "target_date", "family_request", "message"}
        self.plans[plan_id] = plan.model_copy(update={k: v for k, v in fields.items() if k in editable})
        return self.plans[plan_id]

    def approve_plan(self, plan_id: str, approved_by: str, send: bool = True) -> ActionPlan:
        plan = self.plans[plan_id]
        if plan.status == PlanStatus.DRAFT:
            plan.status = PlanStatus.APPROVED
            plan.approved_by = approved_by
        if send and plan.status == PlanStatus.APPROVED:
            for sid in plan.recipient_student_ids:
                self.send_to_family(sid, action_plan.personalize(plan, self.students[sid]), "plan",
                                    key=f"rollcall-plan:{plan.id}:{sid}", plan_id=plan.id)
            plan.status = PlanStatus.SENT
        return plan

    # ---------- F5: learning plans ----------
    def learning_plan(self, student_id: str) -> LearningPlan:
        student = self.students[student_id]
        mine = [a for a in self.absences.values() if a.student_id == student_id]
        labeled = [a.classification.barrier for a in mine if a.classification]
        barrier = max(set(labeled), key=labeled.count) if labeled else Barrier.UNKNOWN
        return learning_plan.build(student_id, student.grade, [a.date for a in mine], barrier, TODAY)

    def complete_learning_plan(self, student_id: str) -> int:
        """Mark make-up instruction complete; those days count toward Attendance Recovery."""
        days = self.learning_plan(student_id).missed_days
        self.completed_learning[student_id] = days
        return days

    # ---------- F6: revenue ----------
    def revenue(self) -> RevenueSummary:
        recovered = PRIOR_DAYS_RECOVERED + sum(self.completed_learning.values())
        return revenue.summarize(self.absences.values(), self.settings, seed.WEEK_START, seed.WEEK_END, recovered)


store = Store()
