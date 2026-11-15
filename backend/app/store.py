"""In-memory store and the end-to-end flow from PRD 1 section 5.

Absence logged -> family replies -> safety gate -> classify & price ->
cluster or case plan drafted -> clerk approves & sends -> learning plan.

State lives in memory for the hackathon demo; `reset()` reloads the seed.
"""
import csv
import io
import os
import statistics
import threading
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

from .data import seed
from .ml import live
from .models import (
    Absence, AbsenceStatus, ActionPlan, Barrier, Classification, Cluster, Correction, Escalation, FamilyReply,
    LearningPlan, Message, PlanStatus, RevenueSettings, RevenueSummary, Student,
)
from .services import action_plan, corrections, learning_plan, outreach, revenue

TODAY = date(2026, 11, 14)
PRIOR_DAYS_RECOVERED = 14  # recovered through Attendance Recovery before this week (synthetic)
COUNSELOR_ONLY_NOTE = ("Safety: with counselor. Automated messages to this household are paused; "
                       "only the counselor contacts the family.")
SCHOOL_ID = "demo"
RECOVERY_SESSION = "attendance_recovery"  # the only session type that recovers funding (PRD 1 F5)

# DEMO_STORY=live holds back three Route 7 replies so the cluster is not there yet at 9am: the
# demo family's live reply (Photon demo student S002, Nov 11) is the third household that tips it.
LIVE_STORY_PENDING = {("S002", date(2026, 11, 10)), ("S004", date(2026, 11, 11)),
                      ("S005", date(2026, 11, 10)), ("S005", date(2026, 11, 11))}


def demo_story() -> str:
    return "live" if os.environ.get("DEMO_STORY", "full").strip().lower() == "live" else "full"


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
        self.version = 0  # bumped on every change that can move clusters (cache key)
        self._cluster_cache: Optional[Tuple[tuple, List[Cluster], dict]] = None
        self.corrections = corrections.CorrectionStore()
        # Delivery function, set by the Photon channel when it starts. Survives reset().
        # (student, phone, text, idempotency_key) -> {"status": sent | queued | failed, "reason", "channel"}
        self.sender: Callable[[Student, str, str, str], dict] = _no_channel
        self._log_lock = threading.Lock()  # the Photon thread and API requests both append messages
        self.reset()

    def touch(self) -> None:
        self.version += 1

    # ---------- setup ----------
    def reset(self) -> None:
        self.touch()
        self.corrections.reset()
        self.recovery_log: List[dict] = []
        self.students: Dict[str, Student] = {s["id"]: Student(**s) for s in seed.students()}
        self.absences: Dict[str, Absence] = {}
        self.escalations: Dict[str, Escalation] = {}
        self.plans: Dict[str, ActionPlan] = {}
        self.messages: List[Message] = []  # every inbound and outbound text, oldest first
        self.completed_learning: Dict[str, int] = {}  # student_id -> days recovered
        self.settings: RevenueSettings = revenue.STATE_DEFAULTS["CA"].model_copy()
        live_story = demo_story() == "live"
        for rec in seed.absences():
            reply = rec.pop("reply")
            if live_story and (rec["student_id"], rec["date"]) in LIVE_STORY_PENDING:
                reply = None
            self.absences[rec["id"]] = Absence(**rec)
            if reply:  # precomputed history: logged with no latency, labeled without model calls, no cluster lookup
                self.record_reply(rec["id"], FamilyReply(**reply), channel="seed", measure=False, use_model=False,
                                  route=False)

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
        if added:
            self.touch()
        return added, errors

    def record_reply(self, absence_id: str, reply: FamilyReply, *, channel: str = "simulated",
                     received: Optional[datetime] = None, external_id: Optional[str] = None,
                     phone: Optional[str] = None, reason: Optional[str] = None, measure: bool = True,
                     use_model: bool = True, route: bool = True) -> str:
        """Classify a family reply, log it, and return the route: escalated | cluster | case | unknown.

        Classification uses this school's past staff corrections (F8). `use_model=False` skips Gemini
        (seed data); `route=False` skips the cluster lookup (seed load).

        `received` is when the family's message arrived (defaults to now). Latency is measured
        from then until the label is stored. The absence keeps its latest reply; the message
        log keeps every one.
        """
        start = received or utc_now()
        result_route = self._route_reply(absence_id, reply, use_model, route)
        absence = self.absences[absence_id]
        detected = absence.classification.language if absence.classification else None
        self.log_message(
            absence_id=absence_id, student_id=absence.student_id, direction="inbound", channel=channel,
            language=detected or reply.language, raw_text_or_transcript=reply.text,
            timestamp=iso(start) if measure else reply.received_at,
            latency_ms=elapsed_ms(start) if measure else None,
            route=result_route, classification=absence.classification, phone=phone, external_id=external_id,
            reason=reason,
        )
        return result_route

    def _route_reply(self, absence_id: str, reply: FamilyReply, use_model: bool = True,
                     route: bool = True) -> str:
        """Run the safety gate and classify."""
        absence = self.absences[absence_id]
        if not reply.message_id:
            reply = reply.model_copy(update={"message_id": f"M-{absence_id}"})
        absence.reply = reply
        self.touch()
        result, _info = corrections.classify(reply.text, reply.language, self.corrections, SCHOOL_ID,
                                             names=self.names_for(absence.student_id), allow_model=use_model)
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
        if not route:
            return "case"
        in_cluster = any(absence_id in c.absence_ids for c in self.clusters())
        return "cluster" if in_cluster else "case"

    # ---------- F8: learning from corrections ----------
    def correct(self, absence_id: str, label: Barrier, corrected_by: str) -> Tuple[Absence, Correction]:
        """Store a staff correction and re-label the absence. Only a counselor can downgrade a safety label
        (PRD 2 section 3.5); a correction to safety opens a counselor escalation."""
        absence = self.absences[absence_id]
        old = absence.classification.barrier if absence.classification else None
        if not (corrected_by or "").strip():
            raise ValueError("corrected_by is required")
        is_counselor = "counselor" in corrected_by.lower()
        if old == Barrier.SAFETY and label != Barrier.SAFETY and not is_counselor:
            raise PermissionError("only a counselor can downgrade a safety label")
        text = absence.reply.text if absence.reply else ""
        lang = absence.reply.language if absence.reply else "en"
        correction = self.corrections.add(corrections.make_correction(
            absence_id, old, label, corrected_by.strip(), text, lang, SCHOOL_ID))
        absence.classification = Classification(
            barrier=label, confidence=1.0, evidence=text.strip(), safety_critical=label == Barrier.SAFETY,
            source="staff_correction", correction_ref=absence_id)
        if label == Barrier.SAFETY:
            absence.status = AbsenceStatus.REASON_CAPTURED
            if absence_id not in self.escalations or self.escalations[absence_id].status != "open":
                self.escalations[absence_id] = Escalation(
                    absence_id=absence_id, student_id=absence.student_id, reason=label.value,
                    evidence=text.strip() or "(staff correction)")
        else:
            absence.status = AbsenceStatus.UNKNOWN if label == Barrier.UNKNOWN else AbsenceStatus.REASON_CAPTURED
            esc = self.escalations.get(absence_id)
            if esc and esc.status == "open":  # a counselor downgraded it (checked above)
                esc.status = f"closed by {corrected_by.strip()}"
        self.touch()
        return absence, correction

    def names_for(self, student_id: str) -> List[str]:
        """Names to hide from the model for this student's messages: student, guardian, teacher."""
        s = self.students[student_id]
        return [s.first_name, s.last_name, *s.guardian_name.split(), s.teacher.split()[-1]]

    def absence_count(self, student_id: str) -> int:
        return sum(1 for a in self.absences.values() if a.student_id == student_id)

    # ---------- safety gate (PRD 3 section 3) ----------
    def suppressed_student_ids(self) -> Set[str]:
        """Students no automated or drafted message may reach: anyone with an open safety
        escalation, plus everyone in that student's household. Only the counselor contacts them."""
        flagged = {e.student_id for e in self.escalations.values() if e.status == "open"}
        households = {self.students[sid].household_id for sid in flagged if sid in self.students}
        return {sid for sid, s in self.students.items() if sid in flagged or s.household_id in households}

    def is_suppressed(self, student_id: str) -> bool:
        return student_id in self.suppressed_student_ids()

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
        s = self.students[absence.student_id]
        if self.is_suppressed(s.id):
            return COUNSELOR_ONLY_NOTE
        return outreach.check_in_text(s, absence.date)

    # ---------- F2: outbound messages ----------
    def household_suppressed(self, student_id: str) -> bool:
        """True while any student in this household has an open safety escalation."""
        return self.is_suppressed(student_id)  # one rule: see suppressed_student_ids()

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

    # ---------- clustering (PRD 2 section 3.2, via app.ml.live) ----------
    def _detect(self) -> Tuple[List[Cluster], dict]:
        key = (self.version, live.config_key())
        if self._cluster_cache is None or self._cluster_cache[0] != key:
            # Students with the counselor stay visible to staff in clusters (draft_plan withholds them from
            # messages); live drops their safety-labeled absences so no cluster cites a disclosure.
            found, meta = live.detect(self.students, list(self.absences.values()), self.settings,
                                      seed.WEEK_START, seed.WEEK_END)
            self._cluster_cache = (key, found, {**meta, "version": self.version})
        return self._cluster_cache[1], self._cluster_cache[2]

    def clusters(self) -> List[Cluster]:
        return self._detect()[0]

    def cluster_meta(self) -> dict:
        return self._detect()[1]

    def update_settings(self, settings: RevenueSettings) -> None:
        self.settings = settings
        self.touch()  # dollar figures on clusters change

    # ---------- F4: school action plans ----------
    def draft_plan(self, cluster_id: Optional[str] = None, absence_id: Optional[str] = None) -> ActionPlan:
        if cluster_id:
            cluster = next((c for c in self.clusters() if c.id == cluster_id), None)
            if cluster is None:
                raise KeyError(f"cluster {cluster_id} not found")
            if cluster.dataset != "demo":
                raise ValueError("clusters from the simulated dataset have no real families; plans are disabled")
            suppressed = self.suppressed_student_ids()
            withheld = [sid for sid in cluster.student_ids if sid in suppressed]
            reachable = [sid for sid in cluster.student_ids if sid not in suppressed]
            if not reachable:
                raise ValueError("every family in this cluster is with the counselor; no automated plan")
            plan = action_plan.for_cluster(cluster.model_copy(update={"student_ids": reachable}),
                                           self.students, TODAY)
            if withheld:
                # Staff-facing note only; never names the student or the disclosure.
                n = len(withheld)
                plan.school_action += (f" Note: {n} student{'s' if n != 1 else ''} in this cluster "
                                       f"{'are' if n != 1 else 'is'} with the counselor (safety) and "
                                       f"excluded from this message.")
        elif absence_id:
            absence = self.absences[absence_id]
            if absence_id in self.escalations or self.is_suppressed(absence.student_id):
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
        # Ties broken by the Barrier enum order so the plan does not depend on set iteration order.
        barrier = max(sorted(set(labeled), key=list(Barrier).index), key=labeled.count) if labeled else Barrier.UNKNOWN
        return learning_plan.build(student_id, student.grade, [a.date for a in mine], barrier, TODAY,
                                   language=student.preferred_language)

    def complete_learning_plan(self, student_id: str, session_type: str = RECOVERY_SESSION,
                               supervised_by: Optional[str] = None) -> dict:
        """Record completed make-up instruction. PRD 1 F5: only a supervised Attendance Recovery session
        for an eligible plan (2+ missed days) counts toward recovered funding. Take-home work or an
        unsupervised session is logged but recovers 0 days. Counting is idempotent per student (the
        plan's missed days, at most once)."""
        plan = self.learning_plan(student_id)
        supervised = bool((supervised_by or "").strip())
        if not plan.eligible:
            counted, reason = False, f"plan not eligible: fewer than {learning_plan.MIN_MISSED_DAYS} missed days"
        elif session_type != RECOVERY_SESSION:
            counted, reason = False, f"'{session_type}' is not an Attendance Recovery session"
        elif not supervised:
            counted, reason = False, "Attendance Recovery counts only when supervised (supervised_by is required)"
        else:
            counted, reason = True, "supervised Attendance Recovery session"
        days = plan.missed_days if counted else 0
        if counted:
            self.completed_learning[student_id] = days
        self.recovery_log.append({"student_id": student_id, "session_type": session_type,
                                  "supervised_by": (supervised_by or "").strip() or None, "counted": counted,
                                  "days": days, "reason": reason,
                                  "at": datetime.now().isoformat(timespec="seconds")})
        return {"days_recovered": days, "counted": counted, "reason": reason,
                "total_days_recovered_for_student": self.completed_learning.get(student_id, 0)}

    # ---------- F6: revenue ----------
    def revenue(self) -> RevenueSummary:
        recovered = PRIOR_DAYS_RECOVERED + sum(self.completed_learning.values())
        return revenue.summarize(self.absences.values(), self.settings, seed.WEEK_START, seed.WEEK_END, recovered)


store = Store()
