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
    Absence, AbsenceStatus, ActionPlan, Barrier, CaseReport, Classification, Cluster, Escalation, FamilyReply,
    LearningPlan, Message, PlanStatus, RevenueSettings, RevenueSummary, Student,
)
from .services import action_plan, case_report, classifier, clustering, learning_plan, outreach, revenue

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
        self.case_reports: Dict[str, List[CaseReport]] = {}  # escalation_id -> saved versions, oldest first
        self.plans: Dict[str, ActionPlan] = {}
        self.messages: List[Message] = []  # every inbound and outbound text, oldest first
        self.completed_learning: Dict[str, int] = {}  # student_id -> days recovered
        self.settings: RevenueSettings = revenue.STATE_DEFAULTS["CA"].model_copy()
        for rec in seed.absences():
            reply = rec.pop("reply")
            self.absences[rec["id"]] = Absence(**rec)
            if reply:  # precomputed history: logged with no latency, and labeled without model calls
                self.record_reply(rec["id"], FamilyReply(**reply), channel="seed", measure=False, use_model=False)

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
                     phone: Optional[str] = None, reason: Optional[str] = None, measure: bool = True,
                     use_model: bool = True) -> str:
        """Classify a family reply, log it, and return the route: escalated | cluster | case | unknown.

        `received` is when the family's message arrived (defaults to now). Latency is measured
        from then until the label is stored. The absence keeps its latest reply; the message
        log keeps every one.
        """
        start = received or utc_now()
        at = iso(start) if measure else reply.received_at
        route, result = self._route_reply(absence_id, reply, at, use_model)
        absence = self.absences[absence_id]
        self.log_message(
            absence_id=absence_id, student_id=absence.student_id, direction="inbound", channel=channel,
            language=result.language or reply.language, raw_text_or_transcript=reply.text,
            timestamp=at, latency_ms=elapsed_ms(start) if measure else None,
            # The message keeps its own label, even when an open escalation holds the absence's label.
            route=route, classification=result, phone=phone, external_id=external_id, reason=reason,
        )
        return route

    def _route_reply(self, absence_id: str, reply: FamilyReply, at: str,
                     use_model: bool = True) -> Tuple[str, Classification]:
        """Run the safety gate and classify. Returns the route and this message's own label."""
        absence = self.absences[absence_id]
        absence.reply = reply
        result: Classification = (classifier.classify(reply.text, self.names_for(absence.student_id)) if use_model
                                  else classifier.keyword_classify(reply.text))
        already_open = self.open_escalation(absence_id)

        if result.safety_critical:
            # Counselor takes over immediately; no automated reply is ever sent.
            absence.classification = result
            absence.status = AbsenceStatus.REASON_CAPTURED
            if already_open is None:
                eid = f"E{len(self.escalations) + 1:03d}"
                self.escalations[eid] = Escalation(
                    escalation_id=eid, absence_id=absence_id, student_id=absence.student_id,
                    household_id=self.students[absence.student_id].household_id,
                    reason=result.barrier.value, evidence=result.evidence, opened_at=at,
                )
            return "escalated", result

        if already_open is not None:
            # Only the counselor can downgrade a safety label. A calmer follow-up ("she twisted
            # her ankle at soccer") is logged with its own label but does not replace this one.
            return "escalated", result

        absence.classification = result
        if result.barrier == Barrier.UNKNOWN:
            absence.status = AbsenceStatus.UNKNOWN
            return "unknown", result

        absence.status = AbsenceStatus.REASON_CAPTURED
        in_cluster = any(absence_id in c.absence_ids for c in self.clusters())
        return ("cluster" if in_cluster else "case"), result

    # ---------- safety escalations ----------
    ACK_WITHIN_MINUTES = 15  # PRD 3 section 3: acknowledge within 15 minutes during school hours

    def open_escalation(self, absence_id: str) -> Optional[Escalation]:
        """The escalation for this absence that the counselor has not closed yet, if any."""
        return next((e for e in self.escalations.values()
                     if e.absence_id == absence_id and e.status != "closed"), None)

    def escalation_view(self, e: Escalation) -> dict:
        """An escalation as the inbox shows it, with how long it has waited for acknowledgement."""
        opened = datetime.fromisoformat(e.opened_at.replace("Z", "+00:00"))
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        waited = (utc_now() - opened).total_seconds() / 60
        saved = self.case_reports.get(e.escalation_id, [])
        return {**e.model_dump(), "student": self.students[e.student_id],
                "minutes_open": max(0, round(waited)),
                "overdue": e.status == "open" and waited > self.ACK_WITHIN_MINUTES,
                "case_report_versions": len(saved),
                "case_report_saved_at": saved[-1].saved_at if saved else None}

    def _escalation(self, escalation_id: str) -> Escalation:
        e = self.escalations[escalation_id]  # KeyError -> 404 in the API
        if e.status == "closed":
            raise ValueError("this escalation is already closed")
        return e

    # ---------- internal case reports ----------
    def draft_case_report(self, escalation_id: str, by: str) -> str:
        """A fresh report from the current record. Not saved. Works on open and closed cases.

        Keeps the counselor notes from the latest saved version, so regenerating after new
        events does not lose what the counselor wrote.
        """
        e = self.escalations[escalation_id]  # KeyError -> 404
        saved = self.case_reports.get(escalation_id, [])
        notes = case_report.notes_from(saved[-1].text) if saved else None
        return case_report.build(e, self.students[e.student_id], self.absences[e.absence_id],
                                 self.thread(e.absence_id), by, iso(utc_now()), notes)

    def save_case_report(self, escalation_id: str, by: str, text: str) -> CaseReport:
        """Store a new version. Earlier versions are kept unchanged."""
        if escalation_id not in self.escalations:
            raise KeyError(escalation_id)
        if not text.strip():
            raise ValueError("the report is empty")
        versions = self.case_reports.setdefault(escalation_id, [])
        report = CaseReport(escalation_id=escalation_id, version=len(versions) + 1, text=text,
                            saved_by=by, saved_at=iso(utc_now()))
        versions.append(report)
        return report

    def acknowledge_escalation(self, escalation_id: str, by: str) -> Escalation:
        e = self._escalation(escalation_id)
        if e.acknowledged_at is None:
            e.acknowledged_at, e.acknowledged_by, e.status = iso(utc_now()), by, "acknowledged"
        return e

    def record_report_decision(self, escalation_id: str, by: str, decision: str) -> Escalation:
        """Record what staff did about mandated reporting. The system never files a report itself."""
        if decision not in ("report_filed", "not_required"):
            raise ValueError("decision must be report_filed or not_required")
        e = self._escalation(escalation_id)
        e.report_status, e.report_by, e.report_at = decision, by, iso(utc_now())
        return e

    def close_escalation(self, escalation_id: str, by: str, outcome: str,
                         relabel_to: Optional[Barrier] = None, note: Optional[str] = None) -> Escalation:
        """Close a safety case. Lifts the household's message suppression if nothing else is open.

        outcome "resolved": a real concern the counselor handled; the absence keeps its safety label.
        outcome "not_safety": a false alarm; the counselor relabels the absence (`relabel_to`).
        """
        e = self._escalation(escalation_id)
        if e.report_status is None:
            raise ValueError("record the mandated-reporting decision before closing")
        if outcome not in ("resolved", "not_safety"):
            raise ValueError("outcome must be resolved or not_safety")
        if outcome == "not_safety":
            if relabel_to is None or relabel_to == Barrier.SAFETY:
                raise ValueError("a false alarm needs the correct, non-safety label")
            absence = self.absences[e.absence_id]
            previous = absence.classification
            absence.classification = Classification(
                barrier=relabel_to, confidence=1.0, evidence=previous.evidence if previous else e.evidence,
                source="counselor", language=previous.language if previous else None,
            )
            absence.status = AbsenceStatus.UNKNOWN if relabel_to == Barrier.UNKNOWN else AbsenceStatus.REASON_CAPTURED
            e.relabeled_to = relabel_to
        now = iso(utc_now())
        if e.acknowledged_at is None:
            e.acknowledged_at, e.acknowledged_by = now, by
        e.status, e.closed_at, e.closed_by, e.outcome, e.note = "closed", now, by, outcome, note
        return e

    def names_for(self, student_id: str) -> List[str]:
        """Names to hide from the model for this student's messages: student, guardian, teacher."""
        s = self.students[student_id]
        return [s.first_name, s.last_name, *s.guardian_name.split(), s.teacher.split()[-1]]

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
        """True while any student in this household has a safety escalation the counselor has not closed."""
        household = self.students[student_id].household_id
        return any(e.status != "closed" and e.household_id == household for e in self.escalations.values())

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
        status_rank = {AbsenceStatus.UNKNOWN: 1, AbsenceStatus.AWAITING_REPLY: 2, AbsenceStatus.REASON_CAPTURED: 3}
        escalated = {e.absence_id for e in self.escalations.values() if e.status != "closed"}

        def key(a: Absence):
            # Open safety cases first. Once the counselor closes one, it is ordered like any other case.
            return (0 if a.id in escalated else 1, status_rank[a.status], -a.date.toordinal(), a.id)

        out = []
        for a in sorted(self.absences.values(), key=key):
            s = self.students[a.student_id]
            out.append({**a.model_dump(), "student": s.model_dump(), "absence_count": self.absence_count(s.id),
                        "escalated": a.id in escalated})
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
            barrier = absence.classification.barrier if absence.classification else Barrier.UNKNOWN
            if self.open_escalation(absence_id) or barrier == Barrier.SAFETY:
                raise ValueError("safety cases are handled by the counselor, not automated plans")
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
