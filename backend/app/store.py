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
    Absence, AbsenceStatus, ActionPlan, Barrier, CaseReport, Classification, Page, Staff, Cluster, Correction, Escalation, FamilyReply,
    LearningPlan, Message, PlanStatus, RevenueSettings, RevenueSummary, Student,
)
from .services import action_plan, case_report, classifier, corrections, learning_plan, oncall, outreach, revenue

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


def data_source() -> str:
    """Where the school's data comes from.

    demo: the hard-coded synthetic school in data/seed.py, in memory only (never saved).
    live: starts empty; students and absences come from SIS CSV imports and are saved to
          TigerData when TIGER_DATABASE_URL is set. Demo mode (DEMO_MODE=1) always uses demo.
    """
    if os.environ.get("DEMO_MODE", "0").strip().lower() in ("1", "true", "on", "yes"):
        return "demo"
    return "live" if os.environ.get("DATA_SOURCE", "demo").strip().lower() == "live" else "demo"


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


def _no_staff_channel(phone: str, text: str, key: str) -> dict:
    """Default page delivery when no messaging channel is running."""
    return {"status": "queued", "reason": "messaging channel is off", "channel": None}


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
        # Page delivery to staff phones, set by the Photon channel. (phone, text, key) -> outcome
        self.staff_sender: Callable[[str, str, str], dict] = _no_staff_channel
        # On-call roster: staff configuration, saved to disk, kept across reset().
        self.roster = oncall.Roster()
        self.reset()

    def touch(self) -> None:
        self.version += 1

    # ---------- setup ----------
    def reset(self) -> None:
        self.touch()
        self.corrections.reset()
        self.recovery_log: List[dict] = []
        live_data = data_source() == "live"
        self.students: Dict[str, Student] = {} if live_data else {s["id"]: Student(**s) for s in seed.students()}
        self.absences: Dict[str, Absence] = {}
        self.escalations: Dict[str, Escalation] = {}
        self.case_reports: Dict[str, List[CaseReport]] = {}  # escalation_id -> saved versions, oldest first
        self.plans: Dict[str, ActionPlan] = {}
        self.messages: List[Message] = []  # every inbound and outbound text, oldest first
        self.demo_seen: Set[str] = set()  # demo mode: "view" steps the presenter has shown (services/demo.py)
        self._held_keys: Dict[str, str] = {}  # message_id -> idempotency key, for messages held for quiet hours
        self.completed_learning: Dict[str, int] = {}  # student_id -> days recovered
        self.settings: RevenueSettings = revenue.STATE_DEFAULTS["CA"].model_copy()
        if live_data:
            return  # real data arrives through CSV imports (and is loaded from TigerData on startup)
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
    # Student columns an SIS export may carry. When present, unknown students are added and known ones updated.
    STUDENT_COLUMNS = ("first_name", "last_name", "grade", "classroom", "teacher", "bus_route", "household_id",
                       "preferred_language", "guardian_name", "guardian_phone")
    STUDENT_REQUIRED = ("first_name", "last_name", "grade", "classroom", "teacher", "household_id",
                        "guardian_name", "guardian_phone")

    def _student_from_row(self, sid: str, row: Dict[str, str]) -> Optional[Student]:
        """The student described by a CSV row, merged over what we already have. None if the row has no student details."""
        given = {k: (row.get(k) or "").strip() for k in self.STUDENT_COLUMNS if (row.get(k) or "").strip()}
        if not given:
            return None
        current = self.students[sid].model_dump() if sid in self.students else {}
        merged = {**current, **given, "id": sid}
        if "bus_route" in row and not (row.get("bus_route") or "").strip():
            merged["bus_route"] = None  # an empty bus_route column means the student does not ride a bus
        missing = [k for k in self.STUDENT_REQUIRED if not merged.get(k)]
        if missing:
            raise ValueError(f"new student needs {', '.join(missing)}")
        return Student.model_validate(merged)

    def import_csv(self, text: str) -> Tuple[int, List[str], int]:
        """Ingest an SIS export: one absence per row.

        Required columns: student_id, date. Optional: sis_code, and the student columns in
        STUDENT_COLUMNS, which add a student we have not seen or update one we have. Without student
        columns, every student_id must already be known. A row with student columns and an empty date is a roster row: it
        adds or updates the student without logging an absence. Returns (absences added, errors, students added).
        """
        added, errors, new_students = 0, [], 0
        reader = csv.DictReader(io.StringIO(text.strip().lstrip("﻿")))
        missing = {"student_id", "date"} - set(reader.fieldnames or [])
        if missing:
            return 0, [f"missing column(s): {', '.join(sorted(missing))}"], 0
        existing = {(a.student_id, a.date) for a in self.absences.values()}
        for line, row in enumerate(reader, start=2):
            sid = (row.get("student_id") or "").strip()
            if not sid:
                errors.append(f"line {line}: empty student_id")
                continue
            try:
                student = self._student_from_row(sid, row)
            except ValueError as e:
                errors.append(f"line {line}: student '{sid}': {str(e).splitlines()[0]}")
                continue
            if student is not None:
                if sid not in self.students:
                    new_students += 1
                if self.students.get(sid) != student:
                    self.students[sid] = student
                    self.touch()
            if sid not in self.students:
                errors.append(f"line {line}: unknown student_id '{sid}' (add the student columns to create it)")
                continue
            if student is not None and not (row.get("date") or "").strip():
                continue  # a roster row: adds or updates the student, logs no absence
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
        return added, errors, new_students

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
        at = iso(start) if measure else reply.received_at
        result_route, result = self._route_reply(absence_id, reply, at, use_model, route)
        absence = self.absences[absence_id]
        self.log_message(
            absence_id=absence_id, student_id=absence.student_id, direction="inbound", channel=channel,
            language=result.language or reply.language, raw_text_or_transcript=reply.text,
            timestamp=at, latency_ms=elapsed_ms(start) if measure else None,
            # The message keeps its own label, even when an open escalation holds the absence's label.
            route=result_route, classification=result, phone=phone, external_id=external_id, reason=reason,
        )
        if result_route == "escalated" and channel != "seed":
            escalation = self.open_escalation(absence_id)
            if escalation is not None:
                seen = set(escalation.safety_types) | set(result.safety_types)
                escalation.safety_types = [t for t in classifier.SAFETY_TYPES if t in seen]
                self._maybe_send_crisis(escalation, result)
                if not escalation.pages:
                    self._page(escalation, oncall.CHAIN[oncall.first_role(outreach.school_now())], "first")
        return result_route

    # ---------- paging staff about safety cases ----------
    def _page(self, e: Escalation, chain: List[str], level: str) -> None:
        """Page everyone on call for the first role in `chain` that has someone on call."""
        at = iso(outreach.school_now())
        targets = self.roster.targets(chain)
        if not targets:
            e.pages.append(Page(level=level, at=at, status="blocked",
                                reason=f"no one on call ({', '.join(chain)}); add staff on the On-call page"))
            return
        text = oncall.page_text(e.escalation_id, self.students[e.student_id].grade, escalated=(level == "escalated"))
        for staff in targets:
            reason = outreach.blocked_reason(staff.phone, text, household_suppressed=False)
            if reason:
                outcome = {"status": "blocked", "reason": reason, "channel": None}
            else:
                outcome = self.staff_sender(staff.phone, text, f"rollcall-page:{e.escalation_id}:{level}:{staff.staff_id}")
            e.pages.append(Page(staff_id=staff.staff_id, name=staff.name, role=staff.role, phone=staff.phone,
                                level=level, at=at, status=outcome["status"], reason=outcome.get("reason")))
            self.log_message(direction="outbound", kind="page", channel=outcome.get("channel"),
                             raw_text_or_transcript=text, timestamp=at, status=outcome["status"],
                             reason=outcome.get("reason"), phone=staff.phone)

    def check_paging(self) -> int:
        """Page the principal about cases still unacknowledged 15 minutes after the first page.

        Runs on every Photon poll and whenever the escalations list is read. Returns how many
        cases were escalated.
        """
        now = outreach.school_now()
        count = 0
        for e in list(self.escalations.values()):
            if e.status != "open" or not e.pages or any(pg.level == "escalated" for pg in e.pages):
                continue
            first = datetime.fromisoformat(e.pages[0].at.replace("Z", "+00:00"))
            if (now - first).total_seconds() > self.ACK_WITHIN_MINUTES * 60:
                self._page(e, oncall.CHAIN["principal"], "escalated")
                count += 1
        return count

    def staff_command(self, phone: str, text: str) -> Optional[str]:
        """Handle a text from a staff phone. Returns the reply, or None if it is not a staff command.

        "ACK" acknowledges the oldest open case this person was paged about; "ACK E004" names one.
        Anything else returns None, so the same phone can also be used to play a family in testing.
        """
        staff = self.roster.by_phone(phone)
        words = outreach.plain(text).split()
        if staff is None or not words or words[0] not in ("ack", "acknowledge", "acknowledged"):
            return None
        named = words[1].upper() if len(words) > 1 else None
        paged = [e for e in self.escalations.values() if e.status == "open"
                 and any(pg.staff_id == staff.staff_id for pg in e.pages)]
        if named:
            target = next((e for e in self.escalations.values() if e.escalation_id == named), None)
            if target is None:
                return f"There is no case {named}."
            if target.status != "open":
                return f"{named} is already acknowledged."
        elif paged:
            target = sorted(paged, key=lambda e: e.opened_at)[0]
        else:
            return "You have no unacknowledged cases."
        self.acknowledge_escalation(target.escalation_id, staff.name)
        return f"Acknowledged {target.escalation_id}. Details are on the Escalations page."

    def _maybe_send_crisis(self, e: Escalation, result: Classification) -> None:
        """Send the fixed crisis-resource message once per case, if the case calls for it."""
        if e.crisis_message_status in ("sent", "withheld"):
            return  # sent once already, or abuse was disclosed: never again for this case
        decision = outreach.crisis_decision(e.safety_types)
        if decision:
            if "abuse_or_neglect" in e.safety_types:
                e.crisis_message_status, e.crisis_message_reason = "withheld", decision
            return
        student = self.students[e.student_id]
        entry = self.send_to_family(
            e.student_id, outreach.crisis_text(result.language, student.preferred_language), "crisis",
            key=f"rollcall-crisis:{e.escalation_id}", absence_id=e.absence_id,
            language=result.language or student.preferred_language, urgent=True)
        e.crisis_message_status, e.crisis_message_reason = entry["status"], entry["reason"]
        e.crisis_message_at = entry["sent_at"]

    def _route_reply(self, absence_id: str, reply: FamilyReply, at: str, use_model: bool = True,
                     route: bool = True) -> Tuple[str, Classification]:
        """Run the safety gate and classify. Returns the route and this message's own label.

        Classification goes through this school's past staff corrections (F8). `route=False` skips
        the cluster lookup (seed load).
        """
        absence = self.absences[absence_id]
        if not reply.message_id:
            reply = reply.model_copy(update={"message_id": f"M-{absence_id}"})
        absence.reply = reply
        self.touch()  # clusters may change
        result, _info = corrections.classify(reply.text, reply.language, self.corrections, SCHOOL_ID,
                                             names=self.names_for(absence.student_id), allow_model=use_model)
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
                    safety_types=result.safety_types,
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
        if not route:
            return "case", result
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
            text = absence.reply.text if absence.reply else ""
            self.corrections.add(corrections.make_correction(
                e.absence_id, Barrier.SAFETY, relabel_to, by, text,
                absence.reply.language if absence.reply else "en", SCHOOL_ID))
            absence.classification = absence.classification.model_copy(update={"correction_ref": e.absence_id})
            self.touch()
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

    # ---------- F8: learning from corrections ----------
    def correct(self, absence_id: str, label: Barrier, corrected_by: str) -> Tuple[Absence, Correction]:
        """Store a staff correction and re-label the absence (F8).

        A correction to safety opens an escalation (and pages on-call staff). A safety label with an
        open escalation cannot be corrected here: the counselor closes the case on the Escalations
        page (reporting decision, then "Not a safety concern"), which records the correction. A closed
        case's safety label can be downgraded by a counselor only (PRD 2 section 3.5).
        """
        absence = self.absences[absence_id]
        old = absence.classification.barrier if absence.classification else None
        if not (corrected_by or "").strip():
            raise ValueError("corrected_by is required")
        if old == Barrier.SAFETY and label != Barrier.SAFETY:
            if self.open_escalation(absence_id):
                raise PermissionError("this safety case is open: the counselor closes it on the Escalations page")
            if "counselor" not in corrected_by.lower():
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
            if self.open_escalation(absence_id) is None:
                eid = f"E{len(self.escalations) + 1:03d}"
                escalation = Escalation(
                    escalation_id=eid, absence_id=absence_id, student_id=absence.student_id,
                    household_id=self.students[absence.student_id].household_id, reason=label.value,
                    evidence=text.strip() or "(staff correction)", opened_at=iso(utc_now()))
                self.escalations[eid] = escalation
                self._page(escalation, oncall.CHAIN[oncall.first_role(outreach.school_now())], "first")
        else:
            absence.status = AbsenceStatus.UNKNOWN if label == Barrier.UNKNOWN else AbsenceStatus.REASON_CAPTURED
        self.touch()
        return absence, correction

    # ---------- safety gate (PRD 3 section 3) ----------
    def suppressed_student_ids(self) -> Set[str]:
        """Students no automated or drafted message may reach: anyone with an open safety
        escalation, plus everyone in that student's household. Only the counselor contacts them."""
        flagged = {e.student_id for e in self.escalations.values() if e.status != "closed"}
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
        """Outbound messages to families, in the shape the Plans page reads. Staff pages are not
        family messages: they show on the escalation card and in the message log."""
        return [self._outbox_entry(m) for m in self.messages if m.direction == "outbound" and m.kind != "page"]

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
        """True while any student in this household has a safety escalation the counselor has not closed."""
        household = self.students[student_id].household_id
        return any(e.status != "closed" and e.household_id == household for e in self.escalations.values())

    def send_to_family(self, student_id: str, text: str, kind: str, key: str,
                       absence_id: Optional[str] = None, plan_id: Optional[str] = None,
                       language: Optional[str] = None, urgent: bool = False) -> dict:
        """Apply the outreach rules, deliver if allowed, and record the outcome in the message log.

        status: sent (accepted by the channel) | queued (allowed, not deliverable yet, or held for
                quiet hours) | blocked (a rule forbids it) | failed (the channel rejected it).
        `urgent` is only for the crisis message: it is not held for quiet hours and is the one
        message allowed through an open safety escalation. Opt-outs and the no-link rule still apply.
        """
        student = self.students[student_id]
        phone = outreach.phone_for(student)
        suppressed = False if urgent else self.household_suppressed(student_id)
        reason = outreach.blocked_reason(phone, text, suppressed)
        until = None if urgent or reason else outreach.quiet_until(outreach.school_now())
        start = utc_now()
        if reason:
            outcome = {"status": "blocked", "reason": reason, "channel": None}
        elif until:
            outcome = {"status": "queued", "channel": None,
                       "reason": f"quiet hours: will send at {until.strftime('%-I:%M %p %b %d')} school time"}
        else:
            outcome = self.sender(student, phone, text, key)
        message = self.log_message(
            absence_id=absence_id, student_id=student_id, direction="outbound", kind=kind,
            language=language or student.preferred_language, raw_text_or_transcript=text, timestamp=iso(start),
            latency_ms=elapsed_ms(start) if outcome["status"] == "sent" else None,
            phone=phone, plan_id=plan_id, deliver_after=iso(until) if until else None, **outcome,
        )
        if until:
            self._held_keys[message.message_id] = key
        return self._outbox_entry(message)

    def flush_due(self) -> int:
        """Send messages held for quiet hours whose time has come. Returns how many were processed.

        The rules are checked again first: a household may have been escalated, or a family may
        have opted out, while the message waited.
        """
        school_now = outreach.school_now()
        if outreach.quiet_until(school_now):
            return 0
        now = school_now.astimezone(timezone.utc)
        # With quiet hours turned off (QUIET_HOURS=off), everything held is released now, not at its 8am time.
        release_all = not outreach.quiet_hours_enabled()
        with self._log_lock:
            due = [m for m in self.messages if m.status == "queued" and m.deliver_after
                   and (release_all or datetime.fromisoformat(m.deliver_after.replace("Z", "+00:00")) <= now)]
        for m in due:
            student = self.students[m.student_id]
            phone = outreach.phone_for(student)
            reason = outreach.blocked_reason(phone, m.raw_text_or_transcript, self.household_suppressed(student.id))
            if reason:
                outcome = {"status": "blocked", "reason": reason, "channel": None}
            else:
                key = self._held_keys.pop(m.message_id, f"rollcall-held:{m.message_id}")
                outcome = self.sender(student, phone, m.raw_text_or_transcript, key)
            m.status, m.reason, m.channel, m.deliver_after = outcome["status"], outcome["reason"], outcome["channel"], None
        return len(due)

    def send_check_in(self, absence_id: str) -> dict:
        """Send the supportive first check-in for an absence that has no reason on file."""
        absence = self.absences[absence_id]
        if absence.reply is not None:
            return {"status": "blocked", "reason": "a reason is already on file for this absence",
                    "absence_id": absence_id}
        return self.send_to_family(absence.student_id, self.check_in_message(absence), "check_in",
                                   key=f"rollcall-checkin:{absence_id}", absence_id=absence_id)

    def check_in_targets(self) -> List[Absence]:
        """Absences awaiting a reply whose family has not been checked in with yet.

        Skipped: a check-in that was sent, one held for quiet hours (it goes out at 8am by itself), and
        a family a rule blocks right now (open safety case, opted out); they count again once it lifts.
        Tried again: one the channel could not deliver yet (the family has not texted the line) or that failed.
        """
        with self._log_lock:
            done = {m.absence_id for m in self.messages
                    if m.kind == "check_in" and m.direction == "outbound"
                    and (m.status == "sent" or (m.status == "queued" and m.deliver_after))}

        def blocked(a: Absence) -> bool:
            student = self.students[a.student_id]
            return outreach.blocked_reason(outreach.phone_for(student), self.check_in_message(a),
                                           self.household_suppressed(student.id)) is not None

        return [a for a in self.absences.values()
                if a.reply is None and a.status == AbsenceStatus.AWAITING_REPLY and a.id not in done and not blocked(a)]

    def send_check_ins(self) -> dict:
        """Send the first check-in to every family in `check_in_targets`. Returns counts by outcome."""
        counts = {"sent": 0, "queued": 0, "blocked": 0, "failed": 0}
        reasons: Dict[str, int] = {}
        for a in self.check_in_targets():
            out = self.send_check_in(a.id)
            counts[out["status"]] = counts.get(out["status"], 0) + 1
            if out["status"] != "sent" and out.get("reason"):
                reasons[out["reason"]] = reasons.get(out["reason"], 0) + 1
        return {"attempted": sum(counts.values()), **counts, "reasons": reasons}

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
            barrier = absence.classification.barrier if absence.classification else Barrier.UNKNOWN
            if self.open_escalation(absence_id) or barrier == Barrier.SAFETY or self.is_suppressed(absence.student_id):
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
