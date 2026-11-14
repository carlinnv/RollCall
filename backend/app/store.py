"""In-memory store and the end-to-end flow from PRD 1 section 5.

Absence logged -> family replies -> safety gate -> classify & price ->
cluster or case plan drafted -> clerk approves & sends -> learning plan.

State lives in memory for the hackathon demo; `reset()` reloads the seed.
"""
import csv
import io
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple

from .data import seed
from .ml import live
from .models import (
    Absence, AbsenceStatus, ActionPlan, Barrier, Classification, Cluster, Correction, Escalation, FamilyReply,
    LearningPlan, PlanStatus, RevenueSettings, RevenueSummary, Student,
)
from .services import action_plan, corrections, learning_plan, revenue

TODAY = date(2026, 11, 14)
PRIOR_DAYS_RECOVERED = 14  # recovered through Attendance Recovery before this week (synthetic)
COUNSELOR_ONLY_NOTE = ("Safety: with counselor. Automated messages to this household are paused; "
                       "only the counselor contacts the family.")
SCHOOL_ID = "demo"
RECOVERY_SESSION = "attendance_recovery"  # the only session type that recovers funding (PRD 1 F5)


class Store:
    def __init__(self) -> None:
        self.version = 0  # bumped on every change that can move clusters (cache key)
        self._cluster_cache: Optional[Tuple[tuple, List[Cluster], dict]] = None
        self.corrections = corrections.CorrectionStore()
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
        self.outbox: List[dict] = []
        self.completed_learning: Dict[str, int] = {}  # student_id -> days recovered
        self.settings: RevenueSettings = revenue.STATE_DEFAULTS["CA"].model_copy()
        for rec in seed.absences():
            reply = rec.pop("reply")
            self.absences[rec["id"]] = Absence(**rec)
            if reply:
                self.record_reply(rec["id"], FamilyReply(**reply), route=False)

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

    def _names(self, student_id: str) -> List[str]:
        s = self.students.get(student_id)
        return [s.first_name, s.last_name, s.guardian_name] if s else []

    def record_reply(self, absence_id: str, reply: FamilyReply, route: bool = True) -> str:
        """Run the safety gate, classify (with this school's past corrections, F8), and return the
        route: escalated | cluster | case | unknown. route=False skips the cluster lookup (seed load)."""
        absence = self.absences[absence_id]
        if not reply.message_id:
            reply = reply.model_copy(update={"message_id": f"M-{absence_id}"})
        absence.reply = reply
        self.touch()
        result, _info = corrections.classify(reply.text, reply.language, self.corrections, SCHOOL_ID,
                                             names=self._names(absence.student_id))
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

    def check_in_message(self, absence: Absence) -> str:
        s = self.students[absence.student_id]
        if self.is_suppressed(s.id):
            return COUNSELOR_ONLY_NOTE
        n = self.absence_count(s.id)
        if s.preferred_language == "es":
            return (f"Hola {s.guardian_name}, extrañamos a {s.first_name} hoy ({n} ausencia(s) este año). "
                    f"¿Hay algo que lo esté dificultando? Puede responder por texto o nota de voz.")
        return (f"Hi {s.guardian_name}, we missed {s.first_name} today ({n} absence(s) this year). "
                f"Is anything getting in the way? You can reply by text or voice note.")

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
            sent_at = datetime.now().isoformat(timespec="seconds")
            suppressed = self.suppressed_student_ids()  # an escalation may have opened after drafting
            for sid in plan.recipient_student_ids:
                if sid in suppressed:
                    continue
                self.outbox.append({**action_plan.outbox_entry(plan, self.students[sid]), "sent_at": sent_at})
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
