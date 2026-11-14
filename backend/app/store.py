"""In-memory store and the end-to-end flow from PRD 1 section 5.

Absence logged -> family replies -> safety gate -> classify & price ->
cluster or case plan drafted -> clerk approves & sends -> learning plan.

State lives in memory for the hackathon demo; `reset()` reloads the seed.
"""
import csv
import io
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from .data import seed
from .models import (
    Absence, AbsenceStatus, ActionPlan, Barrier, Classification, Cluster, Escalation, FamilyReply,
    LearningPlan, PlanStatus, RevenueSettings, RevenueSummary, Student,
)
from .services import action_plan, classifier, clustering, learning_plan, revenue

TODAY = date(2026, 11, 14)
PRIOR_DAYS_RECOVERED = 14  # recovered through Attendance Recovery before this week (synthetic)


class Store:
    def __init__(self) -> None:
        self.reset()

    # ---------- setup ----------
    def reset(self) -> None:
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
                self.record_reply(rec["id"], FamilyReply(**reply))

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

    def record_reply(self, absence_id: str, reply: FamilyReply) -> str:
        """Run the safety gate, classify, and return the route: escalated | cluster | case | unknown."""
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

    def check_in_message(self, absence: Absence) -> str:
        s = self.students[absence.student_id]
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
            sent_at = datetime.now().isoformat(timespec="seconds")
            for sid in plan.recipient_student_ids:
                self.outbox.append({**action_plan.outbox_entry(plan, self.students[sid]), "sent_at": sent_at})
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
