"""Why Absent API."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import Barrier, FamilyReply, RevenueSettings
from .services import classifier, photon_channel
from .store import TODAY, iso, store, utc_now


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Lifespan only runs under a real server (or `with TestClient`), so plain tests never start Photon.
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    photon_channel.start()
    yield
    photon_channel.stop()


app = FastAPI(title="Why Absent API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get(mapping, key, what):
    if key not in mapping:
        raise HTTPException(404, f"{what} {key} not found")
    return mapping[key]


class ReplyIn(BaseModel):
    text: str
    channel: str = "text"
    language: str = "en"


class CsvIn(BaseModel):
    csv: str


class DraftIn(BaseModel):
    cluster_id: Optional[str] = None
    absence_id: Optional[str] = None


class ApproveIn(BaseModel):
    approved_by: str
    send: bool = True


class StaffIn(BaseModel):
    by: str = "Counselor"


class ReportIn(StaffIn):
    decision: str  # report_filed | not_required


class CaseReportIn(StaffIn):
    text: str


class CloseIn(StaffIn):
    outcome: str  # resolved | not_safety
    relabel_to: Optional[Barrier] = None
    note: Optional[str] = None


@app.get("/api/health")
def health():
    return {"ok": True, "today": TODAY, "classifier": classifier.status()}


@app.get("/api/taxonomy")
def taxonomy():
    return [b.value for b in Barrier]


@app.post("/api/reset")
def reset():
    store.reset()
    return {"ok": True}


# ---------- F1 ----------
@app.get("/api/students")
def students():
    return list(store.students.values())


@app.get("/api/absences")
def absences():
    return store.queue()


@app.get("/api/absences/{absence_id}")
def absence(absence_id: str):
    a = _get(store.absences, absence_id, "absence")
    return {**a.model_dump(), "student": store.students[a.student_id], "check_in_message": store.check_in_message(a),
            "escalated": store.open_escalation(absence_id) is not None, "messages": store.thread(absence_id)}


@app.post("/api/absences/import")
def import_absences(body: CsvIn):
    added, errors = store.import_csv(body.csv)
    return {"added": added, "errors": errors}


@app.post("/api/absences/{absence_id}/reply")
def reply(absence_id: str, body: ReplyIn):
    _get(store.absences, absence_id, "absence")
    route = store.record_reply(absence_id, FamilyReply(
        text=body.text, channel=body.channel, language=body.language,
        received_at=iso(utc_now()),
    ))
    return {"route": route, "absence": store.absences[absence_id]}


@app.post("/api/absences/{absence_id}/check-in")
def check_in(absence_id: str):
    """Send the first check-in. Returns status sent | queued | blocked | failed, with the reason."""
    _get(store.absences, absence_id, "absence")
    return store.send_check_in(absence_id)


@app.get("/api/escalations")
def escalations():
    """Every escalation: open ones first (oldest first), then closed ones (most recent first)."""
    items = list(store.escalations.values())
    still_open = sorted((e for e in items if e.status != "closed"), key=lambda e: e.opened_at)
    closed = sorted((e for e in items if e.status == "closed"), key=lambda e: e.closed_at or "", reverse=True)
    return [store.escalation_view(e) for e in still_open + closed]


def _act(action, escalation_id: str, *args):
    _get(store.escalations, escalation_id, "escalation")
    try:
        return store.escalation_view(action(escalation_id, *args))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/escalations/{escalation_id}/acknowledge")
def acknowledge_escalation(escalation_id: str, body: StaffIn):
    return _act(store.acknowledge_escalation, escalation_id, body.by)


@app.post("/api/escalations/{escalation_id}/report")
def report_escalation(escalation_id: str, body: ReportIn):
    """Record the mandated-reporting decision: report_filed or not_required."""
    return _act(store.record_report_decision, escalation_id, body.by, body.decision)


@app.post("/api/escalations/{escalation_id}/case-report/draft")
def draft_case_report(escalation_id: str, body: StaffIn):
    """Generate an internal case report from the record. Not saved until the counselor saves it."""
    _get(store.escalations, escalation_id, "escalation")
    return {"text": store.draft_case_report(escalation_id, body.by)}


@app.get("/api/escalations/{escalation_id}/case-reports")
def case_reports(escalation_id: str):
    """Saved versions of the case report, oldest first."""
    _get(store.escalations, escalation_id, "escalation")
    return store.case_reports.get(escalation_id, [])


@app.post("/api/escalations/{escalation_id}/case-reports")
def save_case_report(escalation_id: str, body: CaseReportIn):
    """Save the counselor's edited report as a new version."""
    _get(store.escalations, escalation_id, "escalation")
    try:
        return store.save_case_report(escalation_id, body.by, body.text)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/escalations/{escalation_id}/close")
def close_escalation(escalation_id: str, body: CloseIn):
    """Close a case: resolved (real concern) or not_safety (false alarm, with relabel_to)."""
    return _act(store.close_escalation, escalation_id, body.by, body.outcome, body.relabel_to, body.note)


# ---------- F2: message log ----------
@app.get("/api/messages")
def messages(absence_id: Optional[str] = None, student_id: Optional[str] = None, direction: Optional[str] = None):
    """Every inbound and outbound text, oldest first. Filters combine."""
    return [m for m in store.messages
            if absence_id in (None, m.absence_id) and student_id in (None, m.student_id)
            and direction in (None, m.direction)]


@app.get("/api/messages/latency")
def message_latency(channel: Optional[str] = None):
    """How fast replies were labeled, against the 10-second target. Optional channel, e.g. imessage."""
    return store.latency_summary(channel)


# ---------- F2: messaging channel ----------
@app.get("/api/channel")
def channel():
    """Photon channel status and the recent message audit trail."""
    return photon_channel.status()


# ---------- clusters ----------
@app.get("/api/clusters")
def clusters():
    return store.clusters()


# ---------- F4 ----------
@app.get("/api/plans")
def plans():
    return list(store.plans.values())


@app.post("/api/plans/draft")
def draft_plan(body: DraftIn):
    try:
        return store.draft_plan(cluster_id=body.cluster_id, absence_id=body.absence_id)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/api/plans/{plan_id}")
def update_plan(plan_id: str, fields: dict):
    _get(store.plans, plan_id, "plan")
    try:
        return store.update_plan(plan_id, fields)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/plans/{plan_id}/approve")
def approve_plan(plan_id: str, body: ApproveIn):
    _get(store.plans, plan_id, "plan")
    return store.approve_plan(plan_id, body.approved_by, body.send)


@app.get("/api/outbox")
def outbox():
    return store.outbox


# ---------- F5 ----------
@app.get("/api/learning-plans/{student_id}")
def get_learning_plan(student_id: str):
    _get(store.students, student_id, "student")
    return store.learning_plan(student_id)


@app.post("/api/learning-plans/{student_id}/complete")
def complete_learning_plan(student_id: str):
    _get(store.students, student_id, "student")
    return {"days_recovered": store.complete_learning_plan(student_id)}


# ---------- F6 ----------
@app.get("/api/revenue")
def get_revenue():
    return store.revenue()


@app.put("/api/revenue/settings")
def put_settings(settings: RevenueSettings):
    store.settings = settings
    return store.revenue()
