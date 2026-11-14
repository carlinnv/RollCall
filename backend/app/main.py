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
            "escalated": absence_id in store.escalations, "messages": store.thread(absence_id)}


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
    return [{**e.model_dump(), "student": store.students[e.student_id]} for e in store.escalations.values()]


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
