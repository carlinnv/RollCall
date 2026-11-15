"""Why Absent API."""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Settings must be in the environment before the store loads the seed (DEMO_STORY, CLASSIFIER, ...).
# Existing environment variables win, so tests and shells can override either file.
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "backend" / ".env")

from .models import Barrier, FamilyReply, RevenueSettings  # noqa: E402
from .services import photon_channel  # noqa: E402
from .store import RECOVERY_SESSION, TODAY, store  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"


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


class CorrectIn(BaseModel):
    label: Barrier
    corrected_by: str


class CompleteIn(BaseModel):
    session_type: str = RECOVERY_SESSION  # attendance_recovery | take_home | ...
    supervised_by: Optional[str] = None  # who supervised the session; required for it to count


@app.get("/api/health")
def health():
    return {"ok": True, "today": TODAY}


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
            "escalated": absence_id in store.escalations}


@app.post("/api/absences/import")
def import_absences(body: CsvIn):
    added, errors = store.import_csv(body.csv)
    return {"added": added, "errors": errors}


@app.post("/api/absences/{absence_id}/reply")
def reply(absence_id: str, body: ReplyIn):
    _get(store.absences, absence_id, "absence")
    route = store.record_reply(absence_id, FamilyReply(
        text=body.text, channel=body.channel, language=body.language,
        received_at=TODAY.isoformat() + "T12:00:00",
    ))
    return {"route": route, "absence": store.absences[absence_id]}


@app.post("/api/absences/{absence_id}/correct")
def correct_label(absence_id: str, body: CorrectIn):
    """F8: staff correct a barrier label. Stored per school and used for future classifications."""
    _get(store.absences, absence_id, "absence")
    try:
        absence, correction = store.correct(absence_id, body.label, body.corrected_by)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"absence": absence, "correction": correction, "escalated": absence_id in store.escalations
            and store.escalations[absence_id].status == "open"}


@app.get("/api/corrections")
def list_corrections():
    return store.corrections.all()
@app.post("/api/absences/{absence_id}/check-in")
def check_in(absence_id: str):
    """Send the first check-in. Returns status sent | queued | blocked | failed, with the reason."""
    _get(store.absences, absence_id, "absence")
    return store.send_check_in(absence_id)


@app.get("/api/escalations")
def escalations():
    return [{**e.model_dump(), "student": store.students[e.student_id]} for e in store.escalations.values()]


# ---------- F2: messaging channel ----------
@app.get("/api/channel")
def channel():
    """Photon channel status and the recent message audit trail."""
    return photon_channel.status()


# ---------- clusters ----------
@app.get("/api/clusters")
def clusters():
    """PRD 2 clusters from app.ml.live (each says which method produced it in `method`)."""
    return store.clusters()


@app.get("/api/clusters/meta")
def clusters_meta():
    """Dataset, method, detector and baseline notes behind /api/clusters."""
    return store.cluster_meta()


# ---------- evaluation artifacts (written by the eval scripts) ----------
def _eval_file(name: str):
    path = EVAL_DIR / name
    if not path.is_file():
        raise HTTPException(404, f"{name} not found; run the evaluation to produce backend/eval/{name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(500, f"could not read {name}: {e}")


@app.get("/api/eval/ablation")
def eval_ablation():
    return _eval_file("ablation.json")


@app.get("/api/eval/retrieval")
def eval_retrieval():
    return _eval_file("retrieval.json")


@app.get("/api/eval/ablation-with-contacts")
def eval_ablation_with_contacts():
    """Comparison only: the same held-out test with research sensor contact circles added to the detector.
    A real school cannot collect sensor data; the default detector uses school records only."""
    return _eval_file("ablation_with_contacts.json")


@app.get("/api/eval/scaling")
def eval_scaling():
    return _eval_file("scaling.json")


# ---------- F7 counselor assistant (read-only; runs over the simulated Lyon school) ----------
class AskIn(BaseModel):
    question: str


@app.post("/api/assistant/ask")
def assistant_ask(body: AskIn):
    """Answer a counselor's question from family messages with citations. Read-only: no tool can send.
    Needs a Gemini key; without one the endpoint returns 503 rather than guessing."""
    from .ml import live
    from .ml.assistant.agent import CounselorAssistant
    from .ml.assistant.corpus import AssistantData
    from .services.gemini_classifier import has_api_key

    question = body.question.strip()
    if not question or len(question) > 500:
        raise HTTPException(400, "question must be 1-500 characters")
    if not has_api_key():
        raise HTTPException(503, "the counselor assistant needs a Gemini key in backend/.env")
    run, alarms, _cal = live._sim_alarms(live.detector_name(), TODAY)
    answer = CounselorAssistant(AssistantData.from_sim_run(run, alarms=alarms)).ask(question, backoff_s=10.0)
    return {"question": answer.question, "answer": answer.answer, "citations": answer.citations,
            "dropped_citations": answer.dropped_citations, "report": answer.report,
            "safety_flags": answer.safety_flags, "error": answer.error, "dataset": "sim (Lyon primary, simulated)"}


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
def complete_learning_plan(student_id: str, body: Optional[CompleteIn] = None):
    """Only a supervised Attendance Recovery session for a 2+ missed-day plan recovers funding."""
    _get(store.students, student_id, "student")
    body = body or CompleteIn()
    return store.complete_learning_plan(student_id, body.session_type, body.supervised_by)


@app.get("/api/recovery-log")
def recovery_log():
    return store.recovery_log


# ---------- F6 ----------
@app.get("/api/revenue")
def get_revenue():
    return store.revenue()


@app.put("/api/revenue/settings")
def put_settings(settings: RevenueSettings):
    store.update_settings(settings)
    return store.revenue()
