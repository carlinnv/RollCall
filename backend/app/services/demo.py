"""Demo mode (PRD 3 section 2, "never cut"): the scripted demo runs with no network.

DEMO_MODE=1 turns on three things:
  * Recorded model answers. The classifier reads Gemini's answers from data/demo_answers.json and
    never calls the API. A reply that was not recorded gets the keyword label.
  * A replayable message script (data/demo_script.json). Each step goes through the same inbound
    path as a real iMessage (photon_channel.handle_inbound), on the "replay" channel.
  * Photon off. Outbound texts (check-ins, plan messages, pages) are logged as sent on the "replay"
    channel and never reach a phone.
It also fixes the story (DEMO_STORY=live) and turns quiet hours off, so the run is the same at any hour.

Record the model answers again after changing the script or the classifier prompt (needs a Gemini key):

    cd backend && .venv/bin/python -m app.services.demo record

The replay state is read from the store (a step is done when its message is in the log), so
/api/reset also rewinds the script.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SCRIPT_FILE = DATA_DIR / "demo_script.json"
ANSWERS_FILE = DATA_DIR / "demo_answers.json"
CHANNEL = "replay"


class NotRecorded(LookupError):
    """Demo mode has no recorded model answer for this text."""


def enabled() -> bool:
    return os.environ.get("DEMO_MODE", "0").strip().lower() in ("1", "true", "on", "yes")


# ---------- recorded model answers ----------
_answers: Optional[dict] = None
_recording: Optional[Dict[str, dict]] = None  # set while `record` runs
_lock = threading.Lock()


def answer_key(prompt_version: str, examples: str, redacted: str) -> str:
    """Key for one model answer. Leaves out the model name, so a recording plays back whatever GEMINI_MODEL says."""
    return hashlib.sha256(f"{prompt_version}|{examples}|{redacted}".encode()).hexdigest()


def recording() -> dict:
    """{"model", "recorded_at", "prompt_version", "answers": {key: answer}} from data/demo_answers.json."""
    global _answers
    if _answers is None:
        try:
            _answers = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _answers = {"answers": {}}
    return _answers


def recorded_answer(key: str) -> dict:
    found = recording()["answers"].get(key)
    if found is None:
        raise NotRecorded(key)
    return found


def remember(key: str, data: dict) -> None:
    """Called by the classifier for every model answer while `record` runs."""
    if _recording is not None:
        with _lock:
            _recording[key] = data


# ---------- the script ----------
def script() -> dict:
    return json.loads(SCRIPT_FILE.read_text(encoding="utf-8"))


def message_id(step_id: str) -> str:
    return f"demo-{step_id}"


def _reply_message(store, step_id: str):
    return next((m for m in store.messages if m.external_id == message_id(step_id) and m.direction == "inbound"), None)


def _absence_for(store, step: dict) -> Optional[str]:
    """The absence a correct step changes: the one its `step` reply landed on."""
    found = _reply_message(store, step["step"])
    return found.absence_id if found else None


def _done(store, step: dict) -> bool:
    if step["kind"] == "reply":
        return _reply_message(store, step["id"]) is not None
    absence_id = _absence_for(store, step)
    if absence_id is None:
        return False
    label = store.absences[absence_id].classification
    return bool(label and label.source == "staff_correction" and label.barrier.value == step["label"])


def _outcome(store, step: dict) -> Optional[dict]:
    absence_id = _reply_message(store, step["id"]).absence_id if step["kind"] == "reply" and _done(store, step) \
        else _absence_for(store, step) if step["kind"] == "correct" else None
    if absence_id is None or absence_id not in store.absences:
        return None
    label = store.absences[absence_id].classification
    return {"absence_id": absence_id, "barrier": label.barrier.value if label else None,
            "source": label.source if label else None, "escalated": store.open_escalation(absence_id) is not None}


def state(store) -> dict:
    data = script()
    steps, upcoming = [], None
    for i, step in enumerate(data["steps"]):
        done = _done(store, step)
        if not done and upcoming is None:
            upcoming = i
        student = store.students.get(step.get("student_id", ""))
        steps.append({**step, "done": done, "outcome": _outcome(store, step) if done else None,
                      "student_name": f"{student.first_name} {student.last_name}" if student else None,
                      "guardian_name": student.guardian_name if student else None})
    rec = recording()
    return {"enabled": enabled(), "title": data.get("title"), "steps": steps, "next": upcoming,
            "recorded_model": rec.get("model"), "recorded_at": rec.get("recorded_at"),
            "recorded_answers": len(rec.get("answers", {}))}


def play_next(store) -> dict:
    """Run the next step that is not done. Returns the step's state afterwards."""
    data = script()
    for step in data["steps"]:
        if not _done(store, step):
            _play(store, step)
            return next(s for s in state(store)["steps"] if s["id"] == step["id"])
    raise LookupError("the demo script is finished; reset to run it again")


def _play(store, step: dict) -> None:
    from ..models import Barrier
    from . import outreach, photon_channel

    if step["kind"] == "correct":
        absence_id = _absence_for(store, step)
        if absence_id is None:
            raise LookupError(f"step {step['step']} has not run yet")
        store.correct(absence_id, Barrier(step["label"]), step.get("by", "clerk"))
        return

    student = store.students[step["student_id"]]
    phone = outreach.phone_for(student)
    received = datetime.now(timezone.utc)
    stamp = received.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    result = photon_channel.handle_inbound(phone, step["text"], stamp, message_id(step["id"]), channel=CHANNEL)
    if result["reply"]:  # the automatic reply a family would get, logged as the real loop does
        from ..store import elapsed_ms, iso, utc_now
        store.log_message(direction="outbound", kind="ack", status="sent", channel=CHANNEL,
                          absence_id=result["absence_id"], student_id=student.id, language=student.preferred_language,
                          raw_text_or_transcript=result["reply"], timestamp=iso(utc_now()),
                          latency_ms=elapsed_ms(received), phone=phone)


# ---------- delivery in demo mode ----------
def replay_sender(student, phone: str, text: str, key: str) -> dict:
    return {"status": "sent", "reason": None, "channel": CHANNEL}


def replay_staff_sender(phone: str, text: str, key: str) -> dict:
    return {"status": "sent", "reason": None, "channel": CHANNEL}


def apply_settings() -> None:
    """Settings demo mode fixes, so the run is the same at any hour. Call before the store loads."""
    if enabled():
        os.environ["DEMO_STORY"] = "live"
        os.environ["QUIET_HOURS"] = "off"


# ---------- recording ----------
def record() -> int:
    """Run the script against live Gemini and save every model answer to data/demo_answers.json."""
    global _recording, _answers
    os.environ["DEMO_MODE"] = "0"
    os.environ["DEMO_STORY"] = "live"
    os.environ["QUIET_HOURS"] = "off"
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[3]
    load_dotenv(root / ".env")
    load_dotenv(root / "backend" / ".env")
    from . import classifier
    from ..store import store

    if not classifier.enabled():
        print("No Gemini key (or CLASSIFIER=keyword): nothing to record.", file=sys.stderr)
        return 1
    _recording = {}
    store.sender, store.staff_sender = replay_sender, replay_staff_sender
    store.reset()
    failed = False
    for step in script()["steps"]:
        outcome = play_next(store)["outcome"] or {}
        expect = step.get("expect", {}).get("barrier")
        ok = (expect is None or outcome.get("barrier") == expect) and outcome.get("source") != "keywords_fallback"
        failed |= not ok
        print(f"{'ok ' if ok else 'BAD'} {step['id']:<10} {outcome.get('barrier')} ({outcome.get('source')})"
              + ("" if ok else f"  expected {expect} from Gemini"))
    if failed:
        print("Not saved: a step did not get its expected label from Gemini. Change the script text or try again.",
              file=sys.stderr)
        return 1
    out = {"model": classifier.model_name(), "prompt_version": classifier.PROMPT_VERSION,
           "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "answers": dict(sorted(_recording.items()))}
    ANSWERS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _answers, _recording = None, None
    print(f"Saved {len(out['answers'])} answers to {ANSWERS_FILE.relative_to(root)}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["record"]:
        print("usage: python -m app.services.demo record", file=sys.stderr)
        sys.exit(2)
    # Run from the imported module: under `python -m` this file is __main__, a second copy whose
    # recording dict the classifier would never see.
    from app.services import demo as _module
    sys.exit(_module.record())
