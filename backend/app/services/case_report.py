"""Internal case report for a safety escalation.

Filled in from the system record only: no model call, so nothing is invented and no names
leave the server. The family's messages are quoted exactly. The counselor edits the draft
(adding their own narrative) and saves it; every save is kept as a new version.

This is a record for the school's files. It is not the mandated report: staff file that
themselves (PRD 3 section 3), and this report only records their decision.
"""
from datetime import date, datetime, timezone
from typing import List, Optional

from ..models import Absence, Escalation, Message, Student

OUTCOME = {"resolved": "Concern handled", "not_safety": "Not a safety concern"}
REPORT_DECISION = {"report_filed": "Report filed", "not_required": "No report required"}
SOURCE = {
    "gemini": "Gemini classifier",
    "keywords": "keyword rules",
    "keywords_fallback": "keyword rules (Gemini unavailable)",
    "safety_keywords": "safety keyword check",
    "counselor": "counselor",
}
KIND = {"reply": "Family", "check_in": "School check-in", "plan": "School plan message", "ack": "School automatic reply"}
NOTES_PLACEHOLDER = "(Write what you learned, who you spoke to, and what happens next.)"


def _when(stamp: Optional[str]) -> str:
    """'Nov 14, 2026 21:02 UTC'. Stamps without a timezone are treated as UTC."""
    if not stamp:
        return "—"
    moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%b %d, %Y %H:%M UTC")


def _day(d: date) -> str:
    return d.strftime("%a %b %d, %Y")


def _message_line(m: Message) -> str:
    who = KIND.get(m.kind, m.kind)
    channel = f" ({m.channel})" if m.channel else ""
    line = f'- {_when(m.timestamp)}  {who}{channel}: "{m.raw_text_or_transcript}"'
    if m.direction == "inbound" and m.classification:
        line += f"  [labeled {m.classification.barrier.value} by {SOURCE.get(m.classification.source, m.classification.source)}]"
    if m.direction == "outbound" and m.status != "sent":
        line += f"  [{m.status}{': ' + m.reason if m.reason else ''}]"
    return line


def build(e: Escalation, student: Student, absence: Absence, messages: List[Message],
          generated_by: str, generated_at: str, notes: Optional[str] = None) -> str:
    """The report as plain text."""
    flagged_by = "—"
    first_flag = next((m for m in messages if m.classification and m.classification.safety_critical), None)
    if first_flag and first_flag.classification:
        flagged_by = SOURCE.get(first_flag.classification.source, first_flag.classification.source)

    timeline = [f"- {_when(e.opened_at)}  Case opened ({e.reason})"]
    if e.acknowledged_at:
        timeline.append(f"- {_when(e.acknowledged_at)}  Acknowledged by {e.acknowledged_by}")
    if e.report_status:
        timeline.append(f"- {_when(e.report_at)}  Mandated-reporting decision: "
                        f"{REPORT_DECISION[e.report_status]} (recorded by {e.report_by})")
    if e.closed_at:
        outcome = OUTCOME.get(e.outcome or "", e.outcome or "")
        if e.outcome == "not_safety" and e.relabeled_to:
            outcome += f"; relabeled {e.relabeled_to.value}"
        timeline.append(f"- {_when(e.closed_at)}  Closed by {e.closed_by}: {outcome}")
    else:
        timeline.append("- Still open")

    paused = (f"from {_when(e.opened_at)} to {_when(e.closed_at)}" if e.closed_at
              else f"since {_when(e.opened_at)} (still paused)")

    lines = [
        "CONFIDENTIAL — SAFETY CASE REPORT",
        f"Escalation {e.escalation_id} · generated {_when(generated_at)} by {generated_by}",
        "",
        "STUDENT",
        f"Name: {student.first_name} {student.last_name} ({student.id})",
        f"Grade {student.grade} · {student.classroom} ({student.teacher})",
        f"Guardian: {student.guardian_name}, {student.guardian_phone}",
        f"Household: {student.household_id}",
        f"Absence: {absence.id}, {_day(absence.date)}",
        "",
        "CONCERN",
        f"Category: {e.reason}",
        f"Flagged by: {flagged_by}",
        f'Family\'s words: "{e.evidence}"',
        "",
        "TIMELINE",
        *timeline,
        "",
        "MESSAGES WITH THE FAMILY (exact text, oldest first)",
        *([_message_line(m) for m in messages] or ["- None recorded"]),
        "",
        "AUTOMATED MESSAGES",
        f"All automated messages to household {student.household_id} were paused {paused}.",
        "",
        "COUNSELOR NOTES",
        notes or NOTES_PLACEHOLDER,
        "",
        "—",
        "Generated from the system record. The system does not file mandated reports; "
        "staff file them and record the decision above.",
    ]
    return "\n".join(lines)


def notes_from(text: str) -> Optional[str]:
    """The counselor's notes from a saved report: the text between COUNSELOR NOTES and the footer."""
    if "COUNSELOR NOTES\n" not in text:
        return None
    notes = text.split("COUNSELOR NOTES\n", 1)[1].split("\n\n—\n", 1)[0].strip()
    return None if notes in ("", NOTES_PLACEHOLDER) else notes
