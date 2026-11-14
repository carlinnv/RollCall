"""Outbound rules: may a message go to a family, and what does the check-in say.

Pure functions with no network and no store access, so every rule is testable
with Photon off. The store supplies the facts; the channel does the sending.

Rules (PRD 3 section 3, PRD 1 Outreach Rules):
- An open safety escalation suppresses every automated message to the household.
- An opted-out number is never messaged.
- Outbound text never contains a link.
- The first check-in asks one supportive question, says who reads replies, and
  states no absence count.
"""
import os
import re
from datetime import date
from typing import Optional

from ..models import Student

DEFAULT_DEMO_STUDENT = "S002"  # Spanish-speaking family on Route 7 with an absence awaiting reply

# Last 10 digits of each number that has asked us to stop. In memory, like the rest of the store.
opted_out: set = set()

_LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|org|net|edu|gov|io|co|ly|us)\b", re.IGNORECASE)

_MONTHS_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
              "octubre", "noviembre", "diciembre"]

CHECK_IN = {
    "en": ("Hi {guardian}, this is the school attendance office. We missed {student} on {day}. "
           "Is anything getting in the way of getting to school? "
           "The attendance clerk reads your replies, and we are here to help."),
    "es": ("Hola {guardian}, le escribe la oficina de asistencia de la escuela. Extrañamos a {student} el {day}. "
           "¿Hay algo que esté dificultando la llegada a la escuela? "
           "La persona encargada de asistencia lee sus respuestas, y queremos ayudar."),
}


def digits(handle: str) -> str:
    """Comparable form of a phone number: its last 10 digits."""
    return re.sub(r"\D", "", handle)[-10:]


def demo_student_id() -> str:
    return os.environ.get("PHOTON_DEMO_STUDENT_ID", DEFAULT_DEMO_STUDENT)


def phone_for(student: Student) -> str:
    """The number we message for this student. The test sender plays the demo student's guardian."""
    sender = os.environ.get("PHOTON_TEST_SENDER")
    if sender and student.id == demo_student_id():
        return sender
    return student.guardian_phone


def has_link(text: str) -> bool:
    return bool(_LINK.search(text))


def blocked_reason(phone: str, text: str, household_suppressed: bool) -> Optional[str]:
    """Why this message must not be sent, or None if it may go."""
    if household_suppressed:
        return "open safety escalation: only the counselor contacts this household"
    if digits(phone) in opted_out:
        return "family opted out of messages"
    if has_link(text):
        return "message contains a link"
    if not text.strip():
        return "message is empty"
    return None


def check_in_text(student: Student, absence_date: date) -> str:
    lang = student.preferred_language if student.preferred_language in CHECK_IN else "en"
    if lang == "es":
        day = f"{absence_date.day} de {_MONTHS_ES[absence_date.month - 1]}"
    else:
        day = f"{absence_date.strftime('%B')} {absence_date.day}"
    return CHECK_IN[lang].format(guardian=student.guardian_name, student=student.first_name, day=day)
