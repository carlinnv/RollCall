"""Synthetic demo data. No real student data is used anywhere in this project.

The scenario is the school week of Mon Nov 9 to Fri Nov 13, 2026 at a fictional
California elementary school. It is built to show:
  * a bus-route cluster (Route 7 did not run Tue/Wed)
  * an illness cluster (Room 104)
  * a sibling caregiving case, a safety escalation, and a mix of single cases
  * absences still awaiting a family reply
"""
from datetime import date
from typing import Dict, List, Optional, Tuple

WEEK_START = date(2026, 11, 9)
WEEK_END = date(2026, 11, 13)

# (id, first, last, grade, classroom, teacher, bus_route, household, lang, guardian)
_STUDENTS: List[Tuple] = [
    # Route 7 riders
    ("S001", "Mateo", "Hernández", 3, "Room 201", "Mr. Okafor", "Route 7", "H01", "es", "Lucía Hernández"),
    ("S002", "Sofia", "Ramírez", 3, "Room 202", "Ms. Chen", "Route 7", "H02", "es", "Carlos Ramírez"),
    ("S003", "Liam", "Nguyen", 4, "Room 301", "Mrs. Patel", "Route 7", "H03", "vi", "Thao Nguyen"),
    ("S004", "Isabella", "Cruz", 3, "Room 201", "Mr. Okafor", "Route 7", "H04", "es", "Ana Cruz"),
    ("S005", "Noah", "Williams", 5, "Room 401", "Mr. Davis", "Route 7", "H05", "en", "Tasha Williams"),
    ("S006", "Camila", "Flores", 4, "Room 302", "Ms. Brooks", "Route 7", "H06", "es", "José Flores"),
    ("S007", "Ethan", "Kim", 3, "Room 202", "Ms. Chen", "Route 7", "H07", "en", "Grace Kim"),
    # Room 104 (grade 2, Ms. Rivera)
    ("S010", "Ava", "Johnson", 2, "Room 104", "Ms. Rivera", "Route 3", "H10", "en", "Mark Johnson"),
    ("S011", "Lucas", "García", 2, "Room 104", "Ms. Rivera", None, "H11", "es", "María García"),
    ("S012", "Mia", "Thompson", 2, "Room 104", "Ms. Rivera", "Route 5", "H12", "en", "Erin Thompson"),
    ("S013", "Elijah", "Martínez", 2, "Room 104", "Ms. Rivera", None, "H13", "es", "Rosa Martínez"),
    ("S014", "Harper", "Lee", 2, "Room 104", "Ms. Rivera", "Route 3", "H14", "en", "Daniel Lee"),
    ("S015", "Aiden", "Patel", 2, "Room 104", "Ms. Rivera", None, "H15", "en", "Priya Patel"),
    # Siblings, caregiving
    ("S020", "Valentina", "Lopez", 4, "Room 301", "Mrs. Patel", "Route 2", "H20", "es", "Gabriela Lopez"),
    ("S021", "Diego", "Lopez", 2, "Room 105", "Mr. Singh", "Route 2", "H20", "es", "Gabriela Lopez"),
    # Single cases
    ("S030", "Olivia", "Brown", 5, "Room 402", "Ms. Alvarez", "Route 4", "H30", "en", "Kevin Brown"),
    ("S031", "James", "Wilson", 4, "Room 302", "Ms. Brooks", None, "H31", "en", "Linda Wilson"),
    ("S032", "Emma", "Davis", 3, "Room 201", "Mr. Okafor", "Route 5", "H32", "en", "Rachel Davis"),
    ("S033", "Benjamin", "Moore", 5, "Room 401", "Mr. Davis", None, "H33", "en", "Steve Moore"),
    ("S034", "Amelia", "Taylor", 3, "Room 202", "Ms. Chen", "Route 4", "H34", "en", "Nicole Taylor"),
    ("S035", "Santiago", "Reyes", 4, "Room 301", "Mrs. Patel", "Route 6", "H35", "es", "Elena Reyes"),
    ("S036", "Charlotte", "Anderson", 5, "Room 402", "Ms. Alvarez", None, "H36", "en", "Paul Anderson"),
    ("S037", "Henry", "Jackson", 3, "Room 201", "Mr. Okafor", "Route 6", "H37", "en", "Monique Jackson"),
    ("S038", "Zoe", "White", 4, "Room 302", "Ms. Brooks", "Route 2", "H38", "en", "Laura White"),
    ("S039", "Samuel", "Harris", 5, "Room 401", "Mr. Davis", "Route 4", "H39", "en", "Andre Harris"),
    ("S040", "Lily", "Clark", 2, "Room 105", "Mr. Singh", None, "H40", "en", "Megan Clark"),
    ("S041", "Gabriel", "Torres", 3, "Room 202", "Ms. Chen", "Route 5", "H41", "es", "Miguel Torres"),
    ("S042", "Chloe", "Lewis", 4, "Room 301", "Mrs. Patel", None, "H42", "en", "Brian Lewis"),
]

# (student_id, date, sis_code, reply) where reply is (channel, language, text) or None
_D = date
_ABSENCES: List[Tuple[str, date, str, Optional[Tuple[str, str, str]]]] = [
    # Route 7 did not run Tuesday and Wednesday
    ("S001", _D(2026, 11, 10), "unexcused", ("text", "es", "El autobús de la ruta 7 no pasó esta mañana y no tenemos carro para llevarlo.")),
    ("S001", _D(2026, 11, 11), "unexcused", ("text", "es", "Otra vez no llegó el autobús.")),
    ("S002", _D(2026, 11, 10), "unexcused", ("text", "es", "No pasó el bus, esperamos 40 minutos en la parada.")),
    ("S002", _D(2026, 11, 11), "unexcused", None),
    ("S003", _D(2026, 11, 10), "unexcused", ("text", "en", "The bus never came to our stop and I had to be at work.")),
    ("S004", _D(2026, 11, 11), "unexcused", ("text", "es", "No hubo autobús hoy, no tengo cómo llevarla a la escuela.")),
    ("S005", _D(2026, 11, 10), "unexcused", ("text", "en", "Bus didn't show up again. No ride available.")),
    ("S005", _D(2026, 11, 11), "unexcused", ("text", "en", "Same thing, missed the bus because it never came.")),
    ("S006", _D(2026, 11, 11), "unexcused", None),
    ("S007", _D(2026, 11, 12), "excused", ("text", "en", "Ethan had a dentist appointment in the morning.")),
    # Room 104 stomach bug
    ("S010", _D(2026, 11, 11), "excused", ("text", "en", "Ava has a fever and has been throwing up since last night.")),
    ("S010", _D(2026, 11, 12), "excused", ("text", "en", "Still sick, the doctor said stomach flu.")),
    ("S011", _D(2026, 11, 11), "excused", ("text", "es", "Lucas está enfermo, tiene fiebre y vómito.")),
    ("S012", _D(2026, 11, 12), "excused", ("text", "en", "Mia came down with the stomach bug going around her class.")),
    ("S013", _D(2026, 11, 12), "unverified", ("text", "es", "Elijah está con dolor de estómago, lo llevamos al médico.")),
    ("S014", _D(2026, 11, 12), "unverified", None),
    ("S015", _D(2026, 11, 13), "excused", ("text", "en", "Aiden is sick with a fever today.")),
    # Siblings home with a younger child
    ("S020", _D(2026, 11, 12), "unexcused", ("text", "es", "Tuve que trabajar y Valentina se quedó a cuidar a su hermanito bebé.")),
    ("S021", _D(2026, 11, 12), "unexcused", ("text", "es", "Diego se quedó en casa con su hermana cuidando al bebé.")),
    # Safety escalation
    ("S030", _D(2026, 11, 9), "unexcused", ("text", "en", "Olivia says some kids have been bullying her at recess and she is scared to go back.")),
    ("S030", _D(2026, 11, 10), "unexcused", None),
    # Housing / food
    ("S031", _D(2026, 11, 10), "unexcused", ("text", "en", "We got evicted last week and are staying at a motel across town right now.")),
    ("S031", _D(2026, 11, 13), "unexcused", None),
    # Disengagement
    ("S033", _D(2026, 11, 9), "unexcused", ("text", "en", "Ben says he hates school, he is falling behind in math and doesn't want to go.")),
    ("S033", _D(2026, 11, 13), "unexcused", ("text", "en", "He refused to go again, says he is too behind.")),
    # Transportation, not on Route 7 (shows text alone over-groups)
    ("S035", _D(2026, 11, 10), "unexcused", ("text", "es", "Se nos descompuso el carro y no pudimos llevarlo.")),
    # Appointments / travel
    ("S034", _D(2026, 11, 9), "excused", ("text", "en", "Amelia had a doctor's appointment for a checkup.")),
    ("S036", _D(2026, 11, 12), "excused", ("text", "en", "We are traveling to a family funeral out of state.")),
    ("S036", _D(2026, 11, 13), "excused", ("text", "en", "Still traveling, back Monday.")),
    # Unclear reply
    ("S037", _D(2026, 11, 11), "unverified", ("text", "en", "He was just out today.")),
    # Awaiting reply
    ("S032", _D(2026, 11, 13), "unverified", None),
    ("S038", _D(2026, 11, 12), "unverified", None),
    ("S039", _D(2026, 11, 9), "unverified", None),
    ("S039", _D(2026, 11, 13), "unverified", None),
    ("S040", _D(2026, 11, 11), "unverified", None),
    ("S041", _D(2026, 11, 13), "unverified", None),
    ("S042", _D(2026, 11, 10), "unverified", None),
]

# Course pacing guides: grade -> list of (date, course, topic, pacing-guide reference)
PACING_GUIDE: Dict[int, List[Tuple[date, str, str, str]]] = {}


def _build_pacing() -> None:
    topics = {
        2: ("Grade 2 Math", [
            "Adding within 100 using place value", "Subtracting within 100 using place value",
            "Word problems with two-digit numbers", "Measuring length in inches and feet",
            "Estimating lengths", "Comparing lengths", "Reading a ruler", "Line plots of measurements",
            "Review: measurement", "Unit 4 check-in",
        ]),
        3: ("Grade 3 Math", [
            "Multiplication as equal groups", "Arrays and multiplication", "Multiplying by 2, 5 and 10",
            "Commutative property", "Division as sharing", "Division as grouping",
            "Relating multiplication and division", "Multiplying by 3 and 4", "Two-step word problems",
            "Unit 3 check-in",
        ]),
        4: ("Grade 4 Math", [
            "Place value to one million", "Rounding multi-digit numbers", "Adding multi-digit numbers",
            "Subtracting multi-digit numbers", "Multiplying by one-digit numbers", "Area model for multiplication",
            "Partial products", "Multiplication word problems", "Factors and multiples", "Unit 2 check-in",
        ]),
        5: ("Grade 5 Math", [
            "Equivalent fractions", "Adding fractions with unlike denominators",
            "Subtracting fractions with unlike denominators", "Adding mixed numbers", "Subtracting mixed numbers",
            "Fraction word problems", "Fractions as division", "Multiplying a fraction by a whole number",
            "Area with fractional side lengths", "Unit 4 check-in",
        ]),
    }
    school_days = [date(2026, 11, d) for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13)]
    for grade, (course, names) in topics.items():
        PACING_GUIDE[grade] = [
            (d, course, name, f"{course} pacing guide, {d.strftime('%b %-d')} lesson {i + 1}")
            for i, (d, name) in enumerate(zip(school_days, names))
        ]


_build_pacing()


def students() -> List[dict]:
    out = []
    for i, (sid, first, last, grade, room, teacher, route, hh, lang, guardian) in enumerate(_STUDENTS):
        out.append(dict(
            id=sid, first_name=first, last_name=last, grade=grade, classroom=room, teacher=teacher,
            bus_route=route, household_id=hh, preferred_language=lang, guardian_name=guardian,
            guardian_phone=f"+1-555-01{i:02d}",  # fictional 555 numbers
        ))
    return out


def absences() -> List[dict]:
    out = []
    for i, (sid, d, code, reply) in enumerate(_ABSENCES):
        rec = dict(id=f"A{i + 1:03d}", student_id=sid, date=d, sis_code=code, reply=None)
        if reply:
            channel, lang, text = reply
            rec["reply"] = dict(channel=channel, language=lang, text=text,
                                received_at=f"{d.isoformat()}T09:{10 + i % 50:02d}:00")
        out.append(rec)
    return out
