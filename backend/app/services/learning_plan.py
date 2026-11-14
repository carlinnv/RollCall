"""Personalized catch-up learning plans (stub).

Maps missed dates to the course pacing guide, so every step is a topic that is
actually in the guide and cites its source line. Workload is adapted to the
absence cause (lighter after illness, smaller chunks for disengagement).
"""
from datetime import date, timedelta
from typing import List

from ..data.seed import PACING_GUIDE
from ..models import Barrier, LearningPlan, LearningPlanStep

BASE_MINUTES = 30
LOAD = {
    Barrier.ILLNESS: (0.6, "Lighter load while recovering from illness: shorter sessions spread over more days."),
    Barrier.DISENGAGEMENT: (0.5, "Short, confidence-building sessions with a teacher check-in after each one."),
    Barrier.HOUSING_FOOD: (0.7, "Flexible pacing; all work can be done during the school day with no materials needed at home."),
    Barrier.SAFETY: (0.6, "Counselor-coordinated pacing; no pressure until the student feels safe returning."),
}
DEFAULT_LOAD = (1.0, "Standard catch-up pacing.")


def build(student_id: str, grade: int, missed: List[date], barrier: Barrier, today: date) -> LearningPlan:
    guide = PACING_GUIDE.get(grade, [])
    by_date = {d: (course, topic, src) for d, course, topic, src in guide}
    factor, note = LOAD.get(barrier, DEFAULT_LOAD)
    minutes = max(10, int(round(BASE_MINUTES * factor / 5.0)) * 5)

    steps: List[LearningPlanStep] = []
    course = guide[0][1] if guide else "Unknown course"
    for d in sorted(set(missed)):
        if d not in by_date:
            continue  # only topics present in the pacing guide; never invent content
        course, topic, src = by_date[d]
        steps.append(LearningPlanStep(
            order=len(steps) + 1,
            topic=topic,
            missed_date=d,
            activity=f"Watch the lesson recap and complete practice set for '{topic}'.",
            minutes=minutes,
            source=src,
        ))

    every = 1 if barrier == Barrier.DISENGAGEMENT else 2
    checkins = [today + timedelta(days=2 * (i // every + 1)) for i in range(0, len(steps), every)]
    return LearningPlan(
        student_id=student_id,
        course=course,
        barrier=barrier,
        missed_days=len(steps),
        load_note=note,
        steps=steps,
        teacher_checkins=sorted(set(checkins)),
        total_minutes=sum(s.minutes for s in steps),
    )
