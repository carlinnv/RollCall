"""Catch-up learning plans (PRD 1 F5).

Maps missed dates to the course pacing guide, so every step is a topic that is actually in the guide
and cites its source line. Workload is adapted to the absence cause (lighter after illness, smaller
chunks for disengagement).

PRD 1 F5 content: a teacher-facing missed-topics list (the steps), one small-group reteach
suggestion, a paper take-home in the home language and an offer of a supervised Attendance Recovery
session. No self-paced video plans for K-5 (the seed school is K-5), so activities are teacher-led
reteach plus paper practice.

Eligibility: plans are for students with 2+ missed days. A plan is still built for one missed day
(teachers can see the topic) but `eligible` is False, no Attendance Recovery session is offered and
completing it recovers no funding (see Store.complete_learning_plan).

Every plan goes through services/claim_check.verify, which sets verification_status and flags.
"""
from datetime import date, timedelta
from typing import List, Optional

from ..data.seed import PACING_GUIDE
from ..models import Barrier, LearningPlan, LearningPlanStep
from . import claim_check

BASE_MINUTES = 30
MIN_MISSED_DAYS = 2
LOAD = {
    Barrier.ILLNESS: (0.6, "Lighter load while recovering from illness: shorter sessions spread over more days."),
    Barrier.DISENGAGEMENT: (0.5, "Short, confidence-building sessions with a teacher check-in after each one."),
    Barrier.HOUSING_FOOD: (0.7, "Flexible pacing; all work can be done during the school day with no materials needed at home."),
    Barrier.SAFETY: (0.6, "Counselor-coordinated pacing; no pressure until the student feels safe returning."),
}
DEFAULT_LOAD = (1.0, "Standard catch-up pacing.")
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "vi": "Vietnamese", "zh": "Chinese"}


def build(student_id: str, grade: int, missed: List[date], barrier: Barrier, today: date,
          language: str = "en", nli: Optional[bool] = None) -> LearningPlan:
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
            activity=f"Small-group reteach of '{topic}' with the teacher, then the matching paper practice sheet.",
            minutes=minutes,
            source=src,
        ))

    every = 1 if barrier == Barrier.DISENGAGEMENT else 2
    checkins = [today + timedelta(days=2 * (i // every + 1)) for i in range(0, len(steps), every)]
    eligible = len(steps) >= MIN_MISSED_DAYS
    lang_name = LANGUAGE_NAMES.get(language, language)
    plan = LearningPlan(
        student_id=student_id,
        course=course,
        barrier=barrier,
        missed_days=len(steps),
        load_note=note,
        steps=steps,
        teacher_checkins=sorted(set(checkins)),
        total_minutes=sum(s.minutes for s in steps),
        eligible=eligible,
        eligibility_note=("" if eligible else
                          f"Fewer than {MIN_MISSED_DAYS} missed lessons: topics are listed for the teacher, but no "
                          f"catch-up plan or Attendance Recovery session is required."),
        reteach_suggestion=(f"Group this student with classmates who also missed '{steps[0].topic}' for one "
                            f"small-group reteach." if eligible else ""),
        take_home=(f"Paper practice packet for {len(steps)} missed lesson(s), printed in {lang_name}."
                   if eligible else ""),
        attendance_recovery_offer=("Offer a supervised Attendance Recovery session; only supervised sessions "
                                   "count toward recovered funding." if eligible else ""),
        home_language=language,
    )
    status, flags, method = claim_check.verify(plan, grade, missed, nli=nli)
    return plan.model_copy(update={"verification_status": status, "flags": flags, "verification_method": method})
