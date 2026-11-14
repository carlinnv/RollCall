"""School action plans (stub templates).

Each plan says what the school will do, who owns it, by when, and what is asked
of the family, plus a plain-language message in the family's language. Tone is
supportive: families are partners, never violators. Plans start as drafts and
must be approved by a clerk before they are sent.
"""
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from ..models import ActionPlan, Barrier, Cluster, Student

# barrier -> (school action, owner, family request) with {link} filled from the cluster
TEMPLATES: Dict[Barrier, Tuple[str, str, str]] = {
    Barrier.TRANSPORTATION: (
        "Contact the transportation office about {link} service gaps and arrange backup rides or bus passes for affected families.",
        "Transportation coordinator",
        "Tell us if you still need a ride option so we can set one up.",
    ),
    Barrier.ILLNESS: (
        "Notify the school nurse of illness in {link}, send home health guidance, and share the return-to-school checklist.",
        "School nurse",
        "Keep your child home until they are fever-free for 24 hours, then let us know when they are back.",
    ),
    Barrier.HOUSING_FOOD: (
        "Connect the family with the McKinney-Vento liaison for transportation, meals, and housing resources.",
        "McKinney-Vento liaison",
        "Share a good time to talk so we can help with school transportation and meals.",
    ),
    Barrier.CAREGIVING: (
        "Share local childcare and after-school program options and check eligibility for family support services.",
        "Family resource coordinator",
        "Let us know which days are hardest so we can suggest childcare options.",
    ),
    Barrier.DISENGAGEMENT: (
        "Schedule a student check-in with the counselor and set up tutoring for the subjects the student finds hardest.",
        "Counselor",
        "Join a short call with the counselor and teacher this week.",
    ),
    Barrier.APPOINTMENT: (
        "Send the catch-up learning plan and remind the family about scheduling appointments outside school hours when possible.",
        "Attendance clerk",
        "Send us any appointment note when you can.",
    ),
    Barrier.UNKNOWN: (
        "Follow up with the family by phone to learn what is getting in the way.",
        "Attendance clerk",
        "Reply or call us so we can understand how to help.",
    ),
}

MESSAGES: Dict[str, Dict[Barrier, str]] = {
    "en": {
        Barrier.TRANSPORTATION: "Hi {guardian}, we know the bus has not been reliable this week. We are working with the transportation office to fix it and can help with a backup ride. {request}",
        Barrier.ILLNESS: "Hi {guardian}, we hope {student} feels better soon. Our nurse has shared some health tips. {request} We will send a catch-up plan so {student} does not fall behind.",
        Barrier.HOUSING_FOOD: "Hi {guardian}, thank you for letting us know. We have support for transportation and meals and want to help. {request}",
        Barrier.CAREGIVING: "Hi {guardian}, thank you for telling us. We know balancing family needs is hard. {request}",
        Barrier.DISENGAGEMENT: "Hi {guardian}, we miss {student} and want school to feel good again. {request}",
        Barrier.APPOINTMENT: "Hi {guardian}, thanks for letting us know. We will send a catch-up plan for {student}. {request}",
        Barrier.UNKNOWN: "Hi {guardian}, we missed {student} at school and want to help with anything that is getting in the way. {request}",
    },
    "es": {
        Barrier.TRANSPORTATION: "Hola {guardian}, sabemos que el autobús no ha sido confiable esta semana. Estamos trabajando con la oficina de transporte para solucionarlo y podemos ayudarle con otro medio de transporte. Díganos si todavía necesita transporte para organizarlo.",
        Barrier.ILLNESS: "Hola {guardian}, esperamos que {student} se mejore pronto. Nuestra enfermera compartió algunos consejos de salud. Mantenga a su hijo en casa hasta que esté sin fiebre por 24 horas y avísenos cuando regrese. Le enviaremos un plan para ponerse al día.",
        Barrier.HOUSING_FOOD: "Hola {guardian}, gracias por avisarnos. Tenemos apoyo para transporte y comidas y queremos ayudarle. Díganos un buen momento para hablar.",
        Barrier.CAREGIVING: "Hola {guardian}, gracias por contarnos. Sabemos que equilibrar las necesidades de la familia es difícil. Díganos qué días son más difíciles para sugerirle opciones de cuidado infantil.",
        Barrier.DISENGAGEMENT: "Hola {guardian}, extrañamos a {student} y queremos que la escuela vuelva a sentirse bien. ¿Puede tener una llamada corta con la consejera y la maestra esta semana?",
        Barrier.APPOINTMENT: "Hola {guardian}, gracias por avisarnos. Le enviaremos un plan para que {student} se ponga al día. Envíenos la nota de la cita cuando pueda.",
        Barrier.UNKNOWN: "Hola {guardian}, extrañamos a {student} en la escuela y queremos ayudar con cualquier cosa que lo esté dificultando. Responda o llámenos.",
    },
}


def _message(lang: str, barrier: Barrier, guardian: str, student: str, request: str) -> Tuple[str, str]:
    lang = lang if lang in MESSAGES else "en"  # TODO: translation service for other languages
    template = MESSAGES[lang].get(barrier, MESSAGES[lang][Barrier.UNKNOWN])
    return lang, template.format(guardian=guardian, student=student, request=request)


def _majority_language(students: List[Student]) -> str:
    langs = [s.preferred_language for s in students]
    return max(set(langs), key=langs.count)


def for_cluster(cluster: Cluster, students: Dict[str, Student], today: date) -> ActionPlan:
    action, owner, request = TEMPLATES.get(cluster.barrier, TEMPLATES[Barrier.UNKNOWN])
    members = [students[s] for s in cluster.student_ids]
    lang, message = _message(_majority_language(members), cluster.barrier, "{guardian}", "{student}", request)
    return ActionPlan(
        id=f"P-{cluster.id}",
        scope="cluster",
        cluster_id=cluster.id,
        barrier=cluster.barrier,
        school_action=action.format(link=cluster.link_value),
        owner=owner,
        target_date=today + timedelta(days=2),
        family_request=request,
        message_language=lang,
        message=message,  # placeholders are filled per family at send time
        recipient_student_ids=cluster.student_ids,
    )


def for_case(absence_id: str, barrier: Barrier, student: Student, today: date) -> ActionPlan:
    action, owner, request = TEMPLATES.get(barrier, TEMPLATES[Barrier.UNKNOWN])
    lang, message = _message(student.preferred_language, barrier, student.guardian_name, student.first_name, request)
    return ActionPlan(
        id=f"P-case-{absence_id}",
        scope="case",
        absence_id=absence_id,
        barrier=barrier,
        school_action=action.format(link=student.classroom),
        owner=owner,
        target_date=today + timedelta(days=3),
        family_request=request,
        message_language=lang,
        message=message,
        recipient_student_ids=[student.id],
    )


def personalize(plan: ActionPlan, student: Student) -> str:
    """Fill a cluster message for one family. Language follows the family when we have a template."""
    lang = student.preferred_language if student.preferred_language in MESSAGES else "en"
    if lang == plan.message_language:
        return plan.message.replace("{guardian}", student.guardian_name).replace("{student}", student.first_name)
    return _message(lang, plan.barrier, student.guardian_name, student.first_name, plan.family_request)[1]

