"""Offline, templated multilingual family replies for the PRD 2 simulator.

`make_reply(barrier, language, rng, style)` returns one synthetic reply text that a
family might send when asked why a child was absent. It is deterministic given the
numpy Generator state, needs no network, and never uses real names: the child is
always "my son", "mi hija", "con", ... .

Styles
  * "normal": a full sentence from the per-barrier templates (occasional code-switching).
  * "terse":  a few words ("sick", "no paso el bus", "bị sốt").
  * "noisy":  a normal or terse reply with typos and SMS-style noise
              (dropped accents, swapped/dropped letters, lowercase, no punctuation).

Safety content is deliberately absent: the safety-or-bullying barrier is evaluated with a
separate frozen message suite, so asking for it here raises ValueError. A later step may
swap this generator for model-written text (PRD 2 asks for a different model family than
the classifier); keep the signature unchanged so the simulator does not care.
"""
from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

import numpy as np

from app.models import Barrier

ILLNESS = Barrier.ILLNESS.value
TRANSPORTATION = Barrier.TRANSPORTATION.value
HOUSING = Barrier.HOUSING_FOOD.value
CAREGIVING = Barrier.CAREGIVING.value
SAFETY = Barrier.SAFETY.value
DISENGAGEMENT = Barrier.DISENGAGEMENT.value
APPOINTMENT = Barrier.APPOINTMENT.value
UNKNOWN = Barrier.UNKNOWN.value

LANGUAGES = ("en", "es", "vi", "zh")
STYLES = ("normal", "terse", "noisy")

# Placeholders: {C}/{c} child noun phrase (capitalised / not), {S}/{s} subject pronoun,
# {o} object pronoun, {p} possessive, {g} Spanish gender ending ("o" / "a").
_REFS: Dict[str, Dict[str, Tuple[List[str], Dict[str, str]]]] = {
    "en": {
        "m": (["my son", "he", "my boy", "my kid"], {"s": "he", "o": "him", "p": "his"}),
        "f": (["my daughter", "she", "my girl", "my kid"], {"s": "she", "o": "her", "p": "her"}),
    },
    "es": {
        "m": (["mi hijo", "el niño", "mi niño"], {"s": "él", "o": "lo", "p": "su", "g": "o"}),
        "f": (["mi hija", "la niña", "mi niña"], {"s": "ella", "o": "la", "p": "su", "g": "a"}),
    },
    "vi": {
        "m": (["con", "cháu", "bé", "con trai tôi"], {"s": "con", "o": "con", "p": "của con"}),
        "f": (["con", "cháu", "bé", "con gái tôi"], {"s": "con", "o": "con", "p": "của con"}),
    },
    "zh": {
        "m": (["孩子", "我儿子"], {"s": "他", "o": "他", "p": "他的"}),
        "f": (["孩子", "我女儿"], {"s": "她", "o": "她", "p": "她的"}),
    },
}

# ---------------------------------------------------------------------------------------
# Full-sentence templates: language -> barrier -> list of templates
# ---------------------------------------------------------------------------------------
TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "en": {
        ILLNESS: [
            "{C} has a fever and is staying home today.",
            "{C} is sick, {s} was throwing up all night.",
            "Sorry, {c} has a bad cold and a cough. Keeping {o} home.",
            "{C} woke up with a stomach ache, not coming in today.",
            "{C} has the flu. {S} will be back when the fever is gone.",
            "{C} has a sore throat and a temperature so we are staying home.",
            "{C} got sick last night, vomiting. Home today.",
            "{C} is not feeling well, headache and fever since this morning.",
        ],
        TRANSPORTATION: [
            "The bus never came this morning and I could not drive {o}.",
            "Our car broke down, no way to get {o} to school today.",
            "{C} missed the bus and I have no ride for {o}.",
            "The bus didn't show up at our stop, we waited 40 minutes.",
            "No transportation today, my car is in the shop.",
            "The school bus was not running on our street so {c} stayed home.",
            "I don't have gas money this week to drive {o} in.",
            "Our ride fell through this morning, sorry.",
        ],
        HOUSING: [
            "We had to leave our apartment and are staying with family far away right now.",
            "We are moving between places this week, {c} will be back when we are settled.",
            "Our power and water got shut off and {c} could not get ready for school.",
            "We got evicted and are at a shelter for now, sorry.",
            "Things are hard right now, we are staying in our car for a few days.",
            "We don't have food at home this week and {c} did not feel ok to go.",
            "Lost our place, staying at a motel across town until we find something.",
        ],
        CAREGIVING: [
            "I had to work early and {c} stayed home to watch the baby.",
            "The daycare closed today so {c} had to help with {p} little brother.",
            "Family emergency, {c} is helping take care of grandma.",
            "I'm sick and {c} had to stay and help with the younger kids.",
            "The babysitter cancelled, {c} watched {p} sister while I worked.",
            "We had to go take care of a family member, sorry.",
            "{C} stayed home to help me with the little ones today.",
        ],
        DISENGAGEMENT: [
            "{C} refuses to go to school, says {p} classes are too hard.",
            "{C} doesn't want to go anymore, says nobody helps {o} in math.",
            "I can't get {o} out of bed, {s} says school is pointless.",
            "{C} is behind on {p} work and is too stressed to come.",
            "{C} says the teacher doesn't explain and {s} feels lost.",
            "{C} is struggling with homework and didn't want to face class today.",
            "{C} just won't go, {s} says {s} is bored and failing anyway.",
        ],
        APPOINTMENT: [
            "{C} has a dentist appointment this morning.",
            "Doctor appointment today, {s} will come in after lunch.",
            "We are out of town for a family trip until Monday.",
            "{C} has a checkup at the clinic today.",
            "We had to travel to see family, back next week.",
            "Eye doctor appointment today, sorry for the short notice.",
            "We have an appointment downtown for paperwork and {c} has to come with us.",
        ],
        UNKNOWN: [
            "{C} won't be in today.",
            "{S} is staying home today, thanks.",
            "Sorry {c} missed school.",
            "Personal reasons.",
            "{C} will be back tomorrow.",
            "Ok thanks for letting me know.",
            "Got the message. {S} is home today.",
        ],
    },
    "es": {
        ILLNESS: [
            "{C} tiene fiebre y se queda en casa hoy.",
            "{C} está enferm{g}, vomitó toda la noche.",
            "{C} tiene gripe y tos, no va a ir a la escuela.",
            "Perdón, {c} amaneció con dolor de estómago.",
            "{C} tiene calentura, mañana regresa si se siente mejor.",
            "{C} está malit{g} de la garganta.",
            "{C} se siente mal, tiene dolor de cabeza y fiebre.",
        ],
        TRANSPORTATION: [
            "No pasó el autobús esta mañana y no tengo carro.",
            "Se descompuso el carro y no tengo cómo llevar{o} a la escuela.",
            "{C} perdió el camión y no hay quien {o} lleve.",
            "El bus no llegó a la parada, esperamos mucho tiempo.",
            "No tengo transporte hoy, el carro está en el taller.",
            "No tengo dinero para gasolina esta semana para llevar{o}.",
            "El camión de la escuela no pasó por nuestra calle.",
        ],
        HOUSING: [
            "Nos sacaron del apartamento y estamos con familia lejos.",
            "Nos estamos mudando esta semana, {c} regresa cuando estemos instalados.",
            "Nos cortaron la luz y el agua, {c} no pudo alistarse.",
            "Estamos en un albergue por ahora, disculpe.",
            "No tenemos dónde quedarnos estos días.",
            "No hay comida en la casa esta semana y {c} no quiso ir así.",
        ],
        CAREGIVING: [
            "Tuve que trabajar temprano y {c} se quedó cuidando al bebé.",
            "La guardería cerró hoy y {c} tuvo que cuidar a su hermanito.",
            "Emergencia familiar, {c} está ayudando con la abuela.",
            "Estoy enferma y {c} me ayudó con los niños chiquitos.",
            "La niñera canceló y {c} cuidó a su hermana.",
            "Tuvimos que cuidar a un familiar enfermo.",
        ],
        DISENGAGEMENT: [
            "{C} no quiere ir a la escuela, dice que las clases están muy difíciles.",
            "{C} ya no quiere ir, dice que nadie le ayuda en matemáticas.",
            "No puedo sacar{o} de la cama, dice que la escuela no sirve.",
            "{C} va atrasad{g} con las tareas y está muy estresad{g}.",
            "{C} dice que no entiende al maestro y se siente perdid{g}.",
            "No quiso ir hoy, está cansad{g} de la escuela.",
        ],
        APPOINTMENT: [
            "{C} tiene cita con el dentista esta mañana.",
            "Cita con el doctor, llega después del almuerzo.",
            "Estamos fuera de la ciudad por un viaje familiar hasta el lunes.",
            "{C} tiene chequeo en la clínica hoy.",
            "Tuvimos que viajar a ver a la familia, regresamos la próxima semana.",
            "Tenemos cita para papeles en el centro y {c} tiene que venir con nosotros.",
        ],
        UNKNOWN: [
            "{C} no va a ir hoy.",
            "Se queda en casa hoy, gracias.",
            "Perdón que {c} faltó.",
            "Por motivos personales.",
            "{C} regresa mañana.",
            "Ok gracias por avisar.",
        ],
    },
    "vi": {
        ILLNESS: [
            "{C} bị sốt, hôm nay ở nhà.",
            "{C} bị ốm, nôn cả đêm.",
            "{C} bị cảm và ho, xin cho nghỉ hôm nay.",
            "{C} đau bụng, không đi học được.",
        ],
        TRANSPORTATION: [
            "Xe buýt không đến, tôi không có xe chở {c} đi học.",
            "Xe của tôi bị hư, không đưa {c} đi học được.",
            "{C} bị lỡ xe buýt sáng nay.",
            "Hôm nay không có ai chở {c} đi học.",
        ],
        HOUSING: [
            "Gia đình tôi phải dọn nhà gấp, đang ở nhờ nhà người quen.",
            "Chúng tôi bị đuổi nhà, đang ở nơi tạm trú.",
            "Nhà bị cắt điện nước, {c} không chuẩn bị đi học được.",
        ],
        CAREGIVING: [
            "Tôi phải đi làm sớm nên {c} ở nhà trông em.",
            "Nhà trẻ đóng cửa hôm nay nên {c} phải trông em.",
            "Gia đình có việc gấp, {c} phải giúp chăm sóc bà.",
        ],
        DISENGAGEMENT: [
            "{C} không muốn đi học, nói bài khó quá.",
            "{C} không chịu dậy đi học.",
            "{C} bị chậm bài và rất căng thẳng.",
        ],
        APPOINTMENT: [
            "{C} có hẹn nha sĩ sáng nay.",
            "{C} đi khám bác sĩ, chiều sẽ vào lớp.",
            "Gia đình đi xa thăm người thân, tuần sau về.",
        ],
        UNKNOWN: [
            "Hôm nay {c} nghỉ.",
            "{C} ở nhà hôm nay, cảm ơn.",
            "Mai {c} đi học lại.",
        ],
    },
    "zh": {
        ILLNESS: ["{C}发烧了，今天在家休息。", "{C}生病了，昨晚一直吐。", "{C}感冒咳嗽，请假一天。"],
        TRANSPORTATION: ["校车没来，我没办法送{c}上学。", "车坏了，今天没法送{c}去学校。", "{C}没赶上校车。"],
        HOUSING: ["我们正在搬家，暂时住在亲戚家。", "我们被房东赶出来了，现在住在临时住所。",
                  "家里停水停电了，{c}没法准备上学。"],
        CAREGIVING: ["我要早上班，{C}在家照顾弟弟。", "托儿所今天关门，{C}要在家看妹妹。",
                     "家里有急事，{C}要帮忙照顾奶奶。"],
        DISENGAGEMENT: ["{C}不想上学，说课太难了。", "{C}早上不肯起床去学校。", "{C}功课跟不上，压力很大。"],
        APPOINTMENT: ["{C}今天上午看牙医。", "{C}去看医生，下午再去学校。", "我们出远门探亲，下周回来。"],
        UNKNOWN: ["{C}今天不去学校。", "{C}今天在家，谢谢。", "好的，谢谢。"],
    },
}

TERSE: Dict[str, Dict[str, List[str]]] = {
    "en": {
        ILLNESS: ["sick", "fever", "sick today", "{c} is sick", "flu", "stomach bug"],
        TRANSPORTATION: ["no ride", "bus didnt come", "missed bus", "car broke down", "no bus today"],
        HOUSING: ["we are moving", "no home rn", "evicted", "staying at shelter"],
        CAREGIVING: ["family emergency", "watching baby", "daycare closed", "had to babysit"],
        DISENGAGEMENT: ["doesnt want to go", "refused to go", "hates school rn", "wont get up"],
        APPOINTMENT: ["dr appt", "dentist", "appointment", "traveling", "out of town"],
        UNKNOWN: ["ok", "home today", "not today", "will be back tmrw", "k"],
    },
    "es": {
        ILLNESS: ["enferm{g}", "fiebre", "está enferm{g}", "gripa", "tiene calentura"],
        TRANSPORTATION: ["no paso el bus", "no hay transporte", "sin carro", "perdio el bus"],
        HOUSING: ["nos estamos mudando", "sin casa", "estamos en albergue"],
        CAREGIVING: ["emergencia familiar", "cuidando al bebé", "cerró la guardería"],
        DISENGAGEMENT: ["no quiso ir", "no quiere ir", "no le gusta la escuela"],
        APPOINTMENT: ["cita médica", "dentista", "de viaje", "cita"],
        UNKNOWN: ["ok", "no va hoy", "gracias", "mañana regresa"],
    },
    "vi": {
        ILLNESS: ["bị sốt", "ốm", "con bị ốm"],
        TRANSPORTATION: ["lỡ xe buýt", "không có xe"],
        HOUSING: ["dọn nhà", "không có chỗ ở"],
        CAREGIVING: ["trông em", "việc gia đình gấp"],
        DISENGAGEMENT: ["không muốn đi học", "không chịu đi"],
        APPOINTMENT: ["đi khám", "hẹn bác sĩ"],
        UNKNOWN: ["ok", "nghỉ hôm nay"],
    },
    "zh": {
        ILLNESS: ["发烧", "生病了"],
        TRANSPORTATION: ["没车", "校车没来"],
        HOUSING: ["搬家"],
        CAREGIVING: ["看孩子", "家里有事"],
        DISENGAGEMENT: ["不想上学"],
        APPOINTMENT: ["看医生", "看牙"],
        UNKNOWN: ["好的", "今天不去"],
    },
}

# Code-switching: a native-language term swapped for the English word families often use.
_CODE_SWITCH: Dict[str, List[Tuple[str, str]]] = {
    "es": [("fiebre", "fever"), ("calentura", "fever"), ("gripe", "flu"), ("dolor de estómago", "stomach ache"),
           ("autobús", "bus"), ("camión", "bus"), ("carro", "car"), ("cita", "appointment"),
           ("dentista", "dentist"), ("guardería", "daycare"), ("bebé", "baby"), ("escuela", "school"),
           ("tareas", "homework"), ("maestro", "teacher"), ("niñera", "babysitter"), ("albergue", "shelter")],
    "vi": [("bác sĩ", "doctor"), ("xe buýt", "bus"), ("nha sĩ", "dentist"), ("sốt", "fever"),
           ("nhà trẻ", "daycare"), ("bài", "homework")],
    "zh": [("校车", "bus"), ("医生", "doctor"), ("发烧", "fever")],
}

_ABBREV: Dict[str, List[Tuple[str, str]]] = {
    "en": [("you", "u"), ("tomorrow", "tmrw"), ("because", "bc"), ("doesn't", "doesnt"), ("don't", "dont"),
           ("today", "2day"), ("please", "pls"), ("thanks", "thx"), ("I'm", "im"), ("appointment", "appt")],
    "es": [("que", "q"), ("porque", "xq"), ("también", "tb"), ("por favor", "porfa"), ("está", "esta")],
    "vi": [("không", "ko"), ("được", "dc"), ("hôm nay", "hnay")],
}


def _strip_accents(text: str) -> str:
    # Vietnamese đ has no combining form; map it explicitly as phone keyboards do.
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn"))


def _typo(text: str, rng: np.random.Generator) -> str:
    """Swap two adjacent letters, drop one, or double one inside a random word."""
    words = text.split(" ")
    idx = [i for i, w in enumerate(words) if sum(ch.isalpha() for ch in w) >= 4]
    if not idx:
        return text
    i = idx[int(rng.integers(len(idx)))]
    w = words[i]
    j = int(rng.integers(1, len(w) - 2)) if len(w) > 3 else 1
    op = int(rng.integers(3))
    if op == 0 and w[j].isalpha() and w[j + 1].isalpha():
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif op == 1 and w[j].isalpha():
        w = w[:j] + w[j + 1:]
    elif w[j].isalpha():
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def _noise(text: str, language: str, rng: np.random.Generator) -> str:
    if language in ("es", "vi") and rng.random() < 0.6:
        text = _strip_accents(text)
    for src, dst in _ABBREV.get(language, []):
        if src in text and rng.random() < 0.5:
            text = text.replace(src, dst)
    if language != "zh":
        for _ in range(int(rng.integers(1, 3))):
            text = _typo(text, rng)
        if rng.random() < 0.6:
            text = text.lower()
    if rng.random() < 0.6:
        text = text.rstrip(".!?。")
    if rng.random() < 0.15:
        text = text.replace(", ", " ").replace("，", " ")
    return text


def _code_switch(text: str, language: str, rng: np.random.Generator) -> str:
    hits = [(a, b) for a, b in _CODE_SWITCH.get(language, []) if a in text]
    if not hits:
        return text
    a, b = hits[int(rng.integers(len(hits)))]
    return text.replace(a, b, 1)


def _fill(template: str, language: str, rng: np.random.Generator, capitalize: bool = True) -> str:
    refs = _REFS[language]
    gender = "m" if rng.random() < 0.5 else "f"
    nouns, pron = refs[gender]
    c = nouns[int(rng.integers(len(nouns)))]
    vals = dict(pron)
    vals["c"] = c
    vals["C"] = c[:1].upper() + c[1:]
    vals["S"] = pron["s"][:1].upper() + pron["s"][1:]
    vals.setdefault("g", "o")
    out = template.format(**vals)
    return out[:1].upper() + out[1:] if capitalize else out


def make_reply(barrier: str, language: str, rng: np.random.Generator, style: str = "normal") -> str:
    """One synthetic family reply expressing `barrier` in `language`.

    `barrier` is an app.models.Barrier value (safety is not generated here). Unsupported
    languages fall back to English templates. Never returns an empty string.
    """
    if barrier == SAFETY:
        raise ValueError("safety replies are not generated here; use the frozen safety suite")
    lang = language if language in TEMPLATES else "en"
    if barrier not in TEMPLATES[lang]:
        raise ValueError(f"unknown barrier {barrier!r}")
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}")

    terse = style == "terse" or (style == "noisy" and rng.random() < 0.35)
    pool = TERSE[lang][barrier] if terse else TEMPLATES[lang][barrier]
    text = _fill(pool[int(rng.integers(len(pool)))], lang, rng, capitalize=not terse)
    if lang != "en" and rng.random() < (0.25 if style == "noisy" else 0.12):
        text = _code_switch(text, lang, rng)
    if style == "noisy":
        text = _noise(text, lang, rng)
    elif rng.random() < 0.08:  # even careful writers drop the final period sometimes
        text = text.rstrip(".!?。")
    text = text.strip()
    return text if text else pool[0].replace("{", "").replace("}", "")
