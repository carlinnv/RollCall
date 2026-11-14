"""Barrier classifier (stub).

A keyword baseline over English and Spanish replies. It exists so the product
flow works end to end; swap `classify` for the model-backed classifier without
changing callers. The safety check always runs first and wins over any other
label, so a safety-critical reply is never auto-answered.
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from ..models import Barrier, Classification

SAFETY_TERMS = [
    "bully", "bullied", "bullying", "scared", "afraid", "unsafe", "threat", "hurt", "hit him", "hit her",
    "abuse", "suicid", "self harm", "acoso", "acosan", "miedo", "amenaz", "le pegan", "golpe",
]

RULES: Dict[Barrier, List[str]] = {
    Barrier.TRANSPORTATION: [
        "bus", "ride", "car broke", "no car", "transport", "autobus", "camion", "carro", "parada",
        "descompuso", "no ride",
    ],
    Barrier.ILLNESS: [
        "sick", "fever", "flu", "throwing up", "vomit", "stomach", "cough", "covid", "ill ", "enfermo",
        "enferma", "fiebre", "vomito", "dolor", "gripe", "medico",
    ],
    Barrier.HOUSING_FOOD: [
        "evict", "motel", "shelter", "homeless", "no food", "hungry", "staying with", "desalojo",
        "albergue", "comida", "sin casa",
    ],
    Barrier.CAREGIVING: [
        "babysit", "watch his", "watch her", "take care", "caring for", "cuidar", "cuidando", "hermanito",
        "bebe", "had to work", "tuve que trabajar",
    ],
    Barrier.DISENGAGEMENT: [
        "hates school", "doesn't want", "does not want", "refused", "falling behind", "too behind", "bored",
        "no quiere", "atrasado",
    ],
    Barrier.APPOINTMENT: [
        "appointment", "dentist", "checkup", "travel", "funeral", "out of state", "cita", "viaje",
    ],
}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _sentence_with(text: str, term: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if term in _norm(sentence):
            return sentence.strip()
    return text.strip()


def is_safety_critical(text: str) -> Optional[str]:
    """Return the matching evidence sentence if the reply needs a counselor, else None."""
    norm = _norm(text)
    for term in SAFETY_TERMS:
        if term in norm:
            return _sentence_with(text, term)
    return None


def classify(text: str) -> Classification:
    evidence = is_safety_critical(text)
    if evidence:
        return Classification(barrier=Barrier.SAFETY, confidence=0.95, evidence=evidence, safety_critical=True)

    norm = _norm(text) + " "
    scores: List[Tuple[int, Barrier, str]] = []
    for barrier, terms in RULES.items():
        hits = [t for t in terms if t in norm]
        if hits:
            scores.append((len(hits), barrier, hits[0]))
    if not scores:
        return Classification(barrier=Barrier.UNKNOWN, confidence=0.0, evidence=text.strip())

    scores.sort(key=lambda s: s[0], reverse=True)
    count, barrier, term = scores[0]
    confidence = min(0.6 + 0.15 * count, 0.95)
    return Classification(barrier=barrier, confidence=round(confidence, 2), evidence=_sentence_with(text, term))
