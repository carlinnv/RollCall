"""Barrier classifier: keyword baseline (English, Spanish, Vietnamese).

This is the baseline from PRD 2 section 4 and the fallback for the model path in
`gemini_classifier.py`. Callers use `classify(text)`; the result type is
`models.Classification`.

Rules (PRD 2 section 3.5):
  * The safety check always runs first and wins over every other label, so a
    safety-critical reply is never auto-answered. Safety recall matters more
    than precision, so ambiguous harm words escalate unless the reply clearly
    describes an accident or sports injury.
  * Text is normalized before matching: lower case, curly quotes to ASCII,
    accents removed for English/Spanish terms. Vietnamese terms that carry
    diacritics are matched on the accented text, because stripping them merges
    distinct words (sợ "afraid" would become "so").
  * Terms match whole words (regex word boundaries), so "ill" does not match
    "still", "bus" does not match "busy" and "ride" does not match "pride".
  * Ties are broken by evidence strength (summed term weights, then the
    strongest single term, then the earliest match), never by dict order.
  * Fail closed: when a reply's language is not supported, the label is
    `unknown` (human review) unless the safety check fires.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

from ..models import Barrier, Classification

SUPPORTED_LANGUAGES = {"en", "es", "vi"}

_QUOTES = str.maketrans({"’": "'", "‘": "'", "‛": "'", "′": "'", "´": "'", "`": "'",
                         "“": '"', "”": '"', "„": '"', "″": '"'})


# ---------- normalization ----------
def normalize_quotes(text: str) -> str:
    return text.translate(_QUOTES)


def _norm_ascii(text: str) -> str:
    """Lower case, ASCII quotes, accents removed (đ is kept: it does not decompose)."""
    text = unicodedata.normalize("NFKD", normalize_quotes(text).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def _norm_accented(text: str) -> str:
    """Lower case, ASCII quotes, accents kept (NFC) for Vietnamese terms."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", normalize_quotes(text).lower()))


# Kept for callers of the old private helper.
_norm = _norm_ascii


@dataclass(frozen=True)
class Term:
    """One keyword rule. `pattern` is a regex body; word boundaries are added around it."""

    pattern: str
    weight: int = 2  # evidence strength: 1 weak, 2 typical, 3 specific
    accented: bool = False  # match on the accent-preserving view (Vietnamese)
    regex: Pattern = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        body = unicodedata.normalize("NFC", self.pattern)
        object.__setattr__(self, "regex", re.compile(rf"(?<!\w)(?:{body})(?!\w)"))


def _terms(weight: int, *patterns: str) -> List[Term]:
    """Build terms; any pattern with non-ASCII letters is matched on accented text."""
    return [Term(p, weight, accented=any(ord(c) > 127 for c in p)) for p in patterns]


class _Views:
    """Both normalized views of one piece of text."""

    def __init__(self, text: str) -> None:
        self.ascii = _norm_ascii(text)
        self.accented = _norm_accented(text)

    def search(self, term: Term) -> Optional[re.Match]:
        return term.regex.search(self.accented if term.accented else self.ascii)


# ---------- safety ----------
_PRON = r"(?:him|her|me|them|us|my (?:son|daughter|kid|child|boy|girl)|(?:the|my) kids?)"

# Escalate whenever these appear.
SAFETY_STRONG: List[Term] = (
    _terms(3,
           # bullying, threats, violence (en)
           r"bull(?:y|ies|ied|ying)", r"cyber ?bull\w*", r"threat\w*", r"harass\w*", r"intimidat\w*",
           r"pick(?:s|ed|ing)? on (?:him|her|them|me)", r"mak(?:e|es|ing) fun of", r"teas(?:e|es|ed|ing) (?:him|her|me)",
           rf"(?:hurt|hurts|hurting|hit|hits|hitting|beat|beats|beating|beat up|punch\w*|kick\w*|push\w*|shov\w*"
           rf"|chok\w*|slap\w*|smack\w*|grab\w*|touch\w*|attack\w*|jump\w*) {_PRON}(?!\w)",
           r"(?:someone|somebody|some kids|kids|a kid|a boy|a girl|classmates?|students?|older kids|they|a man|an adult)"
           r" (?:is |are |was |were |keeps? |has been |have been )?(?:hurt\w*|hit\w*|beat\w*|push\w*|punch\w*|kick\w*"
           r"|touch\w*|attack\w*)",
           r"beat(?:en)? up", r"got jumped", r"fight\w*(?! off| through| a (?:cold|flu|bug|fever|virus|cough))", r"weapon\w*", r"gun\w*", r"knife|knives",
           r"inappropriate\w*", r"molest\w*", r"abus\w*", r"neglect\w*", r"unsafe", r"not safe", r"danger\w*",
           r"scared", r"afraid", r"terrified", r"frightened",
           r"home alone", r"left alone", r"(?:nobody|no one|no adult) (?:is )?(?:at )?home",
           # self-harm (en), direct and indirect
           r"suicid\w*", r"self[ -]?harm\w*", r"cut(?:s|ting)? (?:himself|herself|myself|themselves)",
           r"(?:hurt|hurts|hurting|harm|harms|harming|kill|kills|killing) (?:himself|herself|myself|themselves)",
           r"(?:want|wants|wanted|wanting|wanna) to (?:die|be dead|disappear|end it)", r"wanna die",
           r"(?:wish(?:es|ed)?|says?) (?:he|she|i|they) (?:was|were|is) dead", r"better off dead", r"end (?:his|her|my|their) life",
           r"(?:doesn't|doesnt|does not|don't|dont|do not|didn't|didnt|did not|no longer|never) wants? to (?:live|be alive|be here anymore|wake up)",
           r"(?:not|no) (?:want(?:s|ing)? )?to live", r"no reason to live", r"kill (?:me|him|her|them)",
           # Spanish
           r"acos\w*", r"bull[yi]\w*", r"amenaz\w*", r"intimid\w*", r"miedo", r"asustad\w*", r"abus(?:o|an|aron|a|ando)\w*",
           r"maltrat\w*", r"(?:le|la|lo|me|nos|les|los|las) peg(?:a|an|o|aron|aba|aban|ue|uen|ando)",
           r"peg(?:an|aron)(?:le|lo|la|les)", r"pegandol[eao]s?", r"(?:lo|la|le|me|los|las) golpe(?:a|an|o|aron|aba|aban)",
           r"(?:lo|la|le|me|los|las) (?:lastim|empuj|patea|toc)(?:a|an|o|aron|aba|aban|ando)\w*",
           r"(?:le|me|nos|les) (?:hacen|hicieron|hace|hizo) dano", r"se burlan", r"pele(?:a|an|ar|aron|ando|as)", r"armas?", r"cuchillo",
           r"no (?:esta|estan|se siente) segur[oa]s?", r"inseguro", r"sol[oa]s? en (?:la )?casa", r"(?:lo|la) dejan sol[oa]",
           r"suicid\w*", r"quitarse la vida", r"hacerse dano", r"(?:se|me) (?:quiere|quiero|queria) morir\w*",
           r"(?:quiere|quiero|queria) morir\w*", r"(?:matarse|matarme|suicidarse|cortarse)",
           r"(?:ya )?no (?:quiere|quiero|queria|quieren) (?:vivir|estar viv[oa]|seguir viviendo)",
           r"(?:lo|la|me|nos) (?:van|va) a matar",
           # Vietnamese (accented, plus common unaccented typing)
           r"bắt nạt", r"bị bắt nạt", r"ăn hiếp", r"sợ", r"đánh", r"bị đánh", r"đe dọa", r"đe doạ", r"dọa", r"doạ",
           r"xâm hại", r"lạm dụng", r"quấy rối", r"sờ mó", r"ngược đãi", r"không an toàn", r"ở nhà một mình",
           r"tự tử", r"tự sát", r"tự làm hại", r"tự hại", r"muốn chết", r"không muốn sống", r"chẳng muốn sống",
           r"bat nat", r"bi danh", r"muon chet", r"khong muon song", r"tu sat", r"so di hoc", r"bi doa",
           )
)

# Harm words that are also used for accidents and sports injuries: escalate
# unless the reply clearly describes one (body part, sport, fall, sprain).
SAFETY_CONTEXTUAL: List[Term] = _terms(
    2,
    r"hurt\w*", r"injur\w*", r"bruis\w*", r"bleed\w*", r"black eye", r"hit", r"was hit", r"got hit",
    r"lastim\w*", r"golpe\w*", r"herid[oa]s?", r"moreton\w*", r"bị thương", r"bầm",
)
INJURY_CONTEXT: List[Term] = _terms(
    1,
    r"ankles?", r"knees?", r"wrists?", r"arms?", r"legs?", r"foot|feet", r"toes?", r"fingers?", r"elbows?",
    r"shoulders?", r"back", r"necks?", r"hands?", r"hips?", r"sprain\w*", r"broke (?:his|her|my) \w+",
    r"broken (?:arm|leg|wrist|ankle|finger|toe|foot)", r"fractur\w*", r"cast", r"stitches",
    r"soccer|football|basketball|baseball|softball|volleyball|hockey|gym|pe|recess game|practice|game|match"
    r"|tournament|sports?|bike|skateboard\w*|trampoline|playground|playing",
    r"fell|falls?|falling|tripped|slipped", r"accident\w*", r"car accident",
    r"tobillos?", r"rodillas?", r"munecas?", r"brazos?", r"piernas?", r"pies?", r"dedos?", r"espalda",
    r"futbol|basquetbol|beisbol|jugando|partido|entrenamiento|bicicleta",
    r"se cayo|se callo|cayo|caida|accidente|esguince|torcedura|yeso",
    r"chân", r"tay", r"mắt cá", r"đầu gối", r"bóng đá", r"té", r"ngã", r"tai nạn",
)


def _sentences(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    return parts or [text.strip()]


def _evidence(text: str, term: Term) -> str:
    """The sentence of `text` in which `term` matches (the whole reply as a last resort)."""
    for sentence in _sentences(text):
        if _Views(sentence).search(term):
            return sentence
    return text.strip()


@dataclass
class SafetyHit:
    evidence: str
    term: str
    strength: str  # "strong" | "contextual"


def safety_check(text: str) -> Optional[SafetyHit]:
    """Return why the reply needs a counselor, or None. Errs toward escalation."""
    views = _Views(text)
    for term in SAFETY_STRONG:
        if views.search(term):
            return SafetyHit(_evidence(text, term), term.pattern, "strong")
    for term in SAFETY_CONTEXTUAL:
        if views.search(term):
            sentence = _evidence(text, term)
            # Only a sentence that itself names a body part, sport or fall clears it.
            if any(_Views(sentence).search(c) for c in INJURY_CONTEXT):
                continue
            return SafetyHit(sentence, term.pattern, "contextual")
    return None


def is_safety_critical(text: str) -> Optional[str]:
    """Return the matching evidence sentence if the reply needs a counselor, else None."""
    hit = safety_check(text)
    return hit.evidence if hit else None


# ---------- barriers ----------
RULES: Dict[Barrier, List[Term]] = {
    Barrier.TRANSPORTATION: [
        *_terms(3, r"bus(?:es)?", r"school bus", r"bus stop", r"transport\w*", r"autobus\w*", r"no ride",
                r"car (?:broke|broken|died|wouldn't start|won't start|wont start|trouble|is in the shop)",
                r"no car", r"flat tire", r"no (?:gas|way to get (?:him|her|them) there)", r"descompu\w*",
                r"no tenemos (?:carro|coche|auto|como llevarl[oa])", r"no (?:tengo|tenia|hay) como llevarl[oa]",
                r"xe buýt", r"xe buyt", r"lỡ xe", r"xe hư", r"hư xe", r"không có xe", r"đưa đón"),
        *_terms(2, r"rides?", r"carro", r"coche", r"camion", r"parada", r"raite"),
        *_terms(1, r"never came", r"no paso", r"no llego"),
    ],
    Barrier.ILLNESS: [
        *_terms(3, r"sick", r"(?<!')ill", r"illness", r"fever\w*", r"flu", r"throw(?:ing|s)? up", r"threw up", r"vomit\w*",
                r"stomach (?:bug|ache|flu|virus|pain)", r"stomachache", r"cough\w*", r"covid\w*", r"strep",
                r"(?:a|the) cold", r"ear infection", r"infection", r"headache|migraine", r"asthma", r"diarrh\w*",
                r"nause\w*", r"chicken ?pox", r"pink ?eye", r"not feeling (?:well|good)", r"under the weather",
                r"enferm\w*", r"fiebre", r"calentura", r"vomit\w*", r"gripe|gripa", r"resfri\w*",
                r"dolor de (?:estomago|cabeza|garganta|oido|panza)", r"no se siente bien", r"tos",
                r"ốm", r"bệnh", r"sốt", r"bị cảm", r"cảm cúm", r"cúm", r"nôn", r"ói", r"đau bụng", r"bị ho",
                r"bi om", r"bi sot", r"bi benh", r"bi cam"),
        *_terms(2, r"stomach", r"se siente mal", r"doctor said", r"rash", r"lice", r"piojos", r"đau",
                # accidents and sports injuries (they clear the contextual safety check above)
                r"hurt (?:his|her|my|their) \w+", r"injur\w*", r"sprain\w*", r"fractur\w*", r"stitches",
                r"broke (?:his|her|my|their) (?:arm|leg|wrist|ankle|finger|toe|foot|nose)", r"se lastimo", r"esguince",
                r"se (?:fracturo|quebro|rompio) (?:el|la) \w+", r"bị thương"),
        *_terms(1, r"doctor", r"hospital", r"medico", r"clinic\w*", r"dolor", r"bác sĩ"),
    ],
    Barrier.HOUSING_FOOD: [
        *_terms(3, r"evict\w*", r"motel", r"shelter", r"homeless", r"no food", r"hungry", r"food bank",
                r"lost (?:our|the) (?:house|apartment|home|place)", r"no place to (?:stay|live|sleep)", r"kicked out",
                r"(?:power|electricity|water|lights) (?:was |got |is )?(?:shut|cut) off", r"sleeping in (?:the|our|a) car",
                r"desaloj\w*", r"albergue", r"refugio", r"sin casa", r"sin hogar", r"no tenemos (?:comida|casa|donde vivir)",
                r"no hay comida", r"nos (?:corrieron|sacaron) de",
                r"bị đuổi", r"mất nhà", r"không có nhà", r"không có (?:đồ ăn|thức ăn|gì ăn)", r"đói"),
        *_terms(2, r"staying (?:with|at) (?:friends|family|relatives|my|his|her|a friend)", r"rent", r"hotel",
                r"(?:nos )?quedando con", r"renta"),
        *_terms(1, r"comida", r"food", r"moved", r"mudamos|mudanza"),
    ],
    Barrier.CAREGIVING: [
        *_terms(3, r"babysit\w*", r"watch (?:his|her|their|the) (?:little |baby |younger )?(?:brother|sister|siblings?|baby|kids)",
                r"tak(?:e|ing|es) care of", r"caring for", r"no (?:childcare|child care|babysitter|one to watch)",
                r"cuidar\w*", r"cuidando", r"cuida a", r"translat\w* for", r"interpret\w* for",
                r"trông em", r"chăm sóc", r"giữ em", r"trong em", r"cham soc"),
        *_terms(2, r"watch (?:his|her)", r"had to work", r"tuve que (?:trabajar|ir a trabajar)", r"phải đi làm",
                r"(?:grandma|grandpa|abuela|abuelo|mom|dad|mama|papa) (?:is|was|esta) (?:sick|in the hospital|enferm\w*)"),
        *_terms(1, r"hermanit\w*", r"bebe", r"baby"),
    ],
    Barrier.DISENGAGEMENT: [
        *_terms(3, r"hates? (?:going to )?school", r"hates? (?:his|her) (?:class|teacher)",
                r"(?:doesn't|doesnt|does not|don't|dont|won't|wont|will not|refuses to) (?:want to )?go(?: to school)?",
                r"falling behind", r"too behind", r"behind in", r"skipp\w*", r"ditch\w*",
                r"no quiere ir", r"no le gusta (?:la escuela|ir)", r"se niega", r"atrasad\w*",
                r"không muốn đi học", r"chán học", r"bỏ học", r"không chịu đi học", r"khong muon di hoc"),
        *_terms(2, r"(?:doesn't|doesnt|does not) want", r"refus\w*", r"bored", r"no quiere", r"aburrid\w*",
                r"failing", r"reprob\w*", r"lười học"),
        *_terms(1, r"anxi\w*", r"stress\w*", r"overwhelm\w*", r"ansiedad", r"estres\w*"),
    ],
    Barrier.APPOINTMENT: [
        *_terms(3, r"appointment\w*", r"appt", r"dentist", r"orthodont\w*", r"check ?-?ups?", r"funeral", r"out of (?:state|town|the country)",
                r"cita", r"dentista", r"funeral|velorio|entierro", r"fuera de (?:la ciudad|el estado|del pais)",
                r"lịch hẹn", r"hẹn bác sĩ", r"nha sĩ", r"đám tang", r"đám ma", r"lich hen", r"nha si"),
        *_terms(2, r"travel\w*", r"trip", r"vacation", r"court", r"visiting (?:family|relatives|grandma|grandpa)",
                r"viaje", r"viajando", r"vacaciones", r"de viaje", r"đi du lịch", r"về quê", r"ve que"),
    ],
}

# Fixed last-resort order for exact ties (stable, documented, independent of dict order).
TIE_ORDER: Sequence[Barrier] = (
    Barrier.HOUSING_FOOD, Barrier.ILLNESS, Barrier.TRANSPORTATION, Barrier.CAREGIVING,
    Barrier.APPOINTMENT, Barrier.DISENGAGEMENT,
)


@dataclass
class _Score:
    barrier: Barrier
    total: int
    strongest: int
    first_pos: int
    term: Term
    matched: int


def _score(text: str) -> List[_Score]:
    views = _Views(text)
    out: List[_Score] = []
    for barrier, terms in RULES.items():
        best: Optional[Tuple[int, int, Term]] = None  # (-weight, pos, term)
        total, matched, first = 0, 0, 10 ** 9
        for term in terms:
            m = views.search(term)
            if not m:
                continue
            total += term.weight
            matched += 1
            first = min(first, m.start())
            cand = (-term.weight, m.start(), term)
            if best is None or cand[:2] < best[:2]:
                best = cand
        if best is not None:
            out.append(_Score(barrier, total, -best[0], first, best[2], matched))
    out.sort(key=lambda s: (-s.total, -s.strongest, s.first_pos, TIE_ORDER.index(s.barrier)))
    return out


def language_supported(language: Optional[str]) -> bool:
    """None/""/"auto" means not stated; otherwise the ISO 639-1 prefix must be en, es or vi."""
    if not language or language.lower() in {"auto", "und", "unknown"}:
        return True
    return language.lower().replace("_", "-").split("-")[0] in SUPPORTED_LANGUAGES


def classify(text: str, language: Optional[str] = None) -> Classification:
    hit = safety_check(text)
    if hit:
        confidence = 0.95 if hit.strength == "strong" else 0.8
        return Classification(barrier=Barrier.SAFETY, confidence=confidence, evidence=hit.evidence,
                              safety_critical=True)

    if not language_supported(language):
        # Fail closed: a person reads replies in languages we cannot check.
        return Classification(barrier=Barrier.UNKNOWN, confidence=0.0, evidence=text.strip())

    scores = _score(text)
    if not scores:
        return Classification(barrier=Barrier.UNKNOWN, confidence=0.0, evidence=text.strip())

    top = scores[0]
    confidence = min(0.5 + 0.1 * top.total, 0.95)
    if len(scores) > 1 and scores[1].total >= top.total:
        confidence -= 0.2  # a tie on strength: low confidence so it goes to the clerk
    elif len(scores) > 1:
        confidence -= 0.1
    return Classification(barrier=top.barrier, confidence=round(max(confidence, 0.3), 2),
                          evidence=_evidence(text, top.term))
