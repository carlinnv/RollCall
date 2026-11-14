"""Retrieval over family messages for the F7 counselor assistant (PRD 2 section 3.4).

    index = MessageIndex(messages, group_sizes=data.group_sizes())
    hits = index.search("Why are students on route R03 missing school this week?", k=5, as_of=74)

Pieces (all local, numpy only, no network):
  * Tokenizer: NFKC, lowercase, accents stripped (so "pasó" == "paso" == a typo'd "paso"), crude
    English plural strip, CJK split into characters; unigrams plus adjacent bigrams.
  * BM25 (k1 = 1.2, b = 0.75) over unigrams + bigrams with weighted query terms.
  * Cross-lingual barrier lexicon: English cause words in the query ("bus", "sick", "childcare")
    pull in a small hand-written en/es/vi/zh keyword list for that barrier at a lower weight, so an
    English query matches "no pasó el camión", "xe buýt không đến" and "校车没来". Multi-word entries
    only add their bigrams, which keeps Vietnamese syllables ("nha", "xe") from matching alone.
  * Metadata filter parsed from the query with regex rules: route ("route R03", "bus 3"), section
    ("SEC-04", "class 4"), neighborhood ("NB2"), student ("N012"), language ("Spanish-speaking"),
    time ("week 12" = school days 55-59, "around day 62" = 59-65, "days 60-64", "this week",
    "last 3 days", "today" relative to as_of) and cause. Location/time/language/student are hard
    filters; the cause is a soft boost on the pipeline's observed label (labels are noisy, a hard
    filter would cap recall at label accuracy).
  * "Which class/route/neighborhood ..." questions: zones are ranked by a Poisson LLR of
    cause-relevant messages against the zone's share of students, and the top zone is boosted.
  * Dense: there is no semantic embedding model by default. Gemini embeddings would hit free-tier
    rate limits, so `method="dense"` is a local hashed character n-gram TF-IDF vector baseline
    (cosine), and `MessageIndex(embed_fn=...)` accepts any callable(list[str]) -> array to plug in
    real embeddings later. It is off unless asked for.
"""
from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .corpus import Message, week_days, week_of

ILLNESS = "illness"
TRANSPORTATION = "transportation"
HOUSING = "housing or food insecurity"
CAREGIVING = "caregiving or family obligation"
SAFETY = "safety or bullying"
DISENGAGEMENT = "disengagement or academic struggle"
APPOINTMENT = "appointment or travel"

# ----------------------------------------------------------------------------------------------
# lexicon

# English words in a query that name a cause (matched on normalized tokens / bigrams).
CAUSE_TRIGGERS: Dict[str, Tuple[str, ...]] = {
    ILLNESS: ("sick", "ill", "illness", "fever", "flu", "cold", "cough", "vomit", "vomiting", "stomach",
              "headache", "outbreak", "contagion", "contagious", "virus", "symptom", "sickness", "disease"),
    TRANSPORTATION: ("transportation", "transport", "bus", "buse", "ride", "car", "drive", "driving",
                     "gas", "commute"),
    HOUSING: ("housing", "homeless", "homelessness", "eviction", "evicted", "shelter", "motel", "moving",
              "food", "hungry", "utilitie", "utility", "insecurity"),
    CAREGIVING: ("caregiving", "childcare", "babysit", "babysitting", "babysitter", "daycare", "sibling",
                 "baby", "grandma", "caring", "caregiver"),
    DISENGAGEMENT: ("disengaged", "disengagement", "refuse", "refusing", "bored", "failing", "struggling",
                    "motivation", "unmotivated", "stressed", "academic"),
    APPOINTMENT: ("appointment", "dentist", "doctor", "clinic", "checkup", "travel", "traveling", "trip",
                  "vacation"),
    SAFETY: ("safety", "bullying", "bullied", "bully", "threat", "threatened", "unsafe", "abuse"),
}
# Multilingual keywords per barrier (document side). Hand-written; overlaps the simulator's
# vocabulary by design of the domain, see the limitations note in the benchmark output.
BARRIER_LEXICON: Dict[str, Dict[str, Tuple[str, ...]]] = {
    ILLNESS: {
        "en": ("sick", "fever", "flu", "cold", "cough", "throwing up", "vomiting", "stomach ache", "stomach bug",
               "sore throat", "headache", "temperature", "not feeling well"),
        "es": ("enfermo", "enferma", "fiebre", "gripe", "gripa", "tos", "vomito", "dolor de estomago",
               "dolor de cabeza", "calentura", "garganta", "se siente mal", "malito", "malita"),
        "vi": ("bị sốt", "bị ốm", "bị cảm", "đau bụng", "nôn cả", "sốt", "ốm"),
        "zh": ("发烧", "生病", "感冒", "咳嗽", "一直吐"),
    },
    TRANSPORTATION: {
        "en": ("bus", "missed the bus", "no ride", "car broke down", "ride", "gas money", "transportation",
               "bus stop", "drive"),
        "es": ("autobus", "camion", "carro", "transporte", "descompuso", "taller", "gasolina", "parada",
               "perdio el bus", "no paso el bus"),
        "vi": ("xe buýt", "lỡ xe", "không có xe", "chở con", "xe của tôi bị hư"),
        "zh": ("校车", "没车", "车坏了", "送孩子"),
    },
    CAREGIVING: {
        "en": ("babysit", "babysitter", "daycare", "watch the baby", "watching baby", "little brother",
               "younger kids", "family emergency", "take care of", "grandma", "little ones"),
        "es": ("guarderia", "cuidando", "cuidar", "bebe", "hermanito", "emergencia familiar", "abuela",
               "ninera", "ninos chiquitos"),
        "vi": ("trông em", "nhà trẻ", "chăm sóc", "việc gia đình gấp", "gia đình có việc gấp"),
        "zh": ("照顾", "托儿所", "看孩子", "家里有事", "弟弟", "妹妹"),
    },
    HOUSING: {
        "en": ("evicted", "shelter", "motel", "moving", "apartment", "staying with family", "lost our place",
               "shut off", "no food", "no home"),
        "es": ("albergue", "mudando", "apartamento", "nos sacaron", "la luz", "donde quedarnos", "comida",
               "sin casa"),
        "vi": ("dọn nhà", "đuổi nhà", "tạm trú", "cắt điện", "ở nhờ", "không có chỗ ở"),
        "zh": ("搬家", "赶出来", "临时住所", "停水停电", "亲戚家"),
    },
    DISENGAGEMENT: {
        "en": ("refuses", "refused", "doesnt want to go", "want to go anymore", "wont go", "bored", "failing",
               "behind on", "stressed", "too hard", "pointless", "out of bed", "homework", "feels lost"),
        "es": ("no quiere ir", "no quiso ir", "dificiles", "atrasado", "atrasada", "estresado", "estresada",
               "perdido", "perdida", "no le gusta", "no sirve"),
        "vi": ("không muốn đi học", "không chịu", "chậm bài", "căng thẳng", "khó quá"),
        "zh": ("不想上学", "不肯", "跟不上", "压力", "太难"),
    },
    APPOINTMENT: {
        "en": ("appointment", "appt", "dentist", "doctor", "checkup", "clinic", "family trip", "travel",
               "out of town", "eye doctor"),
        "es": ("cita", "dentista", "doctor", "chequeo", "clinica", "viaje", "fuera de la ciudad", "viajar"),
        "vi": ("có hẹn", "nha sĩ", "bác sĩ", "đi khám", "đi xa"),
        "zh": ("看牙", "看医生", "出远门", "牙医"),
    },
    SAFETY: {
        "en": ("bullied", "bullying", "threatened", "scared", "hurt", "unsafe"),
        "es": ("acoso", "amenaza", "miedo", "lastimaron"),
        "vi": ("bắt nạt", "đe dọa", "sợ"),
        "zh": ("欺负", "威胁", "害怕"),
    },
}

STOPWORDS = frozenset("""
a an the and or of on in at to for from by with about is are was were be been being has have had do does did
why what which who whom whose where when how any there this that these those it its their them they our we
i me my you your he she his her us around near during since after before between than then so as
student students kid's child children families parent parents school schools missing miss absent absence
absences reporting reported report reports said say says saying tell told week weeks day days today yesterday
class classes section sections route routes neighborhood neighborhoods neighbourhood area zone cluster clusters
problem problems issue issues wave school-wide schoolwide home staying keeping keep out show shows showing
spanish vietnamese chinese mandarin cantonese english speaking language languages all last past many
much more most some people message messages reply replies text texts summary summarize explain
""".split())

LANGUAGE_WORDS = {"spanish": "es", "vietnamese": "vi", "chinese": "zh", "mandarin": "zh", "cantonese": "zh",
                  "english": "en"}

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[^\W_]+", re.UNICODE)


# ----------------------------------------------------------------------------------------------
# text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("đ", "d")
    nfd = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", text).replace("'", "")


def _stem(tok: str) -> str:
    if tok.isascii() and len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def words(text: str) -> List[str]:
    """Normalized word tokens (CJK as single characters)."""
    return [_stem(t) for t in _WORD.findall(normalize(text))]


_STOP = frozenset(STOPWORDS | {_stem(normalize(w)) for w in STOPWORDS})


def tokens(text: str, bigrams: bool = True) -> List[str]:
    """Unigrams plus adjacent bigrams ("xe buyt"); bigrams never cross a CJK/Latin boundary oddly."""
    ws = words(text)
    out = list(ws)
    if bigrams:
        out += [f"{a} {b}" for a, b in zip(ws, ws[1:])]
    return out


def phrase_terms(phrase: str) -> List[str]:
    """Query terms for one lexicon phrase: its bigrams when it has 2+ words, else the word."""
    ws = words(phrase)
    if len(ws) >= 2:
        return [f"{a} {b}" for a, b in zip(ws, ws[1:])]
    return ws


def expansion_terms(barrier: str, languages: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Weighted terms that express `barrier` in every (or the given) language."""
    out: Dict[str, float] = {}
    for lang, phrases in BARRIER_LEXICON.get(barrier, {}).items():
        if languages is not None and lang not in languages:
            continue
        for p in phrases:
            for t in phrase_terms(p):
                out[t] = 1.0
    return out


# ----------------------------------------------------------------------------------------------
# query parsing


@dataclass
class ParsedQuery:
    text: str
    route: Optional[str] = None
    section: Optional[str] = None
    neighborhood: Optional[str] = None
    student: Optional[str] = None
    languages: Optional[List[str]] = None
    day_from: Optional[int] = None
    day_to: Optional[int] = None
    causes: List[str] = field(default_factory=list)
    group_by: Optional[str] = None            # "section" | "route" | "neighborhood" for "which X" questions
    content_terms: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def filters(self) -> Dict[str, object]:
        d = {k: getattr(self, k) for k in ("route", "section", "neighborhood", "student", "languages",
                                            "day_from", "day_to", "group_by")}
        d["causes"] = list(self.causes)
        return {k: v for k, v in d.items() if v not in (None, [], "")}


def _match_known(value: int, known: Iterable[str], prefix: str) -> str:
    """Map a number to a known zone id with that numeric part ("3" -> "R03"), else a padded guess."""
    for z in known:
        m = re.search(r"(\d+)$", z)
        if m and int(m.group(1)) == value:
            return z
    return f"{prefix}{value:02d}" if prefix != "NB" else f"NB{value}"


def parse_query(query: str, as_of: Optional[int] = None,
                known: Optional[Mapping[str, Iterable[str]]] = None) -> ParsedQuery:
    """Rule-based metadata extraction. `known` = {"route": [...], "section": [...], "neighborhood": [...]}."""
    known = known or {}
    pq = ParsedQuery(text=query)
    q = query
    spans: List[Tuple[int, int]] = []

    def take(m: re.Match) -> None:
        spans.append(m.span())

    # zones
    m = re.search(r"\b(?:bus\s+)?route\s*#?\s*R?-?(\d{1,3})\b|\bbus\s*#?\s*R?(\d{1,3})\b|\bR(\d{1,3})\b", q, re.I)
    if m:
        num = int(next(g for g in m.groups() if g))
        pq.route = _match_known(num, known.get("route", ()), "R")
        take(m)
    m = re.search(r"\bSEC-?(\d{1,3})\b|\b(?:class(?:room)?|section)\s*#?\s*(\d{1,3})\b", q, re.I)
    if m:
        num = int(next(g for g in m.groups() if g))
        pq.section = _match_known(num, known.get("section", ()), "SEC-")
        take(m)
    m = re.search(r"\bNB-?(\d{1,3})\b|\bneighbou?rhood\s*#?\s*(\d{1,3})\b", q, re.I)
    if m:
        num = int(next(g for g in m.groups() if g))
        pq.neighborhood = _match_known(num, known.get("neighborhood", ()), "NB")
        take(m)
    m = re.search(r"\bN\d{3,4}\b", q)
    if m:
        pq.student = m.group(0)
        take(m)

    # languages ("Spanish-speaking families", "in Vietnamese"); bare "English" only with speaking/in
    langs: List[str] = []
    for m in re.finditer(r"\b(spanish|vietnamese|chinese|mandarin|cantonese|english)\b(-|\s)?(speaking)?", q, re.I):
        w = m.group(1).lower()
        before = q[max(0, m.start() - 3):m.start()].lower()
        if w == "english" and not (m.group(3) or before.endswith("in ")):
            continue
        langs.append(LANGUAGE_WORDS[w])
    if langs:
        pq.languages = sorted(set(langs))

    # time
    lo = hi = None
    m = re.search(r"\bdays?\s+(\d{1,3})\s*(?:-|–|to|through|and)\s*(?:day\s+)?(\d{1,3})\b", q, re.I)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        take(m)
    if lo is None:
        m = re.search(r"\b(?:around|near|about|approximately|circa)\s+(?:school\s+)?day\s+(\d{1,3})\b", q, re.I)
        if m:
            d = int(m.group(1))
            lo, hi = max(0, d - 3), d + 3
            take(m)
    if lo is None:
        m = re.search(r"\b(?:since|after|from)\s+(?:school\s+)?day\s+(\d{1,3})\b", q, re.I)
        if m:
            lo, hi = int(m.group(1)), as_of
            take(m)
    if lo is None:
        m = re.search(r"\b(?:school\s+)?day\s+(\d{1,3})\b", q, re.I)
        if m:
            lo = hi = int(m.group(1))
            take(m)
    if lo is None:
        m = re.search(r"\b(?:school\s+)?week\s+(\d{1,2})\b", q, re.I)
        if m:
            lo, hi = week_days(int(m.group(1)))
            take(m)
    if lo is None and as_of is not None:
        if re.search(r"\bthis week\b", q, re.I):
            lo, hi = week_days(week_of(as_of))[0], as_of
        elif re.search(r"\blast week\b", q, re.I):
            lo, hi = week_days(week_of(as_of) - 1)
        elif (m := re.search(r"\b(?:last|past)\s+(\d{1,2})\s+(?:school\s+)?days\b", q, re.I)):
            lo, hi = max(0, as_of - int(m.group(1)) + 1), as_of
        elif re.search(r"\btoday\b", q, re.I):
            lo = hi = as_of
        elif re.search(r"\byesterday\b", q, re.I):
            lo = hi = as_of - 1
    elif lo is None and re.search(r"\b(this|last) week\b|\btoday\b|\byesterday\b", q, re.I):
        pq.notes.append("relative date ignored: no as_of given")
    pq.day_from, pq.day_to = lo, hi

    # "which class / route / neighborhood"
    m = re.search(r"\bwhich\s+(class(?:room)?|section|grade|bus\s+route|route|bus|neighbou?rhood|area)\b", q, re.I)
    if m:
        w = m.group(1).lower()
        pq.group_by = ("section" if w.startswith(("class", "section", "grade")) else
                       "route" if "route" in w or w == "bus" else "neighborhood")

    # content terms: what is left after removing metadata spans
    rest = q
    for a, b in sorted(spans, reverse=True):
        rest = rest[:a] + " " + rest[b:]
    ws = [w for w in words(rest) if w not in _STOP and not w.isdigit()]
    pq.content_terms = ws
    grams = set(ws) | {f"{a} {b}" for a, b in zip(ws, ws[1:])}
    for barrier, trig in CAUSE_TRIGGERS.items():
        # English trigger words, or any lexicon term (a query may quote "autobús" or "fiebre")
        if any(t in grams for t in trig) or grams & set(expansion_terms(barrier)):
            pq.causes.append(barrier)
    # "missed the bus" style phrases also count
    norm = normalize(q)
    if "family emergency" in norm and CAREGIVING not in pq.causes:
        pq.causes.append(CAREGIVING)
    if pq.group_by == "route" and not pq.causes:
        pq.causes.append(TRANSPORTATION)
    return pq


# ----------------------------------------------------------------------------------------------
# scorers


class BM25:
    """Okapi BM25 over pre-tokenized documents with weighted query terms (numpy only)."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                postings[t][i] = c
        self.postings = {t: (np.fromiter(p.keys(), dtype=np.int64), np.fromiter(p.values(), dtype=float))
                         for t, p in postings.items()}

    def idf(self, term: str) -> float:
        p = self.postings.get(term)
        df = 0 if p is None else len(p[0])
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: Mapping[str, float]) -> np.ndarray:
        s = np.zeros(self.n)
        if not self.n:
            return s
        norm = self.k1 * (1.0 - self.b + self.b * self.dl / max(self.avgdl, 1e-9))
        for term, w in query.items():
            p = self.postings.get(term)
            if p is None or w == 0:
                continue
            idx, tf = p
            s[idx] += w * self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm[idx])
        return s


class HashedVectors:
    """Local 'dense' baseline: hashed char 3-4-gram TF-IDF vectors, cosine similarity.

    Not a semantic embedding: it captures spelling overlap (typos, accents, shared stems) only.
    Pass `embed_fn` to MessageIndex to use real embeddings instead.
    """

    def __init__(self, texts: Sequence[str], dim: int = 4096, ngram: Tuple[int, int] = (3, 4)):
        self.dim, self.ngram = dim, ngram
        raw = np.stack([self._raw(t) for t in texts]) if texts else np.zeros((0, dim), dtype=np.float32)
        df = (raw > 0).sum(axis=0)
        self.idf = np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0
        self.mat = self._finish(raw)

    def _raw(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        s = "#" + "#".join(words(text)) + "#"
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(max(0, len(s) - n + 1)):
                v[zlib.crc32(s[i:i + n].encode("utf-8")) % self.dim] += 1.0
        return v

    def _finish(self, raw: np.ndarray) -> np.ndarray:
        x = np.log1p(raw) * self.idf
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(n, 1e-9)

    def scores(self, weighted_texts: Mapping[str, float]) -> np.ndarray:
        q = np.zeros(self.dim, dtype=np.float32)
        for text, w in weighted_texts.items():
            q += w * self._raw(text)
        if not q.any():
            return np.zeros(self.mat.shape[0])
        return self.mat @ self._finish(q[None, :])[0]


class EmbeddingVectors:
    """Wrapper for a real embedding function (e.g. Gemini embeddings); not used by default."""

    def __init__(self, texts: Sequence[str], embed_fn: Callable[[List[str]], np.ndarray]):
        self.embed_fn = embed_fn
        m = np.asarray(embed_fn(list(texts)), dtype=float)
        self.mat = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def scores(self, weighted_texts: Mapping[str, float]) -> np.ndarray:
        text = " ".join(weighted_texts)
        q = np.asarray(self.embed_fn([text]), dtype=float)[0]
        return self.mat @ (q / max(np.linalg.norm(q), 1e-9))


# ----------------------------------------------------------------------------------------------
# index


@dataclass
class Hit:
    message: Message
    score: float
    rank: int
    reasons: List[str] = field(default_factory=list)

    @property
    def message_id(self) -> str:
        return self.message.message_id


@dataclass
class SearchResult:
    hits: List[Hit]
    parsed: ParsedQuery
    n_candidates: int
    method: str
    top_group: Optional[str] = None

    @property
    def ids(self) -> List[str]:
        return [h.message_id for h in self.hits]


EXPANSION_WEIGHT = 0.35
CAUSE_BOOST = 0.5
GROUP_BOOST = 1.0
DENSITY_BOOST = 0.25
METHODS = ("bm25", "dense", "hybrid")


class MessageIndex:
    def __init__(self, messages: Sequence[Message], group_sizes: Optional[Mapping[str, Mapping[str, int]]] = None,
                 embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None, dense: bool = True):
        self.messages = list(messages)
        self.by_id = {m.message_id: m for m in self.messages}
        self.bm25 = BM25([tokens(m.text) for m in self.messages])
        self._dense = None
        self._dense_enabled = dense
        self._embed_fn = embed_fn
        self.group_sizes = {k: dict(v) for k, v in (group_sizes or {}).items()}
        self.known = {layer: sorted({getattr(m, layer) for m in self.messages if getattr(m, layer)} |
                                    set(self.group_sizes.get(layer, {})))
                      for layer in ("route", "section", "neighborhood")}
        self.day = np.array([m.day for m in self.messages], dtype=np.int64)
        self.received = np.array([m.received_day for m in self.messages], dtype=np.int64)

    @property
    def dense(self):
        if self._dense is None:
            if not self._dense_enabled:
                raise RuntimeError("dense retrieval is disabled for this index")
            texts = [m.text for m in self.messages]
            self._dense = EmbeddingVectors(texts, self._embed_fn) if self._embed_fn else HashedVectors(texts)
        return self._dense

    # -- query construction --
    def query_terms(self, pq: ParsedQuery, expand: bool = True) -> Dict[str, float]:
        terms: Dict[str, float] = {}
        ws = pq.content_terms
        for t in ws + [f"{a} {b}" for a, b in zip(ws, ws[1:])]:
            terms[t] = terms.get(t, 0.0) + 1.0
        if expand:
            for c in pq.causes:
                for t, w in expansion_terms(c, pq.languages).items():
                    terms[t] = max(terms.get(t, 0.0), EXPANSION_WEIGHT * w)
        return terms

    def dense_query(self, pq: ParsedQuery, expand: bool = True) -> Dict[str, float]:
        texts = {" ".join(pq.content_terms): 1.0} if pq.content_terms else {}
        if expand:
            for c in pq.causes:
                for lang, phrases in BARRIER_LEXICON.get(c, {}).items():
                    if pq.languages is None or lang in pq.languages:
                        for p in phrases:
                            texts[p] = EXPANSION_WEIGHT
        return texts

    def candidate_mask(self, pq: ParsedQuery, as_of: Optional[int] = None) -> np.ndarray:
        mask = np.ones(len(self.messages), dtype=bool)
        if as_of is not None:
            mask &= self.received <= as_of
        for layer in ("route", "section", "neighborhood", "student"):
            v = getattr(pq, layer)
            if v is not None:
                mask &= np.array([getattr(m, layer) == v for m in self.messages], dtype=bool)
        if pq.languages:
            mask &= np.array([m.language in pq.languages for m in self.messages], dtype=bool)
        if pq.day_from is not None:
            mask &= self.day >= pq.day_from
        if pq.day_to is not None:
            mask &= self.day <= pq.day_to
        return mask

    def _dominant_label(self, cand: np.ndarray, min_count: int = 2, min_share: float = 0.4) -> Optional[str]:
        """Most common observed label (not unknown) among the candidates, if it clearly dominates."""
        counts = Counter(self.messages[i].observed_label for i in np.flatnonzero(cand))
        counts.pop("unknown", None)
        if not counts:
            return None
        label, c = max(sorted(counts.items()), key=lambda kv: kv[1])
        return label if c >= min_count and c / sum(counts.values()) >= min_share else None

    def _top_group(self, layer: str, cand: np.ndarray, rel: np.ndarray) -> Optional[str]:
        sizes = self.group_sizes.get(layer)
        idx = np.flatnonzero(cand & rel)
        if not len(idx):
            return None
        counts = Counter(getattr(self.messages[i], layer) for i in idx)
        counts.pop(None, None)
        if not counts:
            return None
        total_rel = sum(counts.values())
        if sizes:
            total_size = sum(sizes.values())
            share = {z: sizes.get(z, 1) / total_size for z in counts}
        else:
            share = {z: 1.0 / len(counts) for z in counts}

        def llr(z):
            c, e = counts[z], total_rel * share[z]
            return c * math.log(c / e) if c > e else 0.0
        return max(sorted(counts), key=lambda z: (llr(z), counts[z]))

    # -- search --
    def search(self, query: str, k: int = 5, filters: Optional[Mapping[str, object]] = None,
               method: str = "bm25", expand: bool = True, metadata: bool = True,
               as_of: Optional[int] = None, use_labels: bool = True) -> SearchResult:
        """Top-k messages for `query`.

        method: "bm25" (text only), "dense" (hashed vectors / embed_fn), "hybrid" (BM25 + dense, RRF).
        metadata: apply the parsed (and explicit) filters, cause boost and zone rerank.
        use_labels: let the pipeline's observed labels drive the cause boost / inferred cause
                    (off = text evidence only, e.g. for replies not classified yet).
        filters: explicit values override parsed ones: route, section, neighborhood, student,
                 language(s), day_from, day_to, week, cause(s), group_by.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        pq = parse_query(query, as_of=as_of, known=self.known)
        if filters:
            _apply_filters(pq, filters)
        n = len(self.messages)
        if method == "bm25":
            text = self.bm25.scores(self.query_terms(pq, expand))
        elif method == "dense":
            text = self.dense.scores(self.dense_query(pq, expand))
        else:
            text = _rrf([self.bm25.scores(self.query_terms(pq, expand)),
                         self.dense.scores(self.dense_query(pq, expand))])
        if not metadata:
            cand = np.ones(n, dtype=bool) if as_of is None else self.received <= as_of
            score = text.copy()
            top_group = None
        else:
            cand = self.candidate_mask(pq, as_of)
            score = np.zeros(n)
            mx = text[cand].max() if cand.any() else 0.0
            if mx > 0:
                score = text / mx
            if use_labels and not pq.causes and cand.any():
                inferred = self._dominant_label(cand)
                if inferred:
                    pq.causes = [inferred]
                    pq.notes.append(f"no cause in the query; using the most common reported reason: {inferred}")
            causes = set(pq.causes)
            obs = np.array([m.observed_label in causes for m in self.messages], dtype=bool) \
                if causes and use_labels else np.zeros(n, dtype=bool)
            score = score + CAUSE_BOOST * obs
            rel = (text > 0) | obs if (causes or (text[cand] > 0).any()) else np.ones(n, dtype=bool)
            top_group = None
            if pq.group_by:
                top_group = self._top_group(pq.group_by, cand, rel)
                if top_group is not None:
                    in_g = np.array([getattr(m, pq.group_by) == top_group for m in self.messages], dtype=bool)
                    score = score + GROUP_BOOST * (in_g & rel)
            # same-day concentration among relevant candidates ("why are they missing" bursts)
            dens = np.zeros(n)
            sel = np.flatnonzero(cand & rel)
            if len(sel):
                day_counts = Counter(self.day[sel].tolist())
                peak = max(day_counts.values())
                dens[sel] = [day_counts[d] / peak for d in self.day[sel]]
            score = score + DENSITY_BOOST * dens
        order = [i for i in np.lexsort((np.arange(n), -score)) if cand[i]]
        if not metadata:
            order = [i for i in order if text[i] > 0] or order
        hits = []
        for r, i in enumerate(order[:k]):
            m = self.messages[i]
            reasons = []
            if text[i] > 0:
                reasons.append("text match")
            if metadata and m.observed_label in pq.causes:
                reasons.append(f"label={m.observed_label}")
            if metadata and top_group is not None and getattr(m, pq.group_by) == top_group:
                reasons.append(f"top {pq.group_by} {top_group}")
            hits.append(Hit(m, float(score[i]), r + 1, reasons))
        return SearchResult(hits=hits, parsed=pq, n_candidates=int(cand.sum()), method=method, top_group=top_group)


def _rrf(score_lists: Sequence[np.ndarray], k: int = 60) -> np.ndarray:
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(len(s))
        ranks[order] = np.arange(1, len(s) + 1)
        out += np.where(s > 0, 1.0 / (k + ranks), 0.0)
    return out


def _apply_filters(pq: ParsedQuery, filters: Mapping[str, object]) -> None:
    for key in ("route", "section", "neighborhood", "student", "group_by"):
        if filters.get(key):
            setattr(pq, key, str(filters[key]))
    langs = filters.get("languages") or filters.get("language")
    if langs:
        pq.languages = [langs] if isinstance(langs, str) else list(langs)
    if filters.get("week") is not None:
        pq.day_from, pq.day_to = week_days(int(filters["week"]))
    for key in ("day_from", "day_to"):
        if filters.get(key) is not None:
            setattr(pq, key, int(filters[key]))
    causes = filters.get("causes") or filters.get("cause")
    if causes:
        pq.causes = [causes] if isinstance(causes, str) else list(causes)
