"""Evaluate the barrier and safety classifiers on a labeled JSONL set (PRD 2 section 4).

    cd backend
    .venv/bin/python -m app.ml.eval_classifier --set data/labeled/example_set.jsonl
    .venv/bin/python -m app.ml.eval_classifier --set data/labeled/hand_set.jsonl --model gemini --split test

Each line: {"text", "language", "label", "safety": bool, "split": "dev"|"test"}.

Reports macro-F1 with a bootstrap 95% CI, accuracy per language, a confusion
matrix, safety recall with its 95% lower bound (1 - 3/N when there are no
misses, the "rule of three"; Wilson otherwise) and the false-alarm rate, ECE,
and auto-accept precision/coverage at tau plus a tau sweep. The tau to ship is
the smallest one whose auto-accept precision on the dev split is >= 0.95.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models import Barrier

LABELS = [b.value for b in Barrier]
SHORT = {
    "illness": "ill", "transportation": "trans", "housing or food insecurity": "house",
    "caregiving or family obligation": "care", "safety or bullying": "safety",
    "disengagement or academic struggle": "diseng", "appointment or travel": "appt", "unknown": "unk",
}
TAU_GRID = [round(0.5 + 0.05 * i, 2) for i in range(11)]
PRECISION_TARGET = 0.95


@dataclass
class Item:
    text: str
    language: str
    label: str
    safety: bool
    split: str


@dataclass
class Prediction:
    label: str
    confidence: float
    safety: bool
    needs_review: bool
    source: str


def load_set(path: Path, split: Optional[str] = None) -> List[Item]:
    items: List[Item] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row["label"] not in LABELS:
            raise ValueError(f"{path}:{n}: label {row['label']!r} is not in the taxonomy")
        item = Item(text=row["text"], language=row.get("language", "en"), label=row["label"],
                    safety=bool(row.get("safety", row["label"] == Barrier.SAFETY.value)),
                    split=row.get("split", "test"))
        if split in (None, "all") or item.split == split:
            items.append(item)
    return items


def predict(items: Sequence[Item], model: str = "keyword", classifier: Any = None) -> List[Prediction]:
    from ..services.gemini_classifier import GeminiClassifier, KeywordClassifier, has_api_key

    if classifier is None:
        if model == "gemini":
            if not has_api_key():
                raise SystemExit("GEMINI_API_KEY is not set (env or backend/.env); use --model keyword")
            classifier = GeminiClassifier()
        else:
            classifier = KeywordClassifier()
    out = []
    for item in items:
        c, meta = classifier.classify_with_model(item.text, language=item.language)
        out.append(Prediction(label=c.barrier.value, confidence=c.confidence, safety=c.safety_critical,
                              needs_review=meta.needs_review, source=meta.source))
    return out


# ---------- metrics ----------
def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Optional[Sequence[str]] = None) -> float:
    labels = labels or sorted(set(gold))
    scores = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def bootstrap_ci(gold: Sequence[str], pred: Sequence[str], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI for macro-F1; labels fixed to those in the full gold set."""
    labels = sorted(set(gold))
    rng = random.Random(seed)
    n = len(gold)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        stats.append(macro_f1([gold[i] for i in idx], [pred[i] for i in idx], labels))
    stats.sort()
    lo = stats[int(math.floor(alpha / 2 * n_boot))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return lo, hi


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / (1 + z * z / n))


def safety_metrics(items: Sequence[Item], preds: Sequence[Prediction]) -> Dict[str, Any]:
    pos = [(i, p) for i, p in zip(items, preds) if i.safety]
    neg = [(i, p) for i, p in zip(items, preds) if not i.safety]
    caught = sum(1 for _, p in pos if p.safety)
    misses = [i.text for i, p in pos if not p.safety]
    n = len(pos)
    if n and not misses:
        lower, method = max(0.0, 1 - 3 / n), "rule of three (1 - 3/N)"
    else:
        lower, method = wilson_lower(caught, n), "Wilson 95%"
    false_alarms = [i.text for i, p in neg if p.safety]
    return {
        "n_safety": n, "caught": caught, "recall": caught / n if n else None, "recall_lower_95": lower,
        "lower_bound_method": method, "misses": misses, "n_non_safety": len(neg),
        "false_alarm_rate": len(false_alarms) / len(neg) if neg else None, "false_alarms": false_alarms,
    }


def auto_accept(items: Sequence[Item], preds: Sequence[Prediction], tau: float) -> Dict[str, Any]:
    """A label is auto-accepted when confident, not safety, not unknown, and not a fallback."""
    accepted = [(i, p) for i, p in zip(items, preds)
                if p.confidence >= tau and not p.safety and p.label != Barrier.UNKNOWN.value
                and p.source in {"keyword", "gemini"}]
    correct = sum(1 for i, p in accepted if i.label == p.label)
    n = len(items)
    return {"tau": tau, "accepted": len(accepted), "coverage": len(accepted) / n if n else 0.0,
            "precision": correct / len(accepted) if accepted else None,
            "to_clerk_share": 1 - len(accepted) / n if n else 0.0}


def ece(items: Sequence[Item], preds: Sequence[Prediction], bins: int = 10) -> float:
    buckets: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    for i, p in zip(items, preds):
        buckets[min(bins - 1, int(p.confidence * bins))].append((p.confidence, i.label == p.label))
    n = len(items)
    total = 0.0
    for rows in buckets.values():
        conf = sum(c for c, _ in rows) / len(rows)
        acc = sum(1 for _, ok in rows if ok) / len(rows)
        total += len(rows) / n * abs(conf - acc)
    return total


def choose_tau(items: Sequence[Item], preds: Sequence[Prediction], target: float = PRECISION_TARGET) -> Optional[float]:
    """Smallest tau on the grid whose auto-accept precision reaches the target (max coverage)."""
    for tau in TAU_GRID:
        row = auto_accept(items, preds, tau)
        if row["precision"] is not None and row["precision"] >= target:
            return tau
    return None


def evaluate(items: Sequence[Item], preds: Sequence[Prediction], tau: float = 0.8,
             n_boot: int = 2000) -> Dict[str, Any]:
    gold = [i.label for i in items]
    pred = [p.label for p in preds]
    per_lang: Dict[str, List[bool]] = defaultdict(list)
    for i, p in zip(items, preds):
        per_lang[i.language].append(i.label == p.label)
    confusion = Counter((g, p) for g, p in zip(gold, pred))
    dev = [(i, p) for i, p in zip(items, preds) if i.split == "dev"]
    tau_dev = choose_tau([i for i, _ in dev], [p for _, p in dev]) if dev else None
    return {
        "n": len(items),
        "accuracy": sum(1 for g, p in zip(gold, pred) if g == p) / len(items) if items else 0.0,
        "macro_f1": macro_f1(gold, pred),
        "macro_f1_ci95": bootstrap_ci(gold, pred, n_boot=n_boot) if items else (0.0, 0.0),
        "per_language": {lang: {"n": len(v), "accuracy": sum(v) / len(v)} for lang, v in sorted(per_lang.items())},
        "confusion": {f"{g} -> {p}": c for (g, p), c in sorted(confusion.items())},
        "safety": safety_metrics(items, preds),
        "ece": ece(items, preds),
        "auto_accept": auto_accept(items, preds, tau),
        "tau_sweep": [auto_accept(items, preds, t) for t in TAU_GRID],
        "tau_chosen_on_dev": tau_dev,
        "errors": [{"text": i.text, "gold": i.label, "pred": p.label, "confidence": p.confidence}
                   for i, p in zip(items, preds) if i.label != p.label],
    }


# ---------- report ----------
def _fmt(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


def confusion_table(items: Sequence[Item], preds: Sequence[Prediction]) -> str:
    labels = [lab for lab in LABELS if any(i.label == lab for i in items) or any(p.label == lab for p in preds)]
    counts = Counter((i.label, p.label) for i, p in zip(items, preds))
    width = 7
    lines = ["gold \\ pred".ljust(12) + "".join(SHORT[lab].rjust(width) for lab in labels)]
    for g in labels:
        lines.append(SHORT[g].ljust(12) + "".join(str(counts.get((g, p), "") or ".").rjust(width) for p in labels))
    return "\n".join(lines)


def render(report: Dict[str, Any], items: Sequence[Item], preds: Sequence[Prediction], model: str) -> str:
    s = report["safety"]
    lo, hi = report["macro_f1_ci95"]
    out = [
        f"Classifier evaluation: model={model}  n={report['n']}",
        f"  macro-F1 {report['macro_f1']:.3f}  (bootstrap 95% CI {lo:.3f} to {hi:.3f})   accuracy {report['accuracy']:.3f}"
        f"   ECE {report['ece']:.3f}",
        "  per-language accuracy: " + ", ".join(f"{k} {v['accuracy']:.2f} (n={v['n']})"
                                                 for k, v in report["per_language"].items()),
        "",
        "Confusion matrix (rows gold, columns predicted):",
        confusion_table(items, preds),
        "",
        f"Safety: caught {s['caught']}/{s['n_safety']}  recall {_fmt(s['recall'])}  "
        f"95% lower bound {s['recall_lower_95']:.3f} [{s['lower_bound_method']}]",
        f"        false-alarm rate {_fmt(s['false_alarm_rate'])} ({len(s['false_alarms'])}/{s['n_non_safety']})",
    ]
    out += [f"        MISSED: {t}" for t in s["misses"]]
    out += [f"        false alarm: {t}" for t in s["false_alarms"]]
    a = report["auto_accept"]
    out += ["", f"Auto-accept at tau={a['tau']}: precision {_fmt(a['precision'])}  coverage {a['coverage']:.3f}  "
                f"to clerks {a['to_clerk_share']:.3f}", "  tau sweep (risk-coverage):",
            "    tau   accepted  coverage  precision"]
    out += [f"    {r['tau']:.2f}  {r['accepted']:8d}  {r['coverage']:8.3f}  {_fmt(r['precision']):>9}"
            for r in report["tau_sweep"]]
    chosen = report["tau_chosen_on_dev"]
    out.append(f"  tau chosen on dev split (precision >= {PRECISION_TARGET}): "
               f"{chosen if chosen is not None else 'none reaches the target (or no dev items)'}")
    if report["errors"]:
        out += ["", "Errors:"]
        out += [f"  [{e['gold']} -> {e['pred']} @ {e['confidence']}] {e['text']}" for e in report["errors"]]
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", required=True, type=Path, help="labeled JSONL file")
    parser.add_argument("--model", choices=["keyword", "gemini"], default="keyword")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--tau", type=float, default=0.8, help="auto-accept threshold to report")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json", type=Path, help="also write the full report as JSON here")
    args = parser.parse_args(argv)

    items = load_set(args.set, args.split)
    if not items:
        print("no items in that split", file=sys.stderr)
        return 1
    preds = predict(items, args.model)
    report = evaluate(items, preds, tau=args.tau, n_boot=args.bootstrap)
    print(render(report, items, preds, args.model))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
