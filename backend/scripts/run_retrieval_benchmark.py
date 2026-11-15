#!/usr/bin/env python
"""PRD 2 section 3.4 retrieval benchmark for the F7 counselor assistant. Offline, no API calls.

    python scripts/run_retrieval_benchmark.py                    # test: Lyon, CONFIG_B, 2 runs
    python scripts/run_retrieval_benchmark.py --config A --runs 3 --out /tmp/dev.json   # dev split

Templated queries (route / cause / week / which-zone / cross-lingual) are generated from simulator
ground truth; R(q) comes from planted events and true barriers. Retrieval was tuned on CONFIG_A
(seeds 0-2) and is reported once on CONFIG_B (seeds 1000-1001). Writes eval/retrieval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.assistant.benchmark import DEFAULT_METHOD, METHODS, run_benchmark  # noqa: E402
from app.ml.simulator import CONFIG_A, CONFIG_B  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]

NOTES = [
    "Recall@5 = |top5 ∩ R| / min(5, |R|); MRR over the top 10; P@5 is the citation-precision proxy for an "
    "agent that cites the top 5 hits.",
    "R(q) is ground truth from the simulator (planted-event absences or true barrier + language + week); "
    "retrieval sees only reply text, metadata and the pipeline's noisy observed label.",
    "dense_hash is a local hashed character 3-4-gram TF-IDF vector baseline, not a semantic embedding model; "
    "Gemini embeddings are off by default because of free-tier rate limits (MessageIndex accepts embed_fn).",
    "Queries name the route / section / neighborhood / week, so metadata filtering does most of the work; "
    "text-only methods cannot know a reply's route and score near zero by construction.",
    "The cross-lingual barrier lexicon is hand-written and overlaps the template vocabulary of the "
    "simulator's reply generator (app/ml/reply_text.py), so cross-lingual numbers are optimistic for real "
    "family messages. The '(no labels)' rows show its effect when observed labels are not used.",
    "Weeks are 1-based school weeks (week w = days 5(w-1)..5(w-1)+4); 'this week' is relative to the "
    "query's as_of, and replies received after as_of are hidden.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=("A", "B"), default="B")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--school", default="lyon_primary_2009")
    ap.add_argument("--out", default=str(BACKEND / "eval" / "retrieval.json"))
    args = ap.parse_args()

    cfg = CONFIG_B if args.config == "B" else CONFIG_A
    t0 = time.perf_counter()
    res = run_benchmark(cfg, n_runs=args.runs, methods=METHODS, school_name=args.school)
    res["setup"]["seconds"] = round(time.perf_counter() - t0, 2)
    res["setup"]["default_method"] = DEFAULT_METHOD
    res["setup"]["target_recall@5"] = 0.85
    res["notes"] = NOTES
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    s = res["setup"]
    print(f"{s['school']} config {s['config']} seeds {s['seeds']}: {s['n_queries']} queries "
          f"({s['n_cross_lingual']} cross-lingual), {s['seconds']} s")
    print(f"{'method':40s} {'R@5':>6s} {'MRR':>6s} {'P@5':>6s} {'R@5 xl':>7s}")
    for name, m in res["methods"].items():
        print(f"{name:40s} {m['recall@5']:6.3f} {m['mrr']:6.3f} {m['p@5']:6.3f} {m['recall@5_cross_lingual']:7.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
