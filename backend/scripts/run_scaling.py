#!/usr/bin/env python
"""Runtime vs school size N (PRD 2 section 2, "Scaling benchmark"; Arlequin "responsible compute").

    python scripts/run_scaling.py                      # N in 250 .. 50000 -> eval/scaling.json
    python scripts/run_scaling.py --ns 250 1000 --out /tmp/scaling.json

For each N, in a fresh worker process (so peak memory is per N):
  1. graphs.synthetic_sbm(N): section-block SBM (25 students per section, 3 recorded days, top-10 edges)
  2. graphs.assign_logistics: routes and neighborhoods scaled with N (about 30 students per route and
     40 per neighborhood) so a route stays bus-sized; households with siblings as usual
  3. simulator.simulate: a CONFIG_A school year of --days school days (all planted event types,
     replies with the noisy label proxy)
  4. detectors.Pipeline: zones (section / route / neighborhood / contact ego-nets k = 5, 10, 20 / grade /
     school) and observation indexes, built once per run
  5. one prospective day scan at as_of = --as-of with every detector (content_only, graph_only, combined,
     two_stage), timed as a whole and split into the all-cause scan (graph_only + two_stage) and the
     label scan (content_only + combined)
  6. the same day scan again with --extra-perms (default 999, the PRD's value) on the same zones
Permutations: --perms for N <= --large-n, --perms-large above it (recorded per row; both default to 199,
the ablation's value, as nothing needed reducing). Sparse SciPy work is single-threaded; the machine is
recorded in meta.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.detectors import ALL_DETECTORS, Pipeline, select  # noqa: E402
from app.ml.graphs import assign_logistics, synthetic_sbm  # noqa: E402
from app.ml.scan import ScanParams  # noqa: E402
from app.ml.simulator import CONFIG_A, simulate  # noqa: E402

NS = (250, 1000, 5000, 10000, 25000, 50000)
STUDENTS_PER_ROUTE = 30
STUDENTS_PER_NEIGHBORHOOD = 40


def _peak_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024.0 if sys.platform != "darwin" else kb / 1024.0 / 1024.0


def bench(n: int, days: int, as_of: int, perms: int, seed: int, extra_perms: int = 0) -> dict:
    row = {"n": n, "n_days": days, "as_of": as_of, "perms": perms}
    t = time.perf_counter()
    school = synthetic_sbm(n, seed)
    row["sbm_s"] = time.perf_counter() - t

    t = time.perf_counter()
    school = assign_logistics(school, seed, n_neighborhoods=max(6, n // STUDENTS_PER_NEIGHBORHOOD),
                              n_routes=max(8, n // STUDENTS_PER_ROUTE))
    row["logistics_s"] = time.perf_counter() - t

    t = time.perf_counter()
    run = simulate(school, dataclasses.replace(CONFIG_A, n_days=days, seed=seed))
    row["simulate_s"] = time.perf_counter() - t
    row["n_absences"] = len(run.absences)
    row["n_replies"] = len(run.replies)
    row["n_events"] = len(run.events)

    params = ScanParams(n_perm=perms)
    t = time.perf_counter()
    pipe = Pipeline(run, params)
    row["pipeline_build_s"] = time.perf_counter() - t
    row["n_zones"] = pipe.zones.n_zones
    row["n_contact_zones"] = int(pipe.zones.layer_cols["contact"].size)

    t = time.perf_counter()
    res = pipe.day(as_of, ("graph_only", "two_stage"))
    row["day_scan_all_cause_s"] = time.perf_counter() - t
    t = time.perf_counter()
    res.update(pipe.day(as_of, ("content_only", "combined")))
    row["day_scan_labels_s"] = time.perf_counter() - t
    row["day_scan_s"] = row["day_scan_all_cause_s"] + row["day_scan_labels_s"]
    row["n_candidates"] = {d: len(r.candidates) for d, r in res.items()}
    row["n_alarms_bh"] = {d: len(select(r)) for d, r in res.items()}

    t = time.perf_counter()
    again = pipe.day(as_of, ALL_DETECTORS)     # same day, one call (shares nothing extra; sanity check)
    row["day_scan_one_call_s"] = time.perf_counter() - t
    assert {d: len(r.candidates) for d, r in again.items()} == row["n_candidates"]
    if extra_perms:
        extra = Pipeline(run, dataclasses.replace(params, n_perm=extra_perms), zones=pipe.zones)
        t = time.perf_counter()
        extra.day(as_of, ALL_DETECTORS)
        row["extra_perms"] = extra_perms
        row["day_scan_extra_perms_s"] = time.perf_counter() - t
    row["peak_rss_mb"] = _peak_mb()
    for k, v in list(row.items()):
        if k.endswith("_s"):
            row[k] = round(v, 3)
    row["peak_rss_mb"] = round(row["peak_rss_mb"], 1)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--days", type=int, default=100, help="simulated school days")
    ap.add_argument("--as-of", type=int, default=60, help="day of the timed prospective scan (>= 39)")
    ap.add_argument("--perms", type=int, default=199)
    ap.add_argument("--perms-large", type=int, default=199, help="permutations for N > --large-n")
    ap.add_argument("--extra-perms", type=int, default=999, help="also time the day scan with this many (0 = skip)")
    ap.add_argument("--large-n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/scaling.json")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    rows = []
    for n in args.ns:
        perms = args.perms if n <= args.large_n else args.perms_large
        with ProcessPoolExecutor(max_workers=1) as ex:   # fresh process per N: per-N peak memory
            row = ex.submit(bench, n, args.days, args.as_of, perms, args.seed, args.extra_perms).result()
        rows.append(row)
        print(f"N={n:>6}  perms={perms:>3}  sbm {row['sbm_s']:7.2f}s  simulate({args.days}d) {row['simulate_s']:7.2f}s  "
              f"pipeline {row['pipeline_build_s']:7.2f}s  day scan {row['day_scan_s']:7.2f}s "
              f"(all-cause {row['day_scan_all_cause_s']:.2f}, labels {row['day_scan_labels_s']:.2f})  "
              f"zones {row['n_zones']:>7}  {args.extra_perms} perms {row.get('day_scan_extra_perms_s', float('nan')):.2f}s  "
              f"peak {row['peak_rss_mb']:.0f} MB", flush=True)

    result = {
        "meta": {
            "ns": args.ns, "n_days": args.days, "as_of": args.as_of, "perms": args.perms,
            "perms_large": args.perms_large, "large_n": args.large_n, "extra_perms": args.extra_perms,
            "seed": args.seed,
            "config": "CONFIG_A (all planted event types) with n_days and seed replaced",
            "graph": "graphs.synthetic_sbm(N, seed): 25 students/section, 3 recorded days, top-10 edges, "
                     "default (Lyon-fitted) weights",
            "logistics": f"assign_logistics with n_routes = max(8, N // {STUDENTS_PER_ROUTE}), "
                         f"n_neighborhoods = max(6, N // {STUDENTS_PER_NEIGHBORHOOD})",
            "detectors_in_day_scan": list(ALL_DETECTORS),
            "machine": {"platform": platform.platform(), "python": platform.python_version(),
                        "cpu_count": os.cpu_count(), "processor": platform.processor()},
            "single_process": True,
            "peak_rss_note": "peak resident memory of the worker for that N, including the extra-perms scan",
            "total_s": round(time.perf_counter() - t0, 2),
        },
        "rows": rows,
    }
    path = Path(args.out)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"total {result['meta']['total_s']} s; wrote {path}")
    return result


if __name__ == "__main__":
    main()
