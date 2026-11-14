#!/usr/bin/env python
"""PRD 2 section 4 ablation: content-only vs graph-only vs combined at equal false-alarm rate.

    python scripts/run_ablation.py --school lyon_primary_2009 --tune-school marseille_hs_2013 \
        --runs 10 --perms 199 --out eval/ablation.json
    python scripts/run_ablation.py --quick          # 3 runs, 49 permutations (CI smoke test)

Protocol
  1. Calibrate: null runs (null_config(CONFIG_A): flu waves only) on the tuning school; per detector,
     choose the p-value threshold giving <= 1 false alarm per 20 scanned school days.
  2. Test once: CONFIG_B event runs on the held-out school with those thresholds. Also reported:
     CONFIG_B null runs on the held-out school (does the false-alarm rate transfer?), the
     production operating point (BH q = 0.10), EARS-C1 lead time, a recall vs false-alarm curve,
     and a label-noise sweep (accuracy 0.5 -> 1.0, 1.0 = oracle labels) for the label detectors.
Detectors: content_only, graph_only, combined (PRD), plus graph_labels (all-cause scan as graph_only,
cause = majority reply label among the alarm's absences): does using labels only for attribution help?
Seeds: run i of config X uses seed X.seed + i for both the simulator and assign_logistics.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.detectors import ALL_DETECTORS as DETECTORS, LABEL_DETECTORS, Pipeline  # noqa: E402
from app.ml.evaluate import (alarms_at, calibrate_threshold, false_alarm_rate, lead_vs_baseline,  # noqa: E402
                             ears_outcomes, relabel, score_run, summarize, summarize_ears, truth_index)
from app.ml.graphs import assign_logistics, load_school  # noqa: E402
from app.ml.scan import ScanParams  # noqa: E402
from app.ml.simulator import CONFIG_A, CONFIG_B, null_config, simulate  # noqa: E402

NOISE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_SCHOOLS: dict = {}


def _school(name):
    if name not in _SCHOOLS:
        _SCHOOLS[name] = load_school(name)
    return _SCHOOLS[name]


def make_run(school_name, cfg, seed, accuracy=None):
    school = assign_logistics(_school(school_name), seed)
    run = simulate(school, dataclasses.replace(cfg, seed=seed))
    if accuracy is not None:
        run = relabel(run, accuracy, seed)
    return run


def _job(job):
    school_name, cfg, seed, accuracy, detectors, params = job
    t0 = time.perf_counter()
    run = make_run(school_name, cfg, seed, accuracy)
    t1 = time.perf_counter()
    res = Pipeline(run, params).prospective(detectors=detectors)
    return run, res, t1 - t0, time.perf_counter() - t1


def run_jobs(jobs, n_jobs):
    if n_jobs <= 1:
        return [_job(j) for j in jobs]
    with ProcessPoolExecutor(n_jobs) as ex:
        return list(ex.map(_job, jobs))


def _fa20(fa, days):
    return 20.0 * fa / max(days, 1)


def evaluate_detector(det, results, runs, thr, ears, q=None):
    scores = []
    for res, run in zip(results, runs):
        days = [r.day for r in res[det]]
        scores.append(score_run(alarms_at(res[det], p_max=thr, q=q), run, days))
    out = summarize(scores)
    out["lead_vs_ears_c1"] = lead_vs_baseline(scores, ears)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--school", default="lyon_primary_2009", help="held-out test school")
    ap.add_argument("--tune-school", default="marseille_hs_2013", help="calibration school")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--perms", type=int, default=199)
    ap.add_argument("--baseline", type=int, default=40, help="baseline days B")
    ap.add_argument("--target-fa", type=float, default=1 / 20, help="false alarms per scanned school day")
    ap.add_argument("--noise-runs", type=int, default=None, help="runs per label-noise level (default --runs)")
    ap.add_argument("--no-noise", action="store_true", help="skip the label-noise sweep")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (default 1 = serial)")
    ap.add_argument("--quick", action="store_true", help="3 runs, 49 permutations")
    ap.add_argument("--test-config", choices=("A", "B"), default="B",
                    help="config of the event runs (A only for development on the tuning side)")
    ap.add_argument("--out", default="eval/ablation.json")
    args = ap.parse_args(argv)
    if args.quick:
        args.runs, args.perms = 3, 49
    noise_runs = args.noise_runs if args.noise_runs is not None else args.runs

    t_start = time.perf_counter()
    params = ScanParams(baseline_days=args.baseline, n_perm=args.perms)
    cfg_test = CONFIG_B if args.test_config == "B" else dataclasses.replace(CONFIG_A, seed=CONFIG_A.seed + 500)
    cfg_tune_null = null_config(CONFIG_A)
    cfg_test_null = null_config(cfg_test)
    seeds_tune = [CONFIG_A.seed + i for i in range(args.runs)]
    seeds_test = [cfg_test.seed + i for i in range(args.runs)]
    seeds_noise = seeds_test[:noise_runs]

    jobs = ([(args.tune_school, cfg_tune_null, s, None, DETECTORS, params) for s in seeds_tune]
            + [(args.school, cfg_test, s, None, DETECTORS, params) for s in seeds_test]
            + [(args.school, cfg_test_null, s, None, DETECTORS, params) for s in seeds_test])
    if not args.no_noise:
        jobs += [(args.school, cfg_test, s, acc, LABEL_DETECTORS, params) for acc in NOISE_LEVELS for s in seeds_noise]
    out = run_jobs(jobs, args.jobs)
    t_scan = time.perf_counter() - t_start
    n = args.runs
    tune = out[:n]
    test = out[n:2 * n]
    tnull = out[2 * n:3 * n]
    noise = out[3 * n:]

    tune_runs, tune_res = [o[0] for o in tune], [o[1] for o in tune]
    test_runs, test_res = [o[0] for o in test], [o[1] for o in test]
    tnull_runs, tnull_res = [o[0] for o in tnull], [o[1] for o in tnull]

    # 1. calibration on the tuning school's null runs
    t0 = time.perf_counter()
    thresholds = {d: calibrate_threshold([r[d] for r in tune_res], tune_runs, args.target_fa, cap=params.report_p)
                  for d in DETECTORS}

    # 2. held-out evaluation
    scan_days = [[r.day for r in res["combined"]] for res in test_res]
    ears = [ears_outcomes(run, days) for run, days in zip(test_runs, scan_days)]
    report = {}
    for d in DETECTORS:
        thr = thresholds[d]["threshold"]
        rep = evaluate_detector(d, test_res, test_runs, thr, ears)
        fa, nd = false_alarm_rate([r[d] for r in tnull_res], tnull_runs, p_max=thr)
        rep["test_null_fa_per_20_days"] = _fa20(fa, nd)
        report[d] = rep

    bh = {}
    for d in DETECTORS:
        rep = evaluate_detector(d, test_res, test_runs, None, ears, q=0.10)
        fa_t, nd_t = false_alarm_rate([r[d] for r in tune_res], tune_runs, q=0.10)
        fa_n, nd_n = false_alarm_rate([r[d] for r in tnull_res], tnull_runs, q=0.10)
        bh[d] = {"recall": rep["recall"], "fa_per_20_days": rep["fa_per_20_days"],
                 "pct_detected_by_onset_plus_2": rep["pct_detected_by_onset_plus_2"],
                 "tune_null_fa_per_20_days": _fa20(fa_t, nd_t), "test_null_fa_per_20_days": _fa20(fa_n, nd_n)}

    ears_summary = summarize_ears(test_runs, scan_days)
    ears_summary["test_null_fa_per_20_days"] = summarize_ears(
        tnull_runs, [[r.day for r in res["combined"]] for res in tnull_res])["fa_per_20_days"]

    # recall vs false alarms: p grid (log-spaced plus the Monte Carlo p steps) and each calibrated point
    step = 1.0 / (params.n_perm + 1)
    grid = sorted({float(x) for x in np.logspace(-6, np.log10(params.report_p), 12)}
                  | {k * step for k in range(1, 21) if k * step <= params.report_p})
    curve = {d: [] for d in DETECTORS}
    tune_truth = [truth_index(r) for r in tune_runs]
    for d in DETECTORS:
        points = [(thr, False) for thr in grid] + [(thresholds[d]["threshold"], True)]
        for thr, is_cal in sorted(points, key=lambda x: x[0] if isinstance(x[0], float) else x[0][0]):
            fa, nd = false_alarm_rate([r[d] for r in tune_res], tune_runs, p_max=thr, truths=tune_truth)
            scores = [score_run(alarms_at(res[d], p_max=thr), run, [x.day for x in res[d]])
                      for res, run in zip(test_res, test_runs)]
            s = summarize(scores)
            curve[d].append({"threshold": thr, "calibrated": is_cal, "tune_null_fa_per_20_days": _fa20(fa, nd),
                             "test_fa_per_20_days": s["fa_per_20_days"], "recall": s["recall"]})

    # label-noise sweep (paired: same runs, only the observed labels change)
    sweep = []
    for i, acc in enumerate(() if args.no_noise else NOISE_LEVELS):
        chunk = noise[i * noise_runs:(i + 1) * noise_runs]
        row = {"label_accuracy": acc, "oracle": acc == 1.0}
        for d in LABEL_DETECTORS:
            scores = [score_run(alarms_at(res[d], p_max=thresholds[d]["threshold"]), run,
                                [x.day for x in res[d]]) for run, res, _, _ in chunk]
            s = summarize(scores)
            row[d] = {"recall": s["recall"], "fa_per_20_days": s["fa_per_20_days"],
                      "pct_detected_by_onset_plus_2": s["pct_detected_by_onset_plus_2"],
                      "attribution_confusion": s["attribution_confusion"]}
        sweep.append(row)
    t_eval = time.perf_counter() - t0

    sim_s = sum(o[2] for o in out)
    scan_s = sum(o[3] for o in out)
    n_days = sum(len(o[1][next(iter(o[1]))]) for o in out)
    result = {
        "meta": {
            "school": args.school, "tune_school": args.tune_school, "runs": args.runs, "perms": args.perms,
            "baseline_days": args.baseline, "target_fa_per_20_days": 20 * args.target_fa,
            "scan_params": dataclasses.asdict(params),
            "seeds": {"tune_null": seeds_tune, "test": seeds_test, "test_null": seeds_test, "label_noise": seeds_noise},
            "configs": {"tune_null": dataclasses.asdict(cfg_tune_null), "test": dataclasses.asdict(cfg_test),
                        "test_null": dataclasses.asdict(cfg_test_null)},
            "label_noise_model": "observed = true w.p. accuracy, else a uniformly drawn other barrier "
                                 "(evaluate.relabel; the default rows use the simulator's own noise model)",
            "timings_s": {"total": round(time.perf_counter() - t_start, 2), "simulate_and_scan_wall": round(t_scan, 2),
                          "simulate_cpu": round(sim_s, 2), "scan_cpu": round(scan_s, 2),
                          "evaluate": round(t_eval, 2), "n_runs_scanned": len(out),
                          "scan_ms_per_run_day": round(1000 * scan_s / max(n_days, 1), 2), "jobs": args.jobs},
            "n_events_test": report["combined"]["n_events"],
        },
        "thresholds": thresholds,
        "test": report,
        "bh_q_0.10": bh,
        "ears_c1": ears_summary,
        "curve": curve,
        "label_noise": sweep,
    }
    path = Path(args.out)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    print_table(result)
    print(f"\nwrote {path}")
    return result


def _f(x, nd=2):
    return "  -  " if x is None else f"{x:.{nd}f}"


def _iqr(q):
    if not q or q.get("median") is None:
        return "-"
    return f"{q['median']:+.1f} [{q['q1']:+.1f},{q['q3']:+.1f}]"


def print_table(r):
    m = r["meta"]
    print(f"\nAblation: test {m['school']} (config {m['configs']['test']['name']}, {m['runs']} runs, {m['n_events_test']} events), "
          f"tuned on {m['tune_school']} (null CONFIG_A), {m['perms']} perms, B={m['baseline_days']}")
    print(f"{'detector':<13}{'p_thr':>10}{'FA/20d tune':>12}{'FA/20d tnull':>13}{'FA/20d test':>12}"
          f"{'recall':>8}{'<=t+2':>7}{'delay med':>10}  lead vs EARS (med [IQR])")
    for d in DETECTORS:
        t, s = r["thresholds"][d], r["test"][d]
        print(f"{d:<13}{t['p_threshold']:>10.2e}{t['null_fa_per_20_days']:>12.2f}{s['test_null_fa_per_20_days']:>13.2f}"
              f"{s['fa_per_20_days']:>12.2f}{_f(s['recall']):>8}{_f(s['pct_detected_by_onset_plus_2']):>7}"
              f"{_f(s['delay']['median'], 1):>10}  {_iqr(s['lead_vs_ears_c1']['lead'])}")
    e = r["ears_c1"]
    print(f"{'ears_c1':<13}{'-':>10}{'-':>12}{e['test_null_fa_per_20_days']:>13.2f}{e['fa_per_20_days']:>12.2f}"
          f"{_f(e['recall']):>8}{_f(e['pct_detected_by_onset_plus_2']):>7}{_f(e['delay']['median'], 1):>10}")
    print("\nrecall by event kind:")
    for d in DETECTORS:
        kinds = r["test"][d]["recall_by_kind"]
        print(f"  {d:<13}" + "  ".join(f"{k} {v['detected']}/{v['n']}" for k, v in sorted(kinds.items())))
    print("\nBH q=0.10 (production): " + "; ".join(
        f"{d} recall {_f(v['recall'])}, FA/20d test {_f(v['fa_per_20_days'])} tnull {_f(v['test_null_fa_per_20_days'])}"
        for d, v in r["bh_q_0.10"].items()))
    print("\nattribution (rows: true; cols: illness/transportation/other/missed):")
    for d in DETECTORS:
        cm = r["test"][d]["attribution_confusion"]
        print(f"  {d:<13}" + "  ".join(f"{k[:5]}:{[v[c] for c in ('illness', 'transportation', 'other', 'missed')]}"
                                       for k, v in cm.items()))
    print(f"  contagion TPs via (combined): {r['test']['combined']['contagion_tp_via']}")
    if r["label_noise"]:
        print("\nlabel noise (recall / FA per 20 d):")
        for row in r["label_noise"]:
            tag = " (oracle)" if row["oracle"] else ""
            print(f"  acc {row['label_accuracy']:.1f}{tag:<9} " + "  ".join(
                f"{d} {_f(row[d]['recall'])} / {_f(row[d]['fa_per_20_days'])}" for d in LABEL_DETECTORS))
    print(f"\ntimings: {m['timings_s']}")


if __name__ == "__main__":
    main()
