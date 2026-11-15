import { Fragment } from "react";
import { api } from "../api";
import { useLoad } from "../hooks";
import { ErrorNote, Page, Stat } from "../components/ui";
import { Bar, BarList, DataTable, Dumbbell, Legend, LineChart, Panel, Tip, VizState, pct } from "../components/charts";
import type { BarRow, DumbbellRow, Series } from "../components/charts";

const DETECTORS = [
  { key: "two_stage", label: "Two-stage (ours)", uses: "Network finds the cluster, family replies name the cause", cls: "series-1" },
  { key: "graph_only", label: "Network only", uses: "All absences + school network, no replies", cls: "series-2" },
  { key: "combined", label: "Combined (wait for replies)", uses: "Only absences whose family has replied", cls: "series-3" },
  { key: "content_only", label: "Replies only", uses: "Reply labels, school-wide", cls: "series-4" },
];

const SHORT: Record<string, string> = { two_stage: "Two-stage (ours)", graph_only: "Network only", combined: "Combined", content_only: "Replies only" };

const EVENT: Record<string, string> = {
  route_failure: "Bus route failure",
  road_closure: "Road closure",
  section_caregiving: "Observance in one class",
  caregiving_spike: "Caregiving spike",
  contagion: "Illness outbreak",
  bus_illness: "Illness spread on a bus",
};

const QUERY: Record<string, string> = {
  flu_window: "Flu window",
  neighborhood_cause: "Cause in a neighborhood",
  route_day: "A route on one day",
  route_this_week: "A route this week",
  section_illness: "Illness in a class",
  which_class: "Which class",
  which_neighborhood: "Which neighborhood",
  xl_cause_week: "Cross-lingual: cause this week",
  xl_language_week: "Cross-lingual: by language",
};

const humanize = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const methodName = (s: string) =>
  s.replace(/bm25/g, "BM25").replace(/dense_hash/g, "Dense hash").replace(/hybrid/g, "Hybrid").replace(/\+/g, " + ").replace(/ {2,}/g, " ");
const kilo = (n: number) => (n >= 1000 ? `${n / 1000}K` : `${n}`);
const secs = (s: number) => (s < 1 ? `${s.toFixed(2)} s` : `${s.toFixed(1)} s`);
/** Round `max` up to a multiple of `step`, at least `step`. */
const ceilTo = (vals: number[], step: number) => Math.max(step, Math.ceil((Math.max(0, ...vals) + 1e-9) / step) * step);
const ticksTo = (top: number) => Array.from({ length: Math.round(top / 0.2) + 1 }, (_, i) => i * 0.2);

export default function Evidence() {
  const ab = useLoad(api.ablation);
  const recTest: Record<string, any> = useLoad(api.ablationRecords).data?.test ?? {};
  const rt = useLoad(api.retrieval);
  const sc = useLoad(api.scaling);
  const test: Record<string, any> = ab.data?.test ?? {};
  const types = Object.keys(test.two_stage?.by_event_type ?? {});
  const methods: Record<string, any> = rt.data?.methods ?? {};
  const inUse: string = rt.data?.setup?.default_method ?? "bm25+expansion+filter";
  const best = methods[inUse];
  const target: number = rt.data?.setup?.["target_recall@5"] ?? 0.85;
  const rows: any[] = sc.data?.rows ?? [];
  const curve: Record<string, any[]> = ab.data?.curve ?? {};
  const noise: any[] = ab.data?.label_noise ?? [];

  // Per event type: network only → two-stage, biggest gain first.
  const byType: DumbbellRow[] = types
    .map((t) => ({
      key: t,
      label: EVENT[t] ?? humanize(t),
      sub: `${test.two_stage.by_event_type[t].n} events`,
      from: test.graph_only?.by_event_type?.[t]?.actionable_recall ?? 0,
      to: test.two_stage.by_event_type[t].actionable_recall ?? 0,
    }))
    .sort((a, b) => b.to - b.from - (a.to - a.from) || b.to - a.to);

  const FA_MAX = 4;
  const sweep: Series[] = DETECTORS.filter((d) => curve[d.key]?.length).map((d) => ({
    key: d.key,
    label: SHORT[d.key],
    cls: d.cls,
    bold: d.key === "two_stage",
    points: curve[d.key].map((p) => ({ x: p.test_fa_per_20_days, y: p.actionable_recall })),
  }));
  const calibrated = DETECTORS.flatMap((d) => (curve[d.key] ?? []).filter((p) => p.calibrated).map((p) => ({ series: d.key, x: p.test_fa_per_20_days, y: p.actionable_recall })));
  const sweepTop = ceilTo(sweep.flatMap((s) => s.points.filter((p) => p.x <= FA_MAX).map((p) => p.y)), 0.2);

  const noisy: Series[] = DETECTORS.filter((d) => noise.some((r) => r[d.key])).map((d) => ({
    key: d.key,
    label: SHORT[d.key],
    cls: d.cls,
    bold: d.key === "two_stage",
    points: noise.filter((r) => r[d.key]).map((r) => ({ x: r.label_accuracy, y: r[d.key].actionable_recall })),
  }));
  const noisyTop = ceilTo(noisy.flatMap((s) => s.points.map((p) => p.y)), 0.2);
  const accuracies: number[] = noise.map((r) => r.label_accuracy);

  const scaling: Series[] = rows.length
    ? [
        { key: "scan", label: "One-day scan", cls: "series-1", bold: true, points: rows.map((r) => ({ x: r.n, y: r.day_scan_s })) },
        ...(rows.every((r) => r.day_scan_extra_perms_s)
          ? [{ key: "extra", label: `With ${rows[0].extra_perms} permutations`, cls: "series-2", points: rows.map((r) => ({ x: r.n, y: r.day_scan_extra_perms_s })) }]
          : []),
      ]
    : [];

  const methodBars: BarRow[] = Object.entries(methods)
    .sort((a, b) => b[1]["recall@5"] - a[1]["recall@5"])
    .map(([k, m]) => ({
      key: k,
      label: methodName(k),
      sub: k === inUse ? "in use" : undefined,
      value: m["recall@5"],
      dot: m["recall@5_cross_lingual"],
      cls: k === inUse ? "series-1" : "series-muted",
      dim: k !== inUse,
      display: m["recall@5"].toFixed(2),
      aria: `${methodName(k)}: Recall at 5 ${m["recall@5"].toFixed(2)}, cross-lingual ${m["recall@5_cross_lingual"].toFixed(2)}`,
      tip: (
        <Tip
          title={methodName(k)}
          rows={[
            { label: "Recall@5", value: m["recall@5"].toFixed(2) },
            { label: "Cross-lingual Recall@5", value: m["recall@5_cross_lingual"].toFixed(2) },
            { label: "MRR", value: m.mrr.toFixed(2) },
            { label: "P@5", value: m["p@5"].toFixed(2) },
          ]}
        />
      ),
    }));

  const queryBars: BarRow[] = Object.entries((best?.["recall@5_by_type"] ?? {}) as Record<string, number>)
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => {
      const n = rt.data?.setup?.query_types?.[k];
      return {
        key: k,
        label: QUERY[k] ?? humanize(k),
        sub: n ? `${n} questions` : undefined,
        value: v,
        cls: "series-1",
        display: v.toFixed(2),
        aria: `${QUERY[k] ?? humanize(k)}: Recall at 5 ${v.toFixed(2)}`,
      };
    });

  return (
    <Page
      title="Evidence"
      subtitle="Held-out evaluation on a real, sensor-recorded contact network (SocioPatterns, Lyon primary school) with simulated absences. Tuned on a different school (Marseille); tested once."
    >
      <ErrorNote error={ab.error} />
      <div className="stats">
        <Stat label="Right cause, found in time (ours)" value={pct(test.two_stage?.actionable_recall)} hint="two-stage detector" />
        <Stat label="Network only" value={pct(test.graph_only?.actionable_recall)} hint="same false-alarm rate" />
        <Stat label="Assistant retrieval" value={best ? best["recall@5"].toFixed(2) : "–"} hint={`Recall@5 (target ${target})`} />
        <Stat
          label="Largest district scanned"
          value={rows.length ? rows[rows.length - 1].n.toLocaleString() : "–"}
          hint={rows.length ? `students · ${rows[rows.length - 1].day_scan_s.toFixed(1)} s per day on a laptop` : undefined}
        />
      </div>

      <div className="evi">
        <Panel
          className="wide"
          title="Cluster detection: both layers matter"
          note={
            <>
              All detectors are held to the same false-alarm rate (about 1 per 20 school days). "Right cause" means the school would take the correct action.{" "}
              {test.two_stage?.n_events ?? ab.data?.meta?.n_events_test ?? "All"} planted events, including mixed-up cases fixed in advance (e.g. a road
              closure that looks like a neighborhood problem).
            </>
          }
        >
          {ab.data === null ? (
            <VizState height={300} />
          ) : (
            <>
              <div className="det">
                {DETECTORS.map((d, i) => {
                  const r = test[d.key] ?? {};
                  const rec = recTest[d.key] ?? {};
                  return (
                    <Fragment key={d.key}>
                      {i > 0 && <div className="det-sep" />}
                      <div className="det-name">
                        <b>
                          <i className={`key ${d.cls}`} />
                          {d.label}
                        </b>
                        <span>{d.uses}</span>
                      </div>
                      <div
                        className="det-bars"
                        role="img"
                        aria-label={`${d.label}: found the cluster ${pct(r.detection_recall)}, right cause in time ${pct(r.actionable_recall)}; with school records only ${pct(rec.detection_recall)} and ${pct(rec.actionable_recall)}`}
                      >
                        <div className="det-row" title={`With school records only: ${pct(rec.detection_recall)}`}>
                          <span>Found the cluster</span>
                          <Bar value={r.detection_recall ?? 0} cls={d.cls} light track marker={rec.detection_recall} />
                          <b>{pct(r.detection_recall)}</b>
                        </div>
                        <div className="det-row" title={`With school records only: ${pct(rec.actionable_recall)}`}>
                          <span>Right cause, in time</span>
                          <Bar value={r.actionable_recall ?? 0} cls={d.cls} track marker={rec.actionable_recall} />
                          <b>{pct(r.actionable_recall)}</b>
                        </div>
                      </div>
                    </Fragment>
                  );
                })}
              </div>
              <Legend
                items={[
                  { label: "Found the cluster", kind: "light", cls: "series-muted" },
                  { label: "Right cause, in time", cls: "series-muted" },
                  { label: "Same test with school records only (no sensors)", kind: "tick" },
                ]}
              />
              <DataTable
                head={["Detector", "Uses", "Found the cluster", "Right cause, in time", "Found (records only)", "Right cause (records only)"]}
                rows={DETECTORS.map((d) => [
                  d.label,
                  d.uses,
                  pct(test[d.key]?.detection_recall),
                  pct(test[d.key]?.actionable_recall),
                  pct(recTest[d.key]?.detection_recall),
                  pct(recTest[d.key]?.actionable_recall),
                ])}
              />
              <p className="muted small">
                The detector uses the real sensor contact network plus school records (class rosters, bus routes, households, neighborhoods). Schools cannot
                collect sensor data, so the tick marks repeat the test with school records only: the result is the same, because most contact happens inside the
                classroom.
              </p>
            </>
          )}
        </Panel>

        <Panel title="Right cause by event type" note="Share of planted events where the school would act on the right cause. Each row moves from network only to two-stage.">
          {ab.data === null ? (
            <VizState height={280} />
          ) : byType.length === 0 ? (
            <VizState height={280}>No per-event results in this evaluation file.</VizState>
          ) : (
            <>
              <Dumbbell
                rows={byType}
                fromLabel="Network only"
                toLabel="Two-stage (ours)"
                fromCls="series-2"
                toCls="series-1"
                ariaLabel="Right cause found per event type, network only compared with two-stage"
              />
              <Legend
                items={[
                  { label: "Network only", kind: "dot", cls: "series-2" },
                  { label: "Two-stage (ours)", kind: "dot", cls: "series-1" },
                ]}
              />
              <p className="muted small">Honest limits: illness outbreaks stay hard for every detector, and simulated replies are synthetic (the contact network is real).</p>
            </>
          )}
        </Panel>

        <Panel title="Recall against false alarms" note="How many right causes each detector finds as its alarm threshold is loosened. Up and to the left is better.">
          {ab.data === null ? (
            <VizState height={300} />
          ) : sweep.length === 0 ? (
            <VizState height={300}>No threshold sweep in this evaluation file.</VizState>
          ) : (
            <>
              <LineChart
                series={sweep}
                interpolate
                marks={calibrated}
                height={300}
                x={{ domain: [0, FA_MAX], ticks: [0, 1, 2, 3, 4], fmt: (v) => `${+v.toFixed(1)}`, label: "False alarms per 20 school days (test school)" }}
                y={{ domain: [0, sweepTop], ticks: ticksTo(sweepTop), fmt: (v) => pct(v), label: "Right cause found, in time (share of events)" }}
                tipTitle={(v) => `${v.toFixed(1)} false alarms per 20 days`}
                ariaLabel="Right cause found against false alarms per 20 school days, one line per detector"
              />
              <Legend items={[...sweep.map((s) => ({ label: s.label, kind: "line" as const, cls: s.cls })), { label: "Threshold used in the test", kind: "ring", cls: "series-muted" }]} />
              <p className="muted small">
                The axis stops at {FA_MAX} false alarms per 20 days; looser thresholds are in the evaluation file. Each threshold was set on the other school, so the
                rings do not sit at the same false-alarm rate here.
              </p>
            </>
          )}
        </Panel>

        <Panel title="When reply labels are wrong" note="Right cause found as the labels on family replies get less accurate. 100% is a perfect labeler.">
          {ab.data === null ? (
            <VizState height={280} />
          ) : noisy.length === 0 ? (
            <VizState height={280}>No label-noise results in this evaluation file.</VizState>
          ) : (
            <>
              <LineChart
                series={noisy}
                dots
                height={280}
                x={{ domain: [Math.min(...accuracies), Math.max(...accuracies)], ticks: accuracies, fmt: (v) => pct(v), label: "Share of reply labels that are correct" }}
                y={{ domain: [0, noisyTop], ticks: ticksTo(noisyTop), fmt: (v) => pct(v), label: "Right cause found, in time (share of events)" }}
                tipTitle={(v) => `${pct(v)} of labels correct`}
                ariaLabel="Right cause found against reply label accuracy, one line per detector that uses labels"
              />
              <Legend items={noisy.map((s) => ({ label: s.label, kind: "line" as const, cls: s.cls }))} />
              <DataTable head={["Labels correct", ...noisy.map((s) => s.label)]} rows={accuracies.map((a, i) => [pct(a), ...noisy.map((s) => pct(s.points[i]?.y))])} />
            </>
          )}
        </Panel>

        <Panel title="Scales to a district" note="Time to scan one school day on a laptop, single process. Both axes are logarithmic.">
          <ErrorNote error={sc.error} />
          {sc.data === null ? (
            <VizState height={280} />
          ) : rows.length === 0 ? (
            <VizState height={280}>No scaling results.</VizState>
          ) : (
            <>
              <LineChart
                series={scaling}
                dots
                height={280}
                x={{ domain: [rows[0].n, rows[rows.length - 1].n], ticks: rows.map((r) => r.n), fmt: kilo, label: "Students (log scale)", log: true }}
                y={{ domain: [0.01, 100], ticks: [0.01, 0.1, 1, 10, 100], fmt: (v) => `${v} s`, label: "Seconds (log scale)", log: true }}
                tipTitle={(v) => `${v.toLocaleString()} students`}
                ariaLabel="Seconds to scan one day against number of students, both on log scales"
              />
              <Legend items={scaling.map((s) => ({ label: s.label, kind: "line" as const, cls: s.cls }))} />
              <DataTable
                head={["Students", "One-day scan", `With ${rows[0].extra_perms ?? "more"} permutations`, "Simulate 100 days"]}
                rows={rows.map((r) => [r.n.toLocaleString(), secs(r.day_scan_s), r.day_scan_extra_perms_s != null ? secs(r.day_scan_extra_perms_s) : "–", secs(r.simulate_s)])}
              />
            </>
          )}
        </Panel>

        <Panel
          className="wide"
          title="Counselor assistant retrieval"
          note="Does the assistant find the right family replies for a question? Recall@5 per search method, best first."
        >
          <ErrorNote error={rt.error} />
          {rt.data === null ? (
            <VizState height={300} />
          ) : best ? (
            <div className="evi" style={{ marginBottom: 0 }}>
              <div>
                <BarList rows={methodBars} max={1} track refAt={target} ariaLabel="Recall at 5 by retrieval method" />
                <Legend
                  items={[
                    { label: "Recall@5", cls: "series-muted" },
                    { label: "Cross-lingual Recall@5", kind: "dot", cls: "series-ink" },
                    { label: `Target ${target}`, kind: "ref" },
                  ]}
                />
              </div>
              <div>
                <p className="panel-note" style={{ marginBottom: 6 }}>
                  Method in use, by kind of question
                </p>
                <BarList rows={queryBars} max={1} track refAt={target} ariaLabel="Recall at 5 of the method in use, by question type" />
                <ul className="small muted" style={{ margin: "12px 0 0", paddingLeft: 18 }}>
                  <li>
                    Recall@5: <b>{best["recall@5"].toFixed(2)}</b> (target {target})
                  </li>
                  <li>
                    Cross-lingual Recall@5 (English question, Spanish/Vietnamese/Chinese replies): <b>{best["recall@5_cross_lingual"].toFixed(2)}</b>
                  </li>
                  <li>MRR: {best.mrr.toFixed(2)}</li>
                </ul>
              </div>
            </div>
          ) : (
            <VizState height={300}>No retrieval results.</VizState>
          )}
        </Panel>
      </div>
    </Page>
  );
}
