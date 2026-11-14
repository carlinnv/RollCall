import { api } from "../api";
import { useLoad } from "../hooks";
import { ErrorNote, Page, Stat } from "../components/ui";

const DETECTORS: { key: string; label: string; uses: string }[] = [
  { key: "two_stage", label: "Two-stage (ours)", uses: "Network finds the cluster, family replies name the cause" },
  { key: "graph_only", label: "Network only", uses: "All absences + school network, no replies" },
  { key: "combined", label: "Combined (wait for replies)", uses: "Only absences whose family has replied" },
  { key: "content_only", label: "Replies only", uses: "Reply labels, school-wide" },
];

const pct = (x: number | null | undefined) => (x == null ? "–" : `${Math.round(x * 100)}%`);

export default function Evidence() {
  const ab = useLoad(api.ablation);
  const rt = useLoad(api.retrieval);
  const sc = useLoad(api.scaling);
  const test = ab.data?.test ?? {};
  const types = Object.keys(test.two_stage?.by_event_type ?? {});
  const best = rt.data?.methods?.["bm25+expansion+filter"];
  const rows: any[] = sc.data?.rows ?? [];

  return (
    <Page
      title="Evidence"
      subtitle="Held-out evaluation on a real public contact network (SocioPatterns, Lyon primary school) with simulated absences. Tuned on a different school (Marseille); tested once."
    >
      <ErrorNote error={ab.error} />
      <div className="stats">
        <Stat label="Right cause, found in time (ours)" value={pct(test.two_stage?.actionable_recall)} hint="two-stage detector" />
        <Stat label="Network only" value={pct(test.graph_only?.actionable_recall)} hint="same false-alarm rate" />
        <Stat label="Assistant retrieval" value={best ? best["recall@5"].toFixed(2) : "–"} hint="Recall@5 (target 0.85)" />
        <Stat label="Largest district scanned" value={rows.length ? `${rows[rows.length - 1].n.toLocaleString()} students` : "–"}
          hint={rows.length ? `${rows[rows.length - 1].day_scan_s.toFixed(1)} s per day on a laptop` : undefined} />
      </div>

      <div className="card">
        <h3>Cluster detection: both layers matter</h3>
        <p className="muted small">
          All detectors are held to the same false-alarm rate (about 1 per 20 school days). "Right cause" means the
          school would take the correct action. {test.two_stage?.n_events ?? ab.data?.meta?.n_events_test ?? "All"} planted events, including mixed-up cases
          fixed in advance (e.g. a road closure that looks like a neighborhood problem).
        </p>
        <table>
          <thead>
            <tr><th>Detector</th><th>Uses</th><th>Found the cluster</th><th>Right cause, in time</th></tr>
          </thead>
          <tbody>
            {DETECTORS.map((d) => (
              <tr key={d.key}>
                <td><b>{d.label}</b></td>
                <td className="muted small">{d.uses}</td>
                <td>{pct(test[d.key]?.detection_recall)}</td>
                <td><b>{pct(test[d.key]?.actionable_recall)}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {types.length > 0 && (
        <div className="card">
          <h3>Right cause by event type</h3>
          <table>
            <thead>
              <tr><th>Event</th><th>Network only</th><th>Two-stage (ours)</th></tr>
            </thead>
            <tbody>
              {types.map((t) => (
                <tr key={t}>
                  <td>{t.replace(/_/g, " ")}</td>
                  <td>{pct(test.graph_only?.by_event_type?.[t]?.actionable_recall)}</td>
                  <td><b>{pct(test.two_stage?.by_event_type?.[t]?.actionable_recall)}</b></td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small">Honest limits: illness outbreaks stay hard for every detector, and simulated replies are synthetic (the contact network is real).</p>
        </div>
      )}

      <div className="grid2">
        <div className="card">
          <h3>Counselor assistant retrieval</h3>
          <ErrorNote error={rt.error} />
          {best && (
            <ul>
              <li>Recall@5: <b>{best["recall@5"].toFixed(2)}</b> (target 0.85)</li>
              <li>Cross-lingual Recall@5 (English question, Spanish/Vietnamese/Chinese replies): <b>{best["recall@5_cross_lingual"].toFixed(2)}</b></li>
              <li>MRR: {best.mrr.toFixed(2)}</li>
            </ul>
          )}
        </div>
        <div className="card">
          <h3>Scales to a district</h3>
          <ErrorNote error={sc.error} />
          <table>
            <thead><tr><th>Students</th><th>One-day scan</th><th>Simulate 100 days</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.n}><td>{r.n.toLocaleString()}</td><td>{r.day_scan_s.toFixed(2)} s</td><td>{r.simulate_s.toFixed(1)} s</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Page>
  );
}
