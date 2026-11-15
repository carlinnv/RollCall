import { useNavigate } from "react-router-dom";
import { api, fmtDate, money } from "../api";
import type { Cluster } from "../api";
import { useChanged, useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";
import Timeline from "../components/Timeline";
import { Bullet } from "../components/charts";

const LINK: Record<string, string> = {
  bus_route: "Shared bus route",
  classroom: "Shared classroom",
  household: "Same household",
  grade: "Same grade",
  neighborhood: "Same neighborhood",
  contact: "Close contacts",
  school: "Whole school",
};

function methodLabel(m: string) {
  if (m === "rule_based") return "Rule-based fallback";
  const det = m.replace(/^scan:/, "").replace(/\+synthetic_baseline$/, "");
  return `Scan (${det})${m.endsWith("+synthetic_baseline") ? " · synthetic baseline" : ""}`;
}

function fmtP(p: number | null) {
  if (p === null || p === undefined) return "n/a";
  return p < 0.001 ? p.toExponential(1) : p.toFixed(3);
}

function Stats({ c }: { c: Cluster }) {
  if (c.observed !== null && c.expected !== null) {
    return (
      <div className="cluster-stat">
        <Bullet observed={c.observed} expected={c.expected} cls={`barrier-${(c.likely_cause ?? c.barrier).split(" ")[0]}`} unit="students absent" />
        <p className="small muted" style={{ margin: 0 }}>
          p-value <b className="pv">{fmtP(c.p_value)}</b> · the chance of a group this size seeing this many by luck
        </p>
      </div>
    );
  }
  return (
    <div className="ablation">
      <div>
        <b>{c.observed ?? "–"}</b>
        <span>observed</span>
      </div>
      <div>
        <b>{c.expected !== null ? c.expected.toFixed(1) : "–"}</b>
        <span>expected</span>
      </div>
      <div className="hl">
        <b>{fmtP(c.p_value)}</b>
        <span>p-value</span>
      </div>
    </div>
  );
}

export default function Clusters() {
  const { data, error } = useLoad(api.clusters);
  const meta = useLoad(api.clusterMeta);
  // A cluster that just formed or gained a student is highlighted briefly.
  const changed = useChanged(data, (c) => c.id, (c) => `${c.student_ids.length}|${c.absence_ids.length}`);
  const nav = useNavigate();

  const draft = async (cluster_id: string) => {
    await api.draftPlan({ cluster_id });
    nav("/plans");
  };

  return (
    <Page title="Root causes" subtitle="Absences grouped by shared reason and a shared school link (bus route, classroom, grade).">
      <ErrorNote error={error} />
      <Timeline />
      {meta.data && (
        <p className="muted small">
          Dataset: {meta.data.dataset} · method: {meta.data.method} · detector: {meta.data.detector}
          {meta.data.notes.map((n) => (
            <span key={n}>
              <br />
              {n}
            </span>
          ))}
        </p>
      )}
      {data?.length === 0 && <p className="muted">No clusters detected yet.</p>}
      <div className="cards">
        {data?.map((c) => (
          <div key={c.id} className={changed.has(c.id) ? "card cluster flash" : "card cluster"}>
            <div className="row between">
              <span>
                <BarrierTag barrier={c.likely_cause ?? c.barrier} />{" "}
                {c.cause_provisional && <span className="tag" title="Fewer than 2 replies (or under half the members) back this cause">provisional cause</span>}
              </span>
              <span className="money">{money(c.weekly_revenue_at_risk ?? c.dollars_at_risk_week)} at risk this week</span>
            </div>
            <h3>
              {LINK[c.link_type] ?? c.link_type}: {c.link_value}
            </h3>
            <p>{c.summary}</p>
            <p className="muted small">
              {fmtDate(c.first_date)} – {fmtDate(c.last_date)} · {c.absence_ids.length} absence(s) · {c.student_ids.length} student(s)
              {c.detected_on && <> · detected {fmtDate(c.detected_on)}</>}
            </p>
            <p className="small">
              <span className="tag" title={c.method_note ?? ""}>
                {methodLabel(c.method)}
              </span>{" "}
              <span className="muted">layers: {c.layers_involved.join(" + ") || "–"}</span>
            </p>
            <Stats c={c} />
            <div className="ablation">
              <div>
                <b>{c.ablation.text_only}</b>
                <span>reason text only</span>
              </div>
              <div>
                <b>{c.ablation.network_only}</b>
                <span>school link only</span>
              </div>
              <div className="hl">
                <b>{c.ablation.combined}</b>
                <span>both signals</span>
              </div>
            </div>
            {c.method_note && <p className="muted small">{c.method_note}</p>}
            <p className="muted small">Evidence messages: {c.evidence_message_ids.join(", ") || "none"}</p>
            {c.dataset === "demo" ? (
              <button onClick={() => draft(c.id)}>Draft cluster action plan</button>
            ) : (
              <p className="muted small">Simulated school: no families to message.</p>
            )}
          </div>
        ))}
      </div>
    </Page>
  );
}
