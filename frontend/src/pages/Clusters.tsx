import { useNavigate } from "react-router-dom";
import { api, fmtDate } from "../api";
import type { Cluster } from "../api";
import { useChanged, useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";
import Timeline from "../components/Timeline";
import { Bullet } from "../components/charts";

/** What the students in a group have in common, in plain words. */
const LINK: Record<string, string> = {
  bus_route: "Same bus",
  classroom: "Same class",
  household: "Same family",
  grade: "Same grade",
  neighborhood: "Same neighborhood",
  contact: "Spend time together",
  school: "Whole school",
};

function methodLabel(m: string) {
  if (m === "rule_based") return "Rule-based";
  const det = m.replace(/^scan:/, "").replace(/\+synthetic_baseline$/, "");
  return `Statistical scan (${det})${m.endsWith("+synthetic_baseline") ? ", synthetic baseline" : ""}`;
}

const fmtP = (p: number) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));

/** The p-value as a plain sentence. */
function chance(p: number | null) {
  if (p === null || p === undefined) return null;
  if (p < 0.01) return "Very unlikely to be chance.";
  if (p < 0.05) return "Unlikely to be chance.";
  return "Could still be chance. Keep an eye on it.";
}

function HowWeFound({ c }: { c: Cluster }) {
  return (
    <details className="viz-table">
      <summary>How we found this</summary>
      <ul className="small muted" style={{ margin: "8px 0 0", paddingLeft: 18 }}>
        <li>
          Found by: {methodLabel(c.method)}
          {c.layers_involved.length > 0 && <>, using {c.layers_involved.join(" + ")}</>}
        </li>
        {c.p_value !== null && <li>p-value: {fmtP(c.p_value)} (the chance a group this size has this many absences by luck)</li>}
        <li>
          Absences linked using: the reason alone {c.ablation.text_only}, the shared link alone {c.ablation.network_only}, both together{" "}
          {c.ablation.combined}
        </li>
        {c.method_note && <li>{c.method_note}</li>}
        <li>Family messages: {c.evidence_message_ids.join(", ") || "none"}</li>
      </ul>
    </details>
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
    <Page title="Root causes" subtitle="Students out for the same reason who share a bus, class or family. One fix can help them all.">
      <ErrorNote error={error} />
      <Timeline />
      {data?.length === 0 && (
        <p className="muted">No shared causes yet. One shows up when several absent students share a reason and a bus, class or family.</p>
      )}
      <div className="cards">
        {data?.map((c) => {
          const cause = c.likely_cause ?? c.barrier;
          return (
            <div key={c.id} className={changed.has(c.id) ? "card cluster flash" : "card cluster"}>
              <div className="row">
                <BarrierTag barrier={cause} />
                {c.cause_provisional && (
                  <span className="tag" title="Only a few families have replied so far">
                    not confirmed yet
                  </span>
                )}
              </div>
              <h3>
                {LINK[c.link_type] ?? c.link_type}: {c.link_value}
              </h3>
              <p>{c.summary}</p>
              {c.observed !== null && c.expected !== null && (
                <div className="cluster-stat">
                  <Bullet observed={c.observed} expected={c.expected} cls={`barrier-${cause.split(" ")[0]}`} unit="students absent" compact />
                  {chance(c.p_value) && <p className="small muted" style={{ margin: 0 }}>{chance(c.p_value)}</p>}
                </div>
              )}
              <p className="muted small">
                {c.student_ids.length} student(s) · {fmtDate(c.first_date)} – {fmtDate(c.last_date)}
              </p>
              {c.dataset === "demo" ? (
                <button onClick={() => draft(c.id)}>Make an action plan</button>
              ) : (
                <p className="muted small">Practice school: no families to message.</p>
              )}
              <HowWeFound c={c} />
            </div>
          );
        })}
      </div>
      {meta.data && (
        <details className="viz-table">
          <summary>About this page</summary>
          <p className="small muted" style={{ margin: "8px 0 0" }}>
            Data: {meta.data.dataset} · method: {meta.data.method} · detector: {meta.data.detector}
            {meta.data.notes.map((n) => (
              <span key={n}>
                <br />
                {n}
              </span>
            ))}
          </p>
        </details>
      )}
    </Page>
  );
}
