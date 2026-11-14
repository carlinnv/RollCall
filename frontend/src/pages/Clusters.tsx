import { useNavigate } from "react-router-dom";
import { api, fmtDate, money } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";

const LINK = { bus_route: "Shared bus route", classroom: "Shared classroom", household: "Same household" };

export default function Clusters() {
  const { data, error } = useLoad(api.clusters);
  const nav = useNavigate();

  const draft = async (cluster_id: string) => {
    await api.draftPlan({ cluster_id });
    nav("/plans");
  };

  return (
    <Page title="Root causes" subtitle="Absences grouped by shared reason and a shared school link (bus route, classroom, household).">
      <ErrorNote error={error} />
      {data?.length === 0 && <p className="muted">No clusters detected yet.</p>}
      <div className="cards">
        {data?.map((c) => (
          <div key={c.id} className="card cluster">
            <div className="row between">
              <BarrierTag barrier={c.barrier} />
              <span className="money">{money(c.dollars_at_risk_week)} at risk this week</span>
            </div>
            <h3>
              {LINK[c.link_type]}: {c.link_value}
            </h3>
            <p>{c.summary}</p>
            <p className="muted small">
              {fmtDate(c.first_date)} – {fmtDate(c.last_date)} · {c.absence_ids.length} absence(s) · {c.student_ids.length} student(s)
            </p>
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
            <button onClick={() => draft(c.id)}>Draft cluster action plan</button>
          </div>
        ))}
      </div>
    </Page>
  );
}
