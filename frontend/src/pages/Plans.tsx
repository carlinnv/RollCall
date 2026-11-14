import { useState } from "react";
import { api, fmtDate, type ActionPlan } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";

function PlanCard({ plan, onChange }: { plan: ActionPlan; onChange: () => void }) {
  const [message, setMessage] = useState(plan.message);
  const draft = plan.status === "draft";

  const approve = async () => {
    if (message !== plan.message) await api.updatePlan(plan.id, { message });
    await api.approvePlan(plan.id, "Attendance clerk");
    onChange();
  };

  return (
    <div className="card">
      <div className="row between">
        <div className="row">
          <BarrierTag barrier={plan.barrier} />
          <span className="muted small">{plan.scope === "cluster" ? "Cluster plan" : "Case plan"} · {plan.recipient_student_ids.length} family(ies)</span>
        </div>
        <span className={`status plan-${plan.status}`}>{plan.status}</span>
      </div>
      <dl>
        <dt>School will</dt>
        <dd>{plan.school_action}</dd>
        <dt>Owner</dt>
        <dd>{plan.owner}</dd>
        <dt>Target date</dt>
        <dd>{fmtDate(plan.target_date)}</dd>
        <dt>We ask the family</dt>
        <dd>{plan.family_request}</dd>
      </dl>
      <label className="muted small">
        Message to families ({plan.message_language}){plan.scope === "cluster" && " · {guardian} and {student} are filled per family"}
      </label>
      <textarea rows={4} value={message} disabled={!draft} onChange={(e) => setMessage(e.target.value)} />
      {draft ? (
        <button onClick={approve}>Approve &amp; send</button>
      ) : (
        <p className="muted small">Approved by {plan.approved_by}.</p>
      )}
    </div>
  );
}

export default function Plans() {
  const { data, error, reload } = useLoad(api.plans);
  const outbox = useLoad(api.outbox);
  const refresh = () => {
    reload();
    outbox.reload();
  };

  return (
    <Page title="Action plans" subtitle="Every plan needs a clerk's approval before anything is sent.">
      <ErrorNote error={error} />
      {data?.length === 0 && <p className="muted">No plans yet. Draft one from Root causes or a case.</p>}
      <div className="cards">
        {data?.map((p) => <PlanCard key={p.id + p.status} plan={p} onChange={refresh} />)}
      </div>
      {!!outbox.data?.length && (
        <>
          <h2>Outbound messages</h2>
          <table>
            <thead>
              <tr>
                <th>To</th>
                <th>Message</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {outbox.data.map((m, i) => (
                <tr key={i}>
                  <td className="muted">{m.to}</td>
                  <td className="quote">{m.text}</td>
                  <td className="small">
                    {m.status}
                    {m.reason && <div className="muted">{m.reason}</div>}
                  </td>
                  <td className="muted small">{m.sent_at.replace("T", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </Page>
  );
}
