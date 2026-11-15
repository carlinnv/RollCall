import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type Barrier, type CaseReport, type Escalation } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";

// No sign-in yet, so every action is recorded as the counselor.
const STAFF = "Counselor";

// Labels a counselor can relabel a false alarm to.
const RELABEL: Barrier[] = [
  "illness",
  "transportation",
  "housing or food insecurity",
  "caregiving or family obligation",
  "disengagement or academic struggle",
  "appointment or travel",
  "unknown",
];

const REPORT = { report_filed: "Report filed", not_required: "No report required" };

function when(stamp: string | null) {
  if (!stamp) return "";
  const t = new Date(/Z|[+-]\d\d:\d\d$/.test(stamp) ? stamp : stamp + "Z");
  return t.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function waited(minutes: number) {
  if (minutes < 60) return `${minutes} min`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h`;
  return `${Math.round(minutes / (60 * 24))} days`;
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

/** Generate, edit, save and download the internal case report for one escalation. */
function CaseReportPanel({ e, onChange }: { e: Escalation; onChange: () => void }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState("");
  const [versions, setVersions] = useState<CaseReport[]>([]);
  const [viewing, setViewing] = useState<number | null>(null); // a saved version shown read-only
  const [dirty, setDirty] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const run = async (action: () => Promise<void>) => {
    try {
      setErr(null);
      await action();
    } catch (x) {
      setErr((x as Error).message);
    }
  };

  const regenerate = () =>
    run(async () => {
      if (dirty && !window.confirm("Replace your unsaved edits with a fresh report from the record?")) return;
      setText((await api.draftCaseReport(e.escalation_id, STAFF)).text);
      setViewing(null);
      setDirty(false);
      setNote("Filled in from the record. Notes from your last saved version are kept.");
    });

  const toggle = () =>
    run(async () => {
      if (open) return setOpen(false);
      const saved = await api.caseReports(e.escalation_id);
      setVersions(saved);
      setOpen(true);
      if (saved.length) {
        setText(saved[saved.length - 1].text);
        setNote(`Showing saved version ${saved.length}. Regenerate to pick up anything that happened since.`);
      } else {
        setText((await api.draftCaseReport(e.escalation_id, STAFF)).text);
        setNote("Filled in from the record. Add your notes, then save.");
      }
      setViewing(null);
      setDirty(false);
    });

  const save = () =>
    run(async () => {
      const report = await api.saveCaseReport(e.escalation_id, STAFF, text);
      setVersions((v) => [...v, report]);
      setDirty(false);
      setNote(`Saved as version ${report.version}, ${when(report.saved_at)}.`);
      onChange();
    });

  const show = (version: string) => {
    if (version === "current") return setViewing(null);
    setViewing(Number(version));
  };
  const shown = viewing === null ? text : versions.find((v) => v.version === viewing)?.text ?? "";

  return (
    <div className="report">
      <button className="ghost" onClick={toggle}>
        {open ? "Hide case report" : "Case report"}
        {!open && e.case_report_versions > 0 && ` · ${e.case_report_versions} saved`}
      </button>
      {open && (
        <>
          <p className="muted small">
            Internal record for the school's files, filled in from the case. It is not the mandated report.
          </p>
          {versions.length > 0 && (
            <div className="row small">
              <span className="muted">Version:</span>
              <select value={viewing ?? "current"} onChange={(x) => show(x.target.value)}>
                <option value="current">Editing{dirty ? " (unsaved)" : ""}</option>
                {versions.map((v) => (
                  <option key={v.version} value={v.version}>
                    v{v.version} · {v.saved_by} · {when(v.saved_at)}
                  </option>
                ))}
              </select>
            </div>
          )}
          <textarea
            className="report-text"
            value={shown}
            readOnly={viewing !== null}
            onChange={(x) => {
              setText(x.target.value);
              setDirty(true);
            }}
          />
          <div className="row">
            <button disabled={viewing !== null || !text.trim()} onClick={save}>
              Save version {versions.length + 1}
            </button>
            <button className="ghost" onClick={regenerate}>
              Regenerate from record
            </button>
            <button className="ghost" onClick={() => download(`case-report-${e.escalation_id}${viewing ? `-v${viewing}` : ""}.txt`, shown)}>
              Download .txt
            </button>
          </div>
          {note && <p className="muted small">{note}</p>}
          {err && <p className="error">{err}</p>}
        </>
      )}
    </div>
  );
}

const TYPE: Record<string, string> = {
  self_harm: "self-harm",
  imminent_danger: "imminent danger",
  abuse_or_neglect: "abuse or neglect",
  bullying: "bullying",
  supervision: "supervision",
  other: "other",
};

/** Whether the fixed 988/911 message went to the family, and if not, why. */
function CrisisNote({ e }: { e: Escalation }) {
  if (!e.crisis_message_status) return null;
  if (e.crisis_message_status === "sent")
    return <p className="note">Crisis-resource message (988/911) sent to the family, {when(e.crisis_message_at)}.</p>;
  if (e.crisis_message_status === "withheld")
    return <p className="alert-box">Crisis-resource message withheld: {e.crisis_message_reason}.</p>;
  return (
    <p className="alert-box">
      Crisis-resource message {e.crisis_message_status}: {e.crisis_message_reason}. Contact the family directly.
    </p>
  );
}

/** Who was paged about this case, and whether the page went out. */
function PageList({ e }: { e: Escalation }) {
  if (!e.pages.length) return null;
  return (
    <ul className="pages small">
      {e.pages.map((p, i) => (
        <li key={i} className={p.status === "sent" ? "muted" : "overdue"}>
          {p.level === "escalated" ? "Escalated: paged " : "Paged "}
          {p.name ? `${p.name} (${p.role})` : "nobody"}, {when(p.at)} · {p.status}
          {p.reason && `: ${p.reason}`}
        </li>
      ))}
    </ul>
  );
}

function OpenCase({ e, onChange }: { e: Escalation; onChange: () => void }) {
  const [relabel, setRelabel] = useState<Barrier>("illness");
  const [falseAlarm, setFalseAlarm] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async (action: () => Promise<unknown>) => {
    try {
      setErr(null);
      await action();
      onChange();
    } catch (x) {
      setErr((x as Error).message);
    }
  };

  return (
    <div className="card alert-card">
      <div className="row between">
        <span className="tag alert">{e.reason}</span>
        <span className="muted small">{e.escalation_id}</span>
      </div>
      <h3>
        <Link to={`/cases/${e.absence_id}`}>
          {e.student.first_name} {e.student.last_name}
        </Link>
      </h3>
      <p className="muted small">
        Grade {e.student.grade} · {e.student.classroom} · guardian {e.student.guardian_name} ({e.student.guardian_phone})
      </p>
      <blockquote>{e.evidence}</blockquote>
      {e.safety_types.length > 0 && <p className="muted small">Concern: {e.safety_types.map((t) => TYPE[t] ?? t).join(", ")}</p>}
      <CrisisNote e={e} />
      <PageList e={e} />
      <p className="muted small">All other automated messages to this household are paused until this case is closed.</p>

      <ol className="steps">
        <li>
          {e.acknowledged_at ? (
            <span>
              Acknowledged by {e.acknowledged_by}, {when(e.acknowledged_at)}
            </span>
          ) : (
            <span className="row">
              <span className={e.overdue ? "overdue" : "muted"}>
                {e.overdue ? "Overdue: " : ""}waiting {waited(e.minutes_open)} (acknowledge within 15 min)
              </span>
              <button onClick={() => run(() => api.acknowledgeEscalation(e.escalation_id, STAFF))}>Acknowledge</button>
            </span>
          )}
        </li>
        <li>
          {e.report_status ? (
            <span>
              {REPORT[e.report_status]} ({e.report_by}, {when(e.report_at)})
            </span>
          ) : (
            <span className="row">
              <span className="muted">Mandated report:</span>
              <button className="ghost" onClick={() => run(() => api.reportEscalation(e.escalation_id, STAFF, "report_filed"))}>
                Report filed
              </button>
              <button className="ghost" onClick={() => run(() => api.reportEscalation(e.escalation_id, STAFF, "not_required"))}>
                No report required
              </button>
            </span>
          )}
        </li>
        <li>
          {!e.report_status && <span className="muted">Close the case (after the reporting decision)</span>}
          {e.report_status && !falseAlarm && (
            <span className="row">
              <button onClick={() => run(() => api.closeEscalation(e.escalation_id, STAFF, "resolved"))}>
                Close: concern handled
              </button>
              <button className="ghost" onClick={() => setFalseAlarm(true)}>
                Not a safety concern…
              </button>
            </span>
          )}
          {e.report_status && falseAlarm && (
            <span className="row">
              <span className="muted">Correct label:</span>
              <select value={relabel} onChange={(x) => setRelabel(x.target.value as Barrier)}>
                {RELABEL.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
              <button onClick={() => run(() => api.closeEscalation(e.escalation_id, STAFF, "not_safety", relabel))}>
                Relabel and close
              </button>
              <button className="ghost" onClick={() => setFalseAlarm(false)}>
                Cancel
              </button>
            </span>
          )}
        </li>
      </ol>
      {err && <p className="error">{err}</p>}
      <CaseReportPanel e={e} onChange={onChange} />
    </div>
  );
}

function ClosedCase({ e, onChange }: { e: Escalation; onChange: () => void }) {
  return (
    <>
    <tr>
      <td>
        <Link to={`/cases/${e.absence_id}`}>
          {e.student.first_name} {e.student.last_name}
        </Link>
        <div className="muted small">{e.escalation_id}</div>
      </td>
      <td className="quote">{e.evidence}</td>
      <td>
        {e.outcome === "not_safety" ? (
          <>
            Not a safety concern; relabeled <BarrierTag barrier={e.relabeled_to} />
          </>
        ) : (
          "Concern handled"
        )}
        <div className="muted small">{e.report_status ? REPORT[e.report_status] : ""}</div>
      </td>
      <td className="muted small">
        {e.closed_by}, {when(e.closed_at)}
      </td>
    </tr>
    <tr className="report-row">
      <td colSpan={4}>
        <CaseReportPanel e={e} onChange={onChange} />
      </td>
    </tr>
    </>
  );
}

export default function Escalations() {
  const { data, error, reload } = useLoad(api.escalations);
  const open = (data ?? []).filter((e) => e.status !== "closed");
  const closed = (data ?? []).filter((e) => e.status === "closed");

  return (
    <Page title="Escalations" subtitle="Safety concerns. A counselor takes over; nothing is sent to the household until the case is closed.">
      <ErrorNote error={error} />
      {data && open.length === 0 && <p className="muted">No open escalations.</p>}
      <div className="cards">
        {open.map((e) => (
          <OpenCase key={e.escalation_id} e={e} onChange={reload} />
        ))}
      </div>
      {closed.length > 0 && (
        <>
          <h2>Closed</h2>
          <table>
            <thead>
              <tr>
                <th>Student</th>
                <th>Family said</th>
                <th>Outcome</th>
                <th>Closed</th>
              </tr>
            </thead>
            <tbody>
              {closed.map((e) => (
                <ClosedCase key={e.escalation_id} e={e} onChange={reload} />
              ))}
            </tbody>
          </table>
        </>
      )}
    </Page>
  );
}
