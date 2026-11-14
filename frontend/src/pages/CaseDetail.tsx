import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, fmtDate } from "../api";
import type { Barrier } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page, StatusTag } from "../components/ui";

const SOURCE_TEXT: Record<string, string> = {
  keyword: "keyword classifier",
  gemini: "Gemini",
  correction_override: "a similar past staff correction",
  staff_correction: "staff correction",
};

const ROUTE_TEXT = {
  escalated: "Safety concern detected. A counselor has taken over; no automated reply will be sent.",
  cluster: "Part of a root-cause cluster. Draft the plan from the Root causes view.",
  case: "Individual case. You can draft a case-level action plan.",
  unknown: "Barrier unclear. Follow up with the family by phone.",
};

export default function CaseDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const { data: a, error, reload } = useLoad(() => api.absence(id), [id]);
  const [text, setText] = useState("");
  const [channel, setChannel] = useState<"text" | "voice">("text");
  const [route, setRoute] = useState<keyof typeof ROUTE_TEXT | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const taxonomy = useLoad(api.taxonomy);
  const [label, setLabel] = useState<Barrier | "">("");
  const [by, setBy] = useState("clerk");
  const [corrected, setCorrected] = useState<string | null>(null);

  if (!a) return <Page title="Case"><ErrorNote error={error} /></Page>;
  const s = a.student;

  const submitReply = async () => {
    const res = await api.reply(id, text, channel, s.preferred_language);
    setRoute(res.route);
    setText("");
    reload();
  };

  const correct = async () => {
    if (!label) return;
    try {
      const res = await api.correct(id, label, by);
      setErr(null);
      setCorrected(
        `Saved: ${res.correction.old_label ?? "none"} → ${res.correction.new_label}. Similar replies at this school will use it.` +
          (res.escalated ? " A counselor now has this case." : ""),
      );
      setLabel("");
      reload();
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  const draft = async () => {
    try {
      await api.draftPlan({ absence_id: id });
      nav("/plans");
    } catch (e) {
      setErr((e as Error).message);
    }
  };

  return (
    <Page title={`${s.first_name} ${s.last_name}`} subtitle={`Absent ${fmtDate(a.date)} · Grade ${s.grade} · ${s.classroom} (${s.teacher}) · ${s.bus_route ?? "no bus"}`}>
      <ErrorNote error={error ?? err} />
      <div className="grid2">
        <div className="card">
          <h3>Check-in sent to family</h3>
          <p className="muted small">
            To {s.guardian_name} · {s.guardian_phone} · language: {s.preferred_language}
          </p>
          <p className="bubble out">{a.check_in_message}</p>
          {a.reply ? (
            <>
              <p className="muted small">
                Family reply ({a.reply.channel === "voice" ? "voice note transcript" : "text"}, {a.reply.language})
              </p>
              <p className="bubble in">{a.reply.text}</p>
            </>
          ) : (
            <p className="muted">No reply yet.</p>
          )}
        </div>

        <div className="card">
          <h3>Classification</h3>
          <p>
            <StatusTag status={a.status} /> {a.classification && <BarrierTag barrier={a.classification.barrier} />}
          </p>
          {a.classification && (
            <>
              <p className="muted small">
                Evidence quote · confidence {Math.round(a.classification.confidence * 100)}%
                {a.classification.source && <> · from {SOURCE_TEXT[a.classification.source] ?? a.classification.source}</>}
                {a.classification.source === "correction_override" && a.classification.correction_ref && (
                  <>
                    {" "}
                    (<Link to={`/cases/${a.classification.correction_ref}`}>{a.classification.correction_ref}</Link>, similarity{" "}
                    {a.classification.correction_similarity?.toFixed(2)})
                  </>
                )}
              </p>
              <blockquote>{a.classification.evidence}</blockquote>
            </>
          )}
          {a.escalated && <p className="alert-box">{ROUTE_TEXT.escalated}</p>}
          {route && !a.escalated && <p className="note">{ROUTE_TEXT[route]}</p>}
          <div className="row">
            {a.classification && !a.escalated && (
              <button onClick={draft}>Draft case action plan</button>
            )}
            <Link className="button ghost" to={`/learning/${s.id}`}>
              Learning plan
            </Link>
          </div>
        </div>
      </div>

      {a.reply && (
        <div className="card">
          <h3>Correct label</h3>
          <p className="muted small">
            Corrections are stored for this school and used for similar replies later. Only a counselor can remove a safety label.
          </p>
          <div className="row">
            <select value={label} onChange={(e) => setLabel(e.target.value as Barrier)}>
              <option value="" disabled>
                Correct label to…
              </option>
              {taxonomy.data?.map((b) => (
                <option key={b} value={b} disabled={b === a.classification?.barrier}>
                  {b}
                </option>
              ))}
            </select>
            <input value={by} onChange={(e) => setBy(e.target.value)} placeholder="Your role, e.g. clerk or counselor" />
            <button disabled={!label || !by.trim()} onClick={correct}>
              Save correction
            </button>
          </div>
          {corrected && <p className="note">{corrected}</p>}
        </div>
      )}

      <div className="card">
        <h3>Simulate family reply</h3>
        <p className="muted small">Stands in for the SMS / voice channel during the demo. Any language; voice replies are transcripts.</p>
        <div className="row">
          <select value={channel} onChange={(e) => setChannel(e.target.value as "text" | "voice")}>
            <option value="text">Text</option>
            <option value="voice">Voice note</option>
          </select>
          <input className="grow" value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. El autobús no pasó esta mañana" />
          <button disabled={!text.trim()} onClick={submitReply}>
            Receive reply
          </button>
        </div>
      </div>
    </Page>
  );
}
