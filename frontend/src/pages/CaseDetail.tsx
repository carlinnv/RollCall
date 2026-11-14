import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, fmtDate } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page, StatusTag } from "../components/ui";

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

  if (!a) return <Page title="Case"><ErrorNote error={error} /></Page>;
  const s = a.student;

  const submitReply = async () => {
    const res = await api.reply(id, text, channel, s.preferred_language);
    setRoute(res.route);
    setText("");
    reload();
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
              <p className="muted small">Evidence quote · confidence {Math.round(a.classification.confidence * 100)}%</p>
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
