import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, fmtDate, Message } from "../api";
import type { Barrier } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page, StatusTag } from "../components/ui";

const SOURCE_TEXT: Record<string, string> = {
  keywords: "keyword rules",
  keywords_fallback: "keyword rules (Gemini did not answer in time)",
  safety_keywords: "the safety keyword check",
  gemini: "Gemini",
  correction_override: "a similar past staff correction",
  staff_correction: "staff correction",
  counselor: "the counselor (closed as not a safety concern)",
};

const ROUTE_TEXT = {
  escalated: "Safety concern detected. A counselor has taken over; no automated reply will be sent.",
  cluster: "Part of a root-cause cluster. Draft the plan from the Root causes view.",
  case: "Individual case. You can draft a case-level action plan.",
  unknown: "Barrier unclear. Follow up with the family by phone.",
};

const KIND = { reply: "Family", check_in: "Check-in", plan: "Plan message", ack: "Automatic reply", crisis: "Crisis resources (988/911)", page: "Staff page" };
const CHANNEL = { imessage: "iMessage", sms: "SMS", simulated: "simulated", seed: "sample data", replay: "demo replay" };

/** One line under a message: who, channel, time, and how long it took. */
function meta(m: Message) {
  const parts: string[] = [KIND[m.kind]];
  if (m.channel) parts.push(CHANNEL[m.channel]);
  const at = new Date(m.timestamp.endsWith("Z") ? m.timestamp : m.timestamp + "Z");
  if (!isNaN(at.getTime())) parts.push(at.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }));
  if (m.latency_ms !== null) {
    const seconds = (m.latency_ms / 1000).toFixed(1);
    parts.push(m.direction === "inbound" ? `labeled in ${seconds} s` : m.kind === "ack" ? `replied in ${seconds} s` : `sent in ${seconds} s`);
  }
  if (m.direction === "outbound" && m.status !== "sent") parts.push(m.reason ? `${m.status}: ${m.reason}` : m.status);
  return parts.join(" · ");
}

export default function CaseDetail() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const { data: a, error, reload } = useLoad(() => api.absence(id), [id]);
  const [text, setText] = useState("");
  const [route, setRoute] = useState<keyof typeof ROUTE_TEXT | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const taxonomy = useLoad(api.taxonomy);
  const [label, setLabel] = useState<Barrier | "">("");
  const [by, setBy] = useState("clerk");
  const [corrected, setCorrected] = useState<string | null>(null);

  if (!a) return <Page title="Case"><ErrorNote error={error} /></Page>;
  const s = a.student;

  const submitReply = async () => {
    const res = await api.reply(id, text, "text", s.preferred_language);
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
          <h3>Conversation</h3>
          <p className="muted small">
            With {s.guardian_name} · {s.guardian_phone} · language: {s.preferred_language}
          </p>
          {!a.messages.some((m) => m.kind === "check_in") && (
            <>
              <p className="bubble out">{a.check_in_message}</p>
              <p className="muted small" style={{ textAlign: "right" }}>Check-in · not sent through a channel yet</p>
            </>
          )}
          {a.messages.map((m) => (
            <div key={m.message_id}>
              <p className={`bubble ${m.direction === "inbound" ? "in" : "out"}`}>{m.raw_text_or_transcript}</p>
              <p className="muted small" style={{ textAlign: m.direction === "inbound" ? "left" : "right" }}>
                {meta(m)} {m.classification && <BarrierTag barrier={m.classification.barrier} />}
              </p>
            </div>
          ))}
          {!a.messages.some((m) => m.direction === "inbound") && <p className="muted">No reply yet.</p>}
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
        <p className="muted small">Stands in for the iMessage / SMS channel during the demo. Any language.</p>
        <div className="row">
          <input className="grow" value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. El autobús no pasó esta mañana" />
          <button disabled={!text.trim()} onClick={submitReply}>
            Receive reply
          </button>
        </div>
      </div>
    </Page>
  );
}
