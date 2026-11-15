import { useState } from "react";
import { api } from "../api";
import type { AssistantAnswer } from "../api";
import { Page } from "../components/ui";

const EXAMPLES = [
  "Why are students on route R04 missing school this week?",
  "Which class has an illness cluster around day 70?",
  "Which Spanish-speaking families said the bus did not come in week 12?",
];

export default function Assistant() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<AssistantAnswer | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const ask = async (question: string) => {
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      setRes(await api.ask(question));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Page
      title="Counselor assistant"
      subtitle="Ask about family messages. Answers cite the messages they come from. Read-only: it cannot send anything. Runs over the simulated Lyon school."
    >
      <div className="card">
        <div className="row">
          <input className="grow" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question" maxLength={500} />
          <button disabled={!q.trim() || busy} onClick={() => ask(q)}>{busy ? "Thinking…" : "Ask"}</button>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          {EXAMPLES.map((e) => (
            <button key={e} className="ghost small" disabled={busy} onClick={() => { setQ(e); ask(e); }}>{e}</button>
          ))}
        </div>
      </div>
      {err && <p className="error">{err}</p>}
      {res && (
        <div className="card">
          <h3>Answer</h3>
          {res.error && <p className="error">{res.error}</p>}
          <p style={{ whiteSpace: "pre-wrap" }}>{res.answer}</p>
          {res.citations.length > 0 && <p className="muted small">Cited messages: {res.citations.join(", ")}</p>}
          {res.dropped_citations.length > 0 && (
            <p className="note">Removed {res.dropped_citations.length} citation(s) the assistant could not back with a retrieved message.</p>
          )}
          {res.report && (
            <details>
              <summary>Case report</summary>
              <pre style={{ whiteSpace: "pre-wrap" }}>{res.report}</pre>
            </details>
          )}
        </div>
      )}
    </Page>
  );
}
