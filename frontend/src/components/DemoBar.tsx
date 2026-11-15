import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type DemoStep } from "../api";
import { useLoad } from "../hooks";

const PAGE_NAMES: Record<string, string> = {
  "/": "the case queue",
  "/clusters": "Root causes",
  "/plans": "Action plans",
  "/escalations": "Escalations",
  "/oncall": "On-call",
};

/** What a step does, in one line: who texts what, or which label the clerk sets. */
function action(s: DemoStep) {
  if (s.kind === "correct") return `Clerk corrects the label to "${s.label}"`;
  if (s.kind === "view") return `Open ${PAGE_NAMES[s.show ?? ""] ?? s.show}`;
  return `${s.guardian_name} (${s.student_name}'s family) texts: "${s.text}"`;
}

/** The page a finished step should be shown on. */
export function pageFor(s: DemoStep): string | null {
  if (s.show === "case") return s.outcome ? `/cases/${s.outcome.absence_id}` : null;
  return s.show ?? null;
}

function result(s: DemoStep) {
  if (!s.outcome) return null;
  const { absence_id, barrier, escalated } = s.outcome;
  return (
    <>
      {" → "}
      <Link to={`/cases/${absence_id}`}>{barrier ?? "no label"}</Link>
      {escalated && " · with the counselor"}
    </>
  );
}

/**
 * Presenter controls for DEMO_MODE, pinned to the bottom of every page. Plays the next scripted
 * family message through the same path as a real iMessage. Hidden when demo mode is off.
 */
export default function DemoBar() {
  const { data, reload } = useLoad(api.demo);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();
  const [started, setStarted] = useState(false);

  if (!data?.enabled) return null;

  // Before anything has played: one "Start demo" button. It opens the case queue (Monday, 9am)
  // and shows step 1. Once a step has run, the full bar shows even after a page refresh.
  if (!started && !data.steps.some((s) => s.done)) {
    return (
      <div className="demobar" role="region" aria-label="Demo controls">
        <div className="demobar-main">
          <span className="demobar-badge">Demo</span>
          <div className="demobar-text">
            <strong>{data.title ?? "Scripted demo"}</strong>
            <div className="small muted">{data.steps.length} steps. Each one plays a family message or opens the page to show.</div>
          </div>
          <button
            onClick={() => {
              setStarted(true);
              nav("/");
            }}
          >
            Start demo
          </button>
        </div>
      </div>
    );
  }
  const total = data.steps.length;
  const next = data.next === null ? null : data.steps[data.next];
  const last = [...data.steps].reverse().find((s) => s.done);

  const play = async () => {
    setBusy(true);
    setErr(null);
    try {
      const step = await api.demoNext();
      reload();
      const page = pageFor(step);
      if (page) nav(page); // take the presenter to where the step shows up
    } catch (x) {
      setErr((x as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const restart = async () => {
    if (!window.confirm("Reset the demo data and start the script over?")) return;
    await api.reset();
    window.location.assign("/");
  };

  return (
    <div className="demobar" role="region" aria-label="Demo controls">
      {open && (
        <ol className="demobar-steps">
          {data.steps.map((s, i) => (
            <li key={s.id} className={s.done ? "done" : i === data.next ? "current" : ""}>
              <div>{action(s)}</div>
              <div className="small muted">
                {s.say}
                {result(s)}
              </div>
            </li>
          ))}
        </ol>
      )}
      <div className="demobar-main">
        <span className="demobar-badge">Demo</span>
        <div className="demobar-text">
          {/* The headline is the step on screen (the one just played), so the narration matches the page.
              Before the first step, it is the step the button will play. */}
          {last ? (
            <>
              <strong>
                Step {data.steps.indexOf(last) + 1} of {total}:
              </strong>{" "}
              {action(last)}
              {result(last)}
              <div className="small muted">{last.say}</div>
              <div className="small muted demobar-next">
                {next ? <>Up next: {action(next)}</> : "That was the last step."}
              </div>
            </>
          ) : (
            next && (
              <>
                <strong>Ready. Step 1 of {total}:</strong> {action(next)}
                <div className="small muted">{next.say}</div>
              </>
            )
          )}
          {err && <div className="small error">{err}</div>}
        </div>
        <div className="row">
          <button className="ghost" onClick={() => setOpen(!open)}>
            {open ? "Hide steps" : "All steps"}
          </button>
          {next ? (
            <button onClick={play} disabled={busy}>
              {busy
                ? "Sending…"
                : next.kind === "correct"
                  ? "Correct label"
                  : next.kind === "view"
                    ? `Show ${PAGE_NAMES[next.show ?? ""] ?? "page"}`
                    : "Play next message"}
            </button>
          ) : (
            <button onClick={restart}>Start over</button>
          )}
        </div>
      </div>
    </div>
  );
}
