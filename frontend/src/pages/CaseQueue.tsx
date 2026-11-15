import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtDate, type AbsenceStatus, type QueueRow } from "../api";
import { useChanged, useLoad, useStableOrder } from "../hooks";
import { useAuth } from "../auth";
import { BarrierTag, ErrorNote, Page, Stat, StatusTag } from "../components/ui";

const FILTERS: (AbsenceStatus | "all")[] = ["all", "awaiting reply", "unknown", "reason captured"];

const rowId = (r: QueueRow) => r.id;
/** What must differ for a row to count as changed: its status, label, escalation or reply. */
const rowSignature = (r: QueueRow) => [r.status, r.classification?.barrier, r.escalated, r.reply?.text].join("|");

export default function CaseQueue() {
  const { data, error, reload } = useLoad(api.queue);
  const [filter, setFilter] = useState<AbsenceStatus | "all">("all");
  const [importMsg, setImportMsg] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const { role } = useAuth();
  const canSend = role?.id === "clerk" || role?.id === "admin";
  const pending = useLoad(api.checkInsPending, [], canSend);
  const toSend = pending.data?.to_send ?? 0;
  const [sending, setSending] = useState(false);
  const [sendMsg, setSendMsg] = useState<string | null>(null);

  const onSendCheckIns = async () => {
    const { to_send } = await api.checkInsPending();
    if (to_send === 0) {
      setSendMsg("Every family awaiting a reply already has a check-in.");
      return;
    }
    if (!window.confirm(`Text ${to_send} famil${to_send === 1 ? "y" : "ies"} a check-in asking why their student was out?`)) return;
    setSending(true);
    try {
      const r = await api.sendCheckIns();
      const parts = [`${r.sent} sent`];
      if (r.queued) parts.push(`${r.queued} waiting`);
      if (r.blocked) parts.push(`${r.blocked} blocked`);
      if (r.failed) parts.push(`${r.failed} failed`);
      const why = Object.entries(r.reasons).map(([reason, n]) => `${reason} (${n})`).join("; ");
      setSendMsg(`Check-ins: ${parts.join(", ")}.${why ? ` Not sent: ${why}.` : ""}`);
    } catch (e) {
      setSendMsg(`Could not send check-ins: ${(e as Error).message}`);
    } finally {
      setSending(false);
      pending.reload();
      reload();
    }
  };

  // Live updates highlight the row that changed and leave it where it is, so it is easy to follow.
  const changed = useChanged(data, rowId, rowSignature);
  const { rows: ordered, held, resort } = useStableOrder(data, rowId, (r) => r.escalated);
  const rows = useMemo(() => ordered.filter((r) => filter === "all" || r.status === filter), [ordered, filter]);
  const count = (s: AbsenceStatus) => (data ?? []).filter((r) => r.status === s).length;

  const onImport = async (file: File) => {
    const res = await api.importCsv(await file.text());
    const people = res.students_added ? ` and ${res.students_added} new student(s)` : "";
    setImportMsg(`Imported ${res.added} absence(s)${people}${res.errors.length ? `; ${res.errors.length} row(s) skipped: ${res.errors.join("; ")}` : "."}`);
    reload();
  };

  return (
    <Page
      title="Case queue"
      subtitle="Why each student was out. Ordered by risk of harm, not by dollar value."
      actions={
        <>
          <input ref={fileRef} type="file" accept=".csv" hidden onChange={(e) => e.target.files?.[0] && onImport(e.target.files[0])} />
          {canSend && (
            <button
              onClick={onSendCheckIns}
              disabled={sending || toSend === 0}
              title={toSend ? "Text each family whose absence is awaiting a reply and has not had a check-in" : "No families need a check-in right now"}
            >
              {sending ? "Sending…" : `Send check-ins${toSend ? ` (${toSend})` : ""}`}
            </button>
          )}
          <button className="ghost" onClick={() => fileRef.current?.click()} title="CSV columns: student_id, date, sis_code. Add first_name, last_name, grade, classroom, teacher, bus_route, household_id, preferred_language, guardian_name, guardian_phone to add or update students.">
            Import SIS CSV
          </button>
        </>
      }
    >
      <ErrorNote error={error} />
      {importMsg && <p className="note">{importMsg}</p>}
      {sendMsg && <p className="note">{sendMsg}</p>}
      <div className="stats">
        <Stat label="Absences" value={data?.length ?? "–"} />
        <Stat label="Reason captured" value={count("reason captured")} />
        <Stat label="Awaiting reply" value={count("awaiting reply")} />
        <Stat label="Unknown" value={count("unknown")} hint="Needs a follow-up call" />
      </div>
      <div className="row filters">
        {FILTERS.map((f) => (
          <button key={f} className={f === filter ? "chip active" : "chip"} onClick={() => setFilter(f)}>
            {f}
          </button>
        ))}
      </div>
      {held && (
        <p className="note">
          A case just changed. Rows are holding their place so you can follow it.{" "}
          <button className="link" onClick={resort}>
            Re-sort by priority
          </button>
        </p>
      )}
      <table>
        <thead>
          <tr>
            <th>Student</th>
            <th>Date</th>
            <th>SIS code</th>
            <th>Status</th>
            <th>Barrier</th>
            <th>Family said</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className={[r.escalated && "row-alert", changed.has(r.id) && "flash"].filter(Boolean).join(" ") || undefined}>
              <td>
                <Link to={`/cases/${r.id}`}>
                  {r.student.first_name} {r.student.last_name}
                </Link>
                <div className="muted small">
                  Gr {r.student.grade} · {r.student.classroom} · {r.absence_count} absence(s)
                </div>
              </td>
              <td>{fmtDate(r.date)}</td>
              <td className="muted">{r.sis_code}</td>
              <td>
                <StatusTag status={r.status} />
                {r.escalated && <span className="tag alert">counselor</span>}
              </td>
              <td>{r.classification && <BarrierTag barrier={r.classification.barrier} />}</td>
              <td className="quote">
                {r.reply ? (
                  <>
                    {r.reply.text}
                  </>
                ) : (
                  <span className="muted">—</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Page>
  );
}
