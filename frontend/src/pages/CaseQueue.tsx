import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtDate, type AbsenceStatus, type QueueRow } from "../api";
import { useChanged, useLoad, useStableOrder } from "../hooks";
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
          <button className="ghost" onClick={() => fileRef.current?.click()} title="CSV columns: student_id, date, sis_code. Add first_name, last_name, grade, classroom, teacher, bus_route, household_id, preferred_language, guardian_name, guardian_phone to add or update students.">
            Import SIS CSV
          </button>
        </>
      }
    >
      <ErrorNote error={error} />
      {importMsg && <p className="note">{importMsg}</p>}
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
