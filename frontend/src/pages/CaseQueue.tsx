import { useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, fmtDate, type AbsenceStatus } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page, Stat, StatusTag } from "../components/ui";

const FILTERS: (AbsenceStatus | "all")[] = ["all", "awaiting reply", "unknown", "reason captured"];

export default function CaseQueue() {
  const { data, error, reload } = useLoad(api.queue);
  const [filter, setFilter] = useState<AbsenceStatus | "all">("all");
  const [importMsg, setImportMsg] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const rows = useMemo(() => (data ?? []).filter((r) => filter === "all" || r.status === filter), [data, filter]);
  const count = (s: AbsenceStatus) => (data ?? []).filter((r) => r.status === s).length;

  const onImport = async (file: File) => {
    const res = await api.importCsv(await file.text());
    setImportMsg(`Imported ${res.added} absence(s)${res.errors.length ? `; ${res.errors.length} row(s) skipped: ${res.errors.join("; ")}` : "."}`);
    reload();
  };

  return (
    <Page
      title="Case queue"
      subtitle="Why each student was out. Ordered by risk of harm, not by dollar value."
      actions={
        <>
          <input ref={fileRef} type="file" accept=".csv" hidden onChange={(e) => e.target.files?.[0] && onImport(e.target.files[0])} />
          <button className="ghost" onClick={() => fileRef.current?.click()} title="CSV columns: student_id,date,sis_code">
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
            <tr key={r.id} className={r.escalated ? "row-alert" : undefined}>
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
                    {r.reply.channel === "voice" && <span className="voice">voice</span>} {r.reply.text}
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
