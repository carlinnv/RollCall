import { useEffect, useState } from "react";

type Row = { day: string; key: string; absences: number };

/** Daily absences per bus route, read from the TigerData continuous aggregate. Hidden when the sidecar is off. */
export default function Timeline({ group = "bus_route" }: { group?: string }) {
  const [rows, setRows] = useState<Row[] | null>(null);

  useEffect(() => {
    fetch(`/api/timeline?group=${group}`)
      .then((r) => r.json())
      .then((d) => setRows(d.enabled ? d.rows : []))
      .catch(() => setRows([]));
  }, [group]);

  if (!rows || rows.length === 0) return null;

  const days = [...new Set(rows.map((r) => r.day))].sort();
  const totals = new Map<string, number>();
  rows.forEach((r) => totals.set(r.key, (totals.get(r.key) ?? 0) + r.absences));
  const keys = [...totals.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([k]) => k);
  const max = Math.max(...rows.map((r) => r.absences), 1);
  const count = (k: string, d: string) => rows.find((r) => r.key === k && r.day === d)?.absences ?? 0;

  return (
    <div className="card">
      <div className="row between">
        <h3>Daily absences by {group.replace("_", " ")}</h3>
        <span className="muted small">TigerData hypertable · continuous aggregate</span>
      </div>
      <table>
        <thead>
          <tr>
            <th />
            {days.map((d) => (
              <th key={d}>{new Date(d + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" })}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k}>
              <td>{k}</td>
              {days.map((d) => {
                const n = count(k, d);
                return (
                  <td key={d} style={{ background: n ? `rgba(47, 91, 234, ${0.15 + 0.85 * (n / max)})` : undefined, color: n / max > 0.5 ? "#fff" : undefined, textAlign: "center" }}>
                    {n || ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
