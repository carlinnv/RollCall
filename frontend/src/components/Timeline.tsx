import { useEffect, useState } from "react";
import { fmtDate } from "../api";
import { Heatmap, HeatScale, Panel, Tip } from "./charts";

type Row = { day: string; key: string; absences: number };

/** Most groups and days shown, so the heatmap stays readable. */
const MAX_GROUPS = 6;
const MAX_DAYS = 10;

const LABEL: Record<string, string> = { bus_route: "bus route", classroom: "classroom", barrier: "reason" };

/** Daily absences per group, read from the TigerData continuous aggregate. Hidden when the sidecar is off. */
export default function Timeline({ group = "bus_route" }: { group?: string }) {
  const [rows, setRows] = useState<Row[] | null>(null);

  useEffect(() => {
    fetch(`/api/timeline?group=${group}`)
      .then((r) => r.json())
      .then((d) => setRows(d.enabled ? d.rows : []))
      .catch(() => setRows([]));
  }, [group]);

  if (!rows || rows.length === 0) return null;

  const days = [...new Set(rows.map((r) => r.day))].sort().slice(-MAX_DAYS);
  const shown = rows.filter((r) => days.includes(r.day));
  const totals = new Map<string, number>();
  shown.forEach((r) => totals.set(r.key, (totals.get(r.key) ?? 0) + r.absences));
  const keys = [...totals.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], undefined, { numeric: true }))
    .slice(0, MAX_GROUPS)
    .map(([k]) => k);
  const count = (k: string, d: string) => shown.find((r) => r.key === k && r.day === d)?.absences ?? 0;
  const max = Math.max(3, ...shown.map((r) => r.absences));
  const what = LABEL[group] ?? group.replace("_", " ");

  const cells = keys.map((k) =>
    days.map((d) => {
      const n = count(k, d);
      return {
        value: n,
        aria: `${k}, ${fmtDate(d)}: ${n} absent`,
        tip: <Tip title={`${k} · ${fmtDate(d)}`} rows={[{ label: "Absences", value: n }]} />,
      };
    }),
  );

  return (
    <Panel
      title={`Daily absences by ${what}`}
      note={`Absences each day${keys.length === totals.size ? "" : `, for the ${MAX_GROUPS} busiest ${what}s`}. Darker means more students out.`}
    >
      <Heatmap
        ariaLabel={`Absences by ${what} and day`}
        max={max}
        totalLabel={days.length > 1 ? "Total" : "Day"}
        rows={keys.map((k) => ({ key: k, label: k }))}
        cols={days.map((d) => {
          const [weekday, date] = fmtDate(d).split(",");
          return { key: d, label: weekday, sub: date?.trim() };
        })}
        cells={cells}
        rowTotals={cells.map((r) => r.reduce((a, c) => a + c.value, 0))}
      />
      <HeatScale max={max} />
    </Panel>
  );
}
