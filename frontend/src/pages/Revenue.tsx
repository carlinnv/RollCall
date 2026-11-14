import { useEffect, useState } from "react";
import { api, fmtDate, money, type RevenueSettings, type RevenueSummary } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page, Stat } from "../components/ui";

export default function Revenue() {
  const { data: loaded, error } = useLoad(api.revenue);
  const [data, setData] = useState<RevenueSummary | null>(null);
  const [settings, setSettings] = useState<RevenueSettings | null>(null);

  useEffect(() => {
    if (loaded) {
      setData(loaded);
      setSettings(loaded.settings);
    }
  }, [loaded]);

  const save = async (next: RevenueSettings) => {
    setSettings(next);
    if (next.daily_rate > 0) setData(await api.saveSettings(next));
  };

  if (!data || !settings) return <Page title="Revenue"><ErrorNote error={error} /></Page>;
  const max = Math.max(...data.by_root_cause.map((r) => r.dollars), 1);
  const top = data.by_root_cause.find((r) => r.barrier !== "unknown");

  return (
    <Page title="Revenue" subtitle={`Week of ${fmtDate(data.week_start)} – ${fmtDate(data.week_end)}. Dollars justify the program; risk of harm decides who gets help first.`}>
      <div className="stats">
        <Stat label="Revenue at risk this week" value={money(data.dollars_at_risk_week)} hint={`${data.absences_this_week} absence days`} />
        <Stat label="Recovered to date" value={money(data.dollars_recovered_to_date)} hint={`${data.days_recovered_to_date} days via Attendance Recovery`} />
        <Stat label="Top root cause" value={top?.barrier ?? "–"} hint={top ? `${money(top.dollars)} this week` : undefined} />
      </div>

      <div className="grid2">
        <div className="card">
          <h3>Root causes by dollar impact</h3>
          {data.by_root_cause.map((r) => (
            <div key={r.barrier} className="bar-row">
              <span>
                <BarrierTag barrier={r.barrier} /> <span className="muted small">{r.absences} day(s){r.barrier === "unknown" && " · no reason yet"}</span>
              </span>
              <span className="money">{money(r.dollars)}</span>
              <div className="bar">
                <div style={{ width: `${(r.dollars / max) * 100}%` }} />
              </div>
            </div>
          ))}
        </div>

        <div className="card">
          <h3>Assumptions</h3>
          <label>
            State
            <select value={settings.state} onChange={(e) => save({ ...settings, state: e.target.value })}>
              <option value="CA">California (attendance-funded)</option>
            </select>
          </label>
          <label>
            Daily funding rate per student ($)
            <input
              type="number"
              min={1}
              step={1}
              value={settings.daily_rate}
              onChange={(e) => save({ ...settings, daily_rate: Number(e.target.value) })}
            />
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={settings.recovery_enabled}
              onChange={(e) => save({ ...settings, recovery_enabled: e.target.checked })}
            />
            Count Attendance Recovery make-up days as recovered
          </label>
          <p className="muted small">Default $65/day is the midpoint of a $35–$100 vendor estimate (unverified). Use your district's actual rate.</p>
        </div>
      </div>
    </Page>
  );
}
