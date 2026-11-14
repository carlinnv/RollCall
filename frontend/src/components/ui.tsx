import type { ReactNode } from "react";
import type { AbsenceStatus, Barrier } from "../api";

export function BarrierTag({ barrier }: { barrier: Barrier | null | undefined }) {
  const b = barrier ?? "unknown";
  return <span className={`tag barrier-${b.split(" ")[0]}`}>{b}</span>;
}

export function StatusTag({ status }: { status: AbsenceStatus }) {
  return <span className={`status status-${status.replace(" ", "-")}`}>{status}</span>;
}

export function Page({ title, subtitle, actions, children }: { title: string; subtitle?: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="page">
      <header className="page-head">
        <div>
          <h1>{title}</h1>
          {subtitle && <p className="muted">{subtitle}</p>}
        </div>
        {actions && <div className="row">{actions}</div>}
      </header>
      {children}
    </section>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p className="error">
      {error}. Is the API running on port 8000?
    </p>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}
