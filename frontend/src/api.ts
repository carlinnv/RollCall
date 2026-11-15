// Typed client for the Why Absent FastAPI backend (proxied at /api by Vite).

export type Barrier =
  | "illness"
  | "transportation"
  | "housing or food insecurity"
  | "caregiving or family obligation"
  | "safety or bullying"
  | "disengagement or academic struggle"
  | "appointment or travel"
  | "unknown";

export type AbsenceStatus = "reason captured" | "awaiting reply" | "unknown";
export type PlanStatus = "draft" | "approved" | "sent" | "completed";

export interface Student {
  id: string;
  first_name: string;
  last_name: string;
  grade: number;
  classroom: string;
  teacher: string;
  bus_route: string | null;
  household_id: string;
  preferred_language: string;
  guardian_name: string;
  guardian_phone: string;
}

export interface FamilyReply {
  channel: "text" | "voice";
  text: string;
  language: string;
  received_at: string;
}

export interface Classification {
  barrier: Barrier;
  confidence: number;
  evidence: string;
  safety_critical: boolean;
  source: "gemini" | "keywords" | "keywords_fallback" | "safety_keywords" | "counselor";
  language: string | null;
}

export interface Absence {
  id: string;
  student_id: string;
  date: string;
  sis_code: string;
  status: AbsenceStatus;
  reply: FamilyReply | null;
  classification: Classification | null;
}

export interface QueueRow extends Absence {
  student: Student;
  absence_count: number;
  escalated: boolean;
}

export interface Message {
  message_id: string;
  absence_id: string | null;
  student_id: string | null;
  direction: "inbound" | "outbound";
  channel: "imessage" | "sms" | "simulated" | "seed" | null;
  language: string;
  raw_text_or_transcript: string;
  timestamp: string;
  latency_ms: number | null;
  kind: "reply" | "check_in" | "plan" | "ack";
  status: "received" | "sent" | "queued" | "blocked" | "failed";
  reason: string | null;
  route: string | null;
  classification: Classification | null;
}

export interface AbsenceDetail extends Absence {
  student: Student;
  check_in_message: string;
  escalated: boolean;
  messages: Message[];
}

export interface Cluster {
  id: string;
  barrier: Barrier;
  link_type: "bus_route" | "classroom" | "household";
  link_value: string;
  student_ids: string[];
  absence_ids: string[];
  first_date: string;
  last_date: string;
  dollars_at_risk_week: number;
  ablation: { text_only: number; network_only: number; combined: number };
  summary: string;
}

export interface ActionPlan {
  id: string;
  scope: "cluster" | "case";
  cluster_id: string | null;
  absence_id: string | null;
  barrier: Barrier;
  school_action: string;
  owner: string;
  target_date: string;
  family_request: string;
  message_language: string;
  message: string;
  recipient_student_ids: string[];
  status: PlanStatus;
  approved_by: string | null;
}

export interface OutboxMessage {
  kind: "plan" | "check_in" | "ack";
  plan_id?: string;
  absence_id?: string;
  to: string;
  student_id: string;
  text: string;
  sent_at: string;
  status: "sent" | "queued" | "blocked" | "failed";
  reason: string | null;
  channel: string | null;
}

export interface LearningPlan {
  student_id: string;
  course: string;
  barrier: Barrier;
  missed_days: number;
  load_note: string;
  steps: { order: number; topic: string; missed_date: string; activity: string; minutes: number; source: string }[];
  teacher_checkins: string[];
  total_minutes: number;
}

export interface Escalation {
  escalation_id: string;
  absence_id: string;
  student_id: string;
  household_id: string;
  reason: string;
  evidence: string;
  assigned_to: string;
  status: "open" | "acknowledged" | "closed";
  opened_at: string;
  acknowledged_at: string | null;
  acknowledged_by: string | null;
  report_status: "report_filed" | "not_required" | null;
  report_by: string | null;
  report_at: string | null;
  closed_at: string | null;
  closed_by: string | null;
  outcome: "resolved" | "not_safety" | null;
  relabeled_to: Barrier | null;
  note: string | null;
  minutes_open: number;
  overdue: boolean;
  case_report_versions: number;
  case_report_saved_at: string | null;
  student: Student;
}

export interface CaseReport {
  escalation_id: string;
  version: number;
  text: string;
  saved_by: string;
  saved_at: string;
}

export interface RevenueSettings {
  state: string;
  daily_rate: number;
  recovery_enabled: boolean;
}

export interface RevenueSummary {
  settings: RevenueSettings;
  week_start: string;
  week_end: string;
  absences_this_week: number;
  dollars_at_risk_week: number;
  days_recovered_to_date: number;
  dollars_recovered_to_date: number;
  by_root_cause: { barrier: Barrier; absences: number; dollars: number }[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

const post = <T,>(path: string, body?: unknown) =>
  req<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  queue: () => req<QueueRow[]>("/absences"),
  absence: (id: string) => req<AbsenceDetail>(`/absences/${id}`),
  students: () => req<Student[]>("/students"),
  importCsv: (csv: string) => post<{ added: number; errors: string[] }>("/absences/import", { csv }),
  reply: (id: string, text: string, channel: "text" | "voice", language: string) =>
    post<{ route: "escalated" | "cluster" | "case" | "unknown"; absence: Absence }>(`/absences/${id}/reply`, {
      text,
      channel,
      language,
    }),
  escalations: () => req<Escalation[]>("/escalations"),
  acknowledgeEscalation: (id: string, by: string) => post<Escalation>(`/escalations/${id}/acknowledge`, { by }),
  reportEscalation: (id: string, by: string, decision: "report_filed" | "not_required") =>
    post<Escalation>(`/escalations/${id}/report`, { by, decision }),
  draftCaseReport: (id: string, by: string) => post<{ text: string }>(`/escalations/${id}/case-report/draft`, { by }),
  caseReports: (id: string) => req<CaseReport[]>(`/escalations/${id}/case-reports`),
  saveCaseReport: (id: string, by: string, text: string) => post<CaseReport>(`/escalations/${id}/case-reports`, { by, text }),
  closeEscalation: (id: string, by: string, outcome: "resolved" | "not_safety", relabel_to?: Barrier, note?: string) =>
    post<Escalation>(`/escalations/${id}/close`, { by, outcome, relabel_to, note }),
  clusters: () => req<Cluster[]>("/clusters"),
  plans: () => req<ActionPlan[]>("/plans"),
  draftPlan: (target: { cluster_id?: string; absence_id?: string }) => post<ActionPlan>("/plans/draft", target),
  updatePlan: (id: string, fields: Partial<ActionPlan>) =>
    req<ActionPlan>(`/plans/${id}`, { method: "PATCH", body: JSON.stringify(fields) }),
  approvePlan: (id: string, approved_by: string) => post<ActionPlan>(`/plans/${id}/approve`, { approved_by, send: true }),
  outbox: () => req<OutboxMessage[]>("/outbox"),
  learningPlan: (studentId: string) => req<LearningPlan>(`/learning-plans/${studentId}`),
  completeLearningPlan: (studentId: string) => post<{ days_recovered: number }>(`/learning-plans/${studentId}/complete`),
  revenue: () => req<RevenueSummary>("/revenue"),
  saveSettings: (s: RevenueSettings) => req<RevenueSummary>("/revenue/settings", { method: "PUT", body: JSON.stringify(s) }),
  reset: () => post<{ ok: boolean }>("/reset"),
};

export const money = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const fmtDate = (iso: string) =>
  new Date(iso + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
