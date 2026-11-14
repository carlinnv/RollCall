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
  // keyword | gemini | correction_override | staff_correction
  source?: string;
  correction_ref?: string | null;
  correction_similarity?: number | null;
}

export interface Correction {
  absence_id: string;
  old_label: Barrier | null;
  new_label: Barrier;
  corrected_by: string;
  timestamp: string;
  school_id: string;
  text: string;
  language: string;
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

export interface AbsenceDetail extends Absence {
  student: Student;
  check_in_message: string;
  escalated: boolean;
}

export interface Cluster {
  id: string;
  barrier: Barrier;
  link_type: "bus_route" | "classroom" | "household" | "grade" | "neighborhood" | "contact" | "school";
  link_value: string;
  student_ids: string[];
  absence_ids: string[];
  first_date: string;
  last_date: string;
  dollars_at_risk_week: number;
  ablation: { text_only: number; network_only: number; combined: number };
  summary: string;
  // PRD 2 fields
  layers_involved: string[];
  zone: string | null;
  member_ids: string[];
  likely_cause: Barrier | null;
  cause_provisional: boolean;
  observed: number | null;
  expected: number | null;
  p_value: number | null;
  start_date: string | null;
  detected_on: string | null;
  weekly_revenue_at_risk: number | null;
  evidence_message_ids: string[];
  method: string; // "rule_based" | "scan:<detector>" | "scan:<detector>+synthetic_baseline"
  method_note: string | null;
  dataset: "demo" | "sim";
}

export interface ClusterMeta {
  dataset: "demo" | "sim";
  method: string;
  detector: string;
  notes: string[];
  counts: Record<string, number>;
  version?: number;
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
  plan_id: string;
  to: string;
  student_id: string;
  text: string;
  sent_at: string;
}

export interface ClaimFlag {
  where: string;
  claim: string;
  reason: string;
  rule: string;
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
  eligible: boolean;
  eligibility_note: string;
  reteach_suggestion: string;
  take_home: string;
  attendance_recovery_offer: string;
  home_language: string;
  verification_status: "verified" | "flagged" | "unverified";
  verification_method: string;
  flags: ClaimFlag[];
}

export interface RecoveryResult {
  days_recovered: number;
  counted: boolean;
  reason: string;
  total_days_recovered_for_student: number;
}

export interface Escalation {
  absence_id: string;
  student_id: string;
  reason: string;
  evidence: string;
  assigned_to: string;
  status: string;
  student: Student;
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
  clusters: () => req<Cluster[]>("/clusters"),
  clusterMeta: () => req<ClusterMeta>("/clusters/meta"),
  correct: (id: string, label: Barrier, corrected_by: string) =>
    post<{ absence: Absence; correction: Correction; escalated: boolean }>(`/absences/${id}/correct`, { label, corrected_by }),
  corrections: () => req<Correction[]>("/corrections"),
  taxonomy: () => req<Barrier[]>("/taxonomy"),
  plans: () => req<ActionPlan[]>("/plans"),
  draftPlan: (target: { cluster_id?: string; absence_id?: string }) => post<ActionPlan>("/plans/draft", target),
  updatePlan: (id: string, fields: Partial<ActionPlan>) =>
    req<ActionPlan>(`/plans/${id}`, { method: "PATCH", body: JSON.stringify(fields) }),
  approvePlan: (id: string, approved_by: string) => post<ActionPlan>(`/plans/${id}/approve`, { approved_by, send: true }),
  outbox: () => req<OutboxMessage[]>("/outbox"),
  learningPlan: (studentId: string) => req<LearningPlan>(`/learning-plans/${studentId}`),
  completeLearningPlan: (studentId: string, supervised_by: string, session_type = "attendance_recovery") =>
    post<RecoveryResult>(`/learning-plans/${studentId}/complete`, { supervised_by, session_type }),
  revenue: () => req<RevenueSummary>("/revenue"),
  saveSettings: (s: RevenueSettings) => req<RevenueSummary>("/revenue/settings", { method: "PUT", body: JSON.stringify(s) }),
  reset: () => post<{ ok: boolean }>("/reset"),
};

export const money = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const fmtDate = (iso: string) =>
  new Date(iso + "T12:00:00").toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
