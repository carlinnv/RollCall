import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, fmtDate } from "../api";
import type { RecoveryResult } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";

export default function LearningPlanPage() {
  const { studentId } = useParams();
  const nav = useNavigate();
  const students = useLoad(api.students);
  const plan = useLoad(() => (studentId ? api.learningPlan(studentId) : Promise.resolve(null)), [studentId]);
  const [done, setDone] = useState<RecoveryResult | null>(null);
  const student = students.data?.find((s) => s.id === studentId);
  const p = plan.data;

  const complete = async () => {
    if (!studentId || !student) return;
    // Only a supervised Attendance Recovery session counts; the supervisor is the student's teacher.
    setDone(await api.completeLearningPlan(studentId, student.teacher));
  };

  return (
    <Page
      title="Learning plans"
      subtitle="Catch-up plans built only from the course pacing guide, adapted to why the student was out."
      actions={
        <select value={studentId ?? ""} onChange={(e) => nav(`/learning/${e.target.value}`)}>
          <option value="" disabled>
            Choose a student…
          </option>
          {students.data?.map((s) => (
            <option key={s.id} value={s.id}>
              {s.first_name} {s.last_name} (Gr {s.grade})
            </option>
          ))}
        </select>
      }
    >
      <ErrorNote error={students.error ?? plan.error} />
      {!studentId && <p className="muted">Pick a student to see their plan.</p>}
      {p && student && (
        <div className="card">
          <div className="row between">
            <h3>
              {student.first_name} {student.last_name} · {p.course}
            </h3>
            <BarrierTag barrier={p.barrier} />
          </div>
          <p className="muted">
            {p.missed_days} missed lesson(s) · {p.total_minutes} minutes total · {p.load_note}
          </p>
          {p.verification_status === "verified" ? (
            <p className="note">
              Verified ({p.verification_method}): every topic, date and source line is in the pacing guide.
            </p>
          ) : (
            <div className="alert-box">
              <b>
                {p.verification_status === "flagged" ? `${p.flags.length} unsupported claim(s)` : "Not verified"}
              </b>
              <ul>
                {p.flags.map((f, i) => (
                  <li key={i}>
                    <code>{f.where}</code>: “{f.claim}” — {f.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {!p.eligible && <p className="note">{p.eligibility_note}</p>}
          {p.steps.length === 0 ? (
            <p className="muted">No missed lessons in the pacing guide window.</p>
          ) : (
            <ol className="steps">
              {p.steps.map((s) => (
                <li key={s.order}>
                  <b>{s.topic}</b> <span className="muted small">(missed {fmtDate(s.missed_date)}, {s.minutes} min)</span>
                  <div>{s.activity}</div>
                  <div className="source">Source: {s.source}</div>
                </li>
              ))}
            </ol>
          )}
          {p.eligible && (
            <ul>
              {p.reteach_suggestion && (
                <li>
                  <b>Small-group reteach:</b> {p.reteach_suggestion}
                </li>
              )}
              {p.take_home && (
                <li>
                  <b>Take-home:</b> {p.take_home}
                </li>
              )}
              {p.attendance_recovery_offer && (
                <li>
                  <b>Attendance Recovery:</b> {p.attendance_recovery_offer}
                </li>
              )}
            </ul>
          )}
          {p.teacher_checkins.length > 0 && (
            <p>
              <b>Teacher check-ins:</b> {p.teacher_checkins.map(fmtDate).join(", ")}
            </p>
          )}
          {p.steps.length > 0 &&
            (done === null ? (
              <button onClick={complete} disabled={!p.eligible}>
                Record supervised Attendance Recovery session{student ? ` (${student.teacher})` : ""}
              </button>
            ) : (
              <p className="note">
                {done.counted ? `${done.days_recovered} day(s) counted toward Attendance Recovery.` : `Not counted: ${done.reason}.`}
              </p>
            ))}
        </div>
      )}
    </Page>
  );
}
