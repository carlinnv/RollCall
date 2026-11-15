import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, fmtDate } from "../api";
import { useLoad } from "../hooks";
import { BarrierTag, ErrorNote, Page } from "../components/ui";

export default function LearningPlanPage() {
  const { studentId } = useParams();
  const nav = useNavigate();
  const students = useLoad(api.students, [], false); // the roster does not change during a session
  const plan = useLoad(() => (studentId ? api.learningPlan(studentId) : Promise.resolve(null)), [studentId]);
  const [done, setDone] = useState<number | null>(null);
  const student = students.data?.find((s) => s.id === studentId);
  const p = plan.data;

  const complete = async () => {
    if (!studentId) return;
    setDone((await api.completeLearningPlan(studentId)).days_recovered);
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
          {p.teacher_checkins.length > 0 && (
            <p>
              <b>Teacher check-ins:</b> {p.teacher_checkins.map(fmtDate).join(", ")}
            </p>
          )}
          {p.steps.length > 0 &&
            (done === null ? (
              <button onClick={complete}>Mark make-up instruction complete</button>
            ) : (
              <p className="note">{done} day(s) counted toward Attendance Recovery.</p>
            ))}
        </div>
      )}
    </Page>
  );
}
