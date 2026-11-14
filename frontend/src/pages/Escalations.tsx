import { Link } from "react-router-dom";
import { api } from "../api";
import { useLoad } from "../hooks";
import { ErrorNote, Page } from "../components/ui";

export default function Escalations() {
  const { data, error } = useLoad(api.escalations);
  return (
    <Page title="Escalations" subtitle="Safety and health concerns. A counselor takes over; nothing is sent automatically.">
      <ErrorNote error={error} />
      {data?.length === 0 && <p className="muted">No open escalations.</p>}
      <div className="cards">
        {data?.map((e) => (
          <div key={e.absence_id} className="card alert-card">
            <div className="row between">
              <span className="tag alert">{e.reason}</span>
              <span className="muted small">
                {e.assigned_to} · {e.status}
              </span>
            </div>
            <h3>
              <Link to={`/cases/${e.absence_id}`}>
                {e.student.first_name} {e.student.last_name}
              </Link>
            </h3>
            <p className="muted small">
              Grade {e.student.grade} · {e.student.classroom} · guardian {e.student.guardian_name} ({e.student.guardian_phone})
            </p>
            <blockquote>{e.evidence}</blockquote>
          </div>
        ))}
      </div>
    </Page>
  );
}
