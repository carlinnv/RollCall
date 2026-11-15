import { NavLink, Route, Routes } from "react-router-dom";
import { api } from "./api";
import CaseQueue from "./pages/CaseQueue";
import CaseDetail from "./pages/CaseDetail";
import Clusters from "./pages/Clusters";
import Escalations from "./pages/Escalations";
import Plans from "./pages/Plans";
import LearningPlanPage from "./pages/LearningPlan";
import Revenue from "./pages/Revenue";
import Evidence from "./pages/Evidence";
import Assistant from "./pages/Assistant";

const NAV = [
  { to: "/", label: "Case queue", who: "Clerk" },
  { to: "/clusters", label: "Root causes", who: "Clerk · Counselor" },
  { to: "/plans", label: "Action plans", who: "Clerk" },
  { to: "/escalations", label: "Escalations", who: "Counselor · Nurse" },
  { to: "/learning", label: "Learning plans", who: "Teacher · Student" },
  { to: "/revenue", label: "Revenue", who: "Business manager" },
  { to: "/assistant", label: "Assistant", who: "Counselor" },
  { to: "/evidence", label: "Evidence", who: "How we know it works" },
];

export default function App() {
  const reset = async () => {
    await api.reset();
    window.location.reload();
  };
  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">
          Why Absent
          <span>Find the cause. Fix it together.</span>
        </div>
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === "/"} className="nav-link">
            {n.label}
            <small>{n.who}</small>
          </NavLink>
        ))}
        <button className="ghost reset" onClick={reset} title="Reload the synthetic demo data">
          Reset demo data
        </button>
        <p className="fine">Synthetic data only. No real student records.</p>
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<CaseQueue />} />
          <Route path="/cases/:id" element={<CaseDetail />} />
          <Route path="/clusters" element={<Clusters />} />
          <Route path="/plans" element={<Plans />} />
          <Route path="/escalations" element={<Escalations />} />
          <Route path="/learning" element={<LearningPlanPage />} />
          <Route path="/learning/:studentId" element={<LearningPlanPage />} />
          <Route path="/revenue" element={<Revenue />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/evidence" element={<Evidence />} />
        </Routes>
      </main>
    </div>
  );
}
