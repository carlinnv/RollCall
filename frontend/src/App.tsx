import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { api } from "./api";
import { canOpen, useAuth } from "./auth";
import type { Role } from "./auth";
import Login from "./pages/Login";
import Overview from "./pages/Overview";
import CaseQueue from "./pages/CaseQueue";
import CaseDetail from "./pages/CaseDetail";
import Clusters from "./pages/Clusters";
import Escalations from "./pages/Escalations";
import Plans from "./pages/Plans";
import LearningPlanPage from "./pages/LearningPlan";
import Revenue from "./pages/Revenue";
import OnCall from "./pages/OnCall";
import Evidence from "./pages/Evidence";
import Assistant from "./pages/Assistant";
import DemoBar from "./components/DemoBar";

type IconName = "home" | "queue" | "clusters" | "plans" | "alert" | "bell" | "chat" | "book" | "bars" | "check";

const NAV: ({ group: string } | { to: string; label: string; who: string; icon: IconName })[] = [
  { group: "Daily work" },
  { to: "/", label: "Overview", who: "Everyone", icon: "home" },
  { to: "/queue", label: "Case queue", who: "Clerk", icon: "queue" },
  { to: "/clusters", label: "Root causes", who: "Clerk · Counselor", icon: "clusters" },
  { to: "/plans", label: "Action plans", who: "Clerk", icon: "plans" },
  { group: "Safety and support" },
  { to: "/escalations", label: "Escalations", who: "Counselor · Nurse", icon: "alert" },
  { to: "/oncall", label: "On-call", who: "Counselor · Admin", icon: "bell" },
  { to: "/assistant", label: "Assistant", who: "Counselor", icon: "chat" },
  { group: "Outcomes" },
  { to: "/learning", label: "Learning plans", who: "Teacher · Student", icon: "book" },
  { to: "/revenue", label: "Revenue", who: "Business manager", icon: "bars" },
  { to: "/evidence", label: "Evidence", who: "How we know it works", icon: "check" },
];

const ICONS: Record<IconName, ReactNode> = {
  home: (
    <>
      <rect x="4" y="4" width="7" height="9" rx="1.8" />
      <rect x="13" y="4" width="7" height="5" rx="1.8" />
      <rect x="13" y="11" width="7" height="9" rx="1.8" />
      <rect x="4" y="15" width="7" height="5" rx="1.8" />
    </>
  ),
  queue: <path d="M4 6h16M4 12h16M4 18h10" />,
  clusters: (
    <>
      <circle cx="6" cy="7" r="2.4" />
      <circle cx="18" cy="7" r="2.4" />
      <circle cx="12" cy="18" r="2.4" />
      <path d="M8.2 7h7.6M7.2 9.1l3.6 6.8M16.8 9.1l-3.6 6.8" />
    </>
  ),
  plans: (
    <>
      <rect x="5" y="4" width="14" height="17" rx="2.5" />
      <path d="M9 4.5h6M9 13l2.2 2.2L15.5 11" />
    </>
  ),
  alert: <path d="M12 4l9 15.5H3L12 4zM12 10v4.5M12 17.2v.1" />,
  bell: <path d="M6 16v-5a6 6 0 1112 0v5l1.5 2.2h-15L6 16zM10 20.5a2 2 0 004 0" />,
  chat: <path d="M4 5.5h16v11H10l-5 3.5v-3.5H4v-11z" />,
  book: <path d="M5 5.5A2.5 2.5 0 017.5 3H19v15H7.5A2.5 2.5 0 005 20.5v-15zM5 20.5A2.5 2.5 0 007.5 23H19M9 8h6" />,
  bars: <path d="M5 20V11M12 20V4M19 20v-6" />,
  check: (
    <>
      <circle cx="12" cy="12" r="8.5" />
      <path d="M8.2 12.3l2.6 2.6 5-5.6" />
    </>
  ),
};

function Icon({ name }: { name: IconName }) {
  return (
    <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}

/** Which classifier the API is running, for the pill in the top bar. */
function useEngine() {
  const [engine, setEngine] = useState<{ label: string; live: boolean } | null>(null);
  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((h) => {
        if (h.demo_mode) setEngine({ label: "Demo mode · recorded answers", live: false });
        else if (h.classifier?.engine === "gemini") setEngine({ label: "Gemini live", live: true });
        else setEngine({ label: "Keyword rules", live: false });
      })
      .catch(() => setEngine({ label: "API offline", live: false }));
  }, []);
  return engine;
}

export default function App() {
  const { role } = useAuth();
  const location = useLocation();
  if (location.pathname === "/login") return <Login />;
  if (!role) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return <Shell role={role} />;
}

function NoAccess({ role }: { role: Role }) {
  return (
    <section className="page">
      <header className="page-head">
        <div>
          <h1>Not available for your role</h1>
          <p className="muted">
            Signed in as {role.label}. This page is for other staff.
          </p>
        </div>
      </header>
      <div className="row">
        <Link className="button" to={role.home}>
          Go to my home page
        </Link>
        <Link className="button ghost" to="/login">
          Switch user
        </Link>
      </div>
    </section>
  );
}

function Shell({ role }: { role: Role }) {
  const engine = useEngine();
  const { signOut, district } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const reset = async () => {
    await api.reset();
    window.location.reload();
  };
  const switchUser = () => {
    signOut();
    navigate("/login");
  };
  // Only the pages this role can open, and only the group headings that still have pages under them.
  const nav = NAV.filter((n, i) => {
    if (!("group" in n)) return canOpen(role, n.to);
    for (let j = i + 1; j < NAV.length && !("group" in NAV[j]); j++) {
      const item = NAV[j];
      if (!("group" in item) && canOpen(role, item.to)) return true;
    }
    return false;
  });
  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">
          <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="9" />
            <path d="M9 16.5l4.5 4.5L23 11" />
          </svg>
          <div>
            RollCall
            <span>Find the cause. Fix it together.</span>
          </div>
        </div>
        {nav.map((n) =>
          "group" in n ? (
            <div key={n.group} className="nav-group">
              {n.group}
            </div>
          ) : (
            <NavLink key={n.to} to={n.to} end={n.to === "/"} className="nav-link">
              <Icon name={n.icon} />
              <span>
                {n.label}
                <small>{n.who}</small>
              </span>
            </NavLink>
          ),
        )}
        <div className="whoami">
          <span>
            <b>{role.person}</b>
            <small>{role.label}</small>
          </span>
          <button className="link small" onClick={switchUser}>
            Switch user
          </button>
        </div>
        {role.id === "admin" && (
          <button className="ghost" onClick={reset} title="Reload the synthetic demo data">
            Reset demo data
          </button>
        )}
        <p className="fine">Synthetic data only. No real student records.</p>
      </nav>
      <main>
        <div className="topbar">
          <div className="crumb">
            <b>Lincoln Elementary</b>
            <span>{district?.name ?? "Demo district"} · week of Nov 9–13, 2026</span>
          </div>
          <div className="row">
            {engine && (
              <span className={`pill ${engine.live ? "live" : ""}`}>
                <i />
                {engine.label}
              </span>
            )}
            <span className="pill">Synthetic data</span>
          </div>
        </div>
        {!canOpen(role, pathname) ? (
          <NoAccess role={role} />
        ) : (
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/queue" element={<CaseQueue />} />
          <Route path="/cases/:id" element={<CaseDetail />} />
          <Route path="/clusters" element={<Clusters />} />
          <Route path="/plans" element={<Plans />} />
          <Route path="/escalations" element={<Escalations />} />
          <Route path="/oncall" element={<OnCall />} />
          <Route path="/learning" element={<LearningPlanPage />} />
          <Route path="/learning/:studentId" element={<LearningPlanPage />} />
          <Route path="/revenue" element={<Revenue />} />
          <Route path="/assistant" element={<Assistant />} />
          <Route path="/evidence" element={<Evidence />} />
        </Routes>
        )}
      </main>
      <DemoBar />
    </div>
  );
}
