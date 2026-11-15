import { useState } from "react";
import type { FormEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { DISTRICTS, ROLES, canOpen, useAuth } from "../auth";
import type { RoleId } from "../auth";

function Logo() {
  return (
    <svg className="brand-mark login-logo" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" />
      <path d="M9 16.5l4.5 4.5L23 11" />
    </svg>
  );
}

/** Step 1: find your district. Only the demo district can be chosen. */
function DistrictStep() {
  const { chooseDistrict } = useAuth();
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const matches = DISTRICTS.filter((d) => !q || d.name.toLowerCase().includes(q) || d.where.toLowerCase().includes(q));

  return (
    <div className="login-box">
      <Logo />
      <h1>RollCall</h1>
      <p className="muted">Find your school district to sign in</p>
      <label>
        School district
        <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by district or state" autoFocus />
      </label>
      <ul className="district-list" aria-label="School districts">
        {matches.map((d) => (
          <li key={d.id}>
            <button
              type="button"
              className="district-option"
              disabled={!d.available}
              onClick={() => chooseDistrict(d.id)}
              title={d.available ? undefined : "Not available in the demo"}
            >
              <span className="district-name">{d.name}</span>
              <span className="district-where">{d.available ? d.where : `${d.where} · not available in the demo`}</span>
            </button>
          </li>
        ))}
        {matches.length === 0 && <li className="district-none">No district matches “{query.trim()}”.</li>}
      </ul>
      <p className="fine">Demo sign-in. Synthetic data only, no real student records.</p>
    </div>
  );
}

/** Step 2: choose who to sign in as within the district. */
function RoleStep() {
  const { district, chooseDistrict, signIn } = useAuth();
  const nav = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from;
  const [choice, setChoice] = useState<RoleId | "">("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!choice) return;
    const role = signIn(choice);
    // Go back to the page that asked for sign-in when this role can open it.
    nav(from && from !== "/login" && canOpen(role, from) ? from : role.home, { replace: true });
  };

  return (
    <form className="login-box" onSubmit={submit}>
      <Logo />
      <h1>{district!.name}</h1>
      <p className="muted">
        Sign in to RollCall ·{" "}
        <button type="button" className="link" onClick={() => chooseDistrict(null)}>
          Change district
        </button>
      </p>
      <label>
        Sign in as
        <select value={choice} onChange={(e) => setChoice(e.target.value as RoleId)} required autoFocus>
          <option value="" disabled>
            Choose a user…
          </option>
          {ROLES.map((r) => (
            <option key={r.id} value={r.id}>
              {r.person} · {r.label}
            </option>
          ))}
        </select>
      </label>
      <button type="submit" className="login-submit" disabled={!choice}>
        Sign in
      </button>
      <p className="fine">Demo sign-in. Synthetic data only, no real student records.</p>
    </form>
  );
}

export default function Login() {
  const { district } = useAuth();
  return <div className="login">{district ? <RoleStep /> : <DistrictStep />}</div>;
}
