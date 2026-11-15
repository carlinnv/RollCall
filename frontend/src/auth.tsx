// Demo sign-in: pick a role, no password. The choice is kept in this browser only.
// This hides pages in the UI; it does not protect the API. Real sign-in needs server-side checks.
import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";

export type RoleId = "clerk" | "counselor" | "nurse" | "teacher" | "business" | "admin";

export interface Role {
  id: RoleId;
  label: string;
  /** Demo person signed in under this role. */
  person: string;
  /** Page opened right after sign-in. */
  home: string;
  /** Paths this role can open. "/" means the Overview only; others also cover sub-paths. */
  pages: string[];
}

export interface District {
  id: string;
  name: string;
  where: string;
  /** Only one district has data in the demo; the rest are shown but cannot be chosen. */
  available: boolean;
}

// Fictional districts. Only Lincoln Unified has demo data.
export const DISTRICTS: District[] = [
  { id: "lincoln", name: "Lincoln Unified School District", where: "California · 1 school in demo", available: true },
  { id: "cedar-hollow", name: "Cedar Hollow Unified", where: "California", available: false },
  { id: "brightwater", name: "Brightwater School District", where: "Oregon", available: false },
  { id: "silverpine", name: "Silverpine Unified", where: "Arizona", available: false },
  { id: "oak-hollow", name: "Oak Hollow Public Schools", where: "Nevada", available: false },
];

/** Pages hidden from everyone for now. Remove a path here to bring the page back. */
export const HIDDEN_PAGES = ["/assistant", "/revenue", "/evidence"];

const isHidden = (path: string) => HIDDEN_PAGES.some((p) => path === p || path.startsWith(`${p}/`));

const ALL_ROLES: Role[] = [
  {
    id: "clerk",
    label: "Attendance clerk",
    person: "Dana Ortiz",
    home: "/",
    pages: ["/", "/queue", "/cases", "/clusters", "/plans", "/evidence"],
  },
  {
    id: "counselor",
    label: "Counselor",
    person: "Sam Rivera",
    home: "/escalations",
    pages: ["/", "/queue", "/cases", "/clusters", "/escalations", "/oncall", "/assistant", "/evidence"],
  },
  {
    id: "nurse",
    label: "School nurse",
    person: "Priya Shah",
    home: "/clusters",
    pages: ["/", "/queue", "/cases", "/clusters", "/escalations", "/evidence"],
  },
  {
    id: "teacher",
    label: "Teacher",
    person: "Mr. Chen",
    home: "/learning",
    pages: ["/learning", "/evidence"],
  },
  {
    id: "business",
    label: "Business manager",
    person: "Jordan Lee",
    home: "/revenue",
    pages: ["/revenue", "/evidence"],
  },
  {
    id: "admin",
    label: "Administrator",
    person: "Principal Morgan",
    home: "/",
    pages: ["*"],
  },
];

export function canOpen(role: Role, path: string) {
  if (isHidden(path)) return false;
  return role.pages.some((p) => p === "*" || (p === "/" ? path === "/" : path === p || path.startsWith(`${p}/`)));
}

/** Roles offered at sign-in: only those left with a home page they can open. */
export const ROLES: Role[] = ALL_ROLES.filter((r) => canOpen(r, r.home));

const ROLE_KEY = "rollcall.role";
const DISTRICT_KEY = "rollcall.district";

// Browser storage can be blocked; then the choice lasts for this tab only.
function load(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function save(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

interface Auth {
  district: District | null;
  role: Role | null;
  chooseDistrict: (id: string | null) => void;
  signIn: (id: RoleId) => Role;
  signOut: () => void;
}

const AuthContext = createContext<Auth | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [district, setDistrict] = useState<District | null>(() => DISTRICTS.find((d) => d.available && d.id === load(DISTRICT_KEY)) ?? null);
  // A role only counts once a district is chosen.
  const [role, setRole] = useState<Role | null>(() => (district ? ROLES.find((r) => r.id === load(ROLE_KEY)) ?? null : null));

  /** Pick a district, or pass null to go back to the district list (also signs out). */
  const chooseDistrict = useCallback((id: string | null) => {
    const next = DISTRICTS.find((d) => d.available && d.id === id) ?? null;
    save(DISTRICT_KEY, next?.id ?? null);
    setDistrict(next);
    if (!next) {
      save(ROLE_KEY, null);
      setRole(null);
    }
  }, []);

  const signIn = useCallback((id: RoleId) => {
    const next = ROLES.find((r) => r.id === id)!;
    save(ROLE_KEY, id);
    setRole(next);
    return next;
  }, []);

  /** Sign out but keep the district, like a school portal does. */
  const signOut = useCallback(() => {
    save(ROLE_KEY, null);
    setRole(null);
  }, []);

  return <AuthContext.Provider value={{ district, role, chooseDistrict, signIn, signOut }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const auth = useContext(AuthContext);
  if (!auth) throw new Error("useAuth must be used inside AuthProvider");
  return auth;
}
