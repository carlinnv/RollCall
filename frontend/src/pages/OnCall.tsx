import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type StaffMember, type StaffRole } from "../api";
import { useLoad } from "../hooks";
import { ErrorNote, Page } from "../components/ui";

const ROLES: StaffRole[] = ["counselor", "administrator", "principal"];

/** The phone number cell: shows the number, or an editor after "Edit". */
function PhoneCell({ s, onChange }: { s: StaffMember; onChange: () => void }) {
  const [editing, setEditing] = useState(false);
  const [phone, setPhone] = useState(s.phone);
  const [err, setErr] = useState<string | null>(null);

  const start = () => {
    setPhone(s.phone);
    setErr(null);
    setEditing(true);
  };

  const save = async () => {
    try {
      await api.updateStaff(s.staff_id, { phone: phone.trim() });
      setEditing(false);
      onChange();
    } catch (x) {
      setErr((x as Error).message); // e.g. too few digits, or already on the roster
    }
  };

  if (!editing) {
    return (
      <td>
        <span className="muted">{s.phone}</span>{" "}
        <button className="link small" onClick={start}>
          Edit
        </button>
      </td>
    );
  }
  return (
    <td>
      <div className="row">
        <input
          value={phone}
          autoFocus
          onChange={(x) => setPhone(x.target.value)}
          onKeyDown={(x) => {
            if (x.key === "Enter") save();
            if (x.key === "Escape") setEditing(false);
          }}
          placeholder="+15551234567"
        />
        <button disabled={!phone.trim() || phone.trim() === s.phone} onClick={save}>
          Save
        </button>
        <button className="ghost" onClick={() => setEditing(false)}>
          Cancel
        </button>
      </div>
      {err && <div className="error small">{err}</div>}
    </td>
  );
}

function StaffRow({ s, onChange }: { s: StaffMember; onChange: () => void }) {
  const [note, setNote] = useState<string | null>(null);

  const run = async (action: () => Promise<unknown>) => {
    try {
      setNote(null);
      await action();
      onChange();
    } catch (x) {
      setNote((x as Error).message);
    }
  };

  const test = () =>
    run(async () => {
      const r = await api.testPage(s.staff_id);
      setNote(r.status === "sent" ? "Test page sent. Check the phone." : `Test page ${r.status}: ${r.reason}`);
    });

  const remove = () => {
    if (window.confirm(`Remove ${s.name} from the roster?`)) run(() => api.removeStaff(s.staff_id));
  };

  return (
    <tr>
      <td>
        {s.name}
        <div className="muted small">{s.staff_id}</div>
      </td>
      <td>
        <select value={s.role} onChange={(x) => run(() => api.updateStaff(s.staff_id, { role: x.target.value as StaffRole }))}>
          {ROLES.map((r) => (
            <option key={r} value={r}>
              {r}
            </option>
          ))}
        </select>
      </td>
      <PhoneCell s={s} onChange={onChange} />
      <td>
        <label className="row">
          <input type="checkbox" checked={s.on_call} onChange={(x) => run(() => api.updateStaff(s.staff_id, { on_call: x.target.checked }))} />
          {s.on_call ? "On call" : "Off"}
        </label>
      </td>
      <td>
        {s.reachable ? (
          <span className="tag">reachable</span>
        ) : (
          <span className="muted small">Not yet: from this phone, text its assigned agent number once</span>
        )}
      </td>
      <td>
        <div className="row">
          <button className="ghost" onClick={test}>
            Send test page
          </button>
          <button className="ghost" onClick={remove}>
            Remove
          </button>
        </div>
        {note && <div className="small muted">{note}</div>}
      </td>
    </tr>
  );
}

export default function OnCall() {
  const { data, error, reload } = useLoad(api.oncall);
  const [name, setName] = useState("");
  const [role, setRole] = useState<StaffRole>("counselor");
  const [phone, setPhone] = useState("");
  const [err, setErr] = useState<string | null>(null);

  const add = async () => {
    try {
      setErr(null);
      await api.addStaff({ name, role, phone, on_call: true });
      setName("");
      setPhone("");
      reload();
    } catch (x) {
      setErr((x as Error).message);
    }
  };

  return (
    <Page title="On-call" subtitle="Who is paged when a family's message is flagged as a safety concern.">
      <ErrorNote error={error} />
      {data && (
        <div className="grid2">
          <div className="card">
            <h3>Right now</h3>
            <p>
              {data.would_page_now.length ? (
                <>
                  A new safety case would page <strong>{data.would_page_now.join(", ")}</strong>.
                </>
              ) : (
                <span className="overdue">Nobody is on call. A new safety case would page no one.</span>
              )}
            </p>
            <p className="muted small">
              School time {data.school_time.slice(11, 16)} ({data.timezone}) ·{" "}
              {data.in_school_hours ? "school hours" : "outside school hours"}
            </p>
            {!data.channel_listening && (
              <p className="alert-box">The messaging channel is off, so pages are not being delivered.</p>
            )}
          </div>
          <div className="card">
            <h3>How paging works</h3>
            <ol className="steps">
              <li>
                School hours ({data.school_hours}, {data.school_days}): the on-call <strong>counselor</strong> is paged. Otherwise,
                the on-call <strong>administrator</strong>.
              </li>
              <li>
                Not acknowledged in {data.escalate_after_minutes} minutes: the <strong>principal</strong> is paged.
              </li>
              <li>If nobody is on call for a role, the next role in line is paged.</li>
              <li>
                Staff reply <strong>ACK</strong> to the page, or use the <Link to="/escalations">Escalations</Link> page. Pages never
                include the student's name or the family's words.
              </li>
            </ol>
          </div>
        </div>
      )}

      <div className="card">
        <h3>Roster</h3>
        <p className="muted small">
          On the shared Photon line, add each phone as a sender in the Photon dashboard (iMessage page). Photon assigns that phone its own
          agent number. Text that number once from the phone; pages then arrive from it. If a phone still shows "Not yet", check the number
          here matches the phone exactly.
        </p>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Role</th>
              <th>Phone</th>
              <th>On call</th>
              <th>Can receive pages</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {data?.staff.map((s) => <StaffRow key={s.staff_id} s={s} onChange={reload} />)}
            {data?.staff.length === 0 && (
              <tr>
                <td colSpan={6} className="muted">
                  No one on the roster yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <div className="row" style={{ marginTop: 12 }}>
          <input value={name} onChange={(x) => setName(x.target.value)} placeholder="Name" />
          <select value={role} onChange={(x) => setRole(x.target.value as StaffRole)}>
            {ROLES.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
          <input value={phone} onChange={(x) => setPhone(x.target.value)} placeholder="+15551234567" />
          <button disabled={!name.trim() || !phone.trim()} onClick={add}>
            Add to roster
          </button>
        </div>
        {err && <p className="error">{err}</p>}
      </div>
    </Page>
  );
}
