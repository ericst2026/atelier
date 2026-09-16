import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";

/** Accounts and permissions — the admin's page. A teacher never comes here: they
 *  run a class and admit students to it, and nothing else. */
export default function Admin() {
  const [users, setUsers] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [editing, setEditing] = useState(null);
  const [form, setForm] = useState({ username: "", name: "", role: "student", password: "" });
  const [msg, setMsg] = useState(null);
  const [error, setError] = useState(null);
  const csv = useRef(null);
  const load = useCallback(
    () =>
      api("/users")
        .then((u) => {
          setUsers(u);
          setError(null);
        })
        .catch((e) => setError(`Could not load the accounts: ${e.message}`)),
    []
  );
  useEffect(() => {
    load();
    api("/experiments").then((d) => setExperiments(d.experiments || [])).catch(() => {});
  }, [load]);
  const patch = async (u, body) => {
    try {
      await api(`/users/${u.id}`, { method: "PATCH", body });
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const create = async () => {
    try {
      await api("/users", { method: "POST", body: form });
      setForm({ username: "", name: "", role: "student", password: "" });
      setMsg(`Created ${form.username}.`);
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const importCsv = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const r = await api("/users/import", { method: "POST", form: fd });
      setMsg(`Created ${r.created.length}, skipped ${r.skipped.length}.`);
      load();
    } catch (err) {
      setError(err.message);
    }
    e.target.value = "";
  };
  const waiting = users.filter((u) => !u.active);
  return (
    <main className="page">
      <div className="hero">
        <div>
          <h1>Accounts</h1>
          <p className="muted">Who may sign in, what they are, and which experiments they may run on their own.</p>
        </div>
      </div>

      {error && <div className="panel" style={{ borderColor: "var(--dup)", marginBottom: 14 }}>{error}</div>}

      {waiting.length > 0 && (
        <div className="panel stack" style={{ marginBottom: 14, borderColor: "var(--raw)" }}>
          <h3>Waiting to be let in ({waiting.length})</h3>
          {waiting.map((u) => (
            <div key={u.id} className="row" style={{ justifyContent: "space-between" }}>
              <span>
                {u.name || u.username} <span className="faint small">{u.username} · asked to be a {u.role} · {fmtTime(u.created_at)}</span>
              </span>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm good" onClick={() => patch(u, { active: true })}>
                  Approve
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {editing && (
        // keyed by the account: without it React keeps the previous person's ticks,
        // because the editor sits in the same place in the tree
        <SelfExperiments key={editing.id} user={editing} experiments={experiments} onClose={() => setEditing(null)} onSaved={load} />
      )}

      <div className="panel stack" style={{ marginBottom: 14 }}>
        <h3>Add an account</h3>
        <div className="row">
          <input type="text" placeholder="username" style={{ width: 140 }} value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
          <input type="text" placeholder="name" style={{ width: 160 }} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <select style={{ width: 120 }} value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
            <option value="student">student</option>
            <option value="teacher">teacher</option>
            <option value="admin">admin</option>
          </select>
          <input type="text" placeholder="password" style={{ width: 140 }} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          <button className="btn primary sm" onClick={create} disabled={!form.username || !form.password}>
            Add
          </button>
          <button className="btn sm" onClick={() => csv.current?.click()}>
            Import CSV
          </button>
          <input ref={csv} type="file" accept=".csv" style={{ display: "none" }} onChange={importCsv} />
          <span className="help">CSV columns: username,name,role,password</span>
          {msg && <span className="small" style={{ color: "var(--kept)" }}>{msg}</span>}
        </div>
      </div>

      {users.length === 0 && !error && <div className="empty">No accounts loaded.</div>}
      {users.length > 0 && (
        <div className="tablewrap">
          <table className="data">
            <thead>
              <tr>
                <th>Username</th>
                <th>Name</th>
                <th>Role</th>
                <th>Can sign in</th>
                <th>May run alone</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.username}</td>
                  <td>{u.name}</td>
                  <td>
                    <select value={u.role} onChange={(e) => patch(u, { role: e.target.value })} style={{ width: 110 }}>
                      <option value="student">student</option>
                      <option value="teacher">teacher</option>
                      <option value="admin">admin</option>
                    </select>
                  </td>
                  <td>{u.active ? "yes" : <b style={{ color: "var(--raw)" }}>waiting</b>}</td>
                  <td>
                    <button className="btn sm ghost" onClick={() => setEditing(u)}>
                      {(u.self_experiments || []).length ? `${u.self_experiments.length} experiment${u.self_experiments.length === 1 ? "" : "s"}` : "none"}
                    </button>
                  </td>
                  <td className="row" style={{ gap: 6 }}>
                    <button className="btn sm ghost" onClick={() => { const p = window.prompt(`New password for ${u.username}`); if (p) patch(u, { password: p }); }}>
                      reset password
                    </button>
                    <button className={`btn sm ${u.active ? "ghost" : "good"}`} onClick={() => patch(u, { active: !u.active })}>
                      {u.active ? "deactivate" : "approve"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}

/** Which experiments this person may run outside a class — and, for a teacher,
 *  therefore which they may hold a class on. */
function SelfExperiments({ user, experiments, onClose, onSaved }) {
  const [chosen, setChosen] = useState(user.self_experiments || []);
  const [error, setError] = useState(null);
  useEffect(() => {
    setChosen(user.self_experiments || []);
  }, [user.id, user.self_experiments]);
  const toggle = (slug) => setChosen(chosen.includes(slug) ? chosen.filter((s) => s !== slug) : [...chosen, slug]);
  const save = async () => {
    try {
      await api(`/users/${user.id}/self-experiments`, { method: "PUT", body: { experiments: chosen } });
      onSaved();
      onClose();
    } catch (e) {
      setError(e.message);
    }
  };
  return (
    <div className="panel stack" style={{ marginBottom: 14, borderColor: "var(--sky)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3>{user.name || user.username} may run these on their own</h3>
        <div className="row" style={{ gap: 6 }}>
          <button className="btn sm ghost" onClick={() => setChosen([])}>
            none
          </button>
          <button className="btn sm ghost" onClick={() => setChosen(experiments.map((e) => e.slug))}>
            all
          </button>
          <button className="btn sm primary" onClick={save}>
            Save
          </button>
          <button className="btn sm ghost" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
      <div className="help">
        {user.role === "teacher"
          ? "A teacher may also hold a class on any of these, and on nothing else."
          : "Outside a class this is everything they can run. In class they work on whatever their teacher started."}
      </div>
      {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
      <div className="grid3">
        {experiments.map((e) => (
          <label key={e.slug} className="check">
            <input type="checkbox" checked={chosen.includes(e.slug)} onChange={() => toggle(e.slug)} />
            <span>{e.title}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
