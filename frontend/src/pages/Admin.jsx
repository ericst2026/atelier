import React, { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { useT } from "../i18n";
import { fmtTime } from "../lib/format";

/** Accounts and permissions — the admin's page. A teacher never comes here: they
 *  run a class and admit students to it, and nothing else. */
export default function Admin() {
  const t = useT();
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
        .catch((e) => setError(t("admin.loadFailed", { message: e.message }))),
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
      setMsg(t("admin.created", { username: form.username }));
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
      setMsg(t("admin.imported", { created: r.created.length, skipped: r.skipped.length }));
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
          <h1>{t("admin.title")}</h1>
          <p className="muted">{t("admin.intro")}</p>
        </div>
      </div>

      {error && <div className="panel" style={{ borderColor: "var(--dup)", marginBottom: 14 }}>{error}</div>}

      {waiting.length > 0 && (
        <div className="panel stack" style={{ marginBottom: 14, borderColor: "var(--raw)" }}>
          <h3>{t("admin.waiting.title", { n: waiting.length })}</h3>
          {waiting.map((u) => (
            <div key={u.id} className="row" style={{ justifyContent: "space-between" }}>
              <span>
                {u.name || u.username} <span className="faint small">{u.username} · {t("admin.waiting.askedRole", { role: t(`common.role.${u.role}`) })} · {fmtTime(u.created_at)}</span>
              </span>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm good" onClick={() => patch(u, { active: true })}>
                  {t("admin.waiting.approve")}
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
        <h3>{t("admin.add.title")}</h3>
        <div className="row">
          <input type="text" placeholder={t("admin.add.username")} style={{ width: 140 }} value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
          <input type="text" placeholder={t("admin.add.name")} style={{ width: 160 }} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <select style={{ width: 120 }} value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
            <option value="student">{t("common.role.student")}</option>
            <option value="teacher">{t("common.role.teacher")}</option>
            <option value="admin">{t("common.role.admin")}</option>
          </select>
          <input type="text" placeholder={t("admin.add.password")} style={{ width: 140 }} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
          <button className="btn primary sm" onClick={create} disabled={!form.username || !form.password}>
            {t("common.add")}
          </button>
          <button className="btn sm" onClick={() => csv.current?.click()}>
            {t("admin.add.importCsv")}
          </button>
          <input ref={csv} type="file" accept=".csv" style={{ display: "none" }} onChange={importCsv} />
          <span className="help">{t("admin.add.csvColumns")}</span>
          {msg && <span className="small" style={{ color: "var(--kept)" }}>{msg}</span>}
        </div>
      </div>

      {users.length === 0 && !error && <div className="empty">{t("admin.empty")}</div>}
      {users.length > 0 && (
        <div className="tablewrap">
          <table className="data">
            <thead>
              <tr>
                <th>{t("admin.table.username")}</th>
                <th>{t("admin.table.name")}</th>
                <th>{t("admin.table.role")}</th>
                <th>{t("admin.table.canSignIn")}</th>
                <th>{t("admin.table.mayRunAlone")}</th>
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
                      <option value="student">{t("common.role.student")}</option>
                      <option value="teacher">{t("common.role.teacher")}</option>
                      <option value="admin">{t("common.role.admin")}</option>
                    </select>
                  </td>
                  <td>{u.active ? t("admin.table.active") : <b style={{ color: "var(--raw)" }}>{t("admin.table.waiting")}</b>}</td>
                  <td>
                    <button className="btn sm ghost" onClick={() => setEditing(u)}>
                      {(u.self_experiments || []).length ? t("admin.table.experiments", { count: u.self_experiments.length }) : t("common.none")}
                    </button>
                  </td>
                  <td className="row" style={{ gap: 6 }}>
                    <button className="btn sm ghost" onClick={() => { const p = window.prompt(t("admin.table.newPasswordPrompt", { username: u.username })); if (p) patch(u, { password: p }); }}>
                      {t("admin.table.resetPassword")}
                    </button>
                    <button className={`btn sm ${u.active ? "ghost" : "good"}`} onClick={() => patch(u, { active: !u.active })}>
                      {u.active ? t("admin.table.deactivate") : t("admin.table.approve")}
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
  const t = useT();
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
        <h3>{t("admin.self.title", { name: user.name || user.username })}</h3>
        <div className="row" style={{ gap: 6 }}>
          <button className="btn sm ghost" onClick={() => setChosen([])}>
            {t("common.none")}
          </button>
          <button className="btn sm ghost" onClick={() => setChosen(experiments.map((e) => e.slug))}>
            {t("admin.self.all")}
          </button>
          <button className="btn sm primary" onClick={save}>
            {t("common.save")}
          </button>
          <button className="btn sm ghost" onClick={onClose}>
            {t("common.close")}
          </button>
        </div>
      </div>
      <div className="help">
        {user.role === "teacher"
          ? t("admin.self.teacherHelp")
          : t("admin.self.studentHelp")}
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
