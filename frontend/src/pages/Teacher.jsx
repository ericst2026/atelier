import React, { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { StatusPill } from "../components/RunStatus";
import { api, wsUrl } from "../lib/api";
import { fmtBytes, fmtDuration, fmtTime } from "../lib/format";
import { useSocket } from "../lib/ws";

const MODES = [
  ["grafana", "Hardware (Grafana)"],
  ["live", "Live runs"],
  ["progress", "Class progress"],
  ["leaderboard", "Leaderboard"],
  ["run", "A run's results"],
  ["message", "Message"],
];

function LivePanel({ live }) {
  if (!live) return <div className="empty">Waiting for the worker…</div>;
  const gpus = live.gpus?.gpus || [];
  return (
    <div className="stack">
      <div className="row">
        <span className="pill running">
          <i className="dot" /> {live.running.length} running
        </span>
        <span className="pill queued">{live.queued.length} queued</span>
        <span className="pill">{live.worker ? `${live.worker.nodes?.length > 1 ? `${live.worker.nodes.length} nodes` : "worker ok"} · free GPUs ${live.worker.free_gpus.join(",") || "none"}` : "worker offline"}</span>
        {live.nodes?.length > 1 && <span className="pill">{live.nodes.join(" · ")}</span>}
      </div>
      <div className="kpis">
        {Array.from({ length: live.gpu_count }, (_, i) => gpus[i]).map((g, i) => (
          <div key={i} className={`kpi ${g?.job ? "raw" : g ? "kept" : ""}`}>
            <div className="v">{g ? `${g.util}%` : "–"}</div>
            <div className="l">GPU {i}{g ? ` · ${(g.mem_used / 1e9).toFixed(0)}/${(g.mem_total / 1e9).toFixed(0)} GB · ${g.temp}°` : ""}</div>
            <div className="h">{g?.job ? `${g.job.user} · ${g.job.experiment} · run ${g.job.run_id}` : "free"}</div>
          </div>
        ))}
      </div>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>Run</th>
              <th>Who</th>
              <th>What</th>
              <th>GPUs</th>
              <th>Status</th>
              <th>Progress</th>
              <th>Elapsed</th>
            </tr>
          </thead>
          <tbody>
            {[...live.running, ...live.queued, ...live.recent].map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.user}</td>
                <td>
                  {r.experiment} · {r.kind}
                  {r.step ? ` ${r.step}` : ""} <span className="faint">{r.label}</span>
                </td>
                <td>{r.gpu_ids?.join(",") || (r.gpus ? `${r.gpus}?` : "cpu")}</td>
                <td>
                  <StatusPill status={r.status} />
                </td>
                <td>
                  {Math.round(r.progress_pct)}% <span className="faint">{r.progress_msg}</span>
                </td>
                <td>{fmtDuration(r.elapsed)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function UsersPanel() {
  const [users, setUsers] = useState([]);
  const [form, setForm] = useState({ username: "", name: "", role: "student", password: "" });
  const [msg, setMsg] = useState(null);
  const csv = useRef(null);
  const load = useCallback(() => api("/users").then(setUsers), []);
  useEffect(load, [load]);
  const create = async () => {
    try {
      await api("/users", { method: "POST", body: form });
      setForm({ username: "", name: "", role: "student", password: "" });
      load();
    } catch (e) {
      setMsg(e.message);
    }
  };
  const importCsv = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    const r = await api("/users/import", { method: "POST", form: fd });
    setMsg(`Created ${r.created.length}, skipped ${r.skipped.length}`);
    load();
    e.target.value = "";
  };
  const patch = async (u, body) => {
    await api(`/users/${u.id}`, { method: "PATCH", body });
    load();
  };
  return (
    <div className="stack">
      <div className="row">
        <input type="text" placeholder="username" style={{ width: 140 }} value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} />
        <input type="text" placeholder="name" style={{ width: 160 }} value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
        <select style={{ width: 120 }} value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
          <option value="student">student</option>
          <option value="teacher">teacher</option>
        </select>
        <input type="text" placeholder="password" style={{ width: 140 }} value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
        <button className="btn primary sm" onClick={create} disabled={!form.username || !form.password}>
          Add user
        </button>
        <button className="btn sm" onClick={() => csv.current?.click()}>
          Import CSV
        </button>
        <input ref={csv} type="file" accept=".csv" style={{ display: "none" }} onChange={importCsv} />
        <span className="help">CSV columns: username,name,role,password</span>
        {msg && <span className="small" style={{ color: "var(--sky)" }}>{msg}</span>}
      </div>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>Username</th>
              <th>Name</th>
              <th>Role</th>
              <th>Active</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id}>
                <td>{u.username}</td>
                <td>{u.name}</td>
                <td>{u.role}</td>
                <td>{u.active ? "yes" : "no"}</td>
                <td className="row" style={{ gap: 6 }}>
                  <button className="btn sm ghost" onClick={() => { const p = window.prompt(`New password for ${u.username}`); if (p) patch(u, { password: p }); }}>
                    reset password
                  </button>
                  <button className="btn sm ghost" onClick={() => patch(u, { active: !u.active })}>
                    {u.active ? "deactivate" : "activate"}
                  </button>
                  <button className="btn sm ghost" onClick={() => patch(u, { role: u.role === "teacher" ? "student" : "teacher" })}>
                    make {u.role === "teacher" ? "student" : "teacher"}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RunsPanel() {
  const [runs, setRuns] = useState([]);
  const [filter, setFilter] = useState("");
  const load = useCallback(() => api(`/runs?limit=200${filter ? `&status=${filter}` : ""}`).then(setRuns), [filter]);
  useEffect(load, [load]);
  const cancel = async (r) => {
    await api(`/runs/${r.id}/cancel`, { method: "POST" });
    load();
  };
  const del = async (r) => {
    if (!window.confirm(`Delete run ${r.id} and its files?`)) return;
    await api(`/runs/${r.id}`, { method: "DELETE" });
    load();
  };
  return (
    <div className="stack">
      <div className="row">
        <select style={{ width: 200 }} value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">all statuses</option>
          <option value="queued,running">queued + running</option>
          <option value="succeeded">succeeded</option>
          <option value="failed,cancelled">failed + cancelled</option>
        </select>
        <button className="btn sm" onClick={load}>
          Refresh
        </button>
      </div>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>#</th>
              <th>User</th>
              <th>Experiment</th>
              <th>Kind</th>
              <th>Label</th>
              <th>Status</th>
              <th>GPUs</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.username}</td>
                <td>{r.experiment}</td>
                <td>
                  {r.kind}
                  {r.step ? ` ${r.step}` : ""}
                </td>
                <td>{r.label}</td>
                <td>
                  <StatusPill status={r.status} />
                </td>
                <td>{r.gpu_ids?.join(",") || r.gpus}</td>
                <td>{fmtTime(r.created_at)}</td>
                <td className="row" style={{ gap: 6 }}>
                  {(r.status === "queued" || r.status === "running") && (
                    <button className="btn sm danger" onClick={() => cancel(r)}>
                      stop
                    </button>
                  )}
                  {r.status !== "queued" && r.status !== "running" && (
                    <button className="btn sm ghost" onClick={() => del(r)}>
                      delete
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DisplaysPanel() {
  const [displays, setDisplays] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const load = useCallback(() => api("/displays").then(setDisplays), []);
  useEffect(() => {
    load();
    api("/experiments").then((d) => setExperiments(d.experiments));
  }, [load]);
  const save = async (d) => {
    try {
      await api(`/displays/${d.id}`, { method: "PUT", body: { mode: d.mode, payload: d.payload, name: d.name } });
      load();
    } catch (e) {
      window.alert(e.message);
    }
  };
  const edit = (i, patch) => setDisplays(displays.map((d, j) => (j === i ? { ...d, ...patch } : d)));
  return (
    <div className="grid3">
      {displays.map((d, i) => (
        <div key={d.id} className="panel stack" style={{ gap: 8 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>Display {d.id}</h3>
            <a href={`/display/${d.id}`} target="_blank" rel="noreferrer" className="small">
              open /display/{d.id}
            </a>
          </div>
          <input type="text" value={d.name} onChange={(e) => edit(i, { name: e.target.value })} />
          <select value={d.mode} onChange={(e) => edit(i, { mode: e.target.value, payload: {} })}>
            {MODES.map(([k, l]) => (
              <option key={k} value={k}>
                {l}
              </option>
            ))}
          </select>
          {d.mode === "leaderboard" && (
            <select value={d.payload.experiment || ""} onChange={(e) => edit(i, { payload: { ...d.payload, experiment: e.target.value || undefined } })}>
              <option value="">all experiments</option>
              {experiments.map((x) => (
                <option key={x.slug} value={x.slug}>
                  {x.title}
                </option>
              ))}
            </select>
          )}
          {d.mode === "run" && <input type="number" placeholder="run id" value={d.payload.run_id || ""} onChange={(e) => edit(i, { payload: { run_id: Number(e.target.value) } })} />}
          {d.mode === "grafana" && <input type="text" placeholder="dashboard URL (blank = default hardware dashboard)" value={d.payload.url || ""} onChange={(e) => edit(i, { payload: { url: e.target.value || undefined } })} />}
          {d.mode === "message" && (
            <>
              <input type="text" placeholder="title" value={d.payload.title || ""} onChange={(e) => edit(i, { payload: { ...d.payload, title: e.target.value } })} />
              <textarea placeholder="text" value={d.payload.text || ""} onChange={(e) => edit(i, { payload: { ...d.payload, text: e.target.value } })} />
            </>
          )}
          <button className="btn primary sm" onClick={() => save(d)}>
            Push to display {d.id}
          </button>
        </div>
      ))}
    </div>
  );
}

function SubmissionsPanel() {
  const [subs, setSubs] = useState([]);
  useEffect(() => {
    api("/submissions").then(setSubs);
  }, []);
  return (
    <div className="tablewrap">
      <table className="data">
        <thead>
          <tr>
            <th>#</th>
            <th>Student</th>
            <th>Experiment</th>
            <th>Submitted</th>
            <th>Size</th>
            <th>Auto</th>
            <th>Grade</th>
            <th>Published</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {subs.map((s) => (
            <tr key={s.id}>
              <td>{s.id}</td>
              <td>{s.name || s.username}</td>
              <td>{s.experiment}</td>
              <td>{fmtTime(s.created_at)}</td>
              <td>{fmtBytes(s.bytes)}</td>
              <td>{s.auto_score ?? "–"}</td>
              <td>{s.score ?? "–"}</td>
              <td>{s.published ? "yes" : "no"}</td>
              <td>
                <Link to={`/teacher/submissions/${s.id}`}>review</Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Teacher() {
  const [section, setSection] = useState("live");
  const [live, setLive] = useState(null);
  useSocket(wsUrl("/ws/teacher"), (m) => m.type === "live" && setLive(m.live));
  return (
    <main className="page">
      <div className="hero">
        <div>
          <h1>Teacher</h1>
          <p className="muted">The node, the class, the screens.</p>
        </div>
      </div>
      <div className="tabs">
        {[
          ["live", "Live"],
          ["submissions", "Submissions"],
          ["users", "Users"],
          ["runs", "Runs"],
          ["displays", "Displays"],
        ].map(([k, l]) => (
          <button key={k} className={section === k ? "active" : ""} onClick={() => setSection(k)}>
            {l}
          </button>
        ))}
      </div>
      {section === "live" && <LivePanel live={live} />}
      {section === "submissions" && <SubmissionsPanel />}
      {section === "users" && <UsersPanel />}
      {section === "runs" && <RunsPanel />}
      {section === "displays" && <DisplaysPanel />}
    </main>
  );
}
