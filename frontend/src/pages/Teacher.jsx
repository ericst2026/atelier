import React, { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { StatusPill } from "../components/RunStatus";
import ClassHistory from "../components/ClassHistory";
import ClassPanel, { WallPanel } from "../components/ClassPanel";
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

/** The GPUs of one machine, labelled by that machine's own indices. */
function NodeCard({ name, gpus, worker, single }) {
  const free = (worker?.free_gpus || []).filter((g) => (single ? true : String(g).startsWith(`${name}:`))).length;
  const stale = worker?.at ? Date.now() - new Date(`${worker.at}Z`).getTime() > 30000 : true;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>{name}</b>
        <span className="small muted">
          {gpus.length ? `${gpus.length} GPU${gpus.length === 1 ? "" : "s"} · ${free} free` : "no GPUs"}
          {worker ? (stale ? " · heartbeat late" : " · reporting") : " · not checked in"}
        </span>
      </div>
      <div className="kpis">
        {gpus.length === 0 && <div className="kpi"><div className="v">CPU</div><div className="l">runs on processor</div></div>}
        {gpus.map((g) => (
          <div key={g.uid || `${name}:${g.index}`} className={`kpi ${g.job ? "raw" : "kept"}`}>
            <div className="v">{g.util}%</div>
            <div className="l">
              GPU {g.index} · {(g.mem_used / 1e9).toFixed(0)}/{(g.mem_total / 1e9).toFixed(0)} GB · {g.temp}°
            </div>
            <div className="h">{g.job ? `${g.job.user} · ${g.job.experiment} · run ${g.job.run_id}` : "free"}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LivePanel({ live }) {
  if (!live) return <div className="empty">Waiting for the worker…</div>;
  const gpus = live.gpus?.gpus || [];
  // a GPU belongs to a machine; index 0 on one node is not index 0 on another
  const nodeNames = live.nodes?.length ? live.nodes : [...new Set(gpus.map((g) => g.node).filter(Boolean))];
  const workers = live.worker?.nodes || (live.worker ? [live.worker] : []);
  const single = nodeNames.length <= 1;
  return (
    <div className="stack">
      <div className="row">
        <span className="pill running">
          <i className="dot" /> {live.running.length} running
        </span>
        <span className="pill queued">{live.queued.length} queued</span>
        <span className="pill">
          {live.worker ? `${nodeNames.length > 1 ? `${nodeNames.length} nodes` : "worker ok"} · ${gpus.length} GPU${gpus.length === 1 ? "" : "s"} · ${(live.worker.free_gpus || []).length} free` : "worker offline"}
        </span>
      </div>
      {nodeNames.length === 0 && <div className="help">No worker has checked in.</div>}
      <div className={nodeNames.length > 1 ? "grid2" : "stack"}>
        {nodeNames.map((name) => (
          <div key={name} className={nodeNames.length > 1 ? "panel" : ""}>
            <NodeCard
              name={name}
              single={single}
              gpus={gpus.filter((g) => (g.node || name) === name)}
              worker={workers.find((w) => (w.node || "") === name) || (single ? live.worker : null)}
            />
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

function RunsPanel() {
  const [runs, setRuns] = useState([]);
  const [filter, setFilter] = useState("");
  const load = useCallback(() => api(`/runs?limit=200${filter ? `&status=${filter}` : ""}`).then(setRuns), [filter]);
  useEffect(() => {
    load();
  }, [load]);
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
              <th>Class</th>
              <th>Whose class</th>
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
                <td>{r.class_name || <span className="faint">on their own</span>}</td>
                <td>{r.class_teacher || <span className="faint">–</span>}</td>
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

export default function Teacher() {
  const [sp, setSp] = useSearchParams();
  const section = sp.get("section") || "class";
  const setSection = (k) => setSp({ section: k });
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
          ["class", "The class"],
          ["history", "Past classes"],
          ["displays", "The wall"],
          ["runs", "Runs"],
          ["live", "The node"],
        ].map(([k, l]) => (
          <button key={k} className={section === k ? "active" : ""} onClick={() => setSection(k)}>
            {l}
          </button>
        ))}
      </div>
      {section === "class" && <ClassPanel />}
      {section === "history" && <ClassHistory onResumed={() => setSection("class")} />}
      {section === "live" && <LivePanel live={live} />}
      {section === "runs" && <RunsPanel />}
      {section === "displays" && <WallPanel />}
    </main>
  );
}
