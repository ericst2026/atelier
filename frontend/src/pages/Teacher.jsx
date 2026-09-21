import React, { useCallback, useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { StatusPill } from "../components/RunStatus";
import ClassHistory from "../components/ClassHistory";
import ClassPanel, { WallPanel } from "../components/ClassPanel";
import { useT } from "../i18n";
import { api, wsUrl } from "../lib/api";
import { fmtBytes, fmtDuration, fmtTime } from "../lib/format";
import { useSocket } from "../lib/ws";

const MODES = [
  ["grafana", "teacher.modes.grafana"],
  ["live", "teacher.modes.live"],
  ["progress", "teacher.modes.progress"],
  ["leaderboard", "teacher.modes.leaderboard"],
  ["run", "teacher.modes.run"],
  ["message", "teacher.modes.message"],
];

/** The GPUs of one machine, labelled by that machine's own indices. */
function NodeCard({ name, gpus, worker, single }) {
  const t = useT();
  const free = (worker?.free_gpus || []).filter((g) => (single ? true : String(g).startsWith(`${name}:`))).length;
  const stale = worker?.at ? Date.now() - new Date(`${worker.at}Z`).getTime() > 30000 : true;
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>{name}</b>
        <span className="small muted">
          {gpus.length ? t("teacher.gpusFree", { count: gpus.length, free }) : t("teacher.node.noGpus")}
          {worker ? (stale ? t("teacher.node.heartbeatLate") : t("teacher.node.reporting")) : t("teacher.node.notCheckedIn")}
        </span>
      </div>
      <div className="kpis">
        {gpus.length === 0 && <div className="kpi"><div className="v">{t("teacher.node.cpu")}</div><div className="l">{t("teacher.node.runsOnProcessor")}</div></div>}
        {gpus.map((g) => (
          <div key={g.uid || `${name}:${g.index}`} className={`kpi ${g.job ? "raw" : "kept"}`}>
            <div className="v">{g.util}%</div>
            <div className="l">{t("teacher.node.gpu", { index: g.index, used: (g.mem_used / 1e9).toFixed(0), total: (g.mem_total / 1e9).toFixed(0), temp: g.temp })}</div>
            <div className="h">{g.job ? `${g.job.user} · ${g.job.experiment} · ${t("common.runN", { id: g.job.run_id })}` : t("teacher.node.free")}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function LivePanel({ live }) {
  const t = useT();
  if (!live) return <div className="empty">{t("teacher.live.waitingWorker")}</div>;
  const gpus = live.gpus?.gpus || [];
  // a GPU belongs to a machine; index 0 on one node is not index 0 on another
  const nodeNames = live.nodes?.length ? live.nodes : [...new Set(gpus.map((g) => g.node).filter(Boolean))];
  const workers = live.worker?.nodes || (live.worker ? [live.worker] : []);
  const single = nodeNames.length <= 1;
  return (
    <div className="stack">
      <div className="row">
        <span className="pill running">
          <i className="dot" /> {t("teacher.live.running", { n: live.running.length })}
        </span>
        <span className="pill queued">{t("teacher.live.queued", { n: live.queued.length })}</span>
        <span className="pill">
          {live.worker
            ? `${nodeNames.length > 1 ? t("teacher.live.nodes", { n: nodeNames.length }) : t("teacher.live.workerOk")} · ${t("teacher.gpusFree", { count: gpus.length, free: (live.worker.free_gpus || []).length })}`
            : t("teacher.live.workerOffline")}
        </span>
      </div>
      {nodeNames.length === 0 && <div className="help">{t("teacher.live.noWorker")}</div>}
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
              <th>{t("teacher.live.col.run")}</th>
              <th>{t("teacher.live.col.who")}</th>
              <th>{t("teacher.live.col.what")}</th>
              <th>{t("teacher.live.col.gpus")}</th>
              <th>{t("teacher.live.col.status")}</th>
              <th>{t("teacher.live.col.progress")}</th>
              <th>{t("teacher.live.col.elapsed")}</th>
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
  const t = useT();
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
    if (!window.confirm(t("teacher.runs.confirmDelete", { id: r.id }))) return;
    await api(`/runs/${r.id}`, { method: "DELETE" });
    load();
  };
  return (
    <div className="stack">
      <div className="row">
        <select style={{ width: 200 }} value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="">{t("teacher.runs.filter.all")}</option>
          <option value="queued,running">{t("teacher.runs.filter.active")}</option>
          <option value="succeeded">{t("teacher.runs.filter.succeeded")}</option>
          <option value="failed,cancelled">{t("teacher.runs.filter.failed")}</option>
        </select>
        <button className="btn sm" onClick={load}>
          {t("teacher.runs.refresh")}
        </button>
      </div>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              <th>#</th>
              <th>{t("teacher.runs.col.user")}</th>
              <th>{t("teacher.runs.col.experiment")}</th>
              <th>{t("teacher.runs.col.class")}</th>
              <th>{t("teacher.runs.col.whoseClass")}</th>
              <th>{t("teacher.runs.col.kind")}</th>
              <th>{t("teacher.runs.col.label")}</th>
              <th>{t("teacher.runs.col.status")}</th>
              <th>{t("teacher.runs.col.gpus")}</th>
              <th>{t("teacher.runs.col.created")}</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.username}</td>
                <td>{r.experiment}</td>
                <td>{r.class_name || <span className="faint">{t("teacher.runs.onTheirOwn")}</span>}</td>
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
                      {t("teacher.runs.stop")}
                    </button>
                  )}
                  {r.status !== "queued" && r.status !== "running" && (
                    <button className="btn sm ghost" onClick={() => del(r)}>
                      {t("teacher.runs.delete")}
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
  const t = useT();
  const [sp, setSp] = useSearchParams();
  const section = sp.get("section") || "class";
  const setSection = (k) => setSp({ section: k });
  const [live, setLive] = useState(null);
  useSocket(wsUrl("/ws/teacher"), (m) => m.type === "live" && setLive(m.live));
  return (
    <main className="page">
      <div className="hero">
        <div>
          <h1>{t("teacher.title")}</h1>
          <p className="muted">{t("teacher.subtitle")}</p>
        </div>
      </div>
      <div className="tabs">
        {["class", "history", "displays", "runs", "live"].map((k) => (
          <button key={k} className={section === k ? "active" : ""} onClick={() => setSection(k)}>
            {t(`teacher.tabs.${k}`)}
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
