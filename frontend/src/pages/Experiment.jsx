import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Download, Play } from "lucide-react";
import ChartCard from "../components/ChartCard";
import CodeEditor from "../components/CodeEditor";
import FileTree from "../components/FileTree";
import LogView from "../components/LogView";
import Markdown from "../components/Markdown";
import ParamsForm from "../components/ParamsForm";
import Rail from "../components/Rail";
import ResultsView from "../components/ResultsView";
import RunStatus, { StatusPill } from "../components/RunStatus";
import { api, fileUrl } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtBytes, fmtTime } from "../lib/format";
import { liveCharts, useRunStream } from "../lib/runs";

const TABS = [
  ["guide", "Guide"],
  ["materials", "Materials"],
  ["sample", "Sample code"],
  ["project", "My project"],
  ["submissions", "Submissions"],
];

function StepPanel({ spec, step, runs, prevRuns, onStarted }) {
  const { isTeacher } = useAuth();
  const defaults = useMemo(() => Object.fromEntries((step.params || []).map((p) => [p.key, p.default])), [step]);
  const [params, setParams] = useState(defaults);
  const [parentId, setParentId] = useState(prevRuns[0]?.id || null);
  const [gpus, setGpus] = useState(step.gpus);
  const [selected, setSelected] = useState(runs[0]?.id || null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showLog, setShowLog] = useState(false);
  useEffect(() => {
    setParams(defaults);
    setGpus(step.gpus);
    setSelected(runs[0]?.id || null);
    setError(null);
  }, [step.index]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!parentId && prevRuns[0]) setParentId(prevRuns[0].id);
  }, [prevRuns, parentId]);
  const stream = useRunStream(selected);
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const run = await api(`/experiments/${spec.slug}/steps/${step.index}/runs`, { method: "POST", body: { params, parent_run_id: step.needs_previous ? parentId : null, gpus: isTeacher ? gpus : null } });
      setSelected(run.id);
      setShowLog(true);
      onStarted(run);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const finished = stream.status === "succeeded" || stream.status === "failed" || stream.status === "cancelled";
  const live = liveCharts(stream.live);
  return (
    <div className="steplayout">
      <div className="stack">
        <div className="panel">
          <h2 style={{ marginBottom: 6 }}>
            {step.index}. {step.title}
          </h2>
          <Markdown text={step.description} />
        </div>
        <div className="panel stack">
          {step.needs_previous && (
            <label className="field">
              <span>Build on</span>
              <select value={parentId ?? ""} onChange={(e) => setParentId(Number(e.target.value) || null)}>
                <option value="">— pick a finished step {step.index - 1} run —</option>
                {prevRuns.map((r) => (
                  <option key={r.id} value={r.id}>
                    #{r.id} · {fmtTime(r.created_at)} · {(r.metrics || []).slice(0, 2).map((m) => `${m.label} ${typeof m.value === "number" ? Number(m.value.toFixed(3)) : m.value}`).join(" · ")}
                  </option>
                ))}
              </select>
              {prevRuns.length === 0 && <span className="help">Finish step {step.index - 1} first.</span>}
            </label>
          )}
          <ParamsForm params={step.params || []} values={params} onChange={setParams} disabled={busy} />
          {isTeacher && step.gpus > 0 && (
            <label className="field">
              <span>GPUs for this run</span>
              <input type="number" min={0} max={8} value={gpus} onChange={(e) => setGpus(Number(e.target.value))} />
              <span className="help">Students always get {step.gpus}.</span>
            </label>
          )}
          {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
          <button className="btn primary" onClick={start} disabled={busy || (step.needs_previous && !parentId)}>
            <Play size={14} /> {busy ? "Starting…" : `Run step ${step.index}`}
          </button>
          <div className="help">
            {step.gpus ? `${step.gpus} GPU` : "CPU"} · up to {step.timeout_min} min
          </div>
        </div>
        {runs.length > 0 && (
          <div className="panel tight">
            <div className="small muted" style={{ marginBottom: 6 }}>
              Your runs of this step
            </div>
            <div className="stack" style={{ gap: 4 }}>
              {runs.slice(0, 12).map((r) => (
                <button key={r.id} className={`btn sm ${selected === r.id ? "" : "ghost"}`} style={{ justifyContent: "flex-start" }} onClick={() => setSelected(r.id)}>
                  <StatusPill status={r.id === stream.run?.id && stream.status ? stream.status : r.status} /> #{r.id} <span className="faint">{fmtTime(r.created_at)}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="stack">
        {!selected && <div className="empty">Set the parameters and run the step. Output streams here while it runs.</div>}
        {selected && (
          <div className="panel stack">
            <RunStatus run={stream.run} progress={stream.progress} onCancel={stream.cancel} />
            {stream.error && <div style={{ color: "var(--dup)" }}>{stream.error}</div>}
            {!finished && live.length > 0 && (
              <div className="charts">
                {live.map((c) => (
                  <ChartCard key={c.id} spec={c} height={180} allowStretch={false} />
                ))}
              </div>
            )}
            {(!finished || stream.status !== "succeeded" || showLog) && <LogView lines={stream.lines} />}
            {finished && stream.status === "succeeded" && (
              <button className="btn sm ghost" onClick={() => setShowLog(!showLog)}>
                {showLog ? "Hide log" : "Show log"}
              </button>
            )}
          </div>
        )}
        {stream.result && <ResultsView result={stream.result} runId={selected} />}
      </div>
    </div>
  );
}

function MaterialsTab({ spec }) {
  const [trees, setTrees] = useState({});
  const load = (m) => api(`/experiments/${spec.slug}/materials-tree?path=${encodeURIComponent(m.path)}`).then((t) => setTrees((s) => ({ ...s, [m.key]: t.entries })));
  return (
    <div className="grid2">
      {spec.materials.map((m) => (
        <div key={m.key} className="panel stack" style={{ gap: 8 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>{m.name}</h3>
            <span className={`pill ${m.available ? "succeeded" : "failed"}`}>
                {m.available ? (m.local ? "on the server" : `on ${(m.nodes || []).join(", ")}`) : "not installed"}
              </span>
          </div>
          <div className="muted">{m.description}</div>
          <div className="small faint mono">
            {m.kind} · {m.path}
            {m.stats ? ` · ${m.stats.files} files · ${fmtBytes(m.stats.bytes)}` : ""}
          </div>
          {m.available && !m.local && <div className="muted small">Runs will find this. It lives on the GPU node, so it cannot be browsed from here.</div>}
              {m.available && m.local && !trees[m.key] && (
            <button className="btn sm" onClick={() => load(m)}>
              Browse files
            </button>
          )}
          {trees[m.key] && (
            <div className="inset" style={{ padding: 8, maxHeight: 220, overflow: "auto" }}>
              <FileTree entries={trees[m.key]} onSelect={(p) => window.open(fileUrl(`/experiments/${spec.slug}/materials/${m.path}/${p}`), "_blank")} />
            </div>
          )}
        </div>
      ))}
      {spec.materials.length === 0 && <div className="empty">This experiment needs no prepared materials.</div>}
    </div>
  );
}

function SampleTab({ spec }) {
  const nav = useNavigate();
  const [entries, setEntries] = useState([]);
  const [path, setPath] = useState(null);
  const [file, setFile] = useState(null);
  useEffect(() => {
    api(`/experiments/${spec.slug}/sample/tree`).then((t) => {
      setEntries(t.entries);
      const readme = t.entries.find((e) => e.path.toLowerCase() === "readme.md") || t.entries.find((e) => e.type === "file");
      if (readme) setPath(readme.path);
    });
  }, [spec.slug]);
  useEffect(() => {
    if (path) api(`/experiments/${spec.slug}/sample/file?path=${encodeURIComponent(path)}`).then(setFile);
  }, [spec.slug, path]);
  const copy = async () => {
    if (!window.confirm("Replace your project with a fresh copy of the sample?")) return;
    await api(`/workspaces/${spec.slug}/reset`, { method: "POST" });
    nav(`/experiments/${spec.slug}/workspace`);
  };
  return (
    <div className="stack">
      <div className="panel">
        <h3 style={{ marginBottom: 6 }}>What to build</h3>
        <Markdown text={spec.project?.interface || ""} />
        <div className="row" style={{ marginTop: 8 }}>
          <a className="btn sm" href={fileUrl(`/experiments/${spec.slug}/sample/download`)}>
            <Download size={13} /> Download sample (.zip)
          </a>
          <button className="btn sm" onClick={copy}>
            Reset my project to the sample
          </button>
          <Link className="btn sm primary" to={`/experiments/${spec.slug}/workspace`}>
            Open my project
          </Link>
        </div>
      </div>
      <div className="workspace" style={{ gridTemplateColumns: "240px 1fr", height: "60vh" }}>
        <div className="panel tight">
          <FileTree entries={entries} selected={path} onSelect={setPath} />
        </div>
        <div className="panel editor">
          <div className="bar">
            <span className="mono">{path || ""}</span>
          </div>
          <div style={{ flex: 1 }}>{file && !file.binary && <CodeEditor path={path} value={file.content} readOnly />}</div>
        </div>
      </div>
    </div>
  );
}

function SubmissionsTab({ spec }) {
  const { isTeacher } = useAuth();
  const [subs, setSubs] = useState([]);
  useEffect(() => {
    api(`/submissions?experiment=${spec.slug}`).then(setSubs);
  }, [spec.slug]);
  if (!subs.length) return <div className="empty">{isTeacher ? "No submissions yet." : "You have not submitted yet. Open My project, test it, then submit."}</div>;
  return (
    <div className="tablewrap" style={{ maxHeight: "70vh" }}>
      <table className="data">
        <thead>
          <tr>
            <th>#</th>
            {isTeacher && <th>Student</th>}
            <th>Submitted</th>
            <th>Files</th>
            <th>Auto score</th>
            <th>Grade</th>
            <th>Feedback</th>
            <th>Published</th>
            {isTeacher && <th></th>}
          </tr>
        </thead>
        <tbody>
          {subs.map((s) => (
            <tr key={s.id}>
              <td>{s.id}</td>
              {isTeacher && <td>{s.name || s.username}</td>}
              <td>{fmtTime(s.created_at)}</td>
              <td>
                {s.file_count} · {fmtBytes(s.bytes)}
              </td>
              <td>{s.auto_score ?? "–"}</td>
              <td>{s.score ?? "–"}</td>
              <td>{s.feedback}</td>
              <td>{s.published ? "yes" : "no"}</td>
              {isTeacher && (
                <td>
                  <Link to={`/teacher/submissions/${s.id}`}>review</Link>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Experiment() {
  const { slug } = useParams();
  const nav = useNavigate();
  const [sp, setSp] = useSearchParams();
  const tab = sp.get("tab") || "guide";
  const stepNo = Number(sp.get("step") || 1);
  const [spec, setSpec] = useState(null);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);
  const loadRuns = useCallback(() => api(`/runs?experiment=${slug}&kind=step&mine=true&limit=300`).then(setRuns), [slug]);
  useEffect(() => {
    api(`/experiments/${slug}`).then(setSpec).catch((e) => setError(e.message));
    loadRuns();
  }, [slug, loadRuns]);
  useEffect(() => {
    const t = setInterval(loadRuns, 10000);
    return () => clearInterval(t);
  }, [loadRuns]);
  const railState = useMemo(() => {
    const st = {};
    for (const r of runs) {
      const cur = st[r.step] || {};
      if (!cur.latest || r.id > cur.latest) cur.latest = r.id;
      if (!cur.status) cur.status = r.status;
      if (r.status === "succeeded" && !cur.metrics) cur.metrics = r.metrics;
      st[r.step] = cur;
    }
    return st;
  }, [runs]);
  if (error) return <main className="page empty">{error}</main>;
  if (!spec) return <main className="page muted">Loading…</main>;
  const step = spec.steps[stepNo - 1] || spec.steps[0];
  const setTab = (t) => (t === "project" ? nav(`/experiments/${slug}/workspace`) : setSp({ tab: t, step: String(stepNo) }));
  return (
    <main className="page">
      <div className="hero">
        <div>
          <div className="small muted">
            <Link to="/">Experiments</Link> / {spec.title}
          </div>
          <h1>{spec.title}</h1>
          <p className="muted" style={{ maxWidth: 760 }}>
            {spec.summary}
          </p>
        </div>
        <div className="tags">
          {spec.tags.map((t) => (
            <span key={t} className="tag">
              {t}
            </span>
          ))}
        </div>
      </div>
      <Rail steps={spec.steps} state={railState} active={tab === "guide" ? stepNo : null} onSelect={(n) => setSp({ tab: "guide", step: String(n) })} />
      <div className="tabs">
        {TABS.map(([k, label]) => (
          <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>
            {label}
          </button>
        ))}
      </div>
      {tab === "guide" && <StepPanel key={step.index} spec={spec} step={step} runs={runs.filter((r) => r.step === step.index)} prevRuns={runs.filter((r) => r.step === step.index - 1 && r.status === "succeeded")} onStarted={loadRuns} />}
      {tab === "materials" && <MaterialsTab spec={spec} />}
      {tab === "sample" && <SampleTab spec={spec} />}
      {tab === "submissions" && <SubmissionsTab spec={spec} />}
      {tab === "guide" && spec.description && (
        <div className="panel" style={{ marginTop: 20 }}>
          <Markdown text={spec.description} />
          {spec.readme && <Markdown text={spec.readme} />}
        </div>
      )}
    </main>
  );
}
