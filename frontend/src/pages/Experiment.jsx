import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { Play } from "lucide-react";
import ChartCard from "../components/ChartCard";
import CodeEditor from "../components/CodeEditor";
import LogView from "../components/LogView";
import Markdown from "../components/Markdown";
import ParamsForm from "../components/ParamsForm";
import Rail from "../components/Rail";
import ResultsView from "../components/ResultsView";
import RunStatus, { StatusPill } from "../components/RunStatus";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtTime } from "../lib/format";
import { liveCharts, useRunStream } from "../lib/runs";


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
  // a step the student may re-implement: "standard" runs the shipped script, "own" their file
  const [codeSource, setCodeSource] = useState("standard");
  const [code, setCode] = useState(null);
  const [codeSaved, setCodeSaved] = useState(true);
  const [codeNote, setCodeNote] = useState("");
  useEffect(() => {
    setParams(defaults);
    setGpus(step.gpus);
    setSelected(runs[0]?.id || null);
    setError(null);
    setCodeSource("standard");
    setCode(null);
    setCodeNote("");
  }, [step.index]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    // the file, or the prototype to start from when there is none yet
    if (codeSource !== "own" || code !== null) return;
    api(`/experiments/${spec.slug}/steps/${step.index}/code`)
      .then((r) => {
        setCode(r.code);
        setCodeSaved(r.saved);
        if (!r.saved) setCodeNote("This is a prototype: the inputs and outputs are filled in, the work is yours.");
      })
      .catch((e) => setError(e.message));
  }, [codeSource, code, spec.slug, step.index]);
  const saveCode = async () => {
    try {
      const r = await api(`/experiments/${spec.slug}/steps/${step.index}/code`, { method: "PUT", body: { code } });
      setCodeSaved(true);
      setCodeNote(r.error ? `Saved, but Python cannot read it yet — ${r.error}` : "Saved.");
    } catch (e) {
      setError(e.message);
    }
  };
  const resetCode = async () => {
    if (!window.confirm("Replace your code for this step with the prototype? What you wrote is lost.")) return;
    const r = await api(`/experiments/${spec.slug}/steps/${step.index}/code/reset`, { method: "POST" });
    setCode(r.code);
    setCodeSaved(true);
    setCodeNote("Back to the prototype.");
  };
  const submitCode = async () => {
    if (!codeSaved) await saveCode();
    try {
      const sub = await api(`/experiments/${spec.slug}/submissions`, { method: "POST", body: { step: step.index, note: "" } });
      setCodeNote(`Handed in to your teacher as submission #${sub.id}.`);
    } catch (e) {
      setError(e.message);
    }
  };
  useEffect(() => {
    if (!parentId && prevRuns[0]) setParentId(prevRuns[0].id);
  }, [prevRuns, parentId]);
  useEffect(() => {
    // the list reloads every few seconds: keep showing what the student picked,
    // and fall back to the newest run only when theirs is no longer there
    if (runs.length === 0) return;
    if (!selected || !runs.some((r) => r.id === selected)) setSelected(runs[0].id);
  }, [runs, selected]);
  const stream = useRunStream(selected);
  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      const run = await api(`/experiments/${spec.slug}/steps/${step.index}/runs`, { method: "POST", body: { params, parent_run_id: step.needs_previous ? parentId : null, gpus: isTeacher ? gpus : null, code_source: codeSource } });
      setSelected(run.id);
      setShowLog(true);
      onStarted(run);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const editing = step.own_code && codeSource === "own";
  const ownRun = (r) => (r.inputs || {}).code_source === "own";
  const shown = runs.find((r) => r.id === selected);
  const finished = stream.status === "succeeded" || stream.status === "failed" || stream.status === "cancelled";
  const live = liveCharts(stream.live);
  return (
    <div className={`steplayout ${editing ? "withcode" : ""}`}>
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
          {step.own_code && (
            <label className="field">
              <span>Code for this step</span>
              <select value={codeSource} onChange={(e) => setCodeSource(e.target.value)} disabled={busy}>
                <option value="standard">the standard code</option>
                <option value="own">my own code</option>
              </select>
              <span className="help">{codeSource === "own" ? "Your file runs with the same params and inputs as the standard one." : "The implementation this experiment ships with."}</span>
            </label>
          )}
          <ParamsForm params={step.params || []} values={params} onChange={setParams} disabled={busy} experiment={spec.slug} step={step.index} />
          {isTeacher && step.gpus > 0 && (
            <label className="field">
              <span>GPUs for this run</span>
              <input type="number" min={0} max={8} value={gpus} onChange={(e) => setGpus(Number(e.target.value))} />
              <span className="help">Students always get {step.gpus}.</span>
            </label>
          )}
          {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
          <button className="btn primary" onClick={start} disabled={busy || (step.needs_previous && !parentId)}>
            <Play size={14} /> {busy ? "Starting…" : codeSource === "own" ? `Run my code for step ${step.index}` : `Run step ${step.index}`}
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
                  {ownRun(r) && <span className="tag">own</span>}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
      {editing && (
        <div className="stack">
          <div className="panel stack" style={{ gap: 8 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>Your code · step {step.index}</h3>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm" onClick={saveCode} disabled={codeSaved}>
                  {codeSaved ? "Saved" : "Save"}
                </button>
                <button className="btn sm ghost" onClick={resetCode}>
                  Prototype
                </button>
                <button className="btn sm good" onClick={submitCode}>
                  Hand in
                </button>
              </div>
            </div>
            <div className="inset" style={{ height: 420, overflow: "hidden" }}>
              <CodeEditor
                path="step.py"
                value={code ?? "loading…"}
                onChange={(v) => {
                  setCode(v);
                  setCodeSaved(false);
                }}
                onSave={saveCode}
              />
            </div>
            {codeNote && <div className="help">{codeNote}</div>}
          </div>
        </div>
      )}
      <div className="stack">
        {shown && (
          <div className="row small muted" style={{ gap: 8 }}>
            Showing run #{shown.id} · {ownRun(shown) ? "your own code" : "the standard code"} · {fmtTime(shown.created_at)}
            {editing && !ownRun(shown) && <span className="faint">— your own code has not run yet, so this is the standard one</span>}
          </div>
        )}
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


export default function Experiment() {
  const { slug } = useParams();
  const [sp, setSp] = useSearchParams();
  const stepNo = Number(sp.get("step") || 1);
  const [spec, setSpec] = useState(null);
  const [runs, setRuns] = useState([]);
  const [cls, setCls] = useState(null);
  const [error, setError] = useState(null);
  const loadRuns = useCallback(() => api(`/runs?experiment=${slug}&kind=step&mine=true&limit=300`).then(setRuns), [slug]);
  useEffect(() => {
    api(`/experiments/${slug}`).then(setSpec).catch((e) => setError(e.message));
    api("/class").then(setCls).catch(() => {});
    loadRuns();
  }, [slug, loadRuns]);
  useEffect(() => {
    const t = setInterval(loadRuns, 10000);
    return () => clearInterval(t);
  }, [loadRuns]);
  // a class starts afresh: while this experiment is the one being taught, runs from
  // before the class began are left out, so nobody works from last week's results
  const session = cls?.running && cls.session?.experiment === slug ? cls.session : null;
  const shownRuns = useMemo(
    () => (session ? runs.filter((r) => new Date(`${r.created_at}Z`) >= new Date(`${session.started_at}Z`)) : runs),
    [runs, session]
  );
  const railState = useMemo(() => {
    const st = {};
    for (const r of shownRuns) {
      const cur = st[r.step] || {};
      if (!cur.latest || r.id > cur.latest) cur.latest = r.id;
      if (!cur.status) cur.status = r.status;
      if (r.status === "succeeded" && !cur.metrics) cur.metrics = r.metrics;
      st[r.step] = cur;
    }
    return st;
  }, [shownRuns]);
  if (error) return <main className="page empty">{error}</main>;
  if (!spec) return <main className="page muted">Loading…</main>;
  const step = spec.steps[stepNo - 1] || spec.steps[0];
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
      <Rail steps={spec.steps} state={railState} active={stepNo} onSelect={(n) => setSp({ step: String(n) })} />
      <StepPanel key={step.index} spec={spec} step={step} runs={shownRuns.filter((r) => r.step === step.index)} prevRuns={shownRuns.filter((r) => r.step === step.index - 1 && r.status === "succeeded")} onStarted={loadRuns} />
      {spec.description && (
        <div className="panel" style={{ marginTop: 20 }}>
          <Markdown text={spec.description} />
          {spec.readme && <Markdown text={spec.readme} />}
        </div>
      )}
    </main>
  );
}
