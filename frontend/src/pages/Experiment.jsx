import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { useT } from "../i18n";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtTime } from "../lib/format";
import { liveCharts, useRunStream } from "../lib/runs";


/** One step's panel. `draft` is what this step looked like the last time it was
 *  open, and `remember` keeps it that way: moving between the step tabs leaves
 *  each step's settings, the run being read and an open editor as they were. */
function StepPanel({ spec, step, runs, prevRuns, onStarted, locked, classSession, draft, remember }) {
  const { isTeacher } = useAuth();
  const t = useT();
  const defaults = useMemo(
    () => ({ ...Object.fromEntries((step.params || []).map((p) => [p.key, p.default])), ...(locked || {}) }),
    [step, locked]
  );
  const was = draft || {};
  // whatever the class fixes wins over a remembered setting
  const [params, setParams] = useState(() => ({ ...(was.params || defaults), ...(locked || {}) }));
  const [parentId, setParentId] = useState(was.parentId ?? (prevRuns[0]?.id || null));
  const [gpus, setGpus] = useState(was.gpus ?? step.gpus);
  const [selected, setSelected] = useState(was.selected ?? (runs[0]?.id || null));
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [showLog, setShowLog] = useState(was.showLog ?? false);
  // a step the student may re-implement: "standard" runs the shipped script, "own" their file
  const [codeSource, setCodeSource] = useState(was.codeSource || "standard");
  const [code, setCode] = useState(was.code ?? null);
  const [codeSaved, setCodeSaved] = useState(was.codeSaved ?? true);
  const [codeNote, setCodeNote] = useState(was.codeNote || "");
  useEffect(() => {
    remember({ params, parentId, gpus, selected, showLog, codeSource, code, codeSaved, codeNote });
  });
  useEffect(() => {
    // the file, or the prototype to start from when there is none yet
    if (codeSource !== "own" || code !== null) return;
    api(`/experiments/${spec.slug}/steps/${step.index}/code`)
      .then((r) => {
        setCode(r.code);
        setCodeSaved(r.saved);
        if (!r.saved) setCodeNote(t("experiment.code.prototypeNote"));
      })
      .catch((e) => setError(e.message));
  }, [codeSource, code, spec.slug, step.index]);
  const saveCode = async () => {
    try {
      const r = await api(`/experiments/${spec.slug}/steps/${step.index}/code`, { method: "PUT", body: { code } });
      setCodeSaved(true);
      setCodeNote(r.error ? t("experiment.code.savedWithError", { error: r.error }) : t("common.saved"));
    } catch (e) {
      setError(e.message);
    }
  };
  const resetCode = async () => {
    if (!window.confirm(t("experiment.code.resetConfirm"))) return;
    const r = await api(`/experiments/${spec.slug}/steps/${step.index}/code/reset`, { method: "POST" });
    setCode(r.code);
    setCodeSaved(true);
    setCodeNote(t("experiment.code.reset"));
  };
  const submitCode = async () => {
    if (!codeSaved) await saveCode();
    try {
      const sub = await api(`/experiments/${spec.slug}/submissions`, { method: "POST", body: { step: step.index, note: "" } });
      setCodeNote(t("experiment.code.handedIn", { id: sub.id }));
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
      const run = await api(`/experiments/${spec.slug}/steps/${step.index}/runs`, { method: "POST", body: { params, parent_run_id: step.needs_previous ? parentId : null, gpus: isTeacher ? gpus : null, code_source: codeSource, class_session: classSession } });
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
  const live = liveCharts(stream.live, stream.liveXLabel);
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
              <span>{t("experiment.buildOn.label")}</span>
              <select value={parentId ?? ""} onChange={(e) => setParentId(Number(e.target.value) || null)}>
                <option value="">{t("experiment.buildOn.pick", { n: step.index - 1 })}</option>
                {prevRuns.map((r) => (
                  <option key={r.id} value={r.id}>
                    #{r.id} · {fmtTime(r.created_at)} · {(r.metrics || []).slice(0, 2).map((m) => `${m.label} ${typeof m.value === "number" ? Number(m.value.toFixed(3)) : m.value}`).join(" · ")}
                  </option>
                ))}
              </select>
              {prevRuns.length === 0 && <span className="help">{t("experiment.buildOn.finishFirst", { n: step.index - 1 })}</span>}
            </label>
          )}
          {step.own_code && (
            <label className="field">
              <span>{t("experiment.codeSource.label")}</span>
              <select value={codeSource} onChange={(e) => setCodeSource(e.target.value)} disabled={busy}>
                <option value="standard">{t("experiment.codeSource.standard")}</option>
                <option value="own">{t("experiment.codeSource.own")}</option>
              </select>
              <span className="help">{codeSource === "own" ? t("experiment.codeSource.ownHelp") : t("experiment.codeSource.standardHelp")}</span>
            </label>
          )}
          <ParamsForm params={step.params || []} values={params} onChange={setParams} disabled={busy} experiment={spec.slug} step={step.index} locked={locked} />
          {isTeacher && step.gpus > 0 && (
            <label className="field">
              <span>{t("experiment.gpus.label")}</span>
              <input type="number" min={0} max={8} value={gpus} onChange={(e) => setGpus(Number(e.target.value))} />
              <span className="help">{t("experiment.gpus.studentsGet", { n: step.gpus })}</span>
            </label>
          )}
          {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
          <button className="btn primary" onClick={start} disabled={busy || (step.needs_previous && !parentId)}>
            <Play size={14} /> {busy ? t("experiment.run.starting") : codeSource === "own" ? t("experiment.run.own", { n: step.index }) : t("experiment.run.standard", { n: step.index })}
          </button>
          <div className="help">
            {step.gpus ? `${step.gpus} GPU` : "CPU"} · {t("experiment.run.timeout", { min: step.timeout_min })}
          </div>
        </div>
        {runs.length > 0 && (
          <div className="panel tight">
            <div className="small muted" style={{ marginBottom: 6 }}>
              {t("experiment.runs.title")}
            </div>
            <div className="stack" style={{ gap: 4 }}>
              {runs.slice(0, 12).map((r) => (
                <button key={r.id} className={`btn sm ${selected === r.id ? "" : "ghost"}`} style={{ justifyContent: "flex-start" }} onClick={() => setSelected(r.id)}>
                  <StatusPill status={r.id === stream.run?.id && stream.status ? stream.status : r.status} /> #{r.id} <span className="faint">{fmtTime(r.created_at)}</span>
                  {ownRun(r) && <span className="tag">{t("experiment.runs.ownTag")}</span>}
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
              <h3>{t("experiment.code.title", { n: step.index })}</h3>
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm" onClick={saveCode} disabled={codeSaved}>
                  {codeSaved ? t("experiment.code.saved") : t("common.save")}
                </button>
                <button className="btn sm ghost" onClick={resetCode}>
                  {t("experiment.code.prototype")}
                </button>
                <button className="btn sm good" onClick={submitCode}>
                  {t("experiment.code.handIn")}
                </button>
              </div>
            </div>
            <div className="inset" style={{ height: 420, overflow: "hidden" }}>
              <CodeEditor
                path="step.py"
                value={code ?? t("experiment.code.loading")}
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
            {t("experiment.showing.run", { id: shown.id, code: ownRun(shown) ? t("experiment.showing.own") : t("experiment.showing.standard"), time: fmtTime(shown.created_at) })}
            {editing && !ownRun(shown) && <span className="faint">{t("experiment.showing.ownNotRun")}</span>}
          </div>
        )}
        {!selected && <div className="empty">{t("experiment.empty")}</div>}
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
                {showLog ? t("experiment.log.hide") : t("experiment.log.show")}
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
  const t = useT();
  const stepNo = Number(sp.get("step") || 1);
  const [spec, setSpec] = useState(null);
  const [runs, setRuns] = useState([]);
  const [cls, setCls] = useState(null);
  const [error, setError] = useState(null);
  // what each step looked like when you last left it, while this page is open
  const drafts = useRef(new Map());
  const loadRuns = useCallback(() => api(`/runs?experiment=${slug}&kind=step&mine=true&limit=300`).then(setRuns), [slug]);
  useEffect(() => {
    api(`/experiments/${slug}`).then(setSpec).catch((e) => setError(e.message));
    api("/class").then(setCls).catch(() => {});
    loadRuns();
  }, [slug, loadRuns]);
  useEffect(() => {
    const timer = setInterval(loadRuns, 10000);
    return () => clearInterval(timer);
  }, [loadRuns]);
  // the same experiment is one page whether you are in class or working alone, so
  // the link says which: ?class=<id> is the class, and the runs shown are that
  // class's. Without it this is your own work, and the class's runs stay out of it.
  const classId = Number(sp.get("class")) || null;
  const session = cls?.running && cls.session?.experiment === slug ? cls.session : null;
  // the class's teacher and the people they let in; nobody else, admins included,
  // can view this experiment as the class — it is not their class
  const member = Boolean(session && (session.me?.mine || session.me?.admitted));
  const inClass = Boolean(classId && session && session.id === classId && member);
  const shownRuns = useMemo(
    () => runs.filter((r) => (inClass ? (r.inputs || {}).class_session === classId : !(r.inputs || {}).class_session)),
    [runs, inClass, classId]
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
  // in class, what the class was started from is what runs — for its teacher as well
  const lockedFor = (st) => {
    if (!inClass) return null;
    const keys = new Set((st.params || []).map((p) => p.key));
    const fixed = Object.entries(session.params || {}).filter(([k]) => keys.has(k));
    return fixed.length ? Object.fromEntries(fixed) : null;
  };
  if (error) return <main className="page empty">{error}</main>;
  if (!spec) return <main className="page muted">{t("common.loading")}</main>;
  const step = spec.steps[stepNo - 1] || spec.steps[0];
  // each step, in class and on your own, keeps its own settings while you are on
  // this page: switching tabs is reading another step, not starting it over
  const scope = `${step.index}-${inClass ? classId : "own"}`;
  return (
    <main className="page">
      <div className="hero">
        <div>
          <div className="small muted">
            <Link to="/">{t("experiment.breadcrumb")}</Link> / {spec.title}
          </div>
          <h1>{spec.title}</h1>
          <p className="muted" style={{ maxWidth: 760 }}>
            {spec.summary}
          </p>
        </div>
        <div className="tags">
          {spec.tags.map((tag) => (
            <span key={tag} className="tag">
              {tag}
            </span>
          ))}
        </div>
      </div>
      {inClass && (
        <div className="tip" style={{ marginBottom: 12 }}>
          {t("experiment.class.inClass.before", { teacher: session.teacher })}
          <Link to={`/experiments/${slug}`}>{t("experiment.class.inClass.link")}</Link>
          {t("experiment.class.inClass.after")}
        </div>
      )}
      {!inClass && session && member && (
        <div className="tip" style={{ marginBottom: 12 }}>
          {t("experiment.class.running.before", { teacher: session.teacher })}
          <Link to={`/experiments/${slug}?class=${session.id}`}>{t("experiment.class.running.link")}</Link>
          {t("experiment.class.running.after")}
        </div>
      )}
      {classId && !inClass && (
        <div className="tip" style={{ marginBottom: 12 }}>
          {t("experiment.class.notIn")}
        </div>
      )}
      <Rail steps={spec.steps} state={railState} active={stepNo} onSelect={(n) => setSp(classId ? { step: String(n), class: String(classId) } : { step: String(n) })} />
      <StepPanel
        key={scope}
        draft={drafts.current.get(scope)}
        remember={(d) => drafts.current.set(scope, d)}
        spec={spec}
        step={step}
        runs={shownRuns.filter((r) => r.step === step.index)}
        prevRuns={shownRuns.filter((r) => r.step === step.index - 1 && r.status === "succeeded")}
        onStarted={loadRuns}
        locked={lockedFor(step)}
        classSession={inClass ? session.id : null}
      />
      {spec.description && (
        <div className="panel" style={{ marginTop: 20 }}>
          <Markdown text={spec.description} />
          {spec.readme && <Markdown text={spec.readme} />}
        </div>
      )}
    </main>
  );
}
