import React, { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Download, FilePlus, Play, RotateCcw, Save, Send, Trash2, Upload, Wand2 } from "lucide-react";
import CodeEditor from "../components/CodeEditor";
import FileTree from "../components/FileTree";
import LogView from "../components/LogView";
import ResultsView from "../components/ResultsView";
import RunStatus, { StatusPill } from "../components/RunStatus";
import { api, fileUrl } from "../lib/api";
import { fmtBytes, fmtTime } from "../lib/format";
import { useRunStream } from "../lib/runs";

export default function Workspace() {
  const { slug } = useParams();
  const [info, setInfo] = useState(null);
  const [path, setPath] = useState(null);
  const [content, setContent] = useState("");
  const [dirty, setDirty] = useState(false);
  const [binary, setBinary] = useState(false);
  const [command, setCommand] = useState("");
  const [gpus, setGpus] = useState(1);
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState(null);
  const upload = useRef(null);
  const stream = useRunStream(selected);

  const refresh = useCallback(() => api(`/workspaces/${slug}`).then((i) => {
    setInfo(i);
    setCommand((c) => c || i.run_command);
    setGpus(i.entry ? 1 : 0);
  }), [slug]);
  const loadRuns = useCallback(() => api(`/workspaces/${slug}/runs`).then(setRuns), [slug]);
  useEffect(() => {
    refresh();
    loadRuns();
  }, [refresh, loadRuns]);
  useEffect(() => {
    const t = setInterval(loadRuns, 8000);
    return () => clearInterval(t);
  }, [loadRuns]);
  useEffect(() => {
    if (!info || path) return;
    const first = info.entries.find((e) => e.path === info.entry) || info.entries.find((e) => e.path.toLowerCase() === "readme.md") || info.entries.find((e) => e.type === "file");
    if (first) setPath(first.path);
  }, [info, path]);
  useEffect(() => {
    if (!path) return;
    api(`/workspaces/${slug}/file?path=${encodeURIComponent(path)}`).then((f) => {
      setBinary(!!f.binary);
      setContent(f.content || "");
      setDirty(false);
    });
  }, [slug, path]);

  const flash = (m, ok = true) => {
    setMsg({ m, ok });
    setTimeout(() => setMsg(null), 3500);
  };
  const save = async () => {
    if (!path || binary) return;
    await api(`/workspaces/${slug}/file`, { method: "PUT", body: { path, content } });
    setDirty(false);
    flash(`Saved ${path}`);
    refresh();
  };
  const newFile = async () => {
    const p = window.prompt("New file path (e.g. project/utils.py)");
    if (!p) return;
    await api(`/workspaces/${slug}/file`, { method: "PUT", body: { path: p, content: "" } });
    await refresh();
    setPath(p);
  };
  const del = async () => {
    if (!path || !window.confirm(`Delete ${path}?`)) return;
    await api(`/workspaces/${slug}/file?path=${encodeURIComponent(path)}`, { method: "DELETE" });
    setPath(null);
    refresh();
  };
  const reset = async () => {
    if (!window.confirm("Replace everything with the sample project?")) return;
    await api(`/workspaces/${slug}/reset`, { method: "POST" });
    setPath(null);
    refresh();
  };
  const onUpload = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const form = new FormData();
    form.append("file", f);
    try {
      await api(`/workspaces/${slug}/upload?replace=${window.confirm("Replace the whole project with the archive? (Cancel merges it in)") ? "true" : "false"}`, { method: "POST", form });
      flash("Uploaded");
      setPath(null);
      refresh();
    } catch (err) {
      flash(err.message, false);
    }
    e.target.value = "";
  };
  const startRun = async (kind) => {
    try {
      if (dirty) await save();
      const run = kind === "test" ? await api(`/workspaces/${slug}/test`, { method: "POST" }) : await api(`/workspaces/${slug}/run`, { method: "POST", body: { command, gpus } });
      setSelected(run.id);
      loadRuns();
    } catch (err) {
      flash(err.message, false);
    }
  };
  const submit = async () => {
    if (!window.confirm("Submit a snapshot of your project? The grader runs automatically.")) return;
    try {
      if (dirty) await save();
      const s = await api(`/experiments/${slug}/submissions`, { method: "POST", body: { note } });
      flash(`Submitted #${s.id}. It is being graded.`);
      setNote("");
      if (s.last_test_run_id) setSelected(s.last_test_run_id);
      loadRuns();
    } catch (err) {
      flash(err.message, false);
    }
  };
  if (!info) return <main className="page muted">Loading…</main>;
  return (
    <main className="page wide">
      <div className="row" style={{ marginBottom: 10 }}>
        <div className="small muted">
          <Link to="/">Experiments</Link> / <Link to={`/experiments/${slug}`}>{slug}</Link> / my project
        </div>
        <span className="spacer" />
        <span className="small faint">
          {info.stats.files} files · {fmtBytes(info.stats.bytes)}
        </span>
        <button className="btn sm" onClick={newFile}>
          <FilePlus size={13} /> New file
        </button>
        <button className="btn sm" onClick={() => upload.current?.click()}>
          <Upload size={13} /> Upload zip
        </button>
        <input ref={upload} type="file" accept=".zip" style={{ display: "none" }} onChange={onUpload} />
        <a className="btn sm" href={fileUrl(`/workspaces/${slug}/download`)}>
          <Download size={13} /> Download
        </a>
        <button className="btn sm" onClick={reset}>
          <RotateCcw size={13} /> Reset to sample
        </button>
      </div>
      <div className="workspace">
        <div className="panel tight">
          <FileTree entries={info.entries} selected={path} onSelect={setPath} />
        </div>
        <div className="panel editor">
          <div className="bar">
            <span className="mono">{path || "no file selected"}</span>
            {dirty && <span className="faint">unsaved</span>}
            <span className="spacer" />
            <button className="btn sm" onClick={del} disabled={!path}>
              <Trash2 size={13} />
            </button>
            <button className="btn sm good" onClick={save} disabled={!dirty || binary}>
              <Save size={13} /> Save
            </button>
          </div>
          <div style={{ flex: 1, minHeight: 320 }}>{path && (binary ? <div className="empty">Binary file — download it instead.</div> : <CodeEditor path={path} value={content} onChange={(v) => { setContent(v); setDirty(true); }} onSave={save} />)}</div>
        </div>
        <div className="panel stack" style={{ gap: 12 }}>
          <div className="stack" style={{ gap: 6 }}>
            <label className="field">
              <span>Run a command in the project</span>
              <input type="text" value={command} onChange={(e) => setCommand(e.target.value)} className="mono" />
            </label>
            <div className="row" style={{ flexWrap: "nowrap" }}>
              <label className="field" style={{ width: 90 }}>
                <span>GPUs</span>
                <input type="number" min={0} max={8} value={gpus} onChange={(e) => setGpus(Number(e.target.value))} />
              </label>
              <button className="btn primary" onClick={() => startRun("run")} style={{ marginTop: 18 }}>
                <Play size={14} /> Run
              </button>
              <button className="btn" onClick={() => startRun("test")} style={{ marginTop: 18 }} title="Run the experiment grader on your current project">
                <Wand2 size={14} /> Test
              </button>
            </div>
          </div>
          <div className="stack" style={{ gap: 6 }}>
            <label className="field">
              <span>Submit</span>
              <textarea value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note for the teacher (optional)" style={{ minHeight: 50 }} />
            </label>
            <button className="btn good" onClick={submit}>
              <Send size={14} /> Submit project
            </button>
          </div>
          {msg && <div className={`small ${msg.ok ? "" : ""}`} style={{ color: msg.ok ? "var(--kept)" : "var(--dup)" }}>{msg.m}</div>}
          <div className="stack" style={{ gap: 4 }}>
            <div className="small muted">Recent runs</div>
            {runs.length === 0 && <div className="faint small">nothing yet</div>}
            {runs.slice(0, 10).map((r) => (
              <button key={r.id} className={`btn sm ${selected === r.id ? "" : "ghost"}`} style={{ justifyContent: "flex-start" }} onClick={() => setSelected(r.id)}>
                <StatusPill status={r.status} /> #{r.id} {r.kind === "test" ? "test" : ""} <span className="faint" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</span> <span className="faint">{fmtTime(r.created_at)}</span>
              </button>
            ))}
          </div>
          {selected && (
            <div className="stack" style={{ gap: 8 }}>
              <RunStatus run={stream.run} progress={stream.progress} onCancel={stream.cancel} />
              <LogView lines={stream.lines} />
            </div>
          )}
        </div>
      </div>
      {stream.result && (
        <div className="panel" style={{ marginTop: 14 }}>
          <ResultsView result={stream.result} runId={selected} />
        </div>
      )}
    </main>
  );
}
