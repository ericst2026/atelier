import React, { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Download } from "lucide-react";
import CodeEditor from "../components/CodeEditor";
import FileTree from "../components/FileTree";
import LogView from "../components/LogView";
import ResultsView from "../components/ResultsView";
import RunStatus, { StatusPill } from "../components/RunStatus";
import { api, fileUrl } from "../lib/api";
import { fmtBytes, fmtTime } from "../lib/format";
import { useRunStream } from "../lib/runs";

export default function SubmissionReview() {
  const { id } = useParams();
  const [sub, setSub] = useState(null);
  const [entries, setEntries] = useState([]);
  const [path, setPath] = useState(null);
  const [file, setFile] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [score, setScore] = useState("");
  const [feedback, setFeedback] = useState("");
  const [published, setPublished] = useState(false);
  const [msg, setMsg] = useState(null);
  const stream = useRunStream(selected);
  const load = useCallback(() => {
    api(`/submissions/${id}`).then((s) => {
      setSub(s);
      setScore(s.score ?? "");
      setFeedback(s.feedback || "");
      setPublished(!!s.published);
    });
    api(`/submissions/${id}/tree`).then((t) => {
      setEntries(t.entries);
      if (!path) setPath(t.entries.find((e) => e.type === "file" && e.path.endsWith(".py"))?.path || t.entries.find((e) => e.type === "file")?.path || null);
    });
    api(`/submissions/${id}/runs`).then((r) => {
      setRuns(r);
      if (r.length && !selected) setSelected(r[0].id);
    });
  }, [id]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(load, [load]);
  useEffect(() => {
    if (path) api(`/submissions/${id}/file?path=${encodeURIComponent(path)}`).then(setFile);
  }, [id, path]);
  const grade = async () => {
    const s = await api(`/submissions/${id}/grade`, { method: "POST", body: { score: score === "" ? null : Number(score), feedback, published } });
    setSub(s);
    setMsg("Saved");
    setTimeout(() => setMsg(null), 2500);
  };
  if (!sub) return <main className="page muted">Loading…</main>;
  return (
    <main className="page wide">
      <div className="row" style={{ marginBottom: 10 }}>
        <div>
          <div className="small muted">
            <Link to="/teacher">Teacher</Link> / submission {sub.id}
          </div>
          <h1>
            {sub.name || sub.username} · {sub.experiment}
            {sub.step ? ` · step ${sub.step}` : ""}
          </h1>
          <div className="muted small">
            {fmtTime(sub.created_at)} · {sub.file_count} files · {fmtBytes(sub.bytes)} · sha256 {sub.sha256.slice(0, 12)}
            {sub.note && <div>Note: {sub.note}</div>}
          </div>
        </div>
        <span className="spacer" />
        <a className="btn sm" href={fileUrl(`/submissions/${id}/download`)}>
          <Download size={13} /> Download
        </a>
      </div>
      <div className="workspace">
        <div className="panel tight">
          <FileTree entries={entries} selected={path} onSelect={setPath} />
        </div>
        <div className="panel editor">
          <div className="bar">
            <span className="mono">{path}</span>
          </div>
          <div style={{ flex: 1, minHeight: 320 }}>{file && !file.binary && <CodeEditor path={path} value={file.content} readOnly />}</div>
        </div>
        <div className="panel stack">
          <div className="stack" style={{ gap: 6 }}>
            <label className="field">
              <span>Score (0–100)</span>
              <input type="number" min={0} max={100} value={score} onChange={(e) => setScore(e.target.value)} />
            </label>
            <label className="field">
              <span>Feedback</span>
              <textarea value={feedback} onChange={(e) => setFeedback(e.target.value)} />
            </label>
            <button className="btn good" onClick={grade}>
              Save grade
            </button>
            {msg && <div className="small" style={{ color: "var(--kept)" }}>{msg}</div>}
          </div>
          <div className="stack" style={{ gap: 4 }}>
            <div className="small muted">What this code produced when they ran it</div>
            {runs.map((r) => (
              <button key={r.id} className={`btn sm ${selected === r.id ? "" : "ghost"}`} style={{ justifyContent: "flex-start" }} onClick={() => setSelected(r.id)}>
                <StatusPill status={r.status} /> #{r.id} <span className="faint">{fmtTime(r.created_at)}</span> {(r.metrics || []).find((m) => m.key === "score") && <b>score {(r.metrics || []).find((m) => m.key === "score").value}</b>}
              </button>
            ))}
          </div>
          {selected && (
            <>
              <RunStatus run={stream.run} progress={stream.progress} onCancel={stream.cancel} />
              <LogView lines={stream.lines} />
            </>
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
