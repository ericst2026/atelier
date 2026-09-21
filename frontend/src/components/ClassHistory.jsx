import React, { useCallback, useEffect, useState } from "react";
import CodeEditor from "./CodeEditor";
import Kpis from "./Kpi";
import { useT } from "../i18n";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";

/** The classes a teacher has held — an admin sees everyone's. Pick one to work
 *  through its four steps, read what each student handed in, and mark it. */
export default function ClassHistory({ onResumed }) {
  const t = useT();
  const [data, setData] = useState(null);
  const [chosen, setChosen] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(
    () =>
      api("/class/history?limit=100")
        .then(setData)
        .catch((e) => setError(e.message)),
    []
  );
  useEffect(() => {
    load();
  }, [load]);
  const resume = async (id) => {
    setError(null);
    try {
      await api(`/class/${id}/resume`, { method: "POST", body: {} });
      load();
      onResumed && onResumed();
    } catch (e) {
      setError(e.message);
    }
  };
  const sessions = data?.sessions || [];
  return (
    <div className="stack">
      {error && <div className="panel" style={{ borderColor: "var(--dup)" }}>{error}</div>}
      {sessions.length === 0 && <div className="empty">{t("class.history.empty")}</div>}
      {sessions.length > 0 && (
        <div className="tablewrap">
          <table className="data">
            <thead>
              <tr>
                <th>{t("class.history.col.class")}</th>
                <th>{t("class.history.col.experiment")}</th>
                <th>{t("class.history.col.owner")}</th>
                <th>{t("class.history.col.started")}</th>
                <th>{t("class.history.col.state")}</th>
                <th>{t("class.history.col.inClass")}</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((s) => (
                <tr key={s.id}>
                  <td>{s.name || <span className="faint">{t("common.unnamed")}</span>}</td>
                  <td>{s.title || s.experiment}</td>
                  <td>{s.teacher}</td>
                  <td>{fmtTime(s.started_at)}</td>
                  <td>
                    {s.ended_at ? (
                      <span className="muted">{t("class.history.ended", { time: fmtTime(s.ended_at) })}</span>
                    ) : s.paused ? (
                      <b style={{ color: "var(--sun)" }}>{t("common.status.paused")}</b>
                    ) : (
                      <b style={{ color: "var(--kept)" }}>{t("common.status.running")}</b>
                    )}
                  </td>
                  <td>{s.members.filter((m) => m.admitted).length}</td>
                  <td className="row" style={{ gap: 6 }}>
                    {s.paused && (
                      <button className="btn sm good" onClick={() => resume(s.id)}>
                        {t("class.history.resume")}
                      </button>
                    )}
                    <button className={`btn sm ${chosen?.id === s.id ? "" : "ghost"}`} onClick={() => setChosen(chosen?.id === s.id ? null : s)}>
                      {chosen?.id === s.id ? t("common.close") : t("common.open")}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {chosen && <ClassWork session={chosen} />}
    </div>
  );
}

/** One class, step by step: who handed in, and what they handed in. */
function ClassWork({ session }) {
  const t = useT();
  const [step, setStep] = useState(1);
  const [subs, setSubs] = useState([]);
  const [open, setOpen] = useState(null);
  const load = useCallback(() => {
    api(`/submissions?experiment=${session.experiment}&step=${step}`)
      .then(setSubs)
      .catch(() => setSubs([]));
  }, [session.experiment, step]);
  useEffect(() => {
    setOpen(null);
    load();
  }, [load]);
  return (
    <div className="panel stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3>
          {session.name ? `${session.name} · ` : ""}
          {session.title || session.experiment} · {session.teacher} · {fmtTime(session.started_at)}
        </h3>
        <div className="ctl">
          {[1, 2, 3, 4].map((n) => (
            <button key={n} className={step === n ? "on" : ""} onClick={() => setStep(n)}>
              {t("common.stepN", { n })}
            </button>
          ))}
        </div>
      </div>
      {subs.length === 0 ? (
        <div className="help">{t("class.history.nobodyHandedIn", { n: step })}</div>
      ) : (
        <div className="stack" style={{ gap: 4 }}>
          {subs.map((s) => (
            <div key={s.id} className="row" style={{ justifyContent: "space-between" }}>
              <span>
                {s.name || s.username} <span className="faint small">{t("class.history.handedIn", { time: fmtTime(s.created_at) })}</span>
                {s.score !== null && s.score !== undefined && <span className="tag">{s.score}</span>}
              </span>
              <button className={`btn sm ${open === s.id ? "" : "ghost"}`} onClick={() => setOpen(open === s.id ? null : s.id)}>
                {open === s.id ? t("common.close") : t("class.history.readIt")}
              </button>
            </div>
          ))}
        </div>
      )}
      {open && <Submission id={open} onGraded={load} />}
    </div>
  );
}

/** What one student handed in: their code, what it produced, and a mark. */
function Submission({ id, onGraded }) {
  const t = useT();
  const [sub, setSub] = useState(null);
  const [file, setFile] = useState(null);
  const [runs, setRuns] = useState([]);
  const [score, setScore] = useState("");
  const [feedback, setFeedback] = useState("");
  const [msg, setMsg] = useState(null);
  useEffect(() => {
    api(`/submissions/${id}`).then((s) => {
      setSub(s);
      setScore(s.score ?? "");
      setFeedback(s.feedback || "");
    });
    api(`/submissions/${id}/tree`)
      .then((t) => {
        const f = (t.entries || []).find((e) => e.type === "file" && e.path.endsWith(".py"));
        if (f) api(`/submissions/${id}/file?path=${encodeURIComponent(f.path)}`).then(setFile);
      })
      .catch(() => {});
    api(`/submissions/${id}/runs`).then(setRuns).catch(() => setRuns([]));
  }, [id]);
  const grade = async () => {
    const s = await api(`/submissions/${id}/grade`, { method: "POST", body: { score: score === "" ? null : Number(score), feedback } });
    setSub(s);
    setMsg(t("class.history.marked"));
    onGraded && onGraded();
    setTimeout(() => setMsg(null), 2000);
  };
  if (!sub) return <div className="help">{t("common.reading")}</div>;
  const run = runs[0];
  return (
    <div className="inset stack" style={{ padding: 12, gap: 10 }}>
      <b>
        {sub.name || sub.username} · {t("common.stepN", { n: sub.step })}
      </b>
      <div className="grid2">
        <div className="stack" style={{ gap: 6 }}>
          <div className="small muted">{t("class.history.theirCode")}</div>
          <div className="inset" style={{ height: 360, overflow: "hidden" }}>
            {file && !file.binary ? (
              <CodeEditor path="step.py" value={file.content} readOnly />
            ) : (
              <div className="help" style={{ padding: 10 }}>
                {t("class.history.noCode")}
              </div>
            )}
          </div>
        </div>
        <div className="stack" style={{ gap: 6 }}>
          <div className="small muted">{t("class.history.produced")}</div>
          {run ? <Kpis metrics={run.metrics} /> : <div className="help">{t("class.history.notRun")}</div>}
          <label className="field">
            <span>{t("class.history.mark")}</span>
            <input type="number" min={0} max={100} value={score} onChange={(e) => setScore(e.target.value)} />
          </label>
          <label className="field">
            <span>{t("class.history.feedback")}</span>
            <textarea value={feedback} onChange={(e) => setFeedback(e.target.value)} />
          </label>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn good" onClick={grade}>
              {t("class.history.saveMark")}
            </button>
            {msg && <span className="small" style={{ color: "var(--kept)" }}>{msg}</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
