import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ChartCard from "../components/ChartCard";
import Kpis from "../components/Kpi";
import Markdown from "../components/Markdown";
import { wsUrl } from "../lib/api";
import { fmtDuration, fmtNum, fmtTime } from "../lib/format";
import { useSocket } from "../lib/ws";

function Clock() {
  const [t, setT] = useState(new Date());
  useEffect(() => {
    const i = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(i);
  }, []);
  return <span className="clock">{`${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}`}</span>;
}

function Live({ data }) {
  const gpus = data.gpus?.gpus || [];
  return (
    <>
      <div className="gpugrid">
        {Array.from({ length: data.gpu_count }, (_, i) => gpus[i]).map((g, i) => (
          <div key={i} className={`gpu ${g?.job ? "busy" : ""}`}>
            <div className="u">{g ? `${g.util}%` : "–"}</div>
            <div className="m">
              GPU {i}
              {g ? ` · ${(g.mem_used / 1e9).toFixed(0)} / ${(g.mem_total / 1e9).toFixed(0)} GB · ${g.temp}°C` : " · no data"}
            </div>
            <div className="bar">
              <i style={{ width: `${g ? g.util : 0}%` }} />
            </div>
            <div className="m">{g?.job ? `${g.job.user} · ${g.job.experiment}` : "free"}</div>
          </div>
        ))}
      </div>
      <div className="runs">
        {data.running.map((r) => (
          <div key={r.id} className="runrow">
            <span className="who">{r.user}</span>
            <span className="what">
              {r.experiment} · {r.kind}
              {r.step ? ` step ${r.step}` : ""} · {fmtDuration(r.elapsed)}
            </span>
            <div className="progress">
              <i style={{ width: `${Math.max(2, r.progress_pct)}%` }} />
            </div>
            <span className="what" style={{ gridColumn: "1 / -1" }}>
              {r.progress_msg}
            </span>
          </div>
        ))}
        {data.queued.slice(0, 6).map((r) => (
          <div key={r.id} className="runrow" style={{ opacity: 0.6 }}>
            <span className="who">{r.user}</span>
            <span className="what">
              queued · {r.experiment} · {r.gpus} GPU
            </span>
          </div>
        ))}
        {data.running.length === 0 && data.queued.length === 0 && <div className="msg" style={{ gridColumn: "1 / -1", minHeight: 160 }}><p>Nothing is running. The node is yours.</p></div>}
      </div>
    </>
  );
}

function Progress({ data }) {
  return (
    <div className="prog">
      {data.experiments.map((e) => (
        <div key={e.slug} className="exp">
          <div>
            <div style={{ fontSize: 26, fontWeight: 600 }}>{e.title}</div>
            <div className="muted" style={{ fontSize: 16 }}>
              {e.submitted} submitted
            </div>
          </div>
          <div className="bars">
            {e.at_step.map((n, i) => (
              <div key={i} className={`b s${i}`}>
                <div className="c">{n}</div>
                <div className="t">{i === 0 ? "not started" : i === 4 ? "all 4 steps" : `at step ${i}`}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
      <div className="muted" style={{ fontSize: 16 }}>
        {data.students} students
      </div>
    </div>
  );
}

function Leaderboard({ data }) {
  return (
    <div className="board" style={{ display: "grid", gridTemplateColumns: data.boards.length > 1 ? "1fr 1fr" : "1fr", gap: 24 }}>
      {data.boards.map((b) => (
        <div key={b.slug} className="panel">
          <h2 style={{ fontSize: 26, marginBottom: 8 }}>
            {b.title} <span className="muted" style={{ fontSize: 16 }}>· {b.metric}{b.higher_is_better ? " ↑" : " ↓"}</span>
          </h2>
          {b.rows.length === 0 && <div className="muted">No published results yet.</div>}
          <table className="data">
            <tbody>
              {b.rows.map((r, i) => (
                <tr key={r.submission_id}>
                  <td style={{ color: i === 0 ? "var(--sun)" : "inherit", fontWeight: 700, width: 40 }}>{i + 1}</td>
                  <td>{r.user}</td>
                  <td style={{ textAlign: "right", fontWeight: 600 }}>{fmtNum(r.value)}</td>
                  <td className="muted" style={{ textAlign: "right" }}>{r.score !== null && r.score !== undefined ? `${r.score} pts` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

function RunView({ data }) {
  if (!data) return <div className="msg"><p>No run selected.</p></div>;
  const { run, result, live } = data;
  const liveCharts = Object.entries(live?.series || {}).filter(([, pts]) => pts.length > 1).map(([k, pts]) => ({ id: k, title: k, type: "line", x: "x", series: [{ key: "y", label: k, color: "sky" }], data: pts }));
  return (
    <>
      <div className="row" style={{ fontSize: 22 }}>
        <span className={`pill ${run.status}`} style={{ fontSize: 18 }}>
          <i className="dot" /> {run.status}
        </span>
        <span>
          {run.user} · {run.experiment}
          {run.step ? ` · step ${run.step}` : ""} · run {run.id}
        </span>
        {run.status === "running" && <span className="muted">{Math.round(run.progress_pct)}% · {run.progress_msg}</span>}
      </div>
      {result ? (
        <>
          <Kpis metrics={result.metrics} />
          <div className="charts">
            {(result.charts || []).slice(0, 4).map((c) => (
              <ChartCard key={c.id} spec={c} height={300} allowStretch={false} />
            ))}
          </div>
        </>
      ) : (
        <div className="charts">
          {liveCharts.map((c) => (
            <ChartCard key={c.id} spec={c} height={300} allowStretch={false} />
          ))}
        </div>
      )}
      {!result && data.log_tail && <div className="log tall" style={{ fontSize: 16 }}>{data.log_tail.join("\n")}</div>}
    </>
  );
}

/** Displays 1-4 while a class runs: what the step is for, what your own version of
 *  it has to do, and who has handed theirs in. No code — the room reads the task. */
function StepView({ data }) {
  if (!data || data.no_class)
    return (
      <div className="msg">
        <div>
          <h1>No running class</h1>
          <p>{data?.title ? `${data.title} · step ${data.step}` : "This screen follows the class once a teacher starts one."}</p>
        </div>
      </div>
    );
  if (data.error) return <div className="msg"><p>{data.error}</p></div>;
  const c = data.instructions || {};
  return (
    <div className="stepwall">
      <div className="panel code">
        <div className="bar">
          {data.title} · step {data.step}: {data.step_title}
        </div>
        <div className="steptext">
          <Markdown text={data.description || data.summary} />
          {data.own_code && (
            <div className="brief">
              <h3>Writing your own</h3>
              <ul>
                <li>
                  <b>You are given</b> {(c.params || []).length} setting{(c.params || []).length === 1 ? "" : "s"} from the form
                  {(c.inputs || []).length ? ` and ${c.inputs.join(", ")} from the step before` : ""}.
                </li>
                {(c.outputs || []).length > 0 && (
                  <li>
                    <b>You must save</b> {c.outputs.join(", ")} — the next step reads them by name.
                  </li>
                )}
                {(c.metrics || []).length > 0 && (
                  <li>
                    <b>You are judged on</b> {(c.metrics || []).slice(0, 4).map((m) => m.label).join(", ")}.
                  </li>
                )}
              </ul>
            </div>
          )}
        </div>
      </div>
      <div className="panel handed">
        <h2>Handed in ({(data.handed_in || []).length})</h2>
        {(data.handed_in || []).length === 0 && <p className="muted">Nobody yet.</p>}
        <ol>
          {(data.handed_in || []).map((h) => (
            <li key={h.user_id}>{h.name}</li>
          ))}
        </ol>
      </div>
    </div>
  );
}

/** One student, put up by the teacher: their code, or their figures beside the ones
 *  the standard code produced for the same step. */
function StudentView({ data }) {
  if (!data) return null;
  if (data.no_class)
    return (
      <div className="msg">
        <div>
          <h1>No running class</h1>
          <p>This screen follows the class once a teacher starts one.</p>
        </div>
      </div>
    );
  if (data.error) return <div className="msg"><p>{data.error}</p></div>;
  const rows = data.compare || [];
  return (
    <div className="stepwall one">
      <div className="panel code">
        <div className="bar">
          {data.name} · step {data.step}
          {data.step_title ? ` · ${data.step_title}` : ""} · {data.show === "code" ? "their code" : "their results against the standard code"}
        </div>
        {data.show === "code" && <pre>{data.code}</pre>}
        {data.show !== "code" && (
          <div className="compare">
            {rows.length > 0 ? (
              <table className="data">
                <thead>
                  <tr>
                    <th>Figure</th>
                    <th>{data.name}</th>
                    <th>The standard code</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.key}>
                      <td>{r.label}</td>
                      <td>
                        <b>{r.mine === null || r.mine === undefined ? "–" : fmtNum(r.mine)}</b>
                      </td>
                      <td className="muted">{r.standard === null || r.standard === undefined ? "–" : fmtNum(r.standard)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <Kpis metrics={data.result?.metrics} />
            )}
            {!data.standard && <p className="muted">Nothing to compare with yet: the standard code has not been run on this step.</p>}
            {(data.result?.charts || []).length > 0 && (
              <div className="charts">
                {data.result.charts.slice(0, 3).map((c) => (
                  <ChartCard key={c.id} spec={c} height={240} allowStretch={false} />
                ))}
              </div>
            )}
            {data.run?.own_code === false && <p className="muted">This run used the standard code.</p>}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Display() {
  const { n } = useParams();
  const [state, setState] = useState(null);
  const [conn, setConn] = useState("closed");
  useSocket(wsUrl(`/ws/displays/${n}`, false), (m) => m.type === "display" && setState(m.display), { onStatus: setConn });
  const mode = state?.mode;
  return (
    <div className="display">
      {mode !== "grafana" && (
        <div className="dhead">
          <span className="n">{n}</span>
          <h1>{state?.name || "Atelier"}</h1>
          <Clock />
        </div>
      )}
      {!state && <div className="msg"><p>{conn === "open" ? "Waiting for data…" : "Connecting to the server…"}</p></div>}
      {mode === "grafana" && <iframe title="hardware" src={state.data?.url} allow="fullscreen" />}
      {mode === "live" && state.data && <Live data={state.data} />}
      {mode === "progress" && state.data && <Progress data={state.data} />}
      {mode === "leaderboard" && state.data && <Leaderboard data={state.data} />}
      {mode === "run" && <RunView data={state.data} />}
      {mode === "step" && <StepView data={state.data} />}
      {mode === "student" && <StudentView data={state.data} />}
      {mode === "message" && (
        <div className="msg">
          <div>
            <h1>{state.payload?.title || ""}</h1>
            <p>{state.payload?.text || ""}</p>
          </div>
        </div>
      )}
      <div className="foot">
        display {n} · {conn === "open" ? "live" : "reconnecting"} · {state?.updated_at ? fmtTime(state.updated_at) : ""}
      </div>
    </div>
  );
}
