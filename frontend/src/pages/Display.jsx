import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import ChartCard from "../components/ChartCard";
import Kpis from "../components/Kpi";
import { DataTable } from "../components/ResultsView";
import Markdown from "../components/Markdown";
import { brandName } from "../brand";
import { useI18n, useT } from "../i18n";
import { wsUrl } from "../lib/api";
import { buildLiveCharts } from "../lib/liveCharts";
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
  const t = useT();
  const gpus = data.gpus?.gpus || [];
  return (
    <>
      <div className="gpugrid">
        {Array.from({ length: data.gpu_count }, (_, i) => gpus[i]).map((g, i) => (
          <div key={i} className={`gpu ${g?.job ? "busy" : ""}`}>
            <div className="u">{g ? `${g.util}%` : "–"}</div>
            <div className="m">
              {t("display.gpus.gpu", { n: i })}
              {g ? ` · ${(g.mem_used / 1e9).toFixed(0)} / ${(g.mem_total / 1e9).toFixed(0)} GB · ${g.temp}°C` : ` · ${t("display.gpus.noData")}`}
            </div>
            <div className="bar">
              <i style={{ width: `${g ? g.util : 0}%` }} />
            </div>
            <div className="m">{g?.job ? `${g.job.user} · ${g.job.experiment}` : t("display.gpus.free")}</div>
          </div>
        ))}
      </div>
      <div className="runs">
        {data.running.map((r) => (
          <div key={r.id} className="runrow">
            <span className="who">{r.user}</span>
            <span className="what">
              {r.experiment} · {r.kind}
              {r.step ? ` ${t("display.stepN", { n: r.step })}` : ""} · {fmtDuration(r.elapsed)}
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
              {t("common.status.queued")} · {r.experiment} · {t("display.gpus.count", { n: r.gpus })}
            </span>
          </div>
        ))}
        {data.running.length === 0 && data.queued.length === 0 && <div className="msg" style={{ gridColumn: "1 / -1", minHeight: 160 }}><p>{t("display.gpus.idle")}</p></div>}
      </div>
    </>
  );
}

function Progress({ data }) {
  const t = useT();
  return (
    <div className="prog">
      {data.experiments.map((e) => (
        <div key={e.slug} className="exp">
          <div>
            <div style={{ fontSize: 26, fontWeight: 600 }}>{e.title}</div>
            <div className="muted" style={{ fontSize: 16 }}>
              {t("display.progress.submitted", { n: e.submitted })}
            </div>
          </div>
          <div className="bars">
            {e.at_step.map((n, i) => (
              <div key={i} className={`b s${i}`}>
                <div className="c">{n}</div>
                <div className="t">{i === 0 ? t("display.progress.notStarted") : i === 4 ? t("display.progress.allSteps") : t("display.progress.atStep", { n: i })}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
      <div className="muted" style={{ fontSize: 16 }}>
        {t("display.progress.students", { count: data.students })}
      </div>
    </div>
  );
}

function Leaderboard({ data }) {
  const t = useT();
  return (
    <div className="board" style={{ display: "grid", gridTemplateColumns: data.boards.length > 1 ? "1fr 1fr" : "1fr", gap: 24 }}>
      {data.boards.map((b) => (
        <div key={b.slug} className="panel">
          <h2 style={{ fontSize: 26, marginBottom: 8 }}>
            {b.title} <span className="muted" style={{ fontSize: 16 }}>· {b.metric}{b.higher_is_better ? " ↑" : " ↓"}</span>
          </h2>
          {b.rows.length === 0 && <div className="muted">{t("display.board.empty")}</div>}
          <table className="data">
            <tbody>
              {b.rows.map((r, i) => (
                <tr key={r.submission_id}>
                  <td style={{ color: i === 0 ? "var(--sun)" : "inherit", fontWeight: 700, width: 40 }}>{i + 1}</td>
                  <td>{r.user}</td>
                  <td style={{ textAlign: "right", fontWeight: 600 }}>{fmtNum(r.value)}</td>
                  <td className="muted" style={{ textAlign: "right" }}>{r.score !== null && r.score !== undefined ? t("display.board.points", { n: r.score }) : ""}</td>
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
  const t = useT();
  if (!data) return <div className="msg"><p>{t("display.run.none")}</p></div>;
  const { run, result, live } = data;
  const liveCharts = buildLiveCharts(live, { max: 4 });
  return (
    <>
      <div className="row" style={{ fontSize: 22 }}>
        <span className={`pill ${run.status}`} style={{ fontSize: 18 }}>
          <i className="dot" /> {t(`common.status.${run.status}`)}
        </span>
        <span>
          {run.user} · {run.experiment}
          {run.step ? ` · ${t("display.stepN", { n: run.step })}` : ""} · {t("common.runN", { id: run.id })}
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

/** Who has handed this step in. It stays on screen whatever the rest shows, with
 *  the one being looked at marked. Nothing scrolls on a wall, so a long class is
 *  cut off with a count. */
function HandedIn({ list, selected, cap = 12 }) {
  const t = useT();
  const shown = (list || []).slice(0, cap);
  const rest = (list || []).length - shown.length;
  return (
    <div className="panel handed">
      <h2>{t("display.handedIn.title", { n: (list || []).length })}</h2>
      {shown.length === 0 && <p className="muted">{t("display.handedIn.nobody")}</p>}
      <ol>
        {shown.map((h) => (
          <li key={h.user_id} className={h.user_id === selected ? "on" : ""}>
            {h.name}
          </li>
        ))}
      </ol>
      {rest > 0 && <p className="muted">{t("display.handedIn.more", { n: rest })}</p>}
    </div>
  );
}

/** The teacher working through this step at the front: where their run is, and the
 *  figures it has so far. */
function TeacherRun({ run }) {
  const t = useT();
  const done = run.status === "succeeded" || run.status === "failed" || run.status === "cancelled";
  return (
    <div className="teacherrun">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>{t("display.teacherRun")}</b>
        <span className={`pill ${run.status}`}>{t(`common.status.${run.status}`)}</span>
      </div>
      {!done && (
        <>
          <div className="progress">
            <i style={{ width: `${Math.max(2, Math.round(run.progress_pct || 0))}%` }} />
          </div>
          {run.progress_msg && <div className="muted small">{run.progress_msg}</div>}
        </>
      )}
      {(run.metrics || []).length > 0 && <Kpis metrics={run.metrics} />}
    </div>
  );
}

/** The slide for a step, from public/slides/<experiment>/step-<n>.png — "world-sft"
 *  keeps its slides in "sft". The whole picture is shown, scaled to fit the screen
 *  and never cropped. */
export const slidePath = (experiment, step) => `/slides/${String(experiment || "").replace(/^world-/, "")}/step-${step}.png`;

function useSlide(experiment, step) {
  const src = experiment ? slidePath(experiment, step) : null;
  const [ready, setReady] = useState(false);
  useEffect(() => {
    setReady(false);
    if (!src) return undefined;
    // ask for it first: the screen only turns into a slide once there is one
    let live = true;
    const img = new Image();
    img.onload = () => live && setReady(true);
    img.onerror = () => live && setReady(false);
    img.src = src;
    return () => {
      live = false;
    };
  }, [src]);
  return { src, ready };
}

/** Displays 1-4 while a class runs: what the step is for, the picture that explains
 *  it, and what a student's own version of it has to do. */
function StepView({ data }) {
  const t = useT();
  if (!data || data.no_class)
    return (
      <div className="msg">
        <div>
          <h1>{t("display.noClass.title")}</h1>
          <p>{data?.title ? `${data.title} · ${t("display.stepN", { n: data.step })}` : t("display.noClass.text")}</p>
        </div>
      </div>
    );
  if (data.error) return <div className="msg"><p>{data.error}</p></div>;
  const c = data.instructions || {};
  // a step people can write themselves has a list of who handed it in; one they
  // cannot has nothing to list, so the explanation takes the whole screen
  return (
    <div className={`stepwall ${data.own_code ? "" : "one"}`}>
      <div className="panel code">
        <div className="bar">
          {data.class_name ? `${data.class_name} · ` : ""}
          {data.title} · {t("display.stepN", { n: data.step })}: {data.step_title}
          {data.over ? ` · ${t("display.step.finished")}` : ""}
        </div>
        <div className="steptext">
          {data.figure && <img className="figure" src={data.figure} alt="" />}
          <div className="prose">
            <Markdown text={data.description || data.summary} />
          </div>
          {data.teacher_run && <TeacherRun run={data.teacher_run} />}
          {data.own_code && (
            <div className="brief">
              <h3>{t("display.step.writeOwn")}</h3>
              <ul>
                <li>
                  <b>{t("display.step.givenLabel")}</b> {t("display.step.givenSettings", { count: (c.params || []).length })}
                  {(c.inputs || []).length ? t("display.step.givenInputs", { inputs: c.inputs.join(", ") }) : ""}
                  {t("display.step.end")}
                </li>
                {(c.outputs || []).length > 0 && (
                  <li>
                    <b>{t("display.step.saveLabel")}</b> {c.outputs.join(", ")}
                    {t("display.step.end")}
                  </li>
                )}
                {(c.metrics || []).length > 0 && (
                  <li>
                    <b>{t("display.step.judgedLabel")}</b> {(c.metrics || []).slice(0, 4).map((m) => m.label).join(", ")}
                    {t("display.step.end")}
                  </li>
                )}
              </ul>
            </div>
          )}
        </div>
      </div>
      {data.own_code && <HandedIn list={data.handed_in} />}
    </div>
  );
}

/** What a run produced, laid out for a wall: the charts on the left, two abreast
 *  when there are several, and the step's table beside them. A step whose result
 *  draws almost nothing — training draws one loss chart — shows the curves the run
 *  reported while it went instead. */
function WallResult({ result, live, slideTop }) {
  const own = result.charts || [];
  // the curves it reported as it went, minus anything the result already draws
  const drawn = own.map((c) => `${c.id} ${c.title || ""}`.toLowerCase()).join(" ");
  const extra = buildLiveCharts(live, { timeline: false }).filter((c) => !drawn.includes(c.id.replace("live-", "")));
  const charts = own.length >= 2 ? own : [...own, ...extra].slice(0, 4);
  const tables = (result.tables || []).slice(0, 1);
  const height = chartRoom(charts.length >= 3 ? 300 : charts.length === 2 ? 270 : 620, charts.length >= 3 ? 2 : 1, slideTop);
  return (
    <div className="wallresult">
      <Kpis metrics={result.metrics} />
      <div className={`wrbody ${tables.length ? "" : "wide"}`}>
        <div className={`wrcharts ${charts.length >= 3 ? "two" : ""}`}>
          {charts.slice(0, 4).map((c) => (
            <ChartCard key={c.id} spec={c} height={height} allowStretch={false} />
          ))}
        </div>
        {tables.length > 0 && (
          <div className="wrtables">
            {tables.map((t) => (
              <DataTable key={t.id} table={{ ...t, rows: (t.rows || []).slice(0, slideTop ? 7 : 9) }} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/** One person's handed-in work: their figures against the standard code on the same
 *  settings, with the better one marked — or the code itself. */
function StudentView({ data, slideTop }) {
  const t = useT();
  if (!data) return null;
  if (data.no_class)
    return (
      <div className="msg">
        <div>
          <h1>{t("display.noClass.title")}</h1>
          <p>{t("display.noClass.text")}</p>
        </div>
      </div>
    );
  if (data.show === "running") return <RunningView data={data} slideTop={slideTop} />;
  const rows = (data.compare || []).slice(0, 9);
  return (
    <div className="stepwall one">
      <div className="panel code">
        <div className="bar">
          {data.name} · {t("display.stepN", { n: data.step })}
          {data.step_title ? ` · ${data.step_title}` : ""} · {data.show === "code" ? t("display.student.theirCode") : data.their_own ? t("display.student.theirResults") : t("display.student.againstStandard")}
        </div>
        {data.error && <p className="muted" style={{ padding: 20 }}>{data.error}</p>}
        {data.show === "code" && data.code !== undefined && <pre>{data.code}</pre>}
        {data.show !== "code" && !data.error && (
          <div className="compare">
            {rows.length > 0 ? (
              <table className="data">
                <thead>
                  <tr>
                    <th>{t("display.student.figure")}</th>
                    <th>{data.name}</th>
                    <th>{t("display.student.standardCode")}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.key}>
                      <td>{r.label}</td>
                      <td className={r.best === "mine" ? "best" : ""}>{r.mine === null || r.mine === undefined ? "–" : fmtNum(r.mine)}</td>
                      <td className={r.best === "standard" ? "best" : ""}>{r.standard === null || r.standard === undefined ? "–" : fmtNum(r.standard)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <WallResult result={data.result} live={data.live} slideTop={slideTop} />
            )}
            {rows.length > 0 && (data.result?.charts || []).length > 0 && (
              <div className="charts">
                {data.result.charts.slice(0, 2).map((c) => (
                  <ChartCard key={c.id} spec={c} height={220} allowStretch={false} />
                ))}
              </div>
            )}
            {data.standard_state && data.standard_state !== "succeeded" && (
              <p className="muted">{data.standard_state === "missing" ? t("display.student.standardMissing") : t(data.standard_state === "queued" || data.standard_state === "running" ? "display.student.standardState" : "display.student.standardEnded", { state: t(`common.status.${data.standard_state}`) })}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** A chart's height, less whatever a slide behind the screen takes off the top,
 *  shared out over the rows of charts so they all stay on the screen. */
function chartRoom(height, rows, slideTop) {
  const taken = slideTop ? Math.max(0, slideTop - 96) : 0;
  return Math.max(150, Math.round(height - taken / rows));
}

/** Why a run stopped, in words a room can read, from how the process ended. */
function whyItStopped(run, t) {
  if (run.exit_code === -9 || run.exit_code === 137) return t("display.run.outOfMemory");
  if (/timed out/.test(run.error || "")) return t("display.run.timedOut", { n: run.timeout_min });
  return null;
}

/** One student's newest run of the step, as it goes: how far along, the curves so
 *  far, and — when it fails — the error and the end of its log, which is where
 *  the reason is. */
function RunningView({ data, slideTop }) {
  const t = useT();
  const run = data.run;
  const head = (
    <div className="bar">
      {data.name} · {t("display.stepN", { n: data.step })}
      {data.step_title ? ` · ${data.step_title}` : ""} · {t("display.run.theirLive")}
    </div>
  );
  if (!run)
    return (
      <div className="stepwall one">
        <div className="panel code">
          {head}
          <p className="muted" style={{ padding: 20, fontSize: 20 }}>{data.error || t("display.run.noneYet")}</p>
        </div>
      </div>
    );
  const going = run.status === "running" || run.status === "queued";
  const failed = run.status === "failed" || run.status === "cancelled";
  const why = failed ? whyItStopped(run, t) : null;
  // a wall does not scroll: up to four charts, two abreast, fewer when the error
  // needs the room
  const charts = buildLiveCharts(data.live, { max: failed ? 2 : 4 });
  const chartHeight = chartRoom(failed ? 250 : charts.length >= 3 ? 296 : charts.length === 2 ? 420 : 540, charts.length >= 3 ? 2 : 1, slideTop);
  return (
    <div className="stepwall one">
      <div className="panel code">
        {head}
        <div className="liverun">
          <div className="side">
            <div className="row" style={{ gap: 14, fontSize: 20 }}>
              <span className={`pill ${run.status}`} style={{ fontSize: 18 }}>
                <i className="dot" /> {t(`common.status.${run.status}`)}
              </span>
              <span className="muted">
                {t("common.runN", { id: run.id })} · {run.own_code ? t("display.run.theirOwnCode") : t("display.run.standardCode")} · {fmtDuration(run.elapsed)}
                {going ? t("display.run.ofAllowed", { n: run.timeout_min }) : ""}
              </span>
            </div>
            {going && (
              <>
                <div className="progress big">
                  <i style={{ width: `${Math.max(2, Math.round(run.progress_pct || 0))}%` }} />
                </div>
                <div style={{ fontSize: 20 }}>
                  {Math.round(run.progress_pct || 0)}% · {run.progress_msg || (run.status === "queued" ? t("display.run.waitingMachine") : t("display.run.starting"))}
                </div>
              </>
            )}
            {failed && (
              <div className="failbox">
                <b>{run.status === "cancelled" ? t("display.run.stopped") : t("display.run.failed")}</b>
                {why && <div>{why}</div>}
                {run.error && <pre>{run.error.split("\n").slice(-8).join("\n")}</pre>}
              </div>
            )}
            {run.status === "succeeded" && (run.metrics || []).length > 0 && <Kpis metrics={run.metrics} />}
            <div className={`livecharts ${charts.length > 1 ? "two" : ""}`}>
              {charts.map((c) => (
                <ChartCard key={c.id} spec={c} height={chartHeight} allowStretch={false} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** The code the experiment ships with — just the code, the whole screen. */
function StandardView({ data }) {
  const t = useT();
  if (!data) return null;
  if (data.no_class)
    return (
      <div className="msg">
        <div>
          <h1>{t("display.noClass.title")}</h1>
          <p>{t("display.noClass.text")}</p>
        </div>
      </div>
    );
  if (data.error) return <div className="msg"><p>{data.error}</p></div>;
  return (
    <div className="stepwall one">
      <div className="panel code">
        <div className="bar">{t("display.standard.title")}</div>
        <pre>{data.code}</pre>
      </div>
    </div>
  );
}

export default function Display() {
  const { n } = useParams();
  const t = useT();
  const { lang } = useI18n();
  const [state, setState] = useState(null);
  const [conn, setConn] = useState("closed");
  useSocket(wsUrl(`/ws/displays/${n}`, false), (m) => m.type === "display" && setState(m.display), { onStatus: setConn });
  const mode = state?.mode;
  // the screen says what it is showing, not which screen it is
  const step = state?.payload?.step || state?.data?.step;
  const stepTitle = state?.data?.step_title;
  const title =
    mode === "step" || mode === "student" || mode === "standard"
      ? `${t("common.stepN", { n: step || n })}${stepTitle ? ` · ${stepTitle}` : ""}`
      : mode === "leaderboard"
        ? t("display.leaderboard")
        : state?.name || brandName(lang);
  // a step screen is its slide when the experiment has one, and the whole screen
  // is the slide: no heading, no clock, no footer. Without one it says the step
  // in words instead.
  // every screen that belongs to a step can carry that step's slide: on its own
  // (a step screen), or behind what the screen shows when the wall asks for it
  const onStep = mode === "step" || mode === "student" || mode === "standard";
  const { src: slideSrc, ready: hasSlide } = useSlide(onStep ? state?.payload?.experiment || state?.data?.experiment : null, step || n);
  const behind = hasSlide && Boolean(state?.payload?.bg);
  const pad = behind ? { top: Number(state?.payload?.pad_top), side: Number(state?.payload?.pad_side) } : null;
  const slideTop = behind ? (Number.isFinite(pad.top) ? pad.top : 192) : 0;
  const slide = hasSlide && !behind && mode === "step";
  // behind the content: barely a veil where the slide carries its title, darker
  // below it so what the screen shows still reads
  return (
    <div
      className={`display ${slide ? "slideonly" : ""} ${behind ? "onslide" : ""}`}
      style={
        behind
          ? {
              backgroundImage: `linear-gradient(rgba(9, 17, 31, 0.1) 0px, rgba(9, 17, 31, 0.14) 120px, rgba(9, 17, 31, 0.7) 230px, rgba(9, 17, 31, 0.78) 100%), url("${slideSrc}")`,
              paddingTop: Number.isFinite(pad.top) ? pad.top : undefined,
              paddingLeft: Number.isFinite(pad.side) ? pad.side : undefined,
              paddingRight: Number.isFinite(pad.side) ? pad.side : undefined,
            }
          : undefined
      }
    >
      {mode !== "grafana" && !slide && !behind && (
        <div className="dhead">
          <h1>{title}</h1>
          <Clock />
        </div>
      )}
      {!state && <div className="msg"><p>{conn === "open" ? t("display.waiting") : t("display.connecting")}</p></div>}
      {mode === "grafana" && <iframe title={t("display.hardware")} src={state.data?.url} allow="fullscreen" />}
      {mode === "live" && state.data && <Live data={state.data} />}
      {mode === "progress" && state.data && <Progress data={state.data} />}
      {mode === "leaderboard" && state.data && <Leaderboard data={state.data} />}
      {mode === "run" && <RunView data={state.data} />}
      {mode === "step" && (slide ? <div className="slide"><img src={slideSrc} alt="" /></div> : <StepView data={state.data} />)}
      {mode === "student" && <StudentView data={state.data} slideTop={slideTop} />}
      {mode === "standard" && <StandardView data={state.data} />}
      {mode === "message" && (
        <div className="msg">
          <div>
            <h1>{state.payload?.title || ""}</h1>
            <p>{state.payload?.text || ""}</p>
          </div>
        </div>
      )}
      {!slide && !behind && (
        <div className="foot">
          {conn === "open" ? t("display.live") : t("display.reconnecting")} · {state?.updated_at ? fmtTime(state.updated_at) : ""}
        </div>
      )}
    </div>
  );
}
