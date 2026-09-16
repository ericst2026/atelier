import React, { useCallback, useEffect, useState } from "react";
import { Pause, Play, Square } from "lucide-react";
import ParamsForm from "./ParamsForm";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtTime } from "../lib/format";

/** The class: one experiment at a time, who is in it, and what the wall shows.
 *  Students do not pick the experiment — this is where it is chosen. One class
 *  runs in the building, so another teacher's has to end before yours starts. */
export default function ClassPanel() {
  const { user } = useAuth();
  const isAdmin = user?.role === "admin";
  const [state, setState] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [displays, setDisplays] = useState([]);
  const [pick, setPick] = useState("");
  const [name, setName] = useState("");
  const [spec, setSpec] = useState(null); // the chosen experiment, to ask what it starts from
  const [startParams, setStartParams] = useState({});
  const [error, setError] = useState(null);
  const load = useCallback(() => {
    api("/class").then(setState).catch((e) => setError(e.message));
    api("/displays").then(setDisplays).catch(() => {});
  }, []);
  useEffect(() => {
    api("/experiments").then((d) => setExperiments(d.experiments || []));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);
  useEffect(() => {
    setStartParams({});
    if (!pick) return setSpec(null);
    api(`/experiments/${pick}`).then(setSpec).catch(() => setSpec(null));
  }, [pick]);
  const start = async () => {
    setError(null);
    try {
      setState(await api("/class", { method: "POST", body: { experiment: pick, name, params: startParams } }));
      setName("");
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const stop = async () => {
    setError(null);
    try {
      await api("/class", { method: "DELETE" });
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const admit = async (userId, admitted) => {
    setError(null);
    try {
      setState(await api(`/class/members/${userId}`, { method: "POST", body: { admitted } }));
    } catch (e) {
      setError(e.message);
    }
  };
  const deny = async (userId) => {
    setError(null);
    try {
      setState(await api(`/class/members/${userId}`, { method: "DELETE" }));
    } catch (e) {
      setError(e.message);
    }
  };
  const pause = async () => {
    setError(null);
    try {
      setState(await api("/class/pause", { method: "POST", body: {} }));
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const resume = async (id) => {
    setError(null);
    try {
      setState(await api(`/class/${id}/resume`, { method: "POST", body: {} }));
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const push = async (displayId, payload) => {
    await api(`/displays/${displayId}`, { method: "PUT", body: payload });
    load();
  };
  const s = state?.session;
  const running = Boolean(state?.running && s);
  // an admin may take the room; a teacher only touches the class they started
  const ours = Boolean(isAdmin || s?.me?.mine);
  // what an admin has given this teacher to teach; an admin may teach anything
  const teachable = isAdmin ? experiments : experiments.filter((e) => (user?.self_experiments || []).includes(e.slug));
  const members = s?.members || [];
  const waiting = members.filter((m) => !m.admitted);
  const inClass = members.filter((m) => m.admitted);
  return (
    <div className="stack">
      <div className="panel stack">
        <h3>Start an experiment</h3>
        {running && (
          <div className="help">
            {ours ? "Your class" : `${s.teacher}'s class`} is running. Pause or end it before starting another.
          </div>
        )}
        {teachable.length === 0 ? (
          <div className="help">An admin has not given you an experiment to teach yet.</div>
        ) : (
          <div className="stack" style={{ gap: 10 }}>
            <label className="field">
              <span>Name it</span>
              <input
                type="text"
                value={name}
                placeholder="Tuesday 2pm, group B — anything that tells it from the others"
                onChange={(e) => setName(e.target.value)}
                disabled={running}
              />
              <span className="help">The experiment says what kind of class it is; this says which one.</span>
            </label>
            <label className="field">
              <span>Experiment</span>
              <select value={pick} onChange={(e) => setPick(e.target.value)} disabled={running} style={{ maxWidth: 420 }}>
                <option value="">— choose an experiment —</option>
                {teachable.map((e) => (
                  <option key={e.slug} value={e.slug}>
                    {e.title}
                  </option>
                ))}
              </select>
            </label>
            {pick && spec && startFields(spec).length > 0 && (
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b className="small">What the class starts from</b>
            <div className="help">
              Students have not done the experiment this one builds on, so choose it once here. These are fixed for everyone in the class.
            </div>
            <ParamsForm params={startFields(spec)} values={startParams} onChange={setStartParams} experiment={pick} step={1} />
          </div>
        )}
            <button className="btn primary" onClick={start} disabled={!pick || !name.trim() || running}>
              <Play size={14} /> Start the experiment
            </button>
          </div>
        )}
        {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
        <div className="help">
          While a class runs, that is the only experiment students can work on, and only after you let them in.
        </div>
      </div>

      {running && (
        <div className="panel stack" style={{ borderColor: ours ? "var(--kept)" : "var(--line)" }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>
              Running: {s.name || s.title || s.experiment}
              <span className="faint small"> · {s.title || s.experiment} · {s.teacher} · since {fmtTime(s.started_at)}</span>
            </h3>
            {ours && (
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm" onClick={pause} title="Free the room without ending the class">
                  <Pause size={13} /> Pause
                </button>
                <button className="btn sm danger" onClick={stop}>
                  <Square size={13} /> End it
                </button>
              </div>
            )}
          </div>

          {ours && (
        <div className="grid2">
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b>Asking to join ({waiting.length})</b>
            {waiting.length === 0 && <div className="help">Nobody is waiting.</div>}
            {waiting.map((m) => (
              <div key={m.user_id} className="row" style={{ justifyContent: "space-between" }}>
                <span>
                  {m.name || m.username} <span className="faint small">{m.username}</span>
                </span>
                <div className="row" style={{ gap: 6 }}>
                  <button className="btn sm good" onClick={() => admit(m.user_id, true)}>
                    Accept
                  </button>
                  <button className="btn sm danger" onClick={() => deny(m.user_id)}>
                    Deny
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b>In the class ({inClass.length})</b>
            {inClass.length === 0 && <div className="help">Nobody yet.</div>}
            {inClass.map((m) => (
              <div key={m.user_id} className="row" style={{ justifyContent: "space-between" }}>
                <span>
                  {m.name || m.username} <span className="faint small">{m.username}</span>
                </span>
                <button className="btn sm ghost" onClick={() => admit(m.user_id, false)}>
                  Remove
                </button>
              </div>
            ))}
          </div>
        </div>
          )}
        </div>
      )}

    </div>
  );
}

/** The wall: one control per screen, so the teacher can follow a step or put a
 *  student up. Lives in its own tab, but needs the same class state. */
export function WallPanel() {
  const [state, setState] = useState(null);
  const [displays, setDisplays] = useState([]);
  const [classes, setClasses] = useState([]);
  const load = useCallback(() => {
    api("/class").then(setState).catch(() => {});
    api("/displays").then(setDisplays).catch(() => {});
    api("/class/history?limit=50").then((d) => setClasses(d.sessions || [])).catch(() => {});
  }, []);
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, [load]);
  const push = async (displayId, body) => {
    await api(`/displays/${displayId}`, { method: "PUT", body });
    load();
  };
  const session = state?.running ? state.session : null;
  return (
    <div className="stack">
      <div className="panel stack">
        <h3>What the wall shows</h3>
        <div className="help">
          {session
            ? "Screens 1–4 follow the four steps: what the step is for, what your own version has to do, and who has handed it in. Put a student up to compare their figures with the standard code."
            : "No class is running, so the step screens say so. The last screen keeps the hardware dashboard."}
        </div>
        <div className="grid2">
          {displays.map((d) => (
            <DisplayControl key={d.id} display={d} session={session} classes={classes} onPush={push} />
          ))}
        </div>
      </div>
    </div>
  );
}

function DisplayControl({ display, session, classes, onPush }) {
  const isStep = display.id <= 4;
  const payload = display.payload || {};
  const [cls, setCls] = useState(payload.session_id || session?.id || "");
  const [step, setStep] = useState(payload.step || display.id);
  const [show, setShow] = useState(display.mode === "student" ? payload.show || "results" : "explanation");
  const [who, setWho] = useState(payload.user_id || "");
  const chosen = classes.find((c) => c.id === Number(cls)) || (Number(cls) === session?.id ? session : null);
  // only people with something to show: the wall reads what was handed in, so a
  // student who has not handed this step in would put an empty screen up
  const [handed, setHanded] = useState([]);
  useEffect(() => {
    if (!chosen?.experiment) return setHanded([]);
    api(`/submissions?experiment=${chosen.experiment}&step=${Number(step)}`)
      .then((subs) => {
        const seen = new Set();
        setHanded(
          subs
            .filter((x) => !seen.has(x.user_id) && seen.add(x.user_id))
            .map((x) => ({ user_id: x.user_id, name: x.name || x.username }))
        );
      })
      .catch(() => setHanded([]));
  }, [chosen?.experiment, step]);
  // the teacher hands nothing in, so their own last run of the step stands in
  const people = chosen ? [...handed, { user_id: chosen.teacher_id, name: `${chosen.teacher} (yours)` }].filter((x, i, a) => a.findIndex((y) => y.user_id === x.user_id) === i) : [];
  const apply = () => {
    const base = { session_id: Number(cls) || undefined, experiment: chosen?.experiment, step: Number(step), pinned: Boolean(chosen?.ended_at) };
    if (show === "explanation") return onPush(display.id, { mode: "step", payload: base });
    onPush(display.id, { mode: "student", payload: { ...base, user_id: Number(who), show } });
  };
  return (
    <div className="inset stack" style={{ padding: 10, gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>{isStep ? `Step ${payload.step || display.id} screen` : `Screen ${display.id}`}</b>
        <a className="small" href={`/display/${display.id}`} target="_blank" rel="noreferrer">
          open
        </a>
      </div>
      <div className="small muted">
        showing: {display.mode === "student" ? `${payload.show || "results"} of one student` : display.mode === "step" ? "the step" : display.mode}
      </div>
      {isStep ? (
        <>
          <select value={cls} onChange={(e) => setCls(e.target.value)}>
            <option value="">— a class —</option>
            {session && (
              <option value={session.id}>
                {session.name || session.title || session.experiment} (running)
              </option>
            )}
            {classes
              .filter((c) => c.id !== session?.id)
              .map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name || c.title || c.experiment}
                  {c.ended_at ? " (finished)" : c.paused ? " (paused)" : ""}
                </option>
              ))}
          </select>
          <div className="row" style={{ gap: 6 }}>
            <select value={step} onChange={(e) => setStep(e.target.value)} style={{ width: 100 }}>
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>
                  step {n}
                </option>
              ))}
            </select>
            <select value={show} onChange={(e) => setShow(e.target.value)} style={{ flex: 1 }}>
              <option value="explanation">what the step is</option>
              <option value="results">a student's results</option>
              <option value="code">a student's code</option>
            </select>
          </div>
          {show !== "explanation" && (
            <select value={who} onChange={(e) => setWho(e.target.value)}>
              <option value="">{handed.length ? "— who —" : "— nobody has handed this in —"}</option>
              {people.map((m) => (
                <option key={m.user_id} value={m.user_id}>
                  {m.name || m.username}
                </option>
              ))}
            </select>
          )}
          <button className="btn sm primary" onClick={apply} disabled={!cls || (show !== "explanation" && !who)}>
            Put it up
          </button>
        </>
      ) : (
        <div className="row" style={{ gap: 6 }}>
          <button className={`btn sm ${display.mode === "grafana" ? "primary" : "ghost"}`} onClick={() => onPush(display.id, { mode: "grafana", payload: {} })}>
            Hardware
          </button>
          <button className={`btn sm ${display.mode === "leaderboard" ? "primary" : "ghost"}`} onClick={() => onPush(display.id, { mode: "leaderboard", payload: { experiment: session?.experiment } })}>
            Leaderboard
          </button>
        </div>
      )}
    </div>
  );
}

/** The first step's run and material params: what an experiment needs before a
 *  student can start it, which in class the teacher settles once. */
/** What an experiment needs from outside itself, which a student cannot supply:
 *  another experiment's run, and the prepared models or datasets that stand in for
 *  one. A step that builds on the step before it is not here — that is the
 *  student's own work, and they pick their own. */
function startFields(spec) {
  const out = [];
  for (const st of spec?.steps || []) {
    for (const p of st.params || []) {
      const fromElsewhere =
        (p.type === "run" && p.experiment && p.experiment !== spec.slug) ||
        p.type === "material" ||
        (p.type === "select" && /source/.test(p.key));
      if (!fromElsewhere || out.some((x) => x.key === p.key)) continue;
      out.push(st.index === 1 ? p : { ...p, label: `${p.label} (step ${st.index})` });
    }
  }
  return out;
}
