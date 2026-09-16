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
      setState(await api("/class", { method: "POST", body: { experiment: pick, params: startParams } }));
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
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>{running ? `Running: ${s.title || s.experiment}` : "No class is running"}</h3>
          {running && ours ? (
            <div className="row" style={{ gap: 6 }}>
              <button className="btn sm" onClick={pause} title="Free the room without ending the class">
                <Pause size={13} /> Pause
              </button>
              <button className="btn sm danger" onClick={stop}>
                <Square size={13} /> End the class
              </button>
            </div>
          ) : null}
        </div>
        {running && !ours && (
          <div className="help">
            {s.teacher} is running this class, since {fmtTime(s.started_at)}. It has to end before yours can start.
          </div>
        )}
        {teachable.length === 0 ? (
          <div className="help">An admin has not given you an experiment to teach yet.</div>
        ) : (
          <div className="row">
            <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ maxWidth: 380 }}>
              <option value="">— choose an experiment —</option>
              {teachable.map((e) => (
                <option key={e.slug} value={e.slug}>
                  {e.title}
                </option>
              ))}
            </select>
            <button className="btn primary" onClick={start} disabled={!pick || (running && !ours)}>
              <Play size={14} /> {running ? "Switch the class to this" : "Start it for the class"}
            </button>
          </div>
        )}
        {pick && spec && startFields(spec).length > 0 && (
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b className="small">What the class starts from</b>
            <div className="help">
              Students have not done the experiment this one builds on, so choose it once here. These are fixed for everyone in the class.
            </div>
            <ParamsForm params={startFields(spec)} values={startParams} onChange={setStartParams} experiment={pick} step={1} />
          </div>
        )}
        {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
        <div className="help">
          While a class runs, that is the only experiment students can work on, and only after you admit them. Displays 1–4 follow its four steps; the last display goes back to the hardware dashboard.
        </div>
      </div>

      {running && ours && (
        <div className="grid2">
          <div className="panel stack" style={{ gap: 8 }}>
            <h3>Asking to join ({waiting.length})</h3>
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
          <div className="panel stack" style={{ gap: 8 }}>
            <h3>In the class ({inClass.length})</h3>
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
  );
}

/** The wall: one control per screen, so the teacher can follow a step or put a
 *  student up. Lives in its own tab, but needs the same class state. */
export function WallPanel() {
  const [state, setState] = useState(null);
  const [displays, setDisplays] = useState([]);
  const load = useCallback(() => {
    api("/class").then(setState).catch(() => {});
    api("/displays").then(setDisplays).catch(() => {});
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
            <DisplayControl key={d.id} display={d} session={session} onPush={push} />
          ))}
        </div>
      </div>
    </div>
  );
}

function DisplayControl({ display, session, onPush }) {
  const isStep = display.id <= 4;
  const step = display.payload?.step || display.id;
  const [who, setWho] = useState(display.payload?.user_id || "");
  const [show, setShow] = useState(display.payload?.show || "results");
  const people = (session?.members || []).filter((m) => m.admitted);
  return (
    <div className="inset stack" style={{ padding: 10, gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>
          Display {display.id} {isStep ? `· step ${step}` : ""}
        </b>
        <a className="small" href={`/display/${display.id}`} target="_blank" rel="noreferrer">
          open
        </a>
      </div>
      <div className="small muted">
        showing: {display.mode}
        {display.mode === "student" ? ` · ${display.payload?.show || "results"}` : ""}
      </div>
      {isStep ? (
        <>
          <div className="row" style={{ gap: 6 }}>
            <select value={who} onChange={(e) => setWho(e.target.value)} style={{ flex: 1 }}>
              <option value="">— a student —</option>
              {people.map((m) => (
                <option key={m.user_id} value={m.user_id}>
                  {m.name || m.username}
                </option>
              ))}
            </select>
            <select value={show} onChange={(e) => setShow(e.target.value)}>
              <option value="results">results</option>
              <option value="code">their code</option>
            </select>
          </div>
          <div className="row" style={{ gap: 6 }}>
            <button className="btn sm primary" disabled={!who} onClick={() => onPush(display.id, { mode: "student", payload: { session_id: session.id, experiment: session.experiment, step, user_id: Number(who), show } })}>
              Put it up
            </button>
            <button className="btn sm ghost" onClick={() => onPush(display.id, { mode: "step", payload: { session_id: session.id, experiment: session.experiment, step } })}>
              Back to the step
            </button>
          </div>
        </>
      ) : (
        <div className="row" style={{ gap: 6 }}>
          <button className={`btn sm ${display.mode === "grafana" ? "primary" : "ghost"}`} onClick={() => onPush(display.id, { mode: "grafana", payload: {} })}>
            Hardware
          </button>
          <button className={`btn sm ${display.mode === "leaderboard" ? "primary" : "ghost"}`} onClick={() => onPush(display.id, { mode: "leaderboard", payload: { experiment: session.experiment } })}>
            Leaderboard
          </button>
        </div>
      )}
    </div>
  );
}

/** The first step's run and material params: what an experiment needs before a
 *  student can start it, which in class the teacher settles once. */
function startFields(spec) {
  const first = (spec?.steps || [])[0];
  return (first?.params || []).filter((p) => p.type === "run" || p.type === "material" || (p.type === "select" && /source/.test(p.key)));
}
