import React, { useCallback, useEffect, useState } from "react";
import { Play, Square } from "lucide-react";
import { api } from "../lib/api";

/** The class: one experiment at a time, who is in it, and what the wall shows.
 *  Students do not pick the experiment — this is where it is chosen. */
export default function ClassPanel() {
  const [state, setState] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [displays, setDisplays] = useState([]);
  const [pick, setPick] = useState("");
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
  const start = async () => {
    setError(null);
    try {
      setState(await api("/class", { method: "POST", body: { experiment: pick } }));
      load();
    } catch (e) {
      setError(e.message);
    }
  };
  const stop = async () => {
    await api("/class", { method: "DELETE" });
    load();
  };
  const admit = async (userId, admitted) => setState(await api(`/class/members/${userId}`, { method: "POST", body: { admitted } }));
  const push = async (displayId, payload) => {
    await api(`/displays/${displayId}`, { method: "PUT", body: payload });
    load();
  };
  const running = state?.running;
  const waiting = (state?.members || []).filter((m) => !m.admitted);
  const inClass = (state?.members || []).filter((m) => m.admitted);
  return (
    <div className="stack">
      <div className="panel stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>{running ? `Running: ${state.title || state.experiment}` : "No class is running"}</h3>
          {running ? (
            <button className="btn sm danger" onClick={stop}>
              <Square size={13} /> End the class
            </button>
          ) : null}
        </div>
        <div className="row">
          <select value={pick} onChange={(e) => setPick(e.target.value)} style={{ maxWidth: 380 }}>
            <option value="">— choose an experiment —</option>
            {experiments.map((e) => (
              <option key={e.slug} value={e.slug}>
                {e.title}
              </option>
            ))}
          </select>
          <button className="btn primary" onClick={start} disabled={!pick}>
            <Play size={14} /> {running ? "Switch the class to this" : "Start it for the class"}
          </button>
        </div>
        {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
        <div className="help">
          While a class runs, that is the only experiment students can work on, and only after you admit them. Displays 1–4 follow its four steps; the last display goes back to the hardware dashboard.
        </div>
      </div>

      {running && (
        <div className="grid2">
          <div className="panel stack" style={{ gap: 8 }}>
            <h3>Asking to join ({waiting.length})</h3>
            {waiting.length === 0 && <div className="help">Nobody is waiting.</div>}
            {waiting.map((m) => (
              <div key={m.user_id} className="row" style={{ justifyContent: "space-between" }}>
                <span>
                  {m.name || m.username} <span className="faint small">{m.username}</span>
                </span>
                <button className="btn sm good" onClick={() => admit(m.user_id, true)}>
                  Let in
                </button>
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

      {running && (
        <div className="panel stack">
          <h3>What the wall shows</h3>
          <div className="help">Each of displays 1–4 shows its step and who has handed that step in. Pick a student to put their work up instead.</div>
          <div className="grid2">
            {displays.map((d) => (
              <DisplayControl key={d.id} display={d} state={state} onPush={push} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DisplayControl({ display, state, onPush }) {
  const isStep = display.id <= 4;
  const step = display.payload?.step || display.id;
  const [who, setWho] = useState(display.payload?.user_id || "");
  const [show, setShow] = useState(display.payload?.show || "results");
  const people = state?.members?.filter((m) => m.admitted) || [];
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
            <button className="btn sm primary" disabled={!who} onClick={() => onPush(display.id, { mode: "student", payload: { experiment: state.experiment, step, user_id: Number(who), show } })}>
              Put it up
            </button>
            <button className="btn sm ghost" onClick={() => onPush(display.id, { mode: "step", payload: { experiment: state.experiment, step } })}>
              Back to the step
            </button>
          </div>
        </>
      ) : (
        <div className="row" style={{ gap: 6 }}>
          <button className={`btn sm ${display.mode === "grafana" ? "primary" : "ghost"}`} onClick={() => onPush(display.id, { mode: "grafana", payload: {} })}>
            Hardware
          </button>
          <button className={`btn sm ${display.mode === "leaderboard" ? "primary" : "ghost"}`} onClick={() => onPush(display.id, { mode: "leaderboard", payload: { experiment: state.experiment } })}>
            Leaderboard
          </button>
        </div>
      )}
    </div>
  );
}
