import React, { useCallback, useEffect, useState } from "react";
import { Pause, Play, Square } from "lucide-react";
import ParamsForm from "./ParamsForm";
import { useT } from "../i18n";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtTime } from "../lib/format";

/** The class: one experiment at a time, who is in it, and what the wall shows.
 *  Students do not pick the experiment — this is where it is chosen. One class
 *  runs in the building, so another teacher's has to end before yours starts. */
export default function ClassPanel() {
  const { user } = useAuth();
  const t = useT();
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
      {running && (
        <div className="panel stack" style={{ borderColor: ours ? "var(--kept)" : "var(--line)" }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>
              {t("class.panel.running", { name: s.name || s.title || s.experiment })}
              <span className="faint small"> · {s.title || s.experiment} · {s.teacher} · {t("class.panel.since", { time: fmtTime(s.started_at) })}</span>
            </h3>
            {ours && (
              <div className="row" style={{ gap: 6 }}>
                <button className="btn sm" onClick={pause} title={t("class.panel.pauseTitle")}>
                  <Pause size={13} /> {t("class.panel.pause")}
                </button>
                <button className="btn sm danger" onClick={stop}>
                  <Square size={13} /> {t("class.panel.end")}
                </button>
              </div>
            )}
          </div>

          {ours && (
        <div className="grid2">
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b>{t("class.panel.asking", { n: waiting.length })}</b>
            {waiting.length === 0 && <div className="help">{t("class.panel.nobodyWaiting")}</div>}
            {waiting.map((m) => (
              <div key={m.user_id} className="row" style={{ justifyContent: "space-between" }}>
                <span>
                  {m.name || m.username} <span className="faint small">{m.username}</span>
                </span>
                <div className="row" style={{ gap: 6 }}>
                  <button className="btn sm good" onClick={() => admit(m.user_id, true)}>
                    {t("class.panel.accept")}
                  </button>
                  <button className="btn sm danger" onClick={() => deny(m.user_id)}>
                    {t("class.panel.deny")}
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b>{t("class.panel.inClass", { n: inClass.length })}</b>
            {inClass.length === 0 && <div className="help">{t("class.panel.nobodyYet")}</div>}
            {inClass.map((m) => (
              <div key={m.user_id} className="row" style={{ justifyContent: "space-between" }}>
                <span>
                  {m.name || m.username} <span className="faint small">{m.username}</span>
                </span>
                <button className="btn sm ghost" onClick={() => admit(m.user_id, false)}>
                  {t("common.remove")}
                </button>
              </div>
            ))}
          </div>
        </div>
          )}
        </div>
      )}

      <div className="panel stack">
        <h3>{t("class.start.heading")}</h3>
        {running && <div className="help">{ours ? t("class.start.yoursRunning") : t("class.start.othersRunning", { teacher: s.teacher })}</div>}
        {teachable.length === 0 ? (
          <div className="help">{t("class.start.nothingToTeach")}</div>
        ) : (
          <div className="stack" style={{ gap: 10 }}>
            <label className="field">
              <span>{t("class.start.nameIt")}</span>
              <input
                type="text"
                value={name}
                placeholder={t("class.start.namePlaceholder")}
                onChange={(e) => setName(e.target.value)}
                disabled={running}
              />
              <span className="help">{t("class.start.nameHelp")}</span>
            </label>
            <label className="field">
              <span>{t("class.start.experiment")}</span>
              <select value={pick} onChange={(e) => setPick(e.target.value)} disabled={running} style={{ maxWidth: 420 }}>
                <option value="">{t("class.start.choose")}</option>
                {teachable.map((e) => (
                  <option key={e.slug} value={e.slug}>
                    {e.title}
                  </option>
                ))}
              </select>
            </label>
            {pick && spec && startFields(spec, t).length > 0 && (
          <div className="inset stack" style={{ padding: 12, gap: 8 }}>
            <b className="small">{t("class.start.startsFrom")}</b>
            <div className="help">{t("class.start.startsFromHelp")}</div>
            <ParamsForm params={startFields(spec, t)} values={startParams} onChange={setStartParams} experiment={pick} step={1} />
          </div>
        )}
            <button className="btn primary" onClick={start} disabled={!pick || !name.trim() || running}>
              <Play size={14} /> {t("class.start.button")}
            </button>
          </div>
        )}
        {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
        <div className="help">{t("class.start.onlyThis")}</div>
      </div>

    </div>
  );
}

/** The wall: one control per screen, so the teacher can follow a step or put a
 *  student up. Lives in its own tab, but needs the same class state. */
export function WallPanel() {
  const { user } = useAuth();
  const t = useT();
  const [state, setState] = useState(null);
  const [displays, setDisplays] = useState([]);
  const [spec, setSpec] = useState(null); // to know which steps can be handed in
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
  // the wall is the running class's: only its teacher, or an admin, points it
  const canSet = Boolean(session && (session.me?.mine || user?.role === "admin"));
  useEffect(() => {
    if (!session?.experiment) return setSpec(null);
    api(`/experiments/${session.experiment}`).then(setSpec).catch(() => setSpec(null));
  }, [session?.experiment]);
  return (
    <div className="stack">
      <div className="panel stack">
        <h3>{t("class.wall.heading")}</h3>
        <div className="help">{session ? t("class.wall.helpRunning") : t("class.wall.helpIdle")}</div>
        {session && !canSet && <div className="tip">{t("class.wall.notYours", { teacher: session.teacher })}</div>}
        <div className="wallrow">
          {displays.map((d) => (
            <DisplayControl key={d.id} display={d} session={session} spec={spec} onPush={push} canSet={canSet} />
          ))}
        </div>
      </div>
    </div>
  );
}

function DisplayControl({ display, session, spec, onPush, canSet }) {
  const t = useT();
  const isStep = display.id <= 4;
  const payload = display.payload || {};
  const step = display.id; // screen 1 is step 1, and so on
  const [show, setShow] = useState(display.mode === "student" ? payload.show || "results" : display.mode === "standard" ? "standard" : "explanation");
  const [who, setWho] = useState(payload.user_id || "");
  // the step's slide behind whatever the screen shows, instead of a plain background
  const [onSlide, setOnSlide] = useState(Boolean(payload.bg));
  // only this step's own_code decides whether there is anything to hand in
  const ownCode = Boolean((spec?.steps || [])[Number(step) - 1]?.own_code);
  // who has something to show for this step in this class: anyone with a finished
  // run of it has results; only those who handed their own code in have code
  // and anyone who has started a run of it at all can be watched while it goes
  const [who2, setWho2] = useState({ ran: [], handed_in: [], running: [] });
  useEffect(() => {
    if (!session) return setWho2({ ran: [], handed_in: [], running: [] });
    const fetchPeople = () =>
      api(`/class/step/${Number(step)}/people`)
        .then((r) => setWho2({ ran: r.ran || [], handed_in: r.handed_in || [], running: r.running || [] }))
        .catch(() => setWho2({ ran: [], handed_in: [], running: [] }));
    fetchPeople();
    const t = setInterval(fetchPeople, 8000); // a student finishing mid-lesson appears without a reload
    return () => clearInterval(t);
  }, [session?.id, step]);
  const handed = who2.handed_in;
  const union = [...who2.ran, ...who2.handed_in].filter((x, i, a) => a.findIndex((y) => y.user_id === x.user_id) === i);
  const people = show === "code" ? handed : show === "running" ? who2.running : union;
  const apply = () => {
    const base = { session_id: session?.id, experiment: session?.experiment, step: Number(step), ...(onSlide ? { bg: true } : {}) };
    if (show === "explanation") return onPush(display.id, { mode: "step", payload: base });
    if (show === "standard") return onPush(display.id, { mode: "standard", payload: base });
    onPush(display.id, { mode: "student", payload: { ...base, user_id: Number(who), show } });
  };
  const needsPerson = show === "results" || show === "code" || show === "running";
  // what the screen is on now; modes without a label of their own show as they are
  const showing =
    display.mode === "student"
      ? payload.show === "running"
        ? t("class.wall.shown.running")
        : payload.show === "code"
          ? t("class.wall.shown.code")
          : t("class.wall.shown.results")
      : ["standard", "step", "grafana", "leaderboard"].includes(display.mode)
        ? t(`class.wall.shown.${display.mode}`)
        : display.mode;
  // the option can be left behind when the step changes under it
  useEffect(() => {
    if (!ownCode && show === "code") setShow("explanation");
  }, [ownCode, show]);
  return (
    <div className="inset stack" style={{ padding: 10, gap: 8 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>{isStep ? `${t("common.stepN", { n: display.id })}${spec?.steps?.[display.id - 1] ? ` · ${spec.steps[display.id - 1].title}` : ""}` : t("class.wall.leaderboardHardware")}</b>
        <a className="small" href={`/display/${display.id}`} target="_blank" rel="noreferrer">
          {t("class.wall.open")}
        </a>
      </div>
      <div className="small muted">
        {t("class.wall.showing", { what: showing })}
      </div>
      {isStep ? (
        <>
          <select value={show} onChange={(e) => setShow(e.target.value)} disabled={!canSet}>
            <option value="explanation">{t("class.wall.show.explanation")}</option>
            <option value="standard">{t("class.wall.show.standard")}</option>
            <option value="results">{t("class.wall.show.results")}</option>
            {ownCode && <option value="code">{t("class.wall.show.code")}</option>}
            <option value="running">{t("class.wall.show.running")}</option>
          </select>
          {needsPerson && (
            <select value={who} onChange={(e) => setWho(e.target.value)} disabled={!canSet}>
              <option value="">
                {people.length ? t("class.wall.who") : show === "code" ? t("class.wall.nobodyHandedIn") : show === "running" ? t("class.wall.nobodyRan") : t("class.wall.nobodyFinished")}
              </option>
              {people.map((m) => (
                <option key={m.user_id} value={m.user_id}>
                  {m.name}
                  {show === "running" && m.status ? ` · ${m.status === "running" ? t("class.wall.runningPct", { pct: Math.round(m.progress_pct || 0) }) : t(`common.status.${m.status}`)}` : ""}
                </option>
              ))}
            </select>
          )}
          <label className="row small muted" style={{ gap: 6 }}>
            <input type="checkbox" checked={onSlide} onChange={(e) => setOnSlide(e.target.checked)} disabled={!canSet} />
            {t("class.wall.onSlide")}
          </label>
          <button className="btn sm primary" onClick={apply} disabled={!canSet || (needsPerson && !who)}>
            {t("class.wall.putItUp")}
          </button>
        </>
      ) : (
        <div className="row" style={{ gap: 6 }}>
          <button className={`btn sm ${display.mode === "grafana" ? "primary" : "ghost"}`} disabled={!canSet} onClick={() => onPush(display.id, { mode: "grafana", payload: {} })}>
            {t("class.wall.hardware")}
          </button>
          <button className={`btn sm ${display.mode === "leaderboard" ? "primary" : "ghost"}`} disabled={!canSet} onClick={() => onPush(display.id, { mode: "leaderboard", payload: { experiment: session?.experiment } })}>
            {t("class.wall.leaderboard")}
          </button>
        </div>
      )}
    </div>
  );
}

/** What an experiment needs from outside itself, which a student cannot supply:
 *  another experiment's run, and the prepared models or datasets that stand in for
 *  one. A step that builds on the step before it is not here — that is the
 *  student's own work, and they pick their own. */
function startFields(spec, t) {
  const out = [];
  for (const st of spec?.steps || []) {
    for (const p of st.params || []) {
      const fromElsewhere =
        (p.type === "run" && p.experiment && p.experiment !== spec.slug) ||
        p.type === "material" ||
        (p.type === "select" && /source/.test(p.key));
      if (!fromElsewhere || out.some((x) => x.key === p.key)) continue;
      out.push(st.index === 1 ? p : { ...p, label: t("class.start.fromStep", { label: p.label, n: st.index }) });
    }
  }
  return out;
}
