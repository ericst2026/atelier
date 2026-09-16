import React, { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** What a student is allowed to work on right now: the class their teacher is
 *  running, and whatever an admin has let them do on their own. */
export default function ClassBanner() {
  const { user, isTeacher } = useAuth();
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(() => api("/class").then(setState).catch(() => {}), []);
  useEffect(() => {
    load();
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [load]);
  if (!state) return null;
  const mine = user?.self_experiments || [];
  const ask = async () => {
    setBusy(true);
    try {
      setState(await api("/class/join", { method: "POST", body: {} }));
    } finally {
      setBusy(false);
    }
  };
  if (!state.running) {
    return (
      <div className="panel" style={{ marginBottom: 14 }}>
        <b>No class is running.</b>{" "}
        {isTeacher ? (
          <span className="muted">
            Start one from the <Link to="/teacher">teacher page</Link>.
          </span>
        ) : mine.length ? (
          <span className="muted">You can work on your own on: {mine.join(", ")}.</span>
        ) : (
          <span className="muted">You can look around, but running a step needs a class, or an admin's permission to work on your own.</span>
        )}
      </div>
    );
  }
  const admitted = state.me?.admitted;
  const asked = state.me?.asked;
  return (
    <div className="panel" style={{ marginBottom: 14, borderColor: admitted || isTeacher ? "var(--kept)" : "var(--raw)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <b>In class now: {state.title || state.experiment}</b>
          <div className="muted small">
            {isTeacher
              ? `${state.members.filter((m) => m.admitted).length} in the class, ${state.members.filter((m) => !m.admitted).length} waiting`
              : admitted
                ? "You are in. Open the experiment and work through the steps."
                : asked
                  ? "Your teacher has your request — wait to be let in."
                  : "Ask your teacher to let you in."}
          </div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {!isTeacher && !admitted && (
            <button className="btn primary" onClick={ask} disabled={busy || asked}>
              {asked ? "Waiting…" : "Ask to join"}
            </button>
          )}
          <Link className="btn" to={`/experiments/${state.experiment}`}>
            Open it
          </Link>
        </div>
      </div>
      {!isTeacher && mine.length > 0 && <div className="help" style={{ marginTop: 6 }}>On your own you may also run: {mine.join(", ")}.</div>}
    </div>
  );
}
