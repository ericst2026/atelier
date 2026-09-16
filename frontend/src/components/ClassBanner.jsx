import React, { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** What a student is allowed to work on right now: the class their teacher is
 *  running, and whatever an admin has let them do on their own. */
export default function ClassBanner() {
  const { user } = useAuth();
  const [state, setState] = useState(null);
  const [busy, setBusy] = useState(false);
  const load = useCallback(() => api("/class").then(setState).catch(() => {}), []);
  useEffect(() => {
    load();
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [load]);
  if (!state) return null;
  const staff = user?.role === "teacher" || user?.role === "admin";
  const mine = user?.self_experiments || [];
  const ask = async () => {
    setBusy(true);
    try {
      setState(await api("/class/join", { method: "POST", body: {} }));
    } finally {
      setBusy(false);
    }
  };
  const s = state.session;
  if (!state.running || !s) {
    return (
      <div className="panel" style={{ marginBottom: 14 }}>
        <b>No class is running.</b>{" "}
        {staff ? (
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
  const admitted = s.me?.admitted;
  const asked = s.me?.asked;
  const teaching = s.me?.mine;
  const waiting = s.members.filter((m) => !m.admitted).length;
  return (
    <div className="panel" style={{ marginBottom: 14, borderColor: admitted || staff ? "var(--kept)" : "var(--raw)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <b>In class now: {s.title || s.experiment}</b>
          <div className="muted small">
            {staff
              ? `${teaching ? "Yours" : `${s.teacher} is teaching it`} — ${s.members.filter((m) => m.admitted).length} in the class, ${waiting} waiting`
              : admitted
                ? `You are in ${s.teacher}'s class. Open the experiment and work through the steps.`
                : asked
                  ? `${s.teacher} has your request — wait to be let in.`
                  : `Ask ${s.teacher} to let you in.`}
          </div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {!staff && !admitted && (
            <button className="btn primary" onClick={ask} disabled={busy || asked}>
              {asked ? "Waiting…" : "Ask to join"}
            </button>
          )}
          <Link className="btn" to={`/experiments/${s.experiment}`}>
            Open it
          </Link>
        </div>
      </div>
      {!staff && mine.length > 0 && <div className="help" style={{ marginTop: 6 }}>On your own you may also run: {mine.join(", ")}.</div>}
    </div>
  );
}
