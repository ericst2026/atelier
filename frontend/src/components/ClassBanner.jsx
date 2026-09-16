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
  const isAdmin = user?.role === "admin";
  const waiting = s.members.filter((m) => !m.admitted).length;
  // the class is the teacher's to let people into — another teacher sitting in asks
  // like anyone else. Only the teacher running it, and an admin, are in already.
  const canOpen = admitted || teaching || isAdmin;
  const mayAsk = !teaching && !isAdmin && !admitted;
  return (
    <div className="panel" style={{ marginBottom: 14, borderColor: canOpen ? "var(--kept)" : "var(--raw)" }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <b>In class now: {s.title || s.experiment}</b>
          <div className="muted small">
            {teaching
              ? `Yours — ${s.members.filter((m) => m.admitted).length} in the class, ${waiting} waiting`
              : admitted
                ? `You are in ${s.teacher}'s class. Open the experiment and work through the steps.`
                : asked
                  ? `${s.teacher} has your request — wait to be let in.`
                  : isAdmin
                    ? `${s.teacher} is teaching it.`
                    : `Ask ${s.teacher} to let you in.`}
          </div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {mayAsk && (
            <button className="btn primary" onClick={ask} disabled={busy || asked}>
              {asked ? "Waiting…" : "Ask to join"}
            </button>
          )}
          {canOpen ? (
            <Link className="btn" to={`/experiments/${s.experiment}?class=${s.id}`}>
              Open it
            </Link>
          ) : (
            <button className="btn" disabled title="Your teacher has to let you in first">
              Open it
            </button>
          )}
        </div>
      </div>
      {!staff && mine.length > 0 && <div className="help" style={{ marginTop: 6 }}>On your own you may also run: {mine.join(", ")}.</div>}
    </div>
  );
}
