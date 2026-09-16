import React, { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** Your own account: who you are signed in as, what you may run on your own, and
 *  the one thing you can change yourself — your password. */
export default function Account() {
  const { user } = useAuth();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [msg, setMsg] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const mine = user?.self_experiments || [];
  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setMsg(null);
    if (next !== again) {
      setError("The two new passwords are not the same.");
      return;
    }
    setBusy(true);
    try {
      await api("/auth/password", { method: "POST", body: { current, new: next } });
      setMsg("Your password is changed.");
      setCurrent("");
      setNext("");
      setAgain("");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <main className="page">
      <div className="hero">
        <div>
          <h1>Your account</h1>
          <p className="muted">
            Signed in as {user?.name || user?.username} · {user?.role}
          </p>
        </div>
      </div>
      <div className="grid2">
        <form className="panel stack" onSubmit={submit}>
          <h3>Change your password</h3>
          <label className="field">
            <span>Your password now</span>
            <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
          </label>
          <label className="field">
            <span>New password</span>
            <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
            <span className="help">At least four characters.</span>
          </label>
          <label className="field">
            <span>New password again</span>
            <input type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} />
          </label>
          {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
          {msg && <div style={{ color: "var(--kept)" }}>{msg}</div>}
          <button className="btn primary" type="submit" disabled={busy || !current || next.length < 4}>
            {busy ? "Changing…" : "Change it"}
          </button>
        </form>
        <div className="panel stack">
          <h3>What you may run</h3>
          {user?.role === "admin" ? (
            <p className="muted">As an admin you can run anything, and you decide what everyone else may run alone.</p>
          ) : mine.length ? (
            <>
              <p className="muted">On your own, whenever you like:</p>
              <ul className="muted">
                {mine.map((slug) => (
                  <li key={slug}>{slug}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="muted">Nothing on your own yet — an admin grants that per experiment. In class you work on whatever your teacher has started, once they let you in.</p>
          )}
          <Link className="btn sm" to="/">
            Back to the experiments
          </Link>
        </div>
      </div>
    </main>
  );
}
