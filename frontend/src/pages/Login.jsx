import React, { useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { user, ready, login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  if (ready && user) return <Navigate to={loc.state?.from || "/"} replace />;
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await login(username.trim(), password);
      nav(loc.state?.from || "/", { replace: true });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 20 }}>
      <form className="panel" onSubmit={submit} style={{ width: 360, display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <div className="brand" style={{ fontSize: 26 }}>
            Ate<b>lier</b>
          </div>
          <div className="muted">LLM experiments on the classroom GPU node</div>
        </div>
        <label className="field">
          <span>Username</span>
          <input type="text" value={username} autoFocus autoComplete="username" onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="field">
          <span>Password</span>
          <input type="password" value={password} autoComplete="current-password" onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
        <button className="btn primary" type="submit" disabled={busy || !username || !password}>
          {busy ? "Signing in…" : "Sign in"}
        </button>
        <div className="help">Ask your teacher for an account. Wall displays need no sign-in: /display/1 … /display/5.</div>
      </form>
    </div>
  );
}
