import React, { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** Anyone can ask for an account. A student is let in at once; a teacher account
 *  waits for an admin to approve it. */
export default function SignUp() {
  const { user, ready, login } = useAuth();
  const nav = useNavigate();
  const [username, setUsername] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("student");
  const [error, setError] = useState(null);
  const [pending, setPending] = useState(false);
  const [busy, setBusy] = useState(false);
  if (ready && user) return <Navigate to="/" replace />;
  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    // say what is missing instead of sitting there doing nothing
    if (!username.trim()) return setError("Choose a username.");
    if (password.length < 4) return setError("The password needs at least four characters.");
    setBusy(true);
    try {
      await api("/auth/signup", { method: "POST", body: { username: username.trim(), name: name.trim(), password, role } });
      if (role === "teacher") {
        setPending(true); // nothing to sign in to yet
      } else {
        // the account exists now; if signing in fails, say so rather than looking stuck
        try {
          await login(username.trim(), password);
          nav("/", { replace: true });
        } catch (err) {
          setError(`Your account was created, but signing in failed: ${err.message}. Try the sign-in page.`);
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 20 }}>
      <form className="panel" onSubmit={submit} style={{ width: 380, display: "flex", flexDirection: "column", gap: 14 }}>
        <div>
          <div className="brand" style={{ fontSize: 26 }}>
            Ate<b>lier</b>
          </div>
          <div className="muted">Make an account</div>
        </div>
        {pending ? (
          <>
            <div className="tip">
              Your teacher account is waiting for an admin to approve it. You will be able to sign in once they do.
            </div>
            <Link className="btn" to="/login">
              Back to sign in
            </Link>
          </>
        ) : (
          <>
            <label className="field">
              <span>Username</span>
              <input type="text" value={username} autoFocus autoComplete="username" onChange={(e) => setUsername(e.target.value)} />
              <span className="help">Letters, digits, dot, dash and underscore.</span>
            </label>
            <label className="field">
              <span>Your name</span>
              <input type="text" value={name} autoComplete="name" onChange={(e) => setName(e.target.value)} />
              <span className="help">Shown to your teacher and on the wall displays.</span>
            </label>
            <label className="field">
              <span>Password</span>
              <input type="password" value={password} autoComplete="new-password" onChange={(e) => setPassword(e.target.value)} />
            </label>
            <label className="field">
              <span>I am a</span>
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="student">student</option>
                <option value="teacher">teacher</option>
              </select>
              <span className="help">
                {role === "teacher" ? "A teacher account has to be approved by an admin before you can sign in." : "You can sign in straight away. Running experiments on your own needs an admin's permission."}
              </span>
            </label>
            {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
            <button className="btn primary" type="submit" disabled={busy}>
              {busy ? "Creating…" : "Create the account"}
            </button>
            <div className="help">
              Already have one? <Link to="/login">Sign in</Link>.
            </div>
          </>
        )}
      </form>
    </div>
  );
}
