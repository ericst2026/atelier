import React, { useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import Brand from "../components/Brand";
import { LanguageSelect, useT } from "../i18n";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** Anyone can ask for an account. A student is let in at once; a teacher account
 *  waits for an admin to approve it. */
export default function SignUp() {
  const { user, ready, login } = useAuth();
  const t = useT();
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
    if (!username.trim()) return setError(t("auth.signUp.chooseUsername"));
    if (password.length < 4) return setError(t("auth.signUp.passwordShort"));
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
          setError(t("auth.signUp.signInFailed", { message: err.message }));
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
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <Brand size={36} className="big" />
            <div className="muted">{t("auth.signUp.subtitle")}</div>
          </div>
          <LanguageSelect />
        </div>
        {pending ? (
          <>
            <div className="tip">{t("auth.signUp.pending")}</div>
            <Link className="btn" to="/login">
              {t("auth.signUp.backToSignIn")}
            </Link>
          </>
        ) : (
          <>
            <label className="field">
              <span>{t("auth.username")}</span>
              <input type="text" value={username} autoFocus autoComplete="username" onChange={(e) => setUsername(e.target.value)} />
              <span className="help">{t("auth.signUp.usernameHelp")}</span>
            </label>
            <label className="field">
              <span>{t("auth.signUp.yourName")}</span>
              <input type="text" value={name} autoComplete="name" onChange={(e) => setName(e.target.value)} />
              <span className="help">{t("auth.signUp.nameHelp")}</span>
            </label>
            <label className="field">
              <span>{t("auth.password")}</span>
              <input type="password" value={password} autoComplete="new-password" onChange={(e) => setPassword(e.target.value)} />
            </label>
            <label className="field">
              <span>{t("auth.signUp.iAmA")}</span>
              <select value={role} onChange={(e) => setRole(e.target.value)}>
                <option value="student">{t("common.role.student")}</option>
                <option value="teacher">{t("common.role.teacher")}</option>
              </select>
              <span className="help">
                {role === "teacher" ? t("auth.signUp.teacherHelp") : t("auth.signUp.studentHelp")}
              </span>
            </label>
            {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
            <button className="btn primary" type="submit" disabled={busy}>
              {busy ? t("auth.signUp.creating") : t("auth.signUp.create")}
            </button>
            <div className="help">
              {t("auth.signUp.alreadyHave")} <Link to="/login">{t("auth.signIn")}</Link>
              {t("auth.signUp.alreadyHaveAfter")}
            </div>
          </>
        )}
      </form>
    </div>
  );
}
