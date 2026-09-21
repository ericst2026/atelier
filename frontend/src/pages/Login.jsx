import React, { useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import Brand from "../components/Brand";
import { LanguageSelect, useT } from "../i18n";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { user, ready, login } = useAuth();
  const t = useT();
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
        <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <Brand size={36} className="big" />
            <div className="muted">{t("common.tagline")}</div>
          </div>
          <LanguageSelect />
        </div>
        <label className="field">
          <span>{t("auth.username")}</span>
          <input type="text" value={username} autoFocus autoComplete="username" onChange={(e) => setUsername(e.target.value)} />
        </label>
        <label className="field">
          <span>{t("auth.password")}</span>
          <input type="password" value={password} autoComplete="current-password" onChange={(e) => setPassword(e.target.value)} />
        </label>
        {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
        <button className="btn primary" type="submit" disabled={busy || !username || !password}>
          {busy ? t("auth.signingIn") : t("auth.signIn")}
        </button>
        <div className="help">
          {t("auth.noAccount")} <Link to="/signup">{t("auth.makeOne")}</Link>. {t("auth.wallNoSignIn")}
        </div>
      </form>
    </div>
  );
}
