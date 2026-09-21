import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useT } from "../i18n";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** Your own account: who you are signed in as, what you may run on your own, and
 *  the one thing you can change yourself — your password. */
export default function Account() {
  const { user } = useAuth();
  const t = useT();
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
      setError(t("account.passwordsDiffer"));
      return;
    }
    setBusy(true);
    try {
      await api("/auth/password", { method: "POST", body: { current, new: next } });
      setMsg(t("account.changed"));
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
          <h1>{t("account.title")}</h1>
          <p className="muted">
            {t("account.signedInAs", { name: user?.name || user?.username, role: user?.role ? t(`common.role.${user.role}`) : "" })}
          </p>
        </div>
      </div>
      <div className="grid2">
        <form className="panel stack" onSubmit={submit}>
          <h3>{t("account.password.title")}</h3>
          <label className="field">
            <span>{t("account.password.current")}</span>
            <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
          </label>
          <label className="field">
            <span>{t("account.password.new")}</span>
            <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
            <span className="help">{t("account.password.newHelp")}</span>
          </label>
          <label className="field">
            <span>{t("account.password.again")}</span>
            <input type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} />
          </label>
          {error && <div style={{ color: "var(--dup)" }}>{error}</div>}
          {msg && <div style={{ color: "var(--kept)" }}>{msg}</div>}
          <button className="btn primary" type="submit" disabled={busy || !current || next.length < 4}>
            {busy ? t("account.password.changing") : t("account.password.submit")}
          </button>
        </form>
        <div className="panel stack">
          <h3>{t("account.mayRun.title")}</h3>
          {user?.role === "admin" ? (
            <p className="muted">{t("account.mayRun.admin")}</p>
          ) : mine.length ? (
            <>
              <p className="muted">{t("account.mayRun.own")}</p>
              <ul className="muted">
                {mine.map((slug) => (
                  <li key={slug}>{slug}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="muted">{t("account.mayRun.nothing")}</p>
          )}
          <Link className="btn sm" to="/">
            {t("account.mayRun.back")}
          </Link>
        </div>
      </div>
    </main>
  );
}
