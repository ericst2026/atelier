import React, { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useT } from "../i18n";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

/** What a student is allowed to work on right now: the class their teacher is
 *  running, and whatever an admin has let them do on their own. */
export default function ClassBanner() {
  const { user } = useAuth();
  const t = useT();
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
        <b>{t("class.banner.noClass")}</b>{" "}
        {staff ? (
          <span className="muted">
            {t("class.banner.startOne.before")}
            <Link to="/teacher">{t("class.banner.startOne.link")}</Link>
            {t("class.banner.startOne.after")}
          </span>
        ) : mine.length ? (
          <span className="muted">{t("class.banner.ownOnly", { list: mine.join(", ") })}</span>
        ) : (
          <span className="muted">{t("class.banner.lookAround")}</span>
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
          <b>{t("class.banner.inClassNow", { title: s.title || s.experiment })}</b>
          <div className="muted small">
            {teaching
              ? t("class.banner.yours", { n: s.members.filter((m) => m.admitted).length, waiting })
              : admitted
                ? t("class.banner.admitted", { teacher: s.teacher })
                : asked
                  ? t("class.banner.asked", { teacher: s.teacher })
                  : isAdmin
                    ? t("class.banner.adminView", { teacher: s.teacher })
                    : t("class.banner.askTeacher", { teacher: s.teacher })}
          </div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          {mayAsk && (
            <button className="btn primary" onClick={ask} disabled={busy || asked}>
              {asked ? t("class.banner.waiting") : t("class.banner.askToJoin")}
            </button>
          )}
          {canOpen ? (
            <Link className="btn" to={`/experiments/${s.experiment}?class=${s.id}`}>
              {t("class.banner.openIt")}
            </Link>
          ) : (
            <button className="btn" disabled title={t("class.banner.letInFirst")}>
              {t("class.banner.openIt")}
            </button>
          )}
        </div>
      </div>
      {!staff && mine.length > 0 && (
        <div className="help" style={{ marginTop: 6 }}>
          {t("class.banner.alsoRun", { list: mine.join(", ") })}
        </div>
      )}
    </div>
  );
}
