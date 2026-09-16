import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import ClassBanner from "../components/ClassBanner";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";
import { useAuth } from "../lib/auth";

export default function Home() {
  const { user, isTeacher, isAdmin, refresh } = useAuth();
  const [data, setData] = useState(null);
  const [cls, setCls] = useState(null);
  const [past, setPast] = useState([]);
  const [error, setError] = useState(null);
  useEffect(() => {
    api("/experiments").then(setData).catch((e) => setError(e.message));
    const poll = () => {
      refresh(); // a permission granted a minute ago should show up without a reload
      api("/class").then(setCls).catch(() => {});
      api("/class/history?limit=12").then((d) => setPast(d.sessions || [])).catch(() => {});
    };
    poll();
    const t = setInterval(poll, 8000);
    return () => clearInterval(t);
  }, [refresh]);
  const granted = user?.self_experiments || [];
  const byslug = Object.fromEntries((data?.experiments || []).map((e) => [e.slug, e]));
  // the class half: what is running now, and the classes that have been held
  const now = cls?.running ? cls.session : null;
  const held = past.filter((h) => h.ended_at);
  // the self half: what an admin granted this person — and for an admin, everything,
  // since their permission is the role itself and never a granted list
  const all = data?.experiments || [];
  const own = isAdmin ? all : all.filter((e) => granted.includes(e.slug));
  return (
    <main className="page">
      <div className="hero">
        <div>
          <h1>Experiments</h1>
          <p className="muted">Four steps each. Run them on the server with the standard code, or write your own for the steps that allow it.</p>
        </div>
        <div className="muted small">signed in as {user?.name || user?.username}</div>
      </div>
      <ClassBanner />
      {error && <div className="empty">{error}</div>}
      {data && isTeacher && Object.keys(data.errors || {}).length > 0 && (
        <div className="panel" style={{ borderColor: "var(--dup)", marginBottom: 14 }}>
          <b>Some experiments failed to load</b>
          {Object.entries(data.errors).map(([k, v]) => (
            <div key={k} className="small mono">
              {k}: {v}
            </div>
          ))}
        </div>
      )}
      <section className="expsection">
        <div className="expsection-head">
          <h2>In class</h2>
          <span className="muted small">{now ? "one running" : "nothing running"}</span>
        </div>
        {now ? (
          <div className="cards">
            {byslug[now.experiment] ? (
              <ExperimentCard e={byslug[now.experiment]} note={`${now.teacher}'s class · ${now.me?.admitted ? "you are in" : now.me?.asked ? "waiting to be let in" : "ask to join"}`} />
            ) : (
              <div className="panel">
                {now.title || now.experiment} · {now.teacher}
              </div>
            )}
          </div>
        ) : (
          <p className="muted small">No class is running. Your teacher starts one and lets you in.</p>
        )}
        {held.length > 0 && (
          <div className="panel tight" style={{ marginTop: 10 }}>
            <div className="small muted" style={{ marginBottom: 6 }}>
              Classes already held
            </div>
            <div className="stack" style={{ gap: 4 }}>
              {held.map((h) => (
                <div key={h.id} className="row small" style={{ justifyContent: "space-between" }}>
                  <span>
                    {h.title || h.experiment} <span className="faint">· {h.teacher}</span>
                  </span>
                  <span className="faint">{fmtTime(h.started_at)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="expsection">
        <div className="expsection-head">
          <h2>On your own</h2>
          <span className="muted small">
            {own.length} experiment{own.length === 1 ? "" : "s"}
            {isAdmin ? " · you are an admin, so all of them" : " granted"}
          </span>
        </div>
        {own.length === 0 ? (
          <p className="muted small">{isAdmin ? "No experiments are loaded." : "An admin has not given you anything to run by yourself yet."}</p>
        ) : (
          <div className="cards">
            {own.map((e) => (
              <ExperimentCard key={e.slug} e={e} />
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function ExperimentCard({ e, note }) {
  const p = e.progress || {};
  return (
    <Link to={`/experiments/${e.slug}`} className="expcard">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>{e.title}</h2>
        <span className="tag">{e.gpus ? `${e.gpus} GPU` : "CPU"}</span>
      </div>
      <div className="muted">{e.summary}</div>
      {note && <div className="small" style={{ color: "var(--raw)" }}>{note}</div>}
      <div className="steps" title={`${p.best_step || 0} of 4 steps done`}>
        {e.steps.map((s) => (
          <i key={s.index} className={s.index <= (p.best_step || 0) ? "done" : ""} />
        ))}
      </div>
      <div className="row small muted" style={{ justifyContent: "space-between" }}>
        <span>
          {p.best_step ? `step ${p.best_step} of 4 done` : "not started"}
          {p.submissions ? ` · ${p.submissions} submission${p.submissions > 1 ? "s" : ""}` : ""}
        </span>
        <span className="tags">
          {e.tags.map((t) => (
            <span key={t} className="tag">
              {t}
            </span>
          ))}
        </span>
      </div>
    </Link>
  );
}
