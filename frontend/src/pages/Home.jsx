import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";

export default function Home() {
  const { user, isTeacher } = useAuth();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api("/experiments").then(setData).catch((e) => setError(e.message));
  }, []);
  return (
    <main className="page">
      <div className="hero">
        <div>
          <h1>Experiments</h1>
          <p className="muted">Four steps each. Run them on the server, then build your own version in the project workspace.</p>
        </div>
        <div className="muted small">signed in as {user.name || user.username}</div>
      </div>
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
      <div className="cards">
        {(data?.experiments || []).map((e) => {
          const p = e.progress || {};
          return (
            <Link key={e.slug} to={`/experiments/${e.slug}`} className="expcard">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2>{e.title}</h2>
                <span className="tag">{e.gpus ? `${e.gpus} GPU` : "CPU"}</span>
              </div>
              <div className="muted">{e.summary}</div>
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
        })}
      </div>
    </main>
  );
}
