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
      {data &&
        sections(data).map((sec) => (
          <section key={sec.id} className="expsection">
            <div className="expsection-head">
              <h2>{sec.title}</h2>
              <span className="muted small">
                {sec.experiments.length} experiment{sec.experiments.length > 1 ? "s" : ""}
                {sec.done ? ` · ${sec.done} finished` : ""}
              </span>
            </div>
            {sec.summary && <p className="muted small">{sec.summary}</p>}
            <div className="cards">
              {sec.experiments.map((e) => (
                <ExperimentCard key={e.slug} e={e} />
              ))}
            </div>
          </section>
        ))}
    </main>
  );
}

/** Categories in the order categories.yaml lists them, empty ones dropped, anything
 * uncategorised last under "Other". Experiments keep their own order inside a section. */
function sections(data) {
  const cats = [...(data.categories || []), { id: "other", title: "Other", summary: "" }];
  const byCat = {};
  for (const e of data.experiments || []) (byCat[e.category || "other"] ||= []).push(e);
  const known = new Set(cats.map((c) => c.id));
  for (const id of Object.keys(byCat)) if (!known.has(id)) byCat.other = [...(byCat.other || []), ...byCat[id]];
  return cats
    .filter((c) => byCat[c.id]?.length)
    .map((c) => ({ ...c, experiments: byCat[c.id], done: byCat[c.id].filter((e) => (e.progress?.best_step || 0) >= 4).length }));
}

function ExperimentCard({ e }) {
  const p = e.progress || {};
  return (
    <Link to={`/experiments/${e.slug}`} className="expcard">
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
}
