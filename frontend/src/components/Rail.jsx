import React from "react";
import { fmtMetric } from "../lib/format";

/** The four-step rail. steps: spec steps; state: {index: {status, metrics}} */
export default function Rail({ steps, state, active, onSelect }) {
  return (
    <div className="rail">
      {steps.map((s) => {
        const st = state[s.index] || {};
        const cls = st.status === "succeeded" ? "done" : st.status === "running" || st.status === "queued" ? "running" : st.status === "failed" ? "failed" : "";
        const figs = (s.figures || []).map((f) => ({ f, m: (st.metrics || []).find((m) => m.key === f.key) })).filter((x) => x.m);
        return (
          <button key={s.index} className={`step ${cls} ${active === s.index ? "active" : ""}`} onClick={() => onSelect(s.index)}>
            <span className="num">{s.index}</span>
            <span>
              <div className="title">{s.title}</div>
              <div className="muted small">{s.summary}</div>
              {figs.length > 0 && (
                <div className="figs">
                  {figs.map(({ f, m }) => (
                    <span key={f.key}>
                      <b>{fmtMetric(m)}</b> {f.label}
                    </span>
                  ))}
                </div>
              )}
            </span>
          </button>
        );
      })}
    </div>
  );
}
