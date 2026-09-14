import React, { useEffect, useState } from "react";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";

const optVal = (o) => (typeof o === "object" ? o.value : o);
const optLabel = (o) => (typeof o === "object" ? o.label || o.value : String(o));

function RunPicker({ p, value, onChange }) {
  const [runs, setRuns] = useState([]);
  useEffect(() => {
    const q = new URLSearchParams({ status: "succeeded", kind: "step", limit: "50" });
    if (p.experiment) q.set("experiment", p.experiment);
    if (p.step) q.set("step", String(p.step));
    api(`/runs?${q}`).then(setRuns).catch(() => setRuns([]));
  }, [p.experiment, p.step]);
  return (
    <select value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
      <option value="">— none —</option>
      {runs.map((r) => (
        <option key={r.id} value={r.id}>
          #{r.id} · {r.label} · {r.username} · {fmtTime(r.created_at)}
        </option>
      ))}
    </select>
  );
}

/** Schema-driven form. params: [{key,label,type,min,max,step,default,options,help}] */
export default function ParamsForm({ params, values, onChange, disabled }) {
  const set = (k, v) => onChange({ ...values, [k]: v });
  return (
    <div className="stack" style={{ gap: 10 }}>
      {params.map((p) => {
        const v = values[p.key] ?? p.default;
        const t = p.type || "text";
        if (t === "bool")
          return (
            <label key={p.key} className="check">
              <input type="checkbox" checked={!!v} disabled={disabled} onChange={(e) => set(p.key, e.target.checked)} />
              <span>{p.label}</span>
              {p.help && <span className="help">{p.help}</span>}
            </label>
          );
        if (t === "int" || t === "float") {
          const hasRange = p.min !== undefined && p.max !== undefined;
          const step = p.step ?? (t === "int" ? 1 : 0.01);
          return (
            <label key={p.key} className="field">
              <span>
                {p.label} <b style={{ color: "var(--raw)", fontWeight: 600, marginLeft: 6 }}>{v}</b>
              </span>
              <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                {hasRange && <input type="range" min={p.min} max={p.max} step={step} value={v ?? p.min} disabled={disabled} onChange={(e) => set(p.key, t === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))} />}
                <input type="number" style={{ width: hasRange ? 110 : "100%" }} min={p.min} max={p.max} step={step} value={v ?? ""} disabled={disabled} onChange={(e) => set(p.key, e.target.value === "" ? null : t === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value))} />
              </div>
              {p.help && <span className="help">{p.help}</span>}
            </label>
          );
        }
        if (t === "select")
          return (
            <label key={p.key} className="field">
              <span>{p.label}</span>
              <select value={v ?? ""} disabled={disabled} onChange={(e) => set(p.key, e.target.value)}>
                {(p.options || []).map((o) => (
                  <option key={optVal(o)} value={optVal(o)}>
                    {optLabel(o)}
                  </option>
                ))}
              </select>
              {p.help && <span className="help">{p.help}</span>}
            </label>
          );
        if (t === "multiselect") {
          const cur = Array.isArray(v) ? v : [];
          return (
            <div key={p.key} className="field">
              <span>{p.label}</span>
              <div className="row" style={{ gap: 8 }}>
                {(p.options || []).map((o) => (
                  <label key={optVal(o)} className="check small">
                    <input type="checkbox" checked={cur.includes(optVal(o))} disabled={disabled} onChange={(e) => set(p.key, e.target.checked ? [...cur, optVal(o)] : cur.filter((x) => x !== optVal(o)))} />
                    {optLabel(o)}
                  </label>
                ))}
              </div>
              {p.help && <span className="help">{p.help}</span>}
            </div>
          );
        }
        if (t === "textarea")
          return (
            <label key={p.key} className="field">
              <span>{p.label}</span>
              <textarea value={v ?? ""} disabled={disabled} onChange={(e) => set(p.key, e.target.value)} />
              {p.help && <span className="help">{p.help}</span>}
            </label>
          );
        if (t === "run")
          return (
            <label key={p.key} className="field">
              <span>{p.label}</span>
              <RunPicker p={p} value={v} onChange={(x) => set(p.key, x)} />
              {p.help && <span className="help">{p.help}</span>}
            </label>
          );
        return (
          <label key={p.key} className="field">
            <span>{p.label}</span>
            <input type="text" value={v ?? ""} disabled={disabled} onChange={(e) => set(p.key, e.target.value)} />
            {p.help && <span className="help">{p.help}</span>}
          </label>
        );
      })}
    </div>
  );
}
