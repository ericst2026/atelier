import React, { useEffect, useState } from "react";
import { useT } from "../i18n";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";

const optVal = (o) => (typeof o === "object" ? o.value : o);
const optLabel = (o) => (typeof o === "object" ? o.label || o.value : String(o));

function RunPicker({ p, value, onChange }) {
  const t = useT();
  const [runs, setRuns] = useState([]);
  useEffect(() => {
    const q = new URLSearchParams({ status: "succeeded", kind: "step", limit: "50" });
    if (p.experiment) q.set("experiment", p.experiment);
    if (p.step) q.set("step", String(p.step));
    api(`/runs?${q}`).then(setRuns).catch(() => setRuns([]));
  }, [p.experiment, p.step]);
  return (
    <select value={value ?? ""} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}>
      <option value="">{t("params.none")}</option>
      {runs.map((r) => (
        <option key={r.id} value={r.id}>
          #{r.id} · {r.label} · {r.username} · {fmtTime(r.created_at)}
        </option>
      ))}
    </select>
  );
}

const fmtBytes = (b) => (!b ? "" : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(0)} MB` : `${Math.max(1, Math.round(b / 1e3))} kB`);

/** A prepared model or dataset: one of the materials the experiment declares for this
 *  param (or, outside an experiment step, anything suitable under materials/). */
function MaterialPicker({ p, value, onChange, disabled, experiment, step }) {
  const t = useT();
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const q = new URLSearchParams({ kind: p.kind || "model" });
    if (p.formats) q.set("formats", p.formats.join(","));
    if (p.schemas) q.set("schemas", p.schemas.join(","));
    if (experiment && step) {
      q.set("experiment", experiment);
      q.set("step", String(step));
      q.set("param", p.key);
    }
    api(`/system/materials/catalog?${q}`)
      .then((r) => { setItems(r.items || []); setError(""); })
      .catch((e) => { setItems([]); setError(String(e.message || e)); });
  }, [p.key, p.kind, (p.formats || []).join(","), (p.schemas || []).join(","), experiment, step]);
  const top = p.kind === "dataset" ? "materials/datasets/" : "materials/models/";
  if (items && items.length === 0)
    return <div className="help">{error || (experiment ? t("params.noMaterialInExperiment", { kind: t(p.kind === "dataset" ? "params.kind.dataset" : "params.kind.model") }) : t("params.noMaterialUnder", { dir: top }))}</div>;
  return (
    <select value={value ?? ""} disabled={disabled || !items} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">{items ? t("params.choose") : t("params.loading")}</option>
      {(items || []).map((m) => (
        <option key={m.path} value={m.path} disabled={!!m.problem}>
          {m.name}
          {m.format ? ` · ${m.format === "hf" ? "HuggingFace" : t("params.format.course")}` : ""}
          {m.schemas ? ` · ${(m.splits || []).join(", ") || m.schemas.join(", ")}` : ""}
          {m.bytes ? ` · ${fmtBytes(m.bytes)}` : ""}
          {m.problem ? ` · ${m.problem}` : ""}
        </option>
      ))}
    </select>
  );
}

// show_if: {key: value | [values]} — the field is shown only while those params match
const visible = (p, values, params) =>
  !p.show_if ||
  Object.entries(p.show_if).every(([k, want]) => {
    const other = params.find((x) => x.key === k);
    const cur = values[k] ?? other?.default;
    return Array.isArray(want) ? want.includes(cur) : cur === want;
  });

/** Schema-driven form. params: [{key,label,type,min,max,step,default,options,help,show_if}] */
/** `locked` holds what a teacher fixed for the class: those fields are shown as
 *  they will run, and cannot be changed. */
export default function ParamsForm({ params, values, onChange, disabled, experiment, step, locked }) {
  // `t` is the field type below, so the translator is `tr` here
  const tr = useT();
  const set = (k, v) => onChange({ ...values, [k]: v });
  return (
    <div className="stack" style={{ gap: 10 }}>
      {params.filter((p) => visible(p, values, params)).map((p) => {
        const v = values[p.key] ?? p.default;
        const t = p.type || "text";
        if (locked && p.key in locked)
          return (
            <label key={p.key} className="field">
              <span>{p.label}</span>
              <input type="text" value={describe(p, locked[p.key], tr)} readOnly disabled />
              <span className="help">{tr("params.lockedByTeacher")}</span>
            </label>
          );
        if (t === "material")
          return (
            <label key={p.key} className="field">
              <span>{p.label}</span>
              <MaterialPicker p={p} value={v} disabled={disabled} experiment={experiment} step={step} onChange={(x) => set(p.key, x)} />
              {p.help && <span className="help">{p.help}</span>}
            </label>
          );
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

/** A fixed value, in words rather than as an id where we can manage it. */
function describe(p, value, t) {
  if (value === null || value === undefined || value === "") return "—";
  if (p.type === "material") return String(value).split("/").slice(-1)[0];
  if (p.type === "run") return t("params.runN", { id: value });
  const opt = (p.options || []).find((o) => (typeof o === "object" ? o.value : o) === value);
  return opt ? (typeof opt === "object" ? opt.label : opt) : String(value);
}
