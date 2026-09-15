export const fmtInt = (v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Math.round(Number(v)).toLocaleString());
export const fmtNum = (v, d = 3) => {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "–";
  const n = Number(v);
  if (Number.isInteger(n) && Math.abs(n) >= 1000) return n.toLocaleString();
  if (Math.abs(n) >= 1e6) return n.toExponential(2);
  return Number(n.toFixed(d)).toLocaleString(undefined, { maximumFractionDigits: d });
};
/** Short tick labels: 60M, 1.2k, 0.35, 1.5e-4. */
export const fmtAxis = (v) => {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const n = Number(v);
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a >= 1000) return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
  if (a < 0.001) return n.toExponential(1);
  return String(Number(n.toPrecision(3)));
};
export const fmtPct =(v, d = 1) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : `${(Number(v) * 100).toFixed(d)}%`);
export const fmtMs = (ms) => {
  if (ms === null || ms === undefined) return "–";
  const s = Number(ms) / 1000;
  if (s < 1) return `${Math.round(ms)} ms`;
  if (s < 90) return `${s.toFixed(1)} s`;
  if (s < 5400) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(2)} h`;
};
export const fmtBytes = (b) => {
  if (b === null || b === undefined) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = Number(b);
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
};
export const fmtDuration = (sec) => fmtMs(Number(sec || 0) * 1000);
export const fmtTime = (iso) => (iso ? new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`).toLocaleString() : "–");
export const fmtMetric = (m) => {
  const f = m.fmt || "num";
  if (f === "int") return fmtInt(m.value);
  if (f === "pct") return fmtPct(m.value);
  if (f === "ms") return fmtMs(m.value);
  if (f === "bytes") return fmtBytes(m.value);
  if (f === "text") return m.value === null || m.value === undefined ? "–" : String(m.value);
  return fmtNum(m.value);
};
export const fmtCell = (v, fmt) => {
  if (v === null || v === undefined) return "";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (fmt === "int") return fmtInt(v);
  if (fmt === "pct") return fmtPct(v);
  if (fmt === "num") return fmtNum(v);
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : fmtNum(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
};
