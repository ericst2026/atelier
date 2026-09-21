import { t } from "../i18n";

export const fmtInt = (v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : Math.round(Number(v)).toLocaleString());
/** Plain digits, never exponential: 1,234,567 and 0.00015, not 1.23e+6 and 1.5e-4.
 *  Values below the rounding step keep three significant digits instead of
 *  collapsing to 0.000 — a learning rate or a loss delta has to stay readable. */
export const fmtNum = (v, d = 3) => {
  if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return "–";
  const n = Number(v);
  if (!Number.isFinite(n)) return n > 0 ? "∞" : "−∞";
  if (Number.isInteger(n)) return n.toLocaleString();
  const a = Math.abs(n);
  const digits = a > 0 && a < 1 ? Math.max(d, Math.ceil(-Math.log10(a)) + 2) : d;
  return n.toLocaleString(undefined, { maximumFractionDigits: Math.min(20, digits) });
};
/** Short tick labels, and no exponents on an axis: 60M, 1.2K, 0.35, 0.00015.
 *  Intl in standard notation writes the digits out however small the value is. */
export const fmtAxis = (v) => {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "";
  const n = Number(v);
  const a = Math.abs(n);
  if (!Number.isFinite(n)) return n > 0 ? "∞" : "−∞";
  if (a === 0) return "0";
  if (a >= 1000) return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
  return new Intl.NumberFormat("en", { maximumSignificantDigits: 3 }).format(n);
};
export const fmtPct =(v, d = 1) => (v === null || v === undefined || Number.isNaN(Number(v)) ? "–" : `${(Number(v) * 100).toFixed(d)}%`);
export const fmtMs = (ms) => {
  if (ms === null || ms === undefined) return "–";
  const s = Number(ms) / 1000;
  if (s < 1) return t("format.ms", { n: Math.round(ms) });
  if (s < 90) return t("format.s", { n: s.toFixed(1) });
  if (s < 5400) return t("format.min", { n: (s / 60).toFixed(1) });
  return t("format.h", { n: (s / 3600).toFixed(2) });
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
/** The server sends naive UTC; mark it as UTC so it shows in local time. */
const asDate = (iso) => {
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime()) ? null : d;
};
const pad = (n) => String(n).padStart(2, "0");
/** YYYY/MM/DD — one date format across the app, whatever the browser's locale. */
export const fmtDate = (iso) => {
  const d = iso ? asDate(iso) : null;
  return d ? `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())}` : "–";
};
/** YYYY/MM/DD HH:MM, 24-hour. */
export const fmtTime = (iso) => {
  const d = iso ? asDate(iso) : null;
  return d ? `${fmtDate(iso)} ${pad(d.getHours())}:${pad(d.getMinutes())}` : "–";
};
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
  if (typeof v === "boolean") return v ? t("format.yes") : t("format.no");
  if (fmt === "int") return fmtInt(v);
  if (fmt === "pct") return fmtPct(v);
  if (fmt === "num") return fmtNum(v);
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString() : fmtNum(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
};
