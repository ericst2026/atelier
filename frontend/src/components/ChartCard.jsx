import React, { useState } from "react";
import { Area, Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Maximize2, Minimize2 } from "lucide-react";
import { fmtAxis, fmtNum } from "../lib/format";

const COLORS = { raw: "#f4a259", kept: "#5fd3b8", dup: "#ff6b81", hold: "#b79cff", sky: "#7cc4ff", sun: "#ffd166" };
const CYCLE = ["#5fd3b8", "#f4a259", "#7cc4ff", "#b79cff", "#ff6b81", "#ffd166"];
const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const color = (c, i) => (c && COLORS[c]) || c || CYCLE[i % CYCLE.length];

function Tip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="tip">
      <div className="muted">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {fmtNum(p.value)}
        </div>
      ))}
    </div>
  );
}

/** Generic chart from a result.json spec: {id,title,type,x,series:[{key,label,color,axis}],data,x_log,y_log,ref_x,ref_label,note,stretch,y_domain}. */
export default function ChartCard({ spec, height = 240, allowStretch = true }) {
  const [type, setType] = useState(spec.type || "bar");
  const [xLog, setXLog] = useState(!!spec.x_log);
  const [yLog, setYLog] = useState(!!spec.y_log);
  const [stretch, setStretch] = useState(!!spec.stretch);
  const data = (spec.data || []).filter((d) => d && d[spec.x] !== undefined);
  const numericX = data.length > 0 && data.every((d) => typeof d[spec.x] === "number");
  const hasRight = (spec.series || []).some((s) => s.axis === "right");
  const canLogX = numericX && data.every((d) => d[spec.x] > 0);
  // ticks skip themselves when they would collide; long category labels are cut and read in full in the tooltip
  const xProps = numericX
    ? { type: "number", dataKey: spec.x, scale: xLog && canLogX ? "log" : "linear", domain: ["dataMin", "dataMax"], tickFormatter: fmtAxis, allowDataOverflow: true, interval: "preserveStartEnd", minTickGap: 12 }
    : { type: "category", dataKey: spec.x, tickFormatter: (v) => clip(String(v), 14), interval: "preserveStartEnd", minTickGap: 10 };
  const axisWidth = (side) => {
    const keys = (spec.series || []).filter((s) => (s.axis === "right") === (side === "right")).map((s) => s.key);
    let longest = 1;
    for (const d of data) for (const k of keys) if (typeof d[k] === "number") longest = Math.max(longest, fmtAxis(d[k]).length);
    return Math.min(84, Math.max(36, longest * 7 + 14));
  };
  const yProps = { scale: yLog ? "log" : "linear", domain: yLog ? ["auto", "auto"] : spec.y_domain || [0, "auto"], allowDataOverflow: true, tickFormatter: fmtAxis };
  const Comp = type === "line" ? Line : type === "area" ? Area : Bar;
  return (
    <div className={`chartcard ${stretch ? "stretch" : ""}`}>
      <div className="head">
        <h3>{spec.title}</h3>
        <div className="ctl">
          {["bar", "line", "area"].map((t) => (
            <button key={t} className={type === t ? "on" : ""} onClick={() => setType(t)}>
              {t}
            </button>
          ))}
        </div>
        <div className="ctl">
          {canLogX && (
            <button className={xLog ? "on" : ""} onClick={() => setXLog(!xLog)}>
              log x
            </button>
          )}
          <button className={yLog ? "on" : ""} onClick={() => setYLog(!yLog)}>
            log y
          </button>
          {allowStretch && (
            <button onClick={() => setStretch(!stretch)} title="Toggle full width">
              {stretch ? <Minimize2 size={11} /> : <Maximize2 size={11} />}
            </button>
          )}
        </div>
      </div>
      <div style={{ width: "100%", height: stretch ? height + 120 : height }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 8, right: hasRight ? 8 : 20, left: spec.y_label ? 8 : 0, bottom: 4 }}>
            <CartesianGrid stroke="#2a3d5f" strokeDasharray="3 3" />
            <XAxis {...xProps} height={spec.x_label ? 42 : 30} stroke="#5f7192" tick={{ fill: "#93a3bf", fontSize: 11 }} label={spec.x_label ? { value: spec.x_label, position: "insideBottom", offset: 0, fill: "#93a3bf", fontSize: 11 } : undefined} />
            <YAxis yAxisId="left" {...yProps} width={axisWidth("left") + (spec.y_label ? 14 : 0)} stroke="#5f7192" tick={{ fill: "#93a3bf", fontSize: 11 }} label={spec.y_label ? { value: spec.y_label, angle: -90, position: "insideLeft", fill: "#93a3bf", fontSize: 11, style: { textAnchor: "middle" } } : undefined} />
            {hasRight && <YAxis yAxisId="right" orientation="right" {...yProps} width={axisWidth("right")} stroke="#5f7192" tick={{ fill: "#93a3bf", fontSize: 11 }} />}
            <Tooltip content={<Tip />} />
            {(spec.series || []).length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
            {spec.ref_x !== null && spec.ref_x !== undefined && <ReferenceLine yAxisId="left" x={spec.ref_x} stroke="#ffd166" strokeDasharray="4 4" label={{ value: spec.ref_label || "", fill: "#ffd166", fontSize: 11, position: "top" }} />}
            {(spec.series || []).map((s, i) => (
              <Comp key={s.key} yAxisId={s.axis === "right" ? "right" : "left"} dataKey={s.key} name={s.label || s.key} stroke={color(s.color, i)} fill={color(s.color, i)} fillOpacity={type === "area" ? 0.25 : 0.9} strokeWidth={type === "bar" ? 0 : 2} dot={type !== "bar" && data.length < 40} isAnimationActive={false} connectNulls />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {spec.note && <div className="note">{spec.note}</div>}
    </div>
  );
}
