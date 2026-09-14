import React, { useState } from "react";
import { Area, Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Maximize2, Minimize2 } from "lucide-react";
import { fmtNum } from "../lib/format";

const COLORS = { raw: "#f4a259", kept: "#5fd3b8", dup: "#ff6b81", hold: "#b79cff", sky: "#7cc4ff", sun: "#ffd166" };
const CYCLE = ["#5fd3b8", "#f4a259", "#7cc4ff", "#b79cff", "#ff6b81", "#ffd166"];
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
  const xProps = numericX
    ? { type: "number", dataKey: spec.x, scale: xLog && canLogX ? "log" : "linear", domain: ["dataMin", "dataMax"], tickFormatter: (v) => fmtNum(v, 2), allowDataOverflow: true }
    : { type: "category", dataKey: spec.x, interval: data.length > 24 ? Math.floor(data.length / 12) : 0 };
  const yProps = { scale: yLog ? "log" : "linear", domain: yLog ? ["auto", "auto"] : spec.y_domain || [0, "auto"], allowDataOverflow: true, tickFormatter: (v) => fmtNum(v, 2), width: 54 };
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
          <ComposedChart data={data} margin={{ top: 8, right: hasRight ? 8 : 16, left: 0, bottom: 4 }}>
            <CartesianGrid stroke="#2a3d5f" strokeDasharray="3 3" />
            <XAxis {...xProps} stroke="#5f7192" tick={{ fill: "#93a3bf", fontSize: 11 }} label={spec.x_label ? { value: spec.x_label, position: "insideBottom", offset: -2, fill: "#93a3bf", fontSize: 11 } : undefined} />
            <YAxis yAxisId="left" {...yProps} stroke="#5f7192" tick={{ fill: "#93a3bf", fontSize: 11 }} label={spec.y_label ? { value: spec.y_label, angle: -90, position: "insideLeft", fill: "#93a3bf", fontSize: 11 } : undefined} />
            {hasRight && <YAxis yAxisId="right" orientation="right" {...yProps} stroke="#5f7192" tick={{ fill: "#93a3bf", fontSize: 11 }} />}
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
