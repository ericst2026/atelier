import React, { useState } from "react";
import { Area, Bar, CartesianGrid, Cell, ComposedChart, Legend, Line, Pie, PieChart, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis, ZAxis } from "recharts";
import { Maximize2, Minimize2 } from "lucide-react";
import { t as tr, useT } from "../i18n";
import { fmtAxis, fmtNum } from "../lib/format";

const COLORS = { raw: "#f4a259", kept: "#5fd3b8", dup: "#ff6b81", hold: "#b79cff", sky: "#7cc4ff", sun: "#ffd166" };
// Categorical order, checked with the palette validator on this surface: adjacent
// pairs stay apart for deuteranopia and for full colour vision. Purple sits away
// from sky, which is the pair that fails when they neighbour each other.
const CYCLE = ["#5fd3b8", "#f4a259", "#7cc4ff", "#ff6b81", "#b79cff", "#ffd166"];
// Forms where every series is compared with every other (scatter) hold to three
// hues; these three clear the all-pairs check.
const ALL_PAIRS = ["#5fd3b8", "#f4a259", "#b79cff"];
// Part-to-whole is one hue stepped light → dark, so neighbouring slices differ in
// lightness and not only in hue. Biggest share takes the lightest, most present step.
const RAMP = ["#dbeeff", "#aed8ff", "#82c1fa", "#5a9ed6", "#4a80b4"];
const SURFACE = "#15243b";
const GRID = "#22344f"; // one hairline shade off the surface
const AXIS = "#5f7192";
const INK = "#93a3bf";
const SLICE_CAP = RAMP.length; // one step per slice, and a donut is only readable while the slices are few

const clip = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const color = (c, i, palette = CYCLE) => (c && COLORS[c]) || c || palette[i % palette.length];

function Tip({ active, payload, label, total }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="tip">
      <div className="muted">{label ?? payload[0]?.name}</div>
      {payload.map((p, i) => (
        <div key={p.dataKey ?? i} style={{ color: p.color || p.payload?.fill }}>
          {p.name}: {fmtNum(p.value)}
          {total ? <span className="faint"> · {Math.round((p.value / total) * 100)}%</span> : null}
        </div>
      ))}
    </div>
  );
}

/** Part-to-whole: the slices, biggest first, with a small tail folded into "other"
 *  so the ring never turns into a dozen unreadable slivers. */
function slices(data, nameKey, valueKey) {
  const rows = data
    .map((d) => ({ name: String(d[nameKey]), value: Number(d[valueKey]) }))
    .filter((d) => Number.isFinite(d.value) && d.value > 0)
    .sort((a, b) => b.value - a.value);
  if (rows.length <= SLICE_CAP) return rows;
  const head = rows.slice(0, SLICE_CAP - 1);
  const rest = rows.slice(SLICE_CAP - 1);
  return [...head, { name: tr("charts.other", { n: rest.length }), value: rest.reduce((s, r) => s + r.value, 0) }];
}

/** Which forms this chart can be shown as. A part-to-whole chart can always fall
 *  back to a bar (better for close values); a correlation stays a scatter or line. */
function formsFor(spec) {
  const t = spec.type || "bar";
  if (t === "donut" || t === "pie") return ["donut", "bar"];
  if (t === "scatter") return ["scatter", "line"];
  const many = (spec.series || []).length > 1;
  return many ? ["bar", "stacked", "line", "area"] : ["bar", "line", "area"];
}

/** Generic chart from a result.json spec:
 *  {id,title,type,x,series:[{key,label,color,axis}],data,x_log,y_log,ref_x,ref_label,note,stretch,y_domain}
 *  type: bar | stacked | line | area | scatter | donut (pie is a donut without the hole). */
export default function ChartCard({ spec, height = 210, allowStretch = true }) {
  const t = useT();
  const [type, setType] = useState(spec.type === "pie" ? "donut" : spec.type || "bar");
  const [xLog, setXLog] = useState(!!spec.x_log);
  const [yLog, setYLog] = useState(!!spec.y_log);
  const [stretch, setStretch] = useState(!!spec.stretch);
  const data = (spec.data || []).filter((d) => d && d[spec.x] !== undefined);
  const series = spec.series || [];
  const forms = formsFor(spec);
  const isDonut = type === "donut";
  const isScatter = type === "scatter";
  const stacked = type === "stacked";
  const palette = isScatter ? ALL_PAIRS : CYCLE;
  const numericX = data.length > 0 && data.every((d) => typeof d[spec.x] === "number");
  const hasRight = series.some((s) => s.axis === "right");
  const canLogX = numericX && data.every((d) => d[spec.x] > 0);
  const plotH = stretch ? height + 90 : height;

  const pie = isDonut ? slices(data, spec.x, (series[0] || {}).key) : [];
  const pieTotal = pie.reduce((s, d) => s + d.value, 0);

  // ticks skip themselves when they would collide; long category labels are cut and read in full in the tooltip
  const xProps = numericX
    ? { type: "number", dataKey: spec.x, scale: xLog && canLogX ? "log" : "linear", domain: ["dataMin", "dataMax"], tickFormatter: fmtAxis, allowDataOverflow: true, interval: "preserveStartEnd", minTickGap: 12 }
    : { type: "category", dataKey: spec.x, tickFormatter: (v) => clip(String(v), 14), interval: "preserveStartEnd", minTickGap: 10 };
  const axisWidth = (side) => {
    const keys = series.filter((s) => (s.axis === "right") === (side === "right")).map((s) => s.key);
    let longest = 1;
    for (const d of data) for (const k of keys) if (typeof d[k] === "number") longest = Math.max(longest, fmtAxis(d[k]).length);
    return Math.min(84, Math.max(36, longest * 7 + 14));
  };
  const yProps = { scale: yLog ? "log" : "linear", domain: yLog ? ["auto", "auto"] : spec.y_domain || [0, "auto"], allowDataOverflow: true, tickFormatter: fmtAxis };

  return (
    <div className={`chartcard ${stretch ? "stretch" : ""}`}>
      <div className="head">
        <h3>{spec.title}</h3>
        <div className="ctl">
          {forms.map((f) => (
            <button key={f} className={type === f ? "on" : ""} onClick={() => setType(f)}>
              {t(`charts.form.${f}`)}
            </button>
          ))}
        </div>
        <div className="ctl">
          {!isDonut && canLogX && (
            <button className={xLog ? "on" : ""} onClick={() => setXLog(!xLog)}>
              {t("charts.logX")}
            </button>
          )}
          {!isDonut && (
            <button className={yLog ? "on" : ""} onClick={() => setYLog(!yLog)}>
              {t("charts.logY")}
            </button>
          )}
          {allowStretch && (
            <button onClick={() => setStretch(!stretch)} title={t("charts.toggleWidth")}>
              {stretch ? <Minimize2 size={11} /> : <Maximize2 size={11} />}
            </button>
          )}
        </div>
      </div>
      <div style={{ width: "100%", height: plotH }}>
        <ResponsiveContainer>
          {isDonut ? (
            <PieChart margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
              <Pie
                data={pie}
                dataKey="value"
                nameKey="name"
                innerRadius="55%"
                outerRadius="80%"
                paddingAngle={1.5}
                stroke={SURFACE}
                strokeWidth={2}
                isAnimationActive={false}
                label={({ percent, name }) => (percent >= 0.06 ? `${clip(name, 14)} ${Math.round(percent * 100)}%` : "")}
                labelLine={false}
                style={{ fontSize: 11, fill: INK }}
              >
                {pie.map((d, i) => (
                  <Cell key={d.name} fill={RAMP[Math.min(i, RAMP.length - 1)]} />
                ))}
              </Pie>
              <Tooltip content={<Tip total={pieTotal} />} />
            </PieChart>
          ) : (
            <ComposedChart data={data} margin={{ top: 8, right: hasRight ? 8 : 20, left: spec.y_label ? 8 : 0, bottom: 4 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis {...xProps} height={spec.x_label ? 42 : 30} stroke={AXIS} tick={{ fill: INK, fontSize: 11 }} label={spec.x_label ? { value: spec.x_label, position: "insideBottom", offset: 0, fill: INK, fontSize: 11 } : undefined} />
              <YAxis yAxisId="left" {...yProps} width={axisWidth("left") + (spec.y_label ? 14 : 0)} stroke={AXIS} tick={{ fill: INK, fontSize: 11 }} label={spec.y_label ? { value: spec.y_label, angle: -90, position: "insideLeft", fill: INK, fontSize: 11, style: { textAnchor: "middle" } } : undefined} />
              {hasRight && <YAxis yAxisId="right" orientation="right" {...yProps} width={axisWidth("right")} stroke={AXIS} tick={{ fill: INK, fontSize: 11 }} />}
              {isScatter && <ZAxis range={[36, 36]} />}
              <Tooltip content={<Tip />} cursor={{ stroke: AXIS, strokeWidth: 1 }} />
              {series.length > 1 && <Legend wrapperStyle={{ fontSize: 11 }} />}
              {spec.ref_x !== null && spec.ref_x !== undefined && <ReferenceLine yAxisId="left" x={spec.ref_x} stroke="#ffd166" strokeDasharray="4 4" label={{ value: spec.ref_label || "", fill: "#ffd166", fontSize: 11, position: "top" }} />}
              {series.map((s, i) => {
                // a series may pick its own mark: measured points as dots on top of a fitted line
                const form = s.form || type;
                const paint = color(s.color, i, palette);
                if (form === "scatter")
                  return <Scatter key={s.key} yAxisId={s.axis === "right" ? "right" : "left"} dataKey={s.key} name={s.label || s.key} fill={paint} stroke={SURFACE} strokeWidth={2} isAnimationActive={false} />;
                const Mark = form === "line" ? Line : form === "area" ? Area : Bar;
                const stack = stacked || (form === "area" && series.length > 1);
                return (
                  <Mark
                    key={s.key}
                    yAxisId={s.axis === "right" ? "right" : "left"}
                    dataKey={s.key}
                    name={s.label || s.key}
                    stackId={stack ? "a" : undefined}
                    // stacked fills are separated by a hairline of the surface, so
                    // neighbouring segments read as two blocks and not one
                    stroke={stacked ? SURFACE : paint}
                    fill={paint}
                    fillOpacity={form === "area" ? (stacked ? 0.85 : 0.25) : 0.9}
                    strokeWidth={form === "bar" ? (stacked ? 2 : 0) : 2}
                    dot={form !== "bar" && !stacked && data.length < 40}
                    isAnimationActive={false}
                    connectNulls
                  />
                );
              })}
            </ComposedChart>
          )}
        </ResponsiveContainer>
      </div>
      {spec.note && <div className="note">{spec.note}</div>}
    </div>
  );
}
