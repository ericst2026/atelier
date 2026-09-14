import React from "react";
import { fmtMetric } from "../lib/format";

export function Kpi({ metric }) {
  return (
    <div className={`kpi ${metric.accent || ""}`} title={metric.help || ""}>
      <div className="v">{fmtMetric(metric)}</div>
      <div className="l">{metric.label}</div>
      {metric.help && <div className="h">{metric.help}</div>}
    </div>
  );
}

export default function Kpis({ metrics }) {
  if (!metrics || !metrics.length) return null;
  return (
    <div className="kpis">
      {metrics.map((m) => (
        <Kpi key={m.key} metric={m} />
      ))}
    </div>
  );
}
