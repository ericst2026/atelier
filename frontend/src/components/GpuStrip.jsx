import React, { useEffect, useState } from "react";
import { useT } from "../i18n";
import { api } from "../lib/api";

/** One small bar per GPU, across every node. The class shares this hardware, so its pulse lives in the header. */
export default function GpuStrip({ count = 8 }) {
  const t = useT();
  const [gpus, setGpus] = useState([]);
  useEffect(() => {
    let alive = true;
    const tick = () => api("/system/gpus").then((d) => alive && setGpus(d.gpus || [])).catch(() => {});
    tick();
    const timer = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  const bars = Array.from({ length: Math.max(count, gpus.length) }, (_, i) => gpus[i] || null);
  return (
    <div className="gpustrip" title={gpus.length ? gpus.map((g) => `${g.node ? `${g.node} ` : ""}GPU ${g.index}: ${g.util}% · ${(g.mem_used / 1e9).toFixed(1)}/${(g.mem_total / 1e9).toFixed(0)} GB · ${g.temp}°C`).join("\n") : t("widgets.gpu.unavailable")}>
      {bars.map((g, i) => (
        <div key={i} className={`bar ${g && g.util > 80 ? "hot" : g && g.util > 10 ? "busy" : ""}`}>
          <i style={{ height: `${g ? Math.max(4, g.util) : 4}%` }} />
        </div>
      ))}
      <span className="label">{gpus.length ? t("widgets.gpu.busy", { busy: gpus.filter((g) => g.util > 10).length, total: gpus.length }) : t("widgets.gpu.noData")}</span>
    </div>
  );
}
