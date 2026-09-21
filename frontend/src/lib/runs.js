import { useCallback, useEffect, useState } from "react";
import { api, wsUrl } from "./api";
import { useSocket } from "./ws";
import { buildLiveCharts } from "./liveCharts";

const MAX_LINES = 3000;

/** Follow one run live: log lines, progress, live series, status, and the result once finished. */
export function useRunStream(runId) {
  const [run, setRun] = useState(null);
  const [lines, setLines] = useState([]);
  const [progress, setProgress] = useState(null);
  const [live, setLive] = useState({});
  // what the x axis of the live curves counts, when the step says
  const [liveXLabel, setLiveXLabel] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const loadResult = useCallback((id) => {
    api(`/runs/${id}/result`)
      .then(setResult)
      .catch(() => setResult(null));
  }, []);

  useEffect(() => {
    setRun(null);
    setLines([]);
    setProgress(null);
    setLive({});
    setLiveXLabel(null);
    setResult(null);
    setError(null);
    if (!runId) return;
    api(`/runs/${runId}`)
      .then((r) => {
        setRun(r);
        if (r.status === "succeeded" || r.status === "failed") loadResult(runId);
      })
      .catch((e) => setError(e.message));
  }, [runId, loadResult]);

  const onMessage = useCallback(
    (msg) => {
      if (msg.type === "snapshot") {
        setLines(msg.lines || []);
        setProgress({ pct: msg.progress_pct, msg: msg.progress_msg, status: msg.status });
        setLive(msg.live?.series || {});
        setLiveXLabel(msg.live?.x_label || null);
      } else if (msg.type === "log") {
        setLines((prev) => {
          const next = prev.concat(msg.lines || []);
          return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
        });
      } else if (msg.type === "progress") {
        setProgress((p) => ({ ...(p || {}), pct: msg.pct ?? p?.pct, msg: msg.msg ?? p?.msg }));
        if (msg.series && Object.keys(msg.series).length) {
          setLive((prev) => {
            const next = { ...prev };
            const s = { ...msg.series };
            const x = s.step ?? s.x;
            delete s.step;
            delete s.x;
            if (typeof s.x_label === "string") setLiveXLabel(s.x_label);
            for (const [k, v] of Object.entries(s)) {
              if (typeof v !== "number") continue;
              const arr = (next[k] || []).slice();
              arr.push({ x: x ?? arr.length, y: v });
              next[k] = arr;
            }
            return next;
          });
        }
      } else if (msg.type === "status") {
        setProgress((p) => ({ ...(p || {}), status: msg.status }));
        setRun((r) => (r ? { ...r, status: msg.status, error: msg.error ?? r.error, gpu_ids: msg.gpu_ids ?? r.gpu_ids, metrics: msg.metrics ?? r.metrics } : r));
        if (msg.status === "succeeded" || msg.status === "failed") setTimeout(() => loadResult(runId), 300);
      }
    },
    [runId, loadResult]
  );
  const active = !!runId && (!run || run.status === "queued" || run.status === "running");
  useSocket(runId ? wsUrl(`/ws/runs/${runId}`) : null, onMessage, { enabled: !!runId });
  const cancel = useCallback(() => api(`/runs/${runId}/cancel`, { method: "POST" }).then(setRun), [runId]);
  return { run, lines, progress, live, liveXLabel, result, error, cancel, active, status: progress?.status || run?.status };
}

/** Live series → chart specs for ChartCard, grouped by what they measure. */
export function liveCharts(live, xLabel) {
  return buildLiveCharts(live, { xLabel });
}
