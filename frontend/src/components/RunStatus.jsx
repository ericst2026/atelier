import React from "react";
import { useT } from "../i18n";
import { fmtDuration } from "../lib/format";

export function StatusPill({ status }) {
  const t = useT();
  return (
    <span className={`pill ${status}`}>
      <i className="dot" /> {status ? t(`common.status.${status}`) : status}
    </span>
  );
}

export default function RunStatus({ run, progress, onCancel }) {
  const t = useT();
  if (!run) return null;
  const pct = progress?.pct ?? run.progress_pct ?? 0;
  const msg = progress?.msg ?? run.progress_msg ?? "";
  const status = progress?.status || run.status;
  const elapsed = run.started_at ? (new Date(run.finished_at ? `${run.finished_at}Z` : Date.now()) - new Date(`${run.started_at}Z`)) / 1000 : 0;
  return (
    <div className="stack" style={{ gap: 6 }}>
      <div className="row" style={{ gap: 10 }}>
        <StatusPill status={status} />
        <span className="muted small">
          {t("common.runN", { id: run.id })} · {run.gpu_ids?.length ? `GPU ${run.gpu_ids.join(",")}` : run.gpus ? `${run.gpus} GPU` : "CPU"} · {status === "queued" ? t("run.waiting") : fmtDuration(elapsed)}
        </span>
        <span className="spacer" />
        {(status === "queued" || status === "running") && onCancel && (
          <button className="btn sm danger" onClick={onCancel}>
            {t("run.stop")}
          </button>
        )}
      </div>
      {(status === "running" || status === "queued") && (
        <div className="progress">
          <i style={{ width: `${Math.max(2, pct)}%` }} />
        </div>
      )}
      {msg && <div className="small muted">{msg}</div>}
      {run.error && status !== "succeeded" && <div className="small" style={{ color: "var(--dup)" }}>{run.error}</div>}
    </div>
  );
}
