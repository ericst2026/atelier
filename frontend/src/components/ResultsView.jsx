import React from "react";
import { Download } from "lucide-react";
import ChartCard from "./ChartCard";
import Kpis from "./Kpi";
import Markdown from "./Markdown";
import { useT } from "../i18n";
import { fileUrl } from "../lib/api";
import { fmtBytes, fmtCell } from "../lib/format";

export function DataTable({ table }) {
  return (
    <div className="stack" style={{ gap: 6 }}>
      <h3>{table.title}</h3>
      <div className="tablewrap">
        <table className="data">
          <thead>
            <tr>
              {table.columns.map((c) => (
                <th key={c.key}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((r, i) => (
              <tr key={i}>
                {table.columns.map((c) => (
                  <td key={c.key}>{fmtCell(r[c.key], c.fmt)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {table.note && <div className="help">{table.note}</div>}
    </div>
  );
}

export function TokenView({ view }) {
  const t = useT();
  return (
    <div className="stack" style={{ gap: 6 }}>
      <h3>{view.title}</h3>
      <div className="tokens">
        {view.tokens.map((tok, i) => (
          <span key={i} className={`tok ${tok.kind || ""}`} title={`${t("results.token.id", { id: tok.id })}${tok.count ? t("results.token.count", { n: tok.count }) : ""}`}>
            {tok.text}
            {tok.count !== undefined && <small>{tok.count}</small>}
          </span>
        ))}
      </div>
      {view.note && <div className="help">{view.note}</div>}
    </div>
  );
}

/** Renders a result.json: metrics → charts → tokens → tables → artifacts → notes. */
export default function ResultsView({ result, runId, compact = false }) {
  if (!result) return null;
  return (
    <div className="stack" style={{ gap: 14 }}>
      <Kpis metrics={result.metrics} />
      {result.notes && result.notes.length > 0 && <Markdown text={result.notes.join("\n\n")} />}
      {result.charts && result.charts.length > 0 && (
        <div className="charts">
          {result.charts.map((c) => (
            <ChartCard key={c.id} spec={c} height={compact ? 180 : 210} />
          ))}
        </div>
      )}
      {(result.tokens || []).map((v) => (
        <TokenView key={v.id} view={v} />
      ))}
      {(result.tables || []).map((t) => (
        <DataTable key={t.id} table={t} />
      ))}
      {runId && result.artifacts && result.artifacts.length > 0 && (
        <div className="row">
          {result.artifacts.map((a) => (
            <a key={a.path} className="btn sm" href={fileUrl(`/runs/${runId}/artifacts/${a.path}`)} target="_blank" rel="noreferrer">
              <Download size={13} /> {a.label} <span className="faint">{fmtBytes(a.bytes)}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
