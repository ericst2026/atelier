/** Live series → chart specs for ChartCard, grouped by what they measure.
 *
 *  A run reports curves by name (see atelier_sdk.progress). Names that measure the
 *  same thing share a chart — train and validation loss on one, every reward on
 *  another — and anything unrecognised gets a chart of its own. A name written
 *  `group__name` goes on the `group` chart, labelled `name`, whatever it is.
 *  The charts come back most telling first, so a screen with room for only a
 *  few shows the ones that matter. */
import { t } from "../i18n";

// in the order they are shown; the first kind whose test matches a name owns it;
// the title is charts.kind.<id>
const KINDS = [
  { id: "loss", test: (k) => /loss|perplexity|ppl/.test(k) },
  { id: "reward", test: (k) => /reward/.test(k) },
  { id: "accuracy", test: (k) => /accuracy|(^|_)acc($|_)|solved|pass_rate|win_rate|recall/.test(k) },
  { id: "kl", test: (k) => /(^|_)kl($|_)/.test(k) },
  { id: "grad", test: (k) => /grad_norm|gradient/.test(k) },
  { id: "margin", test: (k) => /margin/.test(k) },
  { id: "logp", test: (k) => /logp/.test(k) },
  { id: "lr", test: (k) => k === "lr" || /learning_rate/.test(k) },
  { id: "speed", test: (k) => /per_sec|per_s$|throughput/.test(k) },
];
// groups a step names itself, and single unrecognised curves, go between the
// measures of how well it is learning and the settings it is learning with
const OWN_RANK = KINDS.findIndex((k) => k.id === "grad") + 0.5;

// how a curve is labelled on a chart it shares: a key in charts.label
const LABELS = {
  loss: "trainBatch",
  train_loss: "train",
  val_loss: "validation",
  eval_loss: "validation",
  reward: "mean",
  reward_std: "spread",
  eval_accuracy: "validation",
  mean_token_accuracy: "trainToken",
  eval_mean_token_accuracy: "validationToken",
  tokens_per_sec: "tokensPerSec",
  lr: "learningRate",
  learning_rate: "learningRate",
  grad_norm: "beforeClipping",
};
const PALETTE = ["kept", "hold", "sky", "raw", "sun", "dup"];

const words = (k) => k.replace(/__/g, " ").replace(/_/g, " ").trim();
const kindTitle = (kind) => t(`charts.kind.${kind.id}`);
const titled = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

function labelOf(key, kind) {
  if (LABELS[key]) return t(`charts.label.${LABELS[key]}`);
  // val_loss_clean → clean; loss_rank8 → rank8: the kind is already the title
  if (kind === "loss") return words(key.replace(/^(train_|val_|eval_)?loss_?/, "")) || words(key);
  if (kind === "accuracy") return words(key.replace(/^(eval_)?accuracy_?/, "")) || words(key);
  return words(key);
}

/** Rows keyed by x with a column per curve, for curves sampled at different x. */
function merge(entries) {
  const rows = new Map();
  for (const [k, pts] of entries) {
    for (const p of pts) rows.set(p.x, { ...(rows.get(p.x) || { x: p.x }), [k]: p.y });
  }
  return [...rows.values()].sort((a, b) => a.x - b.x);
}

/** Accepts live.json as written ({series, x_label, timeline}) or a bare series map. */
export function buildLiveCharts(live, { xLabel, max = Infinity, timeline = true } = {}) {
  const series = live && live.series && typeof live.series === "object" ? live.series : live || {};
  const xName = xLabel || live?.x_label || t("charts.axis.step");
  const groups = new Map(); // id → {title, rank, entries, order}
  const add = (id, title, rank, entry) => {
    if (!groups.has(id)) groups.set(id, { title, rank, entries: [], order: groups.size });
    groups.get(id).entries.push(entry);
  };
  for (const [key, pts] of Object.entries(series)) {
    if (!Array.isArray(pts) || pts.length < 2) continue;
    if (key.includes("__")) {
      const [group, ...rest] = key.split("__");
      const label = words(rest.join("__")) || words(group);
      // accuracy__mean sits with the accuracies; documents__story on a chart of its own
      const kind = KINDS.findIndex((k) => k.test(group.toLowerCase()));
      if (kind >= 0) add(KINDS[kind].id, kindTitle(KINDS[kind]), kind, [key, pts, label]);
      else add(`own-${group}`, titled(words(group)), OWN_RANK, [key, pts, label]);
      continue;
    }
    const rank = KINDS.findIndex((k) => k.test(key.toLowerCase()));
    if (rank >= 0) add(KINDS[rank].id, kindTitle(KINDS[rank]), rank, [key, pts, labelOf(key, KINDS[rank].id)]);
    else add(`own-${key}`, titled(words(key)), OWN_RANK, [key, pts, words(key)]);
  }
  const charts = [...groups.entries()]
    .sort(([, a], [, b]) => a.rank - b.rank || a.order - b.order)
    .map(([id, g]) => ({
      id: `live-${id}`,
      title: g.title,
      type: "line",
      x: "x",
      x_label: xName,
      series: g.entries.map(([key, , label], i) => ({ key, label, color: PALETTE[i % PALETTE.length] })),
      data: merge(g.entries.map(([key, pts]) => [key, pts])),
      // a loss or a rate reads better from where it actually is than from zero
      y_domain: ["auto", "auto"],
    }));
  // a step that draws nothing of its own still shows how it is getting on
  const tl = live?.timeline;
  if (timeline && charts.length === 0 && Array.isArray(tl) && tl.length > 1) {
    charts.push({ id: "live-timeline", title: t("charts.timeline.title"), type: "line", x: "x", x_label: t("charts.axis.seconds"), series: [{ key: "y", label: t("charts.timeline.done"), color: "sky" }], data: tl, y_domain: [0, 100] });
  }
  return charts.slice(0, max);
}
