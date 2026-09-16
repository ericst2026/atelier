# Writing an experiment

An experiment is a folder under `experiments/`. Copy `_template/`, edit, and it
appears on the home page within five seconds — the registry re-reads the directory,
no restart and no database change. Folders starting with `_` are ignored. Every
experiment must define **exactly four steps**; the registry refuses anything else.

```
experiments/my-experiment/
  experiment.yaml        the manifest
  steps/step1..4.py      one script per step
  sample/                the starting point copied into every student's workspace
    README.md
    project/…
  grader/grade.py        scores a project, writes a `score` metric
  data/                  small files shipped with the experiment
  lib/                   shared code for your steps (optional)
```

## The manifest

```yaml
slug: my-experiment
title: My experiment
order: 6                 # position on the home page
difficulty: core         # intro | core | advanced
tags: [gpu, data]
summary: One sentence on the card.
description: |
  Markdown shown under the step.

materials:
  - {key: gsm8k, name: GSM8K, path: datasets/rl/gsm8k, kind: dataset, format: jsonl,
     description: Shown on the Materials tab; `path` is relative to the materials folder.}

steps:
  - id: data
    title: Data
    summary: Shown on the rail
    description: |
      Markdown shown above the parameter form.
    script: steps/step1_data.py
    gpus: 0
    timeout_min: 30
    needs_previous: false          # true by default for steps 2–4
    figures: [{key: docs, label: docs}]   # metrics echoed on the rail
    params:
      - {key: max_docs, label: Documents, type: int, min: 50, max: 100000, default: 2000}
      - {key: source, label: Source, type: select, default: a, options: [{value: a, label: A}]}
      - {key: tokenizer_run, label: A tokenizer, type: run, experiment: tokenizer, step: 3}
  # … three more

project:
  entry: project/train.py
  run_command: python project/train.py
  timeout_min: 240
  interface: |
    Markdown: exactly what the grader expects from the student's code.

grader: {script: grader/grade.py, gpus: 1, timeout_min: 30, auto_on_submit: true, score_metric: score}
leaderboard: {metric: accuracy, higher_is_better: true}
```

Parameter types: `int`, `float` (slider + box), `bool`, `select`, `multiselect`,
`text`, `textarea`, and `run` — a picker of finished runs, optionally constrained to
an experiment and step. The chosen run arrives in the next script as
`inputs["<key>_run"] = {id, experiment, step, run_dir, outputs}`.

## A step script

```python
from atelier_sdk import Result, inputs, params, parse_args, progress

parse_args()                                  # reads --run-dir
P = params({"max_docs": 2000})                # defaults + what the student chose
I = inputs()                                  # previous step's outputs + the three paths

for i, doc in enumerate(docs):
    progress(100 * i / n, f"doc {i}", step=i, loss=loss)   # extra kwargs stream to the live chart

R = Result()
R.metric("docs", "Documents", len(docs), "int", "kept", help="shown under the number")
R.chart("lengths", "Document lengths", rows, "bin", [{"key": "count", "label": "Docs", "color": "raw"}], "bar")
R.chart("kinds", "Documents by kind", rows, "kind", [{"key": "count", "label": "Docs"}], "donut")
R.chart("fit", "Loss against size", rows, "params",
        [{"key": "fit", "label": "Fitted", "form": "line"},
         {"key": "measured", "label": "Measured", "form": "scatter"}], "line", x_log=True)
R.table("preview", "Preview", [{"key": "text", "label": "Text"}], rows[:40])
R.tokens("sample", "Your text in tokens", [{"text": "▁the", "id": 12, "kind": "word"}])
R.artifact(path, "corpus.jsonl")
R.output("corpus", str(path))                 # becomes inputs["corpus"] in the next step
R.save()
```

The environment a step runs in: `ATELIER_RUN_DIR`, `ATELIER_MATERIALS`,
`ATELIER_EXPERIMENTS`, `ATELIER_WORKSPACE`, `ATELIER_GPUS`, `CUDA_VISIBLE_DEVICES`
already restricted to the allocated GPUs, HuggingFace pinned offline, and
`PYTHONPATH` covering `experiments/_lib`, `experiments/`, and the student's project.

Metric `fmt`: `num, int, pct, ms, bytes, text`. Accents: `raw, kept, dup, hold, sky, sun`.
A step marked `own_code: true` lets a student run their own implementation of it
instead of the shipped script. They get a prototype generated from your script — the
params it takes, the inputs it is handed, the metrics and outputs it has to leave
behind — and write the body themselves. Their file runs with exactly the same
params, inputs and environment, and they can hand it in for you to mark. Leave the
flag off for a step where re-implementing it teaches nothing.

Chart `type`: `bar` compares magnitudes, `stacked` shows parts of a whole across a
dimension, `line` and `area` show a trend, `scatter` puts one number against
another, and `donut` shows parts of a single whole — at most five slices, the tail
folded into `other`, so prefer a bar when the shares are close or many. A series
can set `form` (`line, scatter, bar, area`) to draw itself differently from the
chart's own type: measured points sitting on a fitted line. Options: `x_log, y_log,
y_domain, ref_x, ref_label, note, stretch`, and `axis: "right"` on a series for a
second axis. The student can switch between the sensible forms and the log axes in
the UI, so pick the honest default and move on. Leave `color` off unless it carries
meaning — the chart picks colours that stay apart for colour-blind readers.

## The grader

```
python grader/grade.py --project <workspace or submission> --run-dir <run dir>
```

Import the student's module with `importlib`, wrap everything in try/except so one
broken submission does not lose the rest of the marks, and emit a `score` metric
between 0 and 100 — it lands in `Submission.auto_score` automatically. Keep a
hidden slice of data for the grader that students never see in the guided steps.

## Shared libraries

`atelier_nlp` carries what the experiments have in common: `bpe` (trainer,
encoder with merge-rank traces, evaluation), `dedup` (MinHash + LSH + union-find),
`text` (GPT-2 style pre-tokenizer, character classes), and `hf` — offline model and
dataset loading, batched generation, GSM8K answer extraction, `sft_train` (TRL
`SFTTrainer`) and `grpo_train` (TRL `GRPOTrainer`), plus a progress callback that
turns trainer logs into live charts.

The GPU experiments target `transformers ≥ 4.44`, `trl ≥ 0.19` (`SFTConfig`,
`GRPOConfig`, `processing_class=`), and `peft ≥ 0.12`. TRL renames trainer
arguments fairly often; if you pin different versions, `experiments/_lib/atelier_nlp/hf.py`
is the only file to adjust.


## The two libraries the course uses

`atelier_world` generates everything students train on. `World(lang, seed)` gives
`documents()` for pretraining, `instructions()` for fine-tuning, `eval_set()` for
held-out evaluation, and `grade(text, answer)` for exact checking. A language pack
is data only — copy `lang_en.py`, translate the lists and templates, add it to
`LANGS`, and the whole course runs in that language.

`atelier_mini` is the model stack in pure PyTorch: `model.py` (RoPE, RMSNorm,
SwiGLU, tied embeddings, four size presets), `tok.py` (the student's BPE plus four
control tokens), `data.py` (packing and batching, and the SFT mask that hides the
prompt from the loss), `gen.py` (batched sampling with left padding so every
sequence's last token lands in the same column), `train.py` (pretraining and SFT)
and `rl.py` (GRPO: group-relative advantages, a clipped ratio and a k3 KL estimate
against a frozen reference).

Neither imports `transformers` or `trl`. An experiment that uses only these two
needs nothing but torch, numpy and the standard library.

## Running the Tokenizer experiment on your own text

Step 1 of **Tokenizer** takes its corpus from one of three places.

**Generated** — the default. Nothing is downloaded and nothing is needed: the corpus
comes from a seed, and the same seed regenerates it exactly. This is the one path
verified end to end on CPU, and the right first thing to run on a new machine.

**A folder of your own text.** Put it under `materials/` on the GPU node and give the
path relative to that:

```
materials/corpora/our-reports/report_001.txt
materials/corpora/our-reports/report_002.txt
materials/corpora/our-reports/tickets.jsonl
```

Then set the source to *a folder of your own text* and the folder to
`corpora/our-reports`. `.txt` and `.md` files become one document each; `.jsonl`
files become one document per line, reading `text`, `content` or `body`.

**Pasted text**, for trying something on a few paragraphs without touching the disk.

One difference worth knowing: the generated source plants a share of near-duplicates
on purpose, so step 2 has something to find. Your own corpus gets none planted —
whatever step 2 finds was already there, which is the more useful result. Watch the
`removed` figure on real data; it is usually higher than people expect.

This is the seam where the course becomes a course about your data. The vocabulary
step 3 produces is what pretraining loads, and everything after it inherits. Point
this at your corpus and the model trained in experiments 2 to 5 is trained on your
language, not ours — with no other change.

The advanced **Tokenizer on public text** experiment also ships usable data: the
sample pools in `experiments/tokenizer/data/pools/` are in the repository, so that
experiment runs with its default source selected and nothing downloaded. WikiText,
source code and the multilingual slice are the optional extras.
