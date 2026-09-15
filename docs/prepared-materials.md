# Your own models and datasets in the world experiments

Every world experiment can use what the course makes for itself or what you prepare.
Each input a step takes has its own source:

| Input | generated | prepared |
|---|---|---|
| **Data** | written by the world generator (stories, records, solved problems) | a dataset you put under `materials/datasets/` |
| **Model** | one of the student's own earlier runs (Pretraining → Fine-tuning → …) | a model you put under `materials/models/` |

The two choices are independent: a generated model with your data, your model with
generated data, or both yours. When a step's source is set to *prepared*, the form
lists the materials that experiment declares in the `materials:` section of its
`experiment.yaml` — and only those; nothing is typed, and the API refuses any other
path. A declared material that no machine has, or that lacks the format or fields the
step reads, is listed but cannot be picked.

So putting a model or dataset under `materials/` is half the job: add it to the
`materials:` list of each experiment that should offer it (see the end of this page).

Students can only pick their own runs, so a model you want the whole class to start
from goes in `materials/models/`, not in a run of yours.

## Where things go

```
materials/
  models/
    our-base/                 an Atelier checkpoint
      model.pt
      tokenizer.json
      meta.json               optional: {"lang": "en", "system": "...", "description": "..."}
    Qwen2.5-0.5B-Instruct/    a HuggingFace model: config.json + *.safetensors + tokenizer files
  datasets/
    our-class/
      questions/              train.jsonl, validation.jsonl (or test.jsonl)
      reports/                .txt / .md files, or a .jsonl with a text field
```

On a split install the materials live on the worker nodes; each worker publishes
what it has and the list in the form shows every node's. The list refreshes about
every five minutes — a teacher can force it with `/api/system/materials/catalog?refresh=true`.

## Models

**Atelier checkpoints** are what the world experiments train and save: `model.pt`
and the `tokenizer.json` it was trained with. To share one of your runs with the
class, copy both into a folder under `materials/models/`:

```bash
mkdir -p /srv/atelier/materials/models/our-sft
cp /srv/atelier/data/runs/<run id>/model.pt /srv/atelier/materials/models/our-sft/
cp <the tokenizer path shown in the run's outputs> /srv/atelier/materials/models/our-sft/tokenizer.json
```

Every world experiment accepts them.

**HuggingFace models** (a folder with `config.json` and weights) work where a step
only needs a model to generate, score or be fine-tuned: fine-tuning trains a LoRA
adapter on them, and evaluation, reasoning and RL use them through TRL and
`transformers`. Experiments that work inside the model's own layers — LoRA from
scratch, quantisation and speculative decoding, interpretability, long-context
position scaling — need an Atelier checkpoint and say so if given anything else.
Each experiment's page lists which formats its picker accepts.

## Datasets

Rows are matched by field name, so most JSONL works unchanged.

| What a step needs | Fields it reads | Example |
|---|---|---|
| **documents** (pretraining, tokenizer, curation) | `text`, `content`, `body` or `document` — or plain `.txt`/`.md` files, one document each | `{"text": "The harbour was quiet …"}` |
| **qa** (fine-tuning, RL, reasoning, evaluation) | a question: `prompt`, `question`, `instruction` (+ `input` or `context`), `query`, `problem`; an answer: `answer`, `answers`, `answerKey`, `output`, `response`, `target`, `completion`, `solution`; optional `steps` (list or lines), `family`/`category`, `choices` (shown as A, B, C… with the letter graded). An answer ending in `#### 72` grades `72` | `{"question": "What is 17 + 25?", "answer": "42"}` |
| **pairs** (preference optimisation) | a prompt, plus `chosen` and `rejected` | `{"prompt": "…", "chosen": "Answer: 42", "rejected": "Answer: 41"}` |

Splits are file names: `train.jsonl`, and `validation.jsonl`, `val.jsonl`,
`dev.jsonl` or `test.jsonl` for the held-out set. With no held-out file, a step sets
a slice aside so evaluation never grades a training row.

Answers are graded by exact match after normalisation (commas, trailing full stops
and spacing ignored; numbers compared as numbers), the same as the world's tasks.
Questions with one short answer grade cleanly; free-text answers mostly come out
wrong even when they are right, so use those for fine-tuning demonstrations rather
than as an evaluation set.

A dataset appears in a step's list only if its first rows have the fields that step
needs.

## Adding the options to an experiment of your own

The experiment declares what can be chosen, and the step form and the step script each
have one piece; world-sft is the reference.

```yaml
materials:
  # optional: a worker needs it only for a run that picks it
  # for: offer it only in these material params (without it: every param of its kind)
  - {key: smol135, name: SmolLM2-135M, path: models/SmolLM2-135M, kind: model, optional: true, for: [model_material], description: A 135M base model.}
  - {key: gsm8k, name: GSM8K, path: datasets/rl/gsm8k, kind: dataset, optional: true, description: Grade-school maths word problems.}

params:
  - {key: model_source, label: Base model, type: select, default: generated, options: [{value: generated, label: one of your runs}, {value: prepared, label: a prepared model}]}
  - {key: base_run, label: Run, type: run, experiment: world-pretrain, step: 3, show_if: {model_source: generated}}
  - {key: model_material, label: Prepared model, type: material, kind: model, formats: [atelier, hf], show_if: {model_source: prepared}}
  - {key: data_source, label: Data, type: select, default: generated, options: [{value: generated, label: generated}, {value: prepared, label: prepared}]}
  - {key: data_material, label: Prepared dataset, type: material, kind: dataset, schemas: [qa], show_if: {data_source: prepared}}
```

```python
from atelier_world.prepared import choose_model, load_lm, model_outputs, read_qa, train_val

info = choose_model(P, I, run_key="base_run", run_model_key="model", hint="Choose a run or a prepared model.")
lm = load_lm(info, device)                          # either format
train, val = train_val(P["data_material"], read_qa, val_size=200)
```

`type: material` lists the experiment's declared materials of its `kind`, checked
against what the machines have: models by `formats`, datasets by `schemas`. `show_if`
hides a field unless another param has the given value. A `for:` naming a param that
does not exist stops the experiment from loading, so a typo cannot silently hide a choice.
