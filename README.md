# Atelier

A classroom platform for building language models. Students sign in from their own
laptops, work through experiments of exactly four steps each, write their own
project against a prepared interface, test it and submit it. Every computation
happens on the server; the laptops only render. Wall displays show the hardware and
the class in real time.

Eighteen experiments, sixteen of which download nothing at all: the corpus is
generated from a seed and the models are trained from scratch, so a class can run
most of the course on an air-gapped machine with no licences to check and no dataset
to manage.

- **Frontend** React 18 + Vite, served by nginx
- **Backend** FastAPI + SQLAlchemy (Postgres or SQLite), a Redis queue, and worker processes that take jobs from it — one per machine, on its GPUs or, on a machine without any, on its CPU
- **Experiments** plain Python packages on disk, discovered from `experiment.yaml` — no code change to add one
- **Offline** nothing reaches the internet at run time

```
laptops ─┐
         ├─ nginx ─┬─ React build
displays ┘         ├─ /api  → FastAPI ─┬─ Postgres ─┬─ worker ── 8 × RTX A6000
                   ├─ /ws   → FastAPI ─┘  Redis ────┼─ worker ── 4 × GPU
                   │                                └─ worker ── CPU only
                   └─ /grafana → Grafana ← Prometheus ← /metrics (NVML) + node-exporter
```

Installation in detail is in [INSTALL.md](INSTALL.md). The short version, everything
on one GPU machine:

```bash
cp .env.example .env          # set ATELIER_JWT_SECRET, ATELIER_DB_PASSWORD, ATELIER_ADMIN_PASSWORD
docker compose up -d --build
```

Then open `http://<server>/` and sign in with the account from `.env`, and create
the class:

```bash
docker compose exec api python -m atelier.seed_users --students 24 --password lab2026
```

There is no GPU count to configure: each worker asks the NVIDIA driver how many GPUs
it has, and a machine with none runs its jobs on CPU.

## The course

Each experiment has exactly four steps. A step is a Python script with declared
parameters; the platform runs it on a worker, streams its log and charts, and passes
its outputs to the next step. After the four steps the student writes their own
implementation against a fixed interface, and a grader scores it.

| # | Experiment | The four steps | Needs downloads |
|---|---|---|---|
| 1 | Tokenizer | Corpus → Dedup → Train → Coverage | no |
| 2 | Pretraining | Pack → Size → Train → Evaluate | no |
| 3 | Fine-tuning | Data → Baseline → Train → Evaluate | no |
| 4 | Reinforcement learning | Reward → Prompts → Train → Evaluate | no |
| 5 | Reasoning | Prompting → Collect → Train → Evaluate | no |
| 6 | Preference optimisation | Pairs → Reference → Train → Evaluate | no |
| 7 | Scaling laws | Plan → Sweep → Fit → Verify | no |
| 8 | Distributed training | Profile → Scale → Optimise → Verify | no |
| 9 | Data curation | Survey → Filter → Deduplicate → Train | no |
| 10 | Parameter-efficient tuning | Setup → Sweep → Compare → Merge | no |
| 11 | Retrieval | Corpus → Encode → Train → Evaluate | no |
| 12 | Retrieval on Wikipedia | Index → Retrieve → Generate → Evaluate | yes |
| 13 | Evaluation | Tasks → Harness → Sensitivity → Report | no |
| 14 | Tool use | Tools → Traces → Train → Evaluate | no |
| 15 | Long context | Measure → Extend → Train → Evaluate | no |
| 16 | Efficiency | Measure → Quantise → Accelerate → Distil | no |
| 17 | Safety | Policy → Measure → Train → Evaluate | yes |
| 18 | Interpretability | Probe → Attention → Logit lens → Steering | no |

They chain. The vocabulary from 1 is what 2 loads; the model from 2 is what 3
fine-tunes; 4, 5, 6 and 10 all start from 3. What a student ships in 18 traces back
to a decision they made in 1.

Three alternates using public models and data sit below the course — curation on real
web text, evaluation on HellaSwag and MMLU, interpretability on Pythia — and below
those, five older experiments built on `transformers` and `trl`. See
[docs/curriculum.md](docs/curriculum.md).

### Where the data comes from

`experiments/_lib/atelier_world/` generates everything: documents, instructions,
solved tasks with their reasoning steps, a searchable library, and a corpus spoiled
in nine labelled ways. Language packs for English and Japanese; adding another is one
file.

`experiments/_lib/atelier_mini/` is the model stack in pure PyTorch — the transformer,
both training loops, GRPO, DPO, LoRA, quantisation, a KV cache, speculative decoding
and a contrastive embedder. No `transformers`, no `trl`, no pretrained weights.
Written to be read rather than called.

When your own data is ready, step 1 of the Tokenizer experiment reads a folder of
your text instead of generating one, and the rest of the course follows from it.

### GPUs, and machines without them

Every experiment runs on CPU as well as on GPUs. A worker on a machine with no GPU
takes jobs like any other: runs that ask for no GPU run there straight away, and runs
that ask for GPUs go to a GPU node whenever one is up and fall back to the CPU only
when none is. Multi-GPU steps run as one process on CPU and say so in their results.
It is slow — training steps sized for eight A6000s can take hours — but nothing
crashes for want of CUDA.

## Repository layout

```
backend/atelier/        FastAPI app, models, routers, and the worker
  routers/              one module per API area
  worker/               the scheduler, the runner, GPU allocation, result upload
experiments/            the course — see below
  _lib/                 shared libraries importable from any step
  _template/            skeleton to copy when writing your own
  <slug>/               one experiment
frontend/src/           React app: pages, components, api/ws clients
deploy/                 nginx, Prometheus, Grafana, and the worker deployments — see below
scripts/                development, user seeding, deployment, and the offline tooling
  deploy/               deploying workers to many nodes, over SSH or to Kubernetes
  offline/              images, materials and registry for an air-gapped install
docs/                   everything in depth
```

## The experiments folder

Every folder here that does not start with `_` is an experiment. The registry scans
this directory, reads each `experiment.yaml`, and the platform rebuilds its pages
within five seconds. **Adding an experiment needs no code change and no restart.**

```
experiments/world-tokenizer/
  experiment.yaml       title, description, the four steps, their parameters, the grader
  steps/step1_*.py      one script per step
  sample/               the starter project a student's workspace is copied from
    README.md           the brief they read
    project/            the files they edit
  grader/grade.py       scores a submission, emitting a `score` metric out of 100
```

A step script is ordinary Python. It reads its parameters, does its work, reports
progress, and writes a result:

```python
from atelier_sdk import Result, params, parse_args, progress

parse_args()
P = params({"vocab_size": 4096})
progress(50, "training")
R = Result()
R.metric("chars_per_token", "Chars per token", 4.21, "num", "kept")
R.chart("curve", "Tokens as the vocabulary grows", rows, "vocab", [...], "line")
R.output("tokenizer", str(path))     # becomes an input to the next step
R.save()
```

Whatever a step puts in `R.output(...)` arrives as `inputs()` in the next one. That is
the whole contract between steps. A step picks its device with
`"cuda" if torch.cuda.is_available() else "cpu"`; the platform also passes
`ATELIER_DEVICE` and `ATELIER_GPUS`.

Four rules the registry enforces: exactly four steps, every script present, a grader
present, a `sample/` directory present. A folder that breaks one is listed as an error
on the teacher's page rather than silently ignored.

To write your own: `cp -r experiments/_template experiments/my-thing`, edit the
manifest, and it appears. [docs/experiments.md](docs/experiments.md) covers the
parameter types, the chart and table shapes, and how graders are run.

## The deploy folder

Configuration rather than code. The nginx, Prometheus and Grafana files are mounted
into the official images on whichever machine runs those containers — on a split
install that is the control machine only. The worker deployments are read by the
scripts in `scripts/deploy/`.

```
deploy/nginx/nginx.conf                  production: the React build, and /api, /ws, /grafana
deploy/nginx/nginx.dev.conf              the laptop stack: the same without TLS or caching
deploy/prometheus/prometheus.yml         one machine: scrapes the API and node-exporter
deploy/prometheus/prometheus.split.yml   several worker nodes, via the targets files below
deploy/prometheus/targets/gpu-nodes.yml        every worker's metrics; the deploy scripts rewrite it
deploy/prometheus/targets/node-exporters.yml   the same machines, for CPU and disk
deploy/grafana/provisioning/             datasource and dashboard wiring, applied at startup
deploy/grafana/dashboards/hardware.json  the dashboard the wall displays show
deploy/workers.example                   the node list for the SSH deploy; copy to deploy/workers
deploy/k8s/base/                         the namespace and the CPU worker DaemonSet
deploy/k8s/worker-gpu.template.yaml      the GPU worker, one DaemonSet per GPU count
```

Two things to know. Prometheus does not expand environment variables in its
configuration, which is why multi-node targets live in separate files rather than
being templated into the main one. And the Grafana dashboard has a fixed uid
(`atelier-hw`) because the display pages link straight to it — if you edit the
dashboard in the Grafana UI, export it back over
`deploy/grafana/dashboards/hardware.json`, or the change is lost on the next restart.

## Running it

| Situation | Command |
|---|---|
| Everything on the GPU node | `docker compose up -d` |
| Control plane and GPU node separate | `docker compose -f docker-compose.api.yml up -d`, and `-f docker-compose.worker.yml` on the node |
| A worker node without GPUs | `-f docker-compose.worker.yml -f docker-compose.worker-cpu.yml` on that node |
| Many worker nodes, from the control machine | `scripts/deploy/deploy-workers.sh up` over SSH, or `scripts/deploy/k8s-workers.sh apply` on Kubernetes |
| Shared storage over NFS | add `-f docker-compose.nfs.yml` to either |
| A laptop, no GPU | `docker compose -f docker-compose.windows.yml up -d --build` |
| Development, no Docker | `./scripts/dev.sh`, or `scripts\dev.cmd` on Windows |

## Documentation

- [INSTALL.md](INSTALL.md) — installing, configuring and operating it, start to finish
- [docs/architecture.md](docs/architecture.md) — how a run flows from click to result
- [docs/experiments.md](docs/experiments.md) — writing your own four-step experiment
- [docs/curriculum.md](docs/curriculum.md) — all eighteen experiments and what each needs
- [docs/api.md](docs/api.md) — every endpoint
- [docs/displays.md](docs/displays.md) — the wall screens
- [docs/topologies.md](docs/topologies.md) — one machine, a control plane and several nodes, nodes without GPUs
- [docs/deploy-workers.md](docs/deploy-workers.md) — deploying workers to many nodes over SSH or Kubernetes, with materials from NFS
- [docs/runtime.md](docs/runtime.md) — CUDA, PyTorch and Python versions, the CPU image, and how to change them
- [docs/offline-install.md](docs/offline-install.md) — air-gapped installation
- [docs/harbor.md](docs/harbor.md) — serving the images from your own registry
- [docs/windows.md](docs/windows.md) — trying it, developing and deploying from Windows

## Status, honestly

The Tokenizer experiment has been run end to end — all four steps, on generated text
and on a folder of custom text. Data curation and Evaluation have had their CPU-only
steps run. The world generator, the BPE trainer, the BM25 baseline and the corpus
tooling are all executed and checked.

Everything that needs a GPU is syntax-checked but unexecuted, because it was written
without a GPU to run it on. The CPU fallback in the experiments is the same: compiled,
not yet run under torch. Before the first lesson, run in this order:
`world-tokenizer`, then `world-pretrain` at the tiny preset for 200 steps, then
`world-sft`. Those three exercise the model, both training loops, the sampler and the
grader harness, and will surface anything that needs fixing in about ten minutes.

The scheduler's CPU-node rules are tested against a simulated queue. The SSH deploy
script is tested end to end against simulated nodes, and the Kubernetes overlay is
checked by rendering it with `kubectl kustomize`; neither has yet met a real cluster
or a real NFS server.
