# Architecture

## Processes

| Process | Job |
|---|---|
| **web** (nginx) | serves the React build, proxies `/api`, `/ws`, `/grafana` |
| **api** (FastAPI) | auth, experiment registry, files, run creation, websockets, `/metrics` |
| **worker** | owns the 8 GPUs; pops jobs from Redis, runs them, streams output |
| **postgres** | users, runs, submissions, display state |
| **redis** | job queue, cancellation set, pub/sub for logs and events |
| **prometheus + node-exporter + grafana** | hardware history and the dashboard on display 1 |

The API never executes anything. It writes a `Run` row, materialises the run
directory and pushes the id to Redis. This keeps a slow or crashed training job
from touching the web tier.

## The life of a run

```
browser          api                redis            worker              disk
  │ POST steps/2/runs                │                 │                   │
  ├───────────────►│ validate params, resolve parent    │                   │
  │                ├─ write params.json inputs.json meta.json ─────────────►│
  │                ├─ RPUSH atelier:jobs ──────────────►│                   │
  │ 201 {id}       │                 │  LPOP ──────────►│                   │
  │◄───────────────┤                 │                  ├ allocate GPUs     │
  │ WS /ws/runs/42 │                 │                  ├ spawn python ────►│
  │◄───────────────┼─ pub/sub ◄──────┼─ log lines, ::progress ──────────────┤
  │                │                 │                  ├ read result.json ◄┤
  │◄── status ─────┼─────────────────┼──────────────────┤ update Run row    │
```

Cancellation is a Redis set: the API adds the id, the worker's watchdog notices
within a second and kills the process group (or `docker kill`s the container).

## The run directory

Everything about one execution lives in `data/runs/<id>/`:

```
params.json   what the student chose
inputs.json   materials_dir, workspace_dir, experiment_dir + the previous step's outputs
meta.json     run id, experiment, step, user, command
run.log       stdout and stderr, streamed to the browser as it appears
live.json     series accumulated from ::progress lines (survives a page reload)
result.json   metrics, charts, tables, token views, artifacts, outputs
<artifacts>   tokenizer.json, ckpt.pt, adapter/, corpus.jsonl …
```

`result.json` is the contract between Python and React: the frontend renders
whatever a step declares, so a new experiment needs no frontend code.

## Scheduling

`GpuPool` hands out the lowest free indices. A job that needs more GPUs than are
free waits, and smaller jobs may pass it — in a classroom a one-GPU test should not
queue behind an eight-GPU pretraining run. Students are limited to 2 concurrent
runs and 2 GPUs per run (`ATELIER_MAX_*`); teachers are not. On restart the worker
fails runs that were mid-flight and re-queues the rest.

## Isolation

Default (`ATELIER_RUNNER_BACKEND=subprocess`) runs student code as a subprocess of
the worker with a scrubbed environment, `CUDA_VISIBLE_DEVICES` set to the allocated
GPUs, HuggingFace forced offline, and the process in its own session so the whole
tree can be killed. For stricter separation set `ATELIER_RUNNER_BACKEND=docker`:
each run gets a container with no network, a memory cap, the materials mounted
read-only, and — for grading — the submission mounted read-only too.

## Isolation caveats

Student code runs as ordinary Python. It can read the materials and its own
workspace, and with the docker backend nothing else. It cannot reach the network
in docker mode; in subprocess mode it shares the worker's network namespace, which
on an offline LAN means it can reach the LAN. That is the right trade-off for a
classroom, not for hostile code.
