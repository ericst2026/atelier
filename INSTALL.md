# Installing and running Atelier

Start to finish: what you need, how to install it on one machine or many, how to run
a class, and how to change it.

If you only want to look at it, skip to [Trying it on a laptop](#7-trying-it-on-a-laptop) —
that needs no GPU and about ten minutes.

---

## 1. What you need

**A GPU machine.** Linux, an NVIDIA driver, Docker with the container toolkit, and
enough disk for the data directory. Any number of GPUs; the worker counts them itself
through the driver. CUDA 12.8 needs driver 525.60.13 or newer.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker version --format '{{.Server.Version}}' && docker compose version --short
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L
df -h /srv
```

That third command is the one that matters: if it fails, the GPUs are visible to the
host but not to containers, and the worker will find none and run everything on CPU.
Install `nvidia-container-toolkit` and restart Docker.

From Windows, `scripts\offline\check-node.cmd -Node you@gpu-node` runs all of these
over SSH.

**Machines without GPUs can be workers too** — Linux and Docker with Compose 2.24 or
newer. They run jobs on CPU; see [Nodes without GPUs](#nodes-without-gpus).

**Optional.** A registry (Harbor or similar) if you would rather pull images than
build them on each node. A second machine if you want the control plane off the GPU
box. An NFS share for materials if you run several nodes. Materials at all only if
you intend to teach the six download-based experiments — the other twelve need
nothing.

---

## 2. The quickest working install

Everything on the GPU machine.

```bash
git clone <this repository> /srv/atelier/app
cd /srv/atelier/app
cp .env.example .env
```

Edit `.env`. Three lines matter before anything else:

```bash
ATELIER_JWT_SECRET=<openssl rand -hex 32>
ATELIER_DB_PASSWORD=<something>
ATELIER_ADMIN_PASSWORD=<not "teacher">
```

`ATELIER_DB_PASSWORD` is the one Postgres password for every compose file: it sets
the database's password and the API's connection to it, so the two cannot drift
apart. Postgres reads it only when it first creates its volume — change it later with
`ALTER USER` inside Postgres, or start from a fresh volume.

There is no GPU count to set: the worker asks the NVIDIA driver at startup and logs
what it found.

Then:

```bash
mkdir -p /srv/atelier/data /srv/atelier/materials
docker compose up -d --build
docker compose logs -f worker      # watch it come up
```

```
worker up on 3f2a9c1d7b4e: 8 gpus (detected), backend=subprocess, storage=shared
```

Building the worker image takes twenty to forty minutes the first time: it is CUDA
plus PyTorch. The API and web images take a minute or two.

Open `http://<server>/`, sign in as the admin account from `.env`, and run the
Tokenizer experiment — it generates its own corpus and needs nothing else. If that
completes all four steps, the install is sound.

### Accounts for the class

```bash
docker compose exec api python -m atelier.seed_users --students 24 --password lab2026
docker compose exec -T api python -m atelier.seed_users --csv - < class.csv
```

The credentials are printed as well as written to a file, because a file inside a
container is easy to lose. Existing usernames are skipped rather than overwritten, so
re-running after adding names to the CSV does the right thing.

Or import a CSV from **Teacher → Users**, with columns `username,name,role,password`.
Role defaults to `student`.

### Wall displays

Point each display's browser at `http://<server>/display/1` through `/display/5` in
kiosk mode. What each shows is configured from the teacher's screen. See
[docs/displays.md](docs/displays.md).

---

## 3. Choosing a topology

| | When |
|---|---|
| **One machine** | The default. One thing to install, one place to look when it breaks. |
| **Control plane split off** | The UI should survive rebooting the GPU node, or the desktop is what is on your desk. |
| **Several worker nodes** | More than one machine to compute on — with GPUs or without. They share one queue. |

### One machine

`docker compose up -d`. Nothing further.

### Control plane on a desktop, GPUs on a server

On the desktop, in `.env`:

```bash
ATELIER_JWT_SECRET=<openssl rand -hex 32>
ATELIER_WORKER_TOKEN=<openssl rand -hex 32, a different one>
ATELIER_DB_PASSWORD=<something>
ATELIER_REDIS_PASSWORD=<something>
```

The JWT secret signs session cookies and the worker token authenticates workers to
the API; keep them different.

```bash
docker compose -f docker-compose.api.yml up -d
```

That publishes 80 for the UI, 5432 for Postgres and 6379 for Redis. The last two are
how workers join — a worker reads and writes the run records in Postgres directly —
which is why they have passwords, and why those ports should be reachable only from
your worker nodes.

On the GPU node, in `.env`, the worker token and both passwords plus:

```bash
ATELIER_CONTROL_HOST=192.168.1.20
ATELIER_NODE_NAME=gpu-a
ATELIER_MATERIALS_DIR=/srv/atelier/materials
```

```bash
docker compose -f docker-compose.worker.yml up -d
docker compose -f docker-compose.worker.yml logs -f worker
```

Watch those logs the first time. The worker checks the API before accepting any job
and refuses to start if the token or URL is wrong:

```
worker up on gpu-a: 8 gpus (detected), backend=subprocess, storage=upload
experiments match the API (a3f91c20e5b74d18)
uploading results to http://192.168.1.20:80
```

Both machines need the same `experiments/` tree; only the control machine needs
`deploy/`, and only the worker nodes need `materials/`. The worker hashes its copy of
the experiments against the API's at startup and warns if they differ.

For more than one node, do not do this by hand — see
[Deploying many nodes](#deploying-many-nodes).

### Nodes without GPUs

A machine with no NVIDIA GPU runs the CPU image and drops the GPU reservation:

```bash
docker compose -f docker-compose.worker.yml -f docker-compose.worker-cpu.yml up -d
```

```
worker up on cpu-a: 0 gpus (detected), backend=subprocess, storage=upload
no GPUs visible on cpu-a: running as a CPU-only node, 1 run(s) at a time. …
```

How the queue treats it:

- **Runs that ask for no GPU** run on it like anywhere else.
- **Runs that ask for GPUs go to a GPU node whenever one is up.** The CPU node hands
  them back to the queue, and runs them itself — on CPU — only when no GPU node has
  sent a heartbeat for fifteen seconds. A cluster of CPU nodes alone runs the whole
  course, slowly. The run log's first line says when a run landed on CPU.
- **One run at a time by default**, because training on CPU uses every core.
  `ATELIER_CPU_SLOTS` raises it; each run gets the cores divided by the slots.

The experiments fall back to CPU on their own. Multi-GPU steps run as one process and
say in their results that the scaling numbers mean nothing there.

**The whole stack on one machine without GPUs** — the single-machine file with its CPU
layer, which does the same for the API and the worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

Plain `docker compose up -d` there fails before anything runs, because that file asks
for the NVIDIA runtime: `failed to create the automatic CDI modifier … libcuda.so.1.1:
not found`.

### Several nodes

They take jobs from one queue. Nodes need not match: different GPU counts, different
cards, machines with no GPU at all. One run never spans machines, so a job asking for
four GPUs waits for four free on a single node.

Full detail, including NFS instead of HTTP upload: [docs/topologies.md](docs/topologies.md).

### Deploying many nodes

Two ways to deploy and update every worker from the control machine, both covered in
[docs/deploy-workers.md](docs/deploy-workers.md).

**Over SSH** — each node needs only SSH and Docker:

```bash
cp deploy/workers.example deploy/workers     # one line per node: atelier@192.168.170.101
export CONTROL_HOST=192.168.1.20
scripts/deploy/deploy-workers.sh check
scripts/deploy/deploy-workers.sh up
```

For each node it finds the GPUs and picks the GPU or the CPU image, copies
`experiments/` and a worker-only `.env` built from yours, pulls or loads the image,
copies the materials from NFS, starts the worker and waits for it to report in, then
rewrites the Prometheus target list. Run `up` again to update; `status`, `logs`,
`sync`, `materials` and `down` do what they say.

**On Kubernetes** — an on-premises cluster with NVIDIA's device plugin and GPU
feature discovery on the GPU nodes:

```bash
scripts/deploy/deploy-workers.sh sync        # experiments/ to every node
scripts/deploy/k8s-workers.sh apply
```

A CPU worker DaemonSet on nodes without GPUs, and one GPU worker DaemonSet per GPU
count found in the cluster, with the settings and secrets generated from `.env`.

---

## 4. Materials, if you need them

Twelve experiments generate their own data. Six use public models and datasets:
retrieval on Wikipedia, safety, and the three public-data alternates, plus the older
`transformers` track.

On a machine with internet:

```bash
python scripts/offline/fetch-materials.py --plan            # what and how big
python scripts/offline/fetch-materials.py --tracks core     # about 13 GB
```

From Windows: `.\scripts\offline\fetch-materials.cmd -Out D:\atelier-materials -Tracks core`.

Materials belong on the worker nodes, not the control machine — they are large and
read constantly during training. Each worker publishes an inventory, so the
experiment pages show *which node* has what, and a run is routed to a node that can
serve it. If no node has them, the run fails immediately with the reason rather than
queuing forever.

**One node:** copy them into `/srv/atelier/materials`.

**Several nodes:** put them on an NFS share once and name it in the control
machine's `.env`:

```bash
ATELIER_MATERIALS_NFS=192.168.1.50:/export/atelier/materials
```

Both deploy scripts then copy them into each node's `/srv/atelier/materials` before
its worker starts — everything on a new node, only what changed on an existing one,
after checking there is room. The worker reads the local copy at disk speed.

---

## 5. Installing without internet on the node

Two ways to get images across.

**A registry.** Build on a connected machine and push:

```powershell
docker login harbor.local
.\scripts\offline\push-to-harbor.cmd -Registry harbor.local -Project atelier -Cpu
```

`-Cpu` adds `atelier-worker-cpu`, the image for nodes without GPUs. On each node, in
`.env`:

```bash
ATELIER_REGISTRY=harbor.local/atelier/
ATELIER_TAG=latest
```

```bash
docker compose pull && docker compose up -d --no-build
```

Every service takes the prefix, including Postgres, Redis, Prometheus and Grafana, so
nothing reaches Docker Hub. See [docs/harbor.md](docs/harbor.md).

**A USB stick.**

```bash
CUDA_VARIANTS="cu128 cpu" ./scripts/offline/export-images.sh /media/usb/atelier-images   # connected machine
./scripts/offline/import-images.sh /media/usb/atelier-images                             # each node
docker compose up -d --no-build
```

Leave `cpu` out if every node has GPUs. With many nodes, the SSH deploy streams the
right tarball to each node itself: `IMAGES_DIR=/media/usb/atelier-images
scripts/deploy/deploy-workers.sh up`.

---

## 6. Choosing CUDA, PyTorch and Python

Defaults suit Ampere and newer, which includes the A6000: CUDA 12.8, PyTorch 2.8,
Python 3.12. All four are build arguments.

```bash
docker build -f backend/Dockerfile.worker \
  --build-arg TORCH_VERSION=2.10.0 -t atelier-worker .
```

Or in `.env`, for `docker compose build`:

```bash
ATELIER_CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
ATELIER_CUDA_WHEEL=cu126
ATELIER_CUDA_MM=12.6
ATELIER_TORCH_VERSION=2.8.0
```

Use `cu126` if the class includes cards older than Turing: PyTorch's `cu128` wheels
drop Maxwell, Pascal and Volta, though CUDA 12.8 itself still supports them.

The same Dockerfile builds the CPU image, with no CUDA libraries in it — a fraction of
the size. `docker-compose.worker-cpu.yml` builds it this way on its own:

```bash
docker build -f backend/Dockerfile.worker \
  --build-arg CUDA_IMAGE=ubuntu:24.04 --build-arg CUDA_WHEEL=cpu -t atelier-worker-cpu .
```

The build ends by asserting the installed wheel matches what you asked for — the
CUDA version, or no CUDA at all for `cpu` — and printing its architecture list. Read
it once after any change:

```
python 3.12.3 · torch 2.8.0 · cuda 12.8
compiled for: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

[docs/runtime.md](docs/runtime.md) has the combination table.

---

## 7. Trying it on a laptop

No GPU, no materials, no Postgres.

```bash
docker compose -f docker-compose.windows.yml up -d --build
```

`http://localhost:8080`, sign in as `teacher` / `teacher`. The Tokenizer experiment
runs end to end; the first steps of Data curation and Evaluation run too. Anything
that trains will fail for want of torch, which is expected — that stack uses the
lightweight API image as its worker. For the training steps on CPU, run the
`atelier-worker-cpu` image as the worker instead.

On Windows, PowerShell blocks `.ps1` files by default. Use the `.cmd` wrappers
(`scripts\dev.cmd`, `scripts\offline\push-to-harbor.cmd`,
`scripts\deploy\deploy-workers.cmd`), or run
`powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1`. See
[docs/windows.md](docs/windows.md).

---

## 8. Developing

Without Docker, against SQLite and a local Redis:

```bash
./scripts/dev.sh          # Linux and macOS
scripts\dev.cmd           # Windows
```

That starts the API, the worker and the Vite dev server, with the frontend on 5173
proxying to the API on 8000. Editing backend code reloads; editing frontend code
hot-reloads. The dev worker runs with `ATELIER_GPU_COUNT=0`, so jobs run on CPU.

Run the backend tests — registry validation, storage safety, the SDK contract:

```bash
cd backend && python tests/test_core.py
```

They are quick and worth running before any commit that touches `experiments/` or
`backend/atelier/registry.py`.

---

## 9. Changing things

### Adding an experiment

```bash
cp -r experiments/_template experiments/my-experiment
```

Edit `experiment.yaml`: the slug, the title, the `order` that decides where it appears,
and four steps. Write `steps/step1.py` through `step4.py`, a `sample/project/` for the
student to edit, and `grader/grade.py`. It appears in the UI within five seconds — no
restart, no rebuild, because `experiments/` is a mounted volume.

Pick the device with `"cuda" if torch.cuda.is_available() else "cpu"` and guard
CUDA-only calls (`torch.cuda.synchronize()`, peak-memory stats), so the step also
runs on a node without GPUs.

If it does not appear, look at **Teacher → Experiments**: manifest errors are listed
there with the reason.

[docs/experiments.md](docs/experiments.md) is the reference for parameter types, chart
shapes and grader conventions.

### Changing an existing one

Edit the files. The next run picks them up — on several nodes, after
`scripts/deploy/deploy-workers.sh sync`. Runs already finished keep the results they
produced, so a change mid-term does not rewrite history — but it does mean two
students can have results from different versions of the same step, which is worth
saying out loud if you change something significant.

### Changing the platform

Backend and frontend live in images, so those need a rebuild:

```bash
docker compose up -d --build api web
```

Split install: rebuild and push, then pull on the nodes — `deploy-workers.sh up` or
`k8s-workers.sh apply` does the pulling. Push a new tag rather than overwriting
`latest`, so a lesson in progress is not disturbed, and update between lessons: a
worker that restarts fails the runs it had in progress.

```powershell
.\scripts\offline\push-to-harbor.cmd -Registry harbor.local -Project atelier -Tag 2026-04-15 -Skip Third -Cpu
```

### Renaming the project

```bash
./scripts/rename-project.sh "Our Lab"
```

Changes the display name, the package name, the environment prefix, the Redis keys,
the metric names, the Grafana uid and the paths.

### Using your own data

Step 1 of the Tokenizer experiment reads a folder under `materials/`:

```
materials/corpora/our-reports/report_001.txt
materials/corpora/our-reports/tickets.jsonl
```

Set the source to *a folder of your own text* and the path to `corpora/our-reports`.
`.txt` and `.md` become one document each; `.jsonl` becomes one per line, reading
`text`, `content` or `body`. Everything downstream inherits the vocabulary that step
produces.

---

## 10. Running a class

**Before the first lesson.** Run `world-tokenizer` end to end, then `world-pretrain`
at the tiny preset for 200 steps, then `world-sft`. Ten minutes, and it exercises the
model, both training loops, the sampler and the grader harness. Check
**Teacher → Experiments** for manifest errors and missing materials.

**Limits** are in `.env`:

```bash
ATELIER_MAX_RUNNING_PER_STUDENT=2
ATELIER_MAX_GPUS_PER_STUDENT_RUN=2
ATELIER_DEFAULT_TIMEOUT_MIN=360
ATELIER_CPU_SLOTS=1
```

Two concurrent runs per student and two GPUs each works for a class of twenty-four on
eight GPUs. Teachers are not limited.

**During a lesson**, the teacher's page shows the queue, every running job, the GPUs
and who holds them, the node each run is on, and lets you cancel a run. The wall
displays show the same thing without the controls.

**CPU nodes are for the light work.** Tokenizer, data curation, graders and the tiny
presets are fine on CPU. The steps sized for GPUs — pretraining, fine-tuning and RL on
Qwen, the larger world presets — can take hours there and may hit their step's time
limit. With GPU nodes up, those steps never land on CPU anyway.

**One caution.** The distributed training experiment asks for up to eight GPUs, so a
student starting it queues behind everything else. It works best as a demonstration
with the class watching, rather than twenty students each requesting the whole node.

---

## 11. When something is wrong

| Symptom | Look at |
|---|---|
| Experiment missing from the home page | Teacher → Experiments; the manifest error is listed |
| Runs stay queued forever | `docker compose logs worker` — is any worker up, and does its node have the materials? |
| `failed to create the automatic CDI modifier` / `libcuda.so.1.1: not found` on `up` | The machine has no NVIDIA driver; add `-f docker-compose.cpu.yml` (or `-f docker-compose.worker-cpu.yml` on a worker node) |
| Worker says `0 gpus` on a GPU machine | GPUs invisible to the container: `nvidia-container-toolkit`, then restart Docker |
| A GPU step ran slowly, log says `cpu (asked for N GPU(s))` | No GPU node was up when it started; it ran on a CPU node |
| "No GPU node has the materials…" | That node lacks them; fetch them, set `ATELIER_MATERIALS_NFS`, or use a generated-data experiment |
| Worker refuses to start (split install) | Token mismatch or wrong `ATELIER_API_URL`; the log says which |
| API cannot log in to Postgres | `ATELIER_DB_PASSWORD` changed after the volume was created; `ALTER USER` or a fresh volume |
| `deploy-workers.sh` fails on a node | `deploy-workers.sh check <node>` lists what that node is missing |
| Materials copy refuses: "room for … MB" | The node's disk is too small for the share; free space or trim the share |
| Kubernetes pod stuck in `Init` | The materials copy: `kubectl -n atelier logs <pod> -c materials` |
| GPU node gets the CPU worker on Kubernetes | No `nvidia.com/gpu.count` label: GPU feature discovery is not running there |
| GPU strip empty | Worker not sampling NVML — check `curl http://<node>:9101/metrics \| grep atelier_gpu` |
| Grafana panel flat | `deploy/prometheus/targets/gpu-nodes.yml` does not list the node; `deploy-workers.sh targets` rewrites it |
| Build fails on the CUDA assertion | `CUDA_WHEEL` and `CUDA_IMAGE` disagree; see docs/runtime.md |
| `.ps1 cannot be loaded` on Windows | Use the `.cmd` wrapper, or `scripts\unblock.cmd` |

Logs worth knowing:

```bash
docker compose logs -f worker          # scheduling, GPU allocation, run failures
docker compose logs -f api             # requests, registry reloads
scripts/deploy/deploy-workers.sh status   # every node's worker and what it found
docker compose exec api python -c "from atelier.registry import Registry; \
  from atelier.config import settings; r=Registry(settings.experiments_dir); r.reload(True); print(r.errors)"
```

---

## 12. Backing up

Everything that matters is in two places: the database and the data directory.

```bash
docker compose exec postgres pg_dump -U atelier atelier > atelier-$(date +%F).sql
tar czf atelier-data-$(date +%F).tar.gz -C /srv/atelier data
```

The data directory holds run outputs, student workspaces and submissions, and grows
with use — mostly model checkpoints. `experiments/` is in version control and
`materials/` can be refetched (or lives on the NFS share), so neither needs backing
up.
