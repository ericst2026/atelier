# Installing and running Atelier

Start to finish: what you need, how to install it on one machine or two, how to run
a class, and how to change it.

If you only want to look at it, skip to [Trying it on a laptop](#trying-it-on-a-laptop) —
that needs no GPU and about ten minutes.

---

## 1. What you need

**The GPU machine.** Linux, an NVIDIA driver, Docker with the container toolkit, and
enough disk for the data directory. Any number of GPUs; the platform reads the count
from the machine itself. CUDA 12.8 needs driver 525.60.13 or newer.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
docker version --format '{{.Server.Version}}' && docker compose version --short
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L
df -h /srv
```

That third command is the one that matters: if it fails, the GPUs are visible to the
host but not to containers, and nothing else will work. Install
`nvidia-container-toolkit` and restart Docker.

From Windows, `scripts\offline\check-node.cmd -Node you@gpu-node` runs all of these
over SSH.

**Optional.** A registry (Harbor or similar) if you would rather pull images than
build them on the node. A second machine if you want the control plane off the GPU
box. Materials only if you intend to teach the six download-based experiments — the
other twelve need nothing.

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
ATELIER_JWT_SECRET=<a long random string>
ATELIER_ADMIN_PASSWORD=<not "teacher">
ATELIER_GPU_COUNT=8
```

Then:

```bash
mkdir -p /srv/atelier/data /srv/atelier/materials
docker compose up -d --build
docker compose logs -f worker      # watch it come up
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
| **Several GPU nodes** | More than one machine with GPUs. They share one queue. |

### One machine

`docker compose up -d`. Nothing further.

### Control plane on a desktop, GPUs on a server

On the desktop, in `.env`:

```bash
ATELIER_WORKER_TOKEN=<a long random string>
ATELIER_DB_PASSWORD=<something>
ATELIER_REDIS_PASSWORD=<something>
ATELIER_JWT_SECRET=<something>
```

```bash
docker compose -f docker-compose.api.yml up -d
```

That publishes 80 for the UI, 5432 for Postgres and 6379 for Redis. The last two are
how the worker joins, which is why they have passwords.

On the GPU node, in `.env`, the same three secrets plus:

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
worker up on gpu-a: 8 gpus, backend=subprocess, storage=upload
experiments match the API (a3f91c20e5b74d18)
uploading results to http://192.168.1.20:80
```

Both machines need the same `experiments/` tree; only the control machine needs
`deploy/`, and only the GPU nodes need `materials/`. Keep the experiments in step with
`rsync -a --delete experiments/ node:/srv/atelier/app/experiments/`, a git checkout on
each, or one copy on NFS mounted read-only on both — see
[docs/topologies.md](docs/topologies.md). The worker hashes its copy against the API's
at startup and warns if they differ.

### Several GPU nodes

The same worker file on each machine, with a different name:

```bash
ATELIER_NODE_NAME=gpu-b ATELIER_GPU_COUNT=4 docker compose -f docker-compose.worker.yml up -d
```

They take jobs from one queue. Nodes need not match: different GPU counts, different
cards. One run never spans machines, so a job asking for four GPUs waits for four free
on a single node.

Full detail, including NFS instead of HTTP upload: [docs/topologies.md](docs/topologies.md).

---

## 4. Materials, if you need them

Twelve experiments generate their own data. Six use public models and datasets:
retrieval on Wikipedia, safety, and the three public-data alternates, plus the older
`transformers` track.

On a machine with internet:

```bash
python scripts/offline/fetch-materials.py --plan            # what and how big
python scripts/offline/fetch-materials.py --tracks core     # about 13 GB
rsync -a ./materials/ node:/srv/atelier/materials/
```

From Windows: `.\scripts\offline\fetch-materials.cmd -Out D:\atelier-materials -Tracks core`.

Materials belong on the GPU nodes, not the control machine — they are large and read
constantly during training. Each worker publishes an inventory, so the experiment
pages show *which node* has what, and a run is routed to a node that can serve it. If
no node has them, the run fails immediately with the reason rather than queuing
forever.

---

## 5. Installing without internet on the node

Two ways to get images across.

**A registry.** Build on a connected machine and push:

```powershell
docker login harbor.local
.\scripts\offline\push-to-harbor.cmd -Registry harbor.local -Project atelier
```

On the node, in `.env`:

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
./scripts/offline/export-images.sh /media/usb/atelier-images   # connected machine
./scripts/offline/import-images.sh /media/usb/atelier-images   # classroom node
docker compose up -d --no-build
```

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

The build ends by asserting the installed wheel matches the CUDA you asked for and
printing its architecture list. Read it once after any change:

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
that trains will fail for want of torch, which is expected — add the CPU wheel to the
image if you want the tiny pretraining preset to work.

On Windows, PowerShell blocks `.ps1` files by default. Use the `.cmd` wrappers
(`scripts\dev.cmd`, `scripts\offline\push-to-harbor.cmd`), or run
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
hot-reloads.

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

If it does not appear, look at **Teacher → Experiments**: manifest errors are listed
there with the reason.

[docs/experiments.md](docs/experiments.md) is the reference for parameter types, chart
shapes and grader conventions.

### Changing an existing one

Edit the files. The next run picks them up. Runs already finished keep the results
they produced, so a change mid-term does not rewrite history — but it does mean two
students can have results from different versions of the same step, which is worth
saying out loud if you change something significant.

### Changing the platform

Backend and frontend live in images, so those need a rebuild:

```bash
docker compose up -d --build api web
```

Split install: rebuild and push, then pull on the node. Push a new tag rather than
overwriting `latest`, so a lesson in progress is not disturbed:

```powershell
.\scripts\offline\push-to-harbor.cmd -Registry harbor.local -Project atelier -Tag 2026-04-15 -Skip Third
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
```

Two concurrent runs per student and two GPUs each works for a class of twenty-four on
eight GPUs. Teachers are not limited.

**During a lesson**, the teacher's page shows the queue, every running job, the GPUs
and who holds them, and lets you cancel a run. The wall displays show the same thing
without the controls.

**One caution.** The distributed training experiment asks for up to eight GPUs, so a
student starting it queues behind everything else. It works best as a demonstration
with the class watching, rather than twenty students each requesting the whole node.

---

## 11. When something is wrong

| Symptom | Look at |
|---|---|
| Experiment missing from the home page | Teacher → Experiments; the manifest error is listed |
| Runs stay queued forever | `docker compose logs worker` — usually GPUs invisible to the container |
| "No GPU node has the materials…" | That node lacks them; fetch them, or use a generated-data experiment |
| Worker refuses to start (split install) | Token mismatch or wrong `ATELIER_API_URL`; the log says which |
| GPU strip empty | Worker not sampling NVML — check `curl http://<node>:9101/metrics \| grep atelier_gpu` |
| Grafana panel flat | `deploy/prometheus/targets/gpu-nodes.yml` does not list the node |
| Build fails on the CUDA assertion | `CUDA_WHEEL` and `CUDA_IMAGE` disagree; see docs/runtime.md |
| `.ps1 cannot be loaded` on Windows | Use the `.cmd` wrapper, or `scripts\unblock.cmd` |

Logs worth knowing:

```bash
docker compose logs -f worker          # scheduling, GPU allocation, run failures
docker compose logs -f api             # requests, registry reloads
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
`materials/` can be refetched, so neither needs backing up.
