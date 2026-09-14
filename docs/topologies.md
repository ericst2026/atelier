# One machine or two

Atelier has three moving parts: the **API** (which serves the UI and owns the
database), the **worker** (which has the GPUs and runs the jobs), and **Redis and
Postgres** between them. Where those sit is a configuration choice.

## All on the GPU node

The default. `docker-compose.yml` starts everything on the Ubuntu machine, students
point a browser at it, and nothing else is involved.

```bash
docker compose up -d
```

Choose this unless you have a reason not to. It is one machine to install, one place
to look when something breaks, and the API and worker share a data directory so
results need no transfer.

## Control plane on the desktop, GPUs on the server

What you asked about: the API, the web UI, Postgres and Redis on the Windows desktop,
the worker on the Ubuntu box with the A6000s.

```
  Windows desktop                        Ubuntu, 8 × A6000
  ┌───────────────────────┐              ┌──────────────────────┐
  │ web  ──▶ api          │◀── results ──│ worker  (gpu-a)      │
  │           │           │   over HTTP  │   runs the jobs      │
  │      postgres  redis  │◀── jobs ─────│   materials on disk  │
  └───────────────────────┘   and state  └──────────────────────┘
        :80  :5432  :6379                 ┌──────────────────────┐
                                     ◀────│ worker  (gpu-b)      │
                              more nodes  │   same queue         │
                              join here   └──────────────────────┘
```

The worker takes jobs from the Redis on the desktop, writes to the Postgres on the
desktop, and posts each run's results back to the API over HTTP when the job
finishes. There is no shared filesystem to set up — no SMB, no NFS, no permissions
to argue with.

### On the desktop

```powershell
# .env
ATELIER_WORKER_TOKEN=a-long-random-string
ATELIER_DB_PASSWORD=something
ATELIER_REDIS_PASSWORD=something
ATELIER_JWT_SECRET=something

docker compose -f docker-compose.api.yml up -d
```

This publishes three ports on the LAN: 80 for the UI, 5432 for Postgres and 6379 for
Redis. The last two are how the worker joins, which is why both have passwords.

### On the GPU node

```bash
# .env
ATELIER_CONTROL_HOST=192.168.1.20      # the desktop
ATELIER_WORKER_TOKEN=a-long-random-string   # the same string
ATELIER_DB_PASSWORD=something               # the same password
ATELIER_REDIS_PASSWORD=something            # the same password
ATELIER_NODE_NAME=gpu-node
ATELIER_MATERIALS_DIR=/srv/atelier/materials

docker compose -f docker-compose.worker.yml up -d
docker compose -f docker-compose.worker.yml logs -f worker
```

The worker checks the API before it accepts any job, and refuses to start if the URL
is wrong or the token does not match. That check is worth watching the first time:

```
worker up on gpu-node: 8 gpus, backend=subprocess, storage=upload
experiments match the API (a3f91c20e5b74d18)
uploading results to http://192.168.1.20:80
```

### Which folders each machine needs

| Folder | Control machine | GPU nodes | Why |
|---|---|---|---|
| `experiments/` | yes | yes | the API reads the manifests to build the pages and copies `sample/` into workspaces; the worker runs the scripts and imports `_lib` |
| `deploy/` | yes | no | mounted by nginx, Prometheus and Grafana, all of which run on the control machine |
| `materials/` | no | yes | large, read constantly during training; each worker publishes an inventory so the API can still report availability |
| `backend/`, `frontend/` | in the images | in the images | not mounted at run time |

So a GPU node needs the repository only for `experiments/` and the compose file. If
you clone the whole thing, nothing is harmed — `deploy/` simply sits unused.

The two copies of `experiments/` must match. The worker hashes its copy at startup
and compares it with the API's, saying so if they differ, because a mismatch produces
failures that look like anything except the real cause.

Three ways to keep them in step, in the order most people end up choosing:

**rsync after each change.** No moving parts, no dependency between machines.

```bash
rsync -a --delete experiments/ node:/srv/atelier/app/experiments/
```

**A git checkout on each**, pulled from the same commit. Best if teachers write
experiments and you want the history.

**One copy on NFS**, read-only on both. Then the fingerprints cannot disagree,
because there is only one tree:

```yaml
# .env on both machines
ATELIER_STORAGE_MODE=shared
ATELIER_NFS_HOST=192.168.1.30
ATELIER_NFS_EXPERIMENTS=/export/atelier/experiments
```

```bash
docker compose -f docker-compose.api.yml -f docker-compose.nfs.yml up -d
docker compose -f docker-compose.worker.yml -f docker-compose.nfs.yml up -d
```

`docker-compose.nfs.yml` mounts `experiments/` read-only on both services alongside
the shared data directory. Read-only is enough: steps read their scripts from there
and write everything else to the data directory, and the images set
`PYTHONDONTWRITEBYTECODE=1`, so Python never tries to write `__pycache__` beside the
source.

What you give up is independence. If the NFS server is down, no node can start a run,
whereas with rsync each machine has its own copy and keeps working. On a classroom
LAN with the server on the GPU node, that trade is usually worth it — one tree,
nothing to synchronise, and editing an experiment takes effect everywhere at once.

### Materials

Only the worker needs them, and they are tens of gigabytes. They stay on the GPU node
and never cross the network.

### Large artifacts

Files above `ATELIER_UPLOAD_MAX_MB` (64 MB by default) are not uploaded. They stay on
the GPU node and the log says so:

```
not uploading model.pt for run 431: 118 MB is over the 64 MB limit, it stays on gpu-node
```

Charts, tables, metrics and JSONL samples are all far below that, so the run page
looks complete. Model checkpoints are what stays behind, and they are also what the
next step in the chain needs — which is fine, because the next step runs on the same
node. Raise the limit if you want students to download checkpoints from the browser;
a gigabyte over a LAN is about ten seconds.

## Several GPU nodes, one control plane

Yes — and it is the same worker file on each machine, with a different name:

```bash
# node A
ATELIER_NODE_NAME=gpu-a docker compose -f docker-compose.worker.yml up -d
# node B
ATELIER_NODE_NAME=gpu-b docker compose -f docker-compose.worker.yml up -d
```

Each worker counts its own GPUs through the NVIDIA driver at startup and logs the
number, so an 8-GPU and a 4-GPU node need no GPU setting at all.

With more than a couple of nodes, deploy them all from the control machine instead —
over SSH with `scripts/deploy/deploy-workers.sh`, or to a Kubernetes cluster with
`scripts/deploy/k8s-workers.sh`. See [deploy-workers.md](deploy-workers.md).

Both take jobs from the one Redis queue. `LPOP` is atomic, so two nodes never claim
the same run, and a node that is full simply stops taking work until a job of its own
finishes. Nodes do not have to match: different GPU counts, different cards, even
different CUDA builds of the worker image, as long as every experiment a student can
reach will run on whichever node picks it up.

What the cluster does with several nodes:

- **The queue is shared, the GPUs are not.** Each worker allocates from its own
  machine's GPUs, so a run asking for four GPUs waits for four free on a single node
  rather than two here and two there. Nothing spans machines except the distributed
  training experiment, which uses the GPUs of whichever node runs it.
- **Each run records the node that ran it**, from the moment it is claimed. The run
  page and the teacher table show it.
- **The hardware strip shows every node's GPUs**, labelled, because index 3 on one
  machine is not index 3 on another.
- **A restarting worker only cleans up after itself.** It fails the runs *it* was
  executing and leaves the other nodes alone — the alternative, which is what a naive
  recovery does, is a node restarting and killing everybody else's jobs.
- **The queue is reconciled under a lock.** If Redis is restarted and loses the
  queue, exactly one node re-enqueues the runs that are queued in the database but
  missing from Redis, so a job is not started twice.

Materials must be on every node that might run an experiment needing them; the
generated-data experiments need nothing. The experiments directory must be identical
everywhere, and each worker says so at startup if it is not.

## Worker nodes without GPUs

A machine with no NVIDIA GPU can be a worker too. Layer the CPU file on the worker
file. It runs the `atelier-worker-cpu` image — CPU-only PyTorch on plain Ubuntu, no
CUDA libraries to copy around — and drops the NVIDIA reservation docker would
otherwise refuse:

```bash
ATELIER_NODE_NAME=cpu-a docker compose -f docker-compose.worker.yml -f docker-compose.worker-cpu.yml up -d
```

The worker finds no GPUs and runs jobs on CPU, with `CUDA_VISIBLE_DEVICES` empty,
`ATELIER_GPUS=0` and `ATELIER_DEVICE=cpu`. The experiments fall back to CPU when they
see that. What it takes:

- **Runs that ask for no GPUs** (tokenizers, data curation, most graders) run on the
  CPU node like anywhere else.
- **Runs that ask for GPUs go to a GPU node whenever one is up.** The CPU node puts
  them back on the queue. When no GPU node has sent a heartbeat for fifteen seconds,
  the CPU node runs them itself, on CPU, and the run log's first line says so. A
  cluster of CPU nodes alone runs every step, slowly.
- **One run at a time by default.** Training on CPU uses every core, so a second run
  mostly slows the first. `ATELIER_CPU_SLOTS` raises it, and each run gets the cores
  divided by the slots as `OMP_NUM_THREADS`. A full CPU node leaves the rest of the
  queue to the other nodes rather than holding on to it.
- **Multi-GPU steps run as a single process.** The distributed training experiment
  measures one CPU process instead of a scaling curve and says so in its results.

## Settings that decide this

| Variable | Values | Effect |
|---|---|---|
| `ATELIER_ROLE` | `all`, `api`, `worker` | what this process does |
| `ATELIER_STORAGE_MODE` | `shared`, `upload` | whether both sides see the same data directory |
| `ATELIER_API_URL` | a URL | where the worker posts results |
| `ATELIER_WORKER_TOKEN` | a secret | must match on both machines |
| `ATELIER_UPLOAD_MAX_MB` | a number | the point past which artifacts stay put |
| `ATELIER_NODE_NAME` | a name | appears on the run page and in the logs |
| `ATELIER_GPU_COUNT` | a number, or unset | **worker only, usually unset** — the worker detects its GPUs itself; set it only to hold some back. The API adds up what the workers report |
| `ATELIER_CPU_SLOTS` | a number, default 1 | **nodes without GPUs** — how many runs they run on CPU at once |

`ATELIER_STORAGE_MODE=shared` is the third option: machines that *do* share a
directory. See the next section.

## Sharing the data directory over NFS

With `ATELIER_STORAGE_MODE=shared`, nothing is uploaded: the worker writes results
and the API reads them from the same place. It is faster and simpler in the steady
state, and there is no size limit, so students can download a checkpoint from the
browser. The cost is a mount that has to work before anything else does.

### On the file server

The GPU node is usually the right server — the data is written there and it has the
disks. On Ubuntu:

```bash
sudo apt install nfs-kernel-server
sudo mkdir -p /export/atelier/data /export/atelier/experiments
sudo chown -R 1000:1000 /export/atelier
```

`/etc/exports`, restricted to the machines that need it:

```
/export/atelier/data         192.168.1.0/24(rw,sync,no_subtree_check,root_squash)
/export/atelier/experiments  192.168.1.0/24(ro,sync,no_subtree_check,root_squash)
```

```bash
sudo exportfs -ra
sudo systemctl enable --now nfs-server
showmount -e localhost
```

### On each machine

Set these in `.env` and layer the override on:

```bash
ATELIER_STORAGE_MODE=shared
ATELIER_NFS_HOST=192.168.1.30
ATELIER_NFS_DATA=/export/atelier/data
ATELIER_NFS_EXPERIMENTS=/export/atelier/experiments
```

```bash
# control machine
docker compose -f docker-compose.api.yml -f docker-compose.nfs.yml up -d
# each GPU node
docker compose -f docker-compose.worker.yml -f docker-compose.nfs.yml up -d
```

Docker mounts the export itself through the volume driver, so the host needs nothing
in `/etc/fstab` and nothing has to be mounted before the daemon starts. Check it
landed:

```bash
docker compose -f docker-compose.worker.yml exec worker sh -c "touch /srv/atelier/data/.probe && ls -l /srv/atelier/data/.probe"
```

### The part that actually bites: user ids

NFS matches on numeric uid, not on names. The container writes as some uid; the
export decides whether that uid may write. The usual failure is a worker that runs
fine and then cannot create a run directory.

Three ways through, in order of how much you will like them later:

1. **Make the ids agree.** The images run as root by default and `root_squash` maps
   root to `nobody`, which is why the naive setup fails. Own the export as `1000:1000`
   and run both services as that uid:

   ```yaml
   # add to docker-compose.nfs.yml, or your own override
   services:
     worker:
       user: "1000:1000"
     api:
       user: "1000:1000"
   ```

2. **`no_root_squash` on the export.** One word, works immediately, and gives any
   container on the LAN root over that directory. Acceptable on an isolated
   classroom network, not somewhere else.

3. **`all_squash` with `anonuid=1000,anongid=1000`**, which maps everybody to one
   owner. Simple, and you lose the ability to tell who wrote what — which, for a
   directory that only two services write to, is not much of a loss.

### Windows as the control machine

If the API is on Windows and the NFS server is the Ubuntu node, Docker Desktop can
mount it with the same override — the volume driver runs in Docker's Linux VM, not
on Windows. It works, but it is the configuration most likely to waste an afternoon,
and `upload` mode exists precisely so you do not have to. Try `upload` first; move to
NFS if you find you want checkpoints in the browser.

### What to share and what not to

| Directory | Share it? |
|---|---|
| `data` | yes — this is the point: runs, workspaces, submissions |
| `experiments` | optional, read-only. Convenient, since the copies must match anyway |
| `materials` | no. Tens of gigabytes, read constantly during training; keep it local to each GPU node |

Sharing `materials` read-only over NFS is the one case where it is worth considering
anyway: it lets teachers browse dataset files from the experiment page. Training reads
would then cross the network, which for a tokenizer pass over a few gigabytes is fine
and for repeated epochs is not. If you want both, keep a local copy on each node and
share a second copy read-only with the control machine.

## Monitoring several nodes

Each worker samples its own NVML and serves `/metrics` on port 9101, so Prometheus on
the control machine scrapes every node the same way. The API serves its own metrics
as before.

The node list is **not** in `prometheus.yml`. Prometheus does not expand environment
variables in its configuration — a `${ATELIER_GPU_HOST}` there is scraped literally
and fails quietly — so the targets live in files Prometheus re-reads by itself:

```yaml
# deploy/prometheus/targets/gpu-nodes.yml
- targets: ["192.168.1.30:9101"]
  labels: { node: gpu-a, role: worker }
- targets: ["192.168.1.31:9101"]
  labels: { node: gpu-b, role: worker }
```

Add a line, and it is scraped within thirty seconds. No restart, no reload signal.
`targets/node-exporters.yml` is the same list on port 9100 for CPU, memory, disk and
network, served by the `node-exporter` service in the `dashboards` profile.

Set `node:` to the same string as `ATELIER_NODE_NAME` on that machine. Every GPU
metric then carries it, the Grafana legends read `gpu-a · GPU 3`, and the labels
line up with what the run pages say.

```bash
# control machine
docker compose -f docker-compose.api.yml --profile dashboards up -d
# each GPU node
docker compose -f docker-compose.worker.yml --profile dashboards up -d
```

Check it took, at `http://<control>/prometheus/targets` or:

```bash
curl -s http://192.168.1.30:9101/metrics | grep atelier_gpu_utilization
```

If that returns nothing, the worker is running but not seeing the GPUs — almost
always a container started without `--gpus all`, or `nvidia-ml-py` unable to reach
the driver.

## Materials, when they live on the GPU nodes

The API has no materials of its own, so it cannot answer "is this installed?" by
looking at its disk. Each worker publishes an inventory instead — which of the paths
the experiments ask for it actually has, and how big they are — and the API reads it.

The experiment page therefore says one of three things:

- **on the server** — this machine has it; the file tree and downloads work
- **on gpu-a, gpu-b** — a run will find it; it cannot be browsed from here
- **not installed** — no machine has it, and a run would fail

`materials_missing` on the experiment payload lists anything no node has, which is
the number to check before a lesson rather than during one.

### Scheduling follows the materials

With several nodes this matters: a node without the materials must not take the job.
Each worker checks before claiming, and if it cannot serve the run it puts it back on
the queue for a node that can, with a short delay so the queue is not spun on.

If *no* node has them, the run fails immediately with the reason, rather than sitting
queued forever while a student waits:

```
No GPU node has the materials this experiment needs: datasets/rag/wikipedia-simple.
Fetch them with scripts/offline/fetch-materials.py on a node, or run an experiment
that generates its own data.
```

This is also why partial installs are workable. Put the download-based experiments'
materials on one big node and leave the others bare: the twelve generated-data
experiments run anywhere, and the six that need downloads route themselves to the
node that has them.

The teacher's system page lists every node's inventory, so "which machine is missing
what" is one look rather than an ssh session.

## What the split costs

- **Latency on the run page.** Results appear when the job finishes rather than the
  instant the file is written. Logs still stream live, a few seconds behind.
- **Checkpoints stay on the node**, as above.
- **Two machines to keep up.** If the desktop is off, the class is down — the GPU
  node cannot serve the UI on its own.
- **Prometheus scrapes across the LAN**, one job per GPU node. See below.

The GPU strip in the UI keeps working: the worker publishes its NVML sample to Redis
every few seconds and the API draws from that when it has no GPUs of its own.
