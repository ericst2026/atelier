# Deploying the workers to many nodes

With more than one or two worker nodes — some with GPUs, some without, not all with
the same cards — starting each worker by hand stops being practical. There are two
ways to do it from one place:

| | SSH deploy | Kubernetes |
|---|---|---|
| Nodes need | SSH and Docker | to be in a cluster, plus NVIDIA's device plugin with GPU feature discovery on GPU nodes |
| GPU or CPU image | found per node over SSH | from each node's labels |
| Run it with | `scripts/deploy/deploy-workers.sh up` | `scripts/deploy/k8s-workers.sh apply` |
| Good for | a classroom, a handful of machines | a school that already runs Kubernetes |

Both run only the workers. The control plane — API, web, Postgres, Redis, dashboards —
stays on the control machine under `docker-compose.api.yml`, and both deployments
share the same layout on every node:

```
/srv/atelier/app           compose files and the worker's .env (SSH deploy only)
/srv/atelier/experiments   identical everywhere; the SSH deploy keeps it that way
/srv/atelier/materials     this node's copy of the materials, from NFS
/srv/atelier/data          this node's run directories
```

## Materials from NFS

Keep one copy of the materials on an NFS share — filled once with
`scripts/offline/fetch-materials.py`, see [offline-install.md](offline-install.md) —
and name it in the control machine's `.env`:

```bash
ATELIER_MATERIALS_NFS=192.168.1.50:/export/atelier/materials
```

Both deployments then copy the materials into the node's `/srv/atelier/materials`
**before its worker starts**, and the worker reads that local copy: an 11 GB model
cache loads at disk speed rather than network speed, and a class does not stall when
the NFS server does.

- **A new node gets everything** before its worker first runs.
- **An existing node gets only what changed** on the share — new or newer files —
  and nothing already on the node is deleted. Add a model to the share and the next
  deploy brings it everywhere.
- **The copy is checked for room first.** A node that cannot hold the materials says
  how many MB they need and how many it has, and its worker is not started.
- **A copy that fails leaves the worker unstarted**, so a node never runs an
  experiment against half its materials.

Export the share read-only to the nodes' addresses. The files must be readable by
the user doing the copy — world-readable is simplest.

Before either one, the control plane must be up and reachable from the nodes on the
HTTP, Postgres and Redis ports, with `ATELIER_WORKER_TOKEN`, `ATELIER_DB_PASSWORD` and
`ATELIER_REDIS_PASSWORD` set in its `.env`. Both scripts read the secrets from that
file, so nothing is typed twice. See [topologies.md](topologies.md).

## SSH deploy

### Once

On every node: Docker with Compose 2.24 or newer, a user in the `docker` group who can
write to `/srv/atelier`, and for GPU nodes the NVIDIA driver and container toolkit.

```bash
sudo mkdir -p /srv/atelier && sudo chown atelier /srv/atelier
```

On the control machine: an SSH key the nodes accept without a prompt, and the node
list.

```bash
ssh-copy-id atelier@192.168.170.101          # for each node
cp deploy/workers.example deploy/workers  # then list your nodes
```

```
atelier@192.168.170.101 name=gpu-a
atelier@192.168.170.101 name=gpu-b
atelier@192.168.170.101 name=cpu-a slots=2
```

A line needs only the SSH destination. The name defaults to the node's hostname, and
whether it is a GPU or a CPU node is found when it is deployed: GPUs visible to
`nvidia-smi` *and* the nvidia runtime in Docker make a GPU node; anything else is a
CPU node, with a warning if it has GPUs Docker cannot use.

### Every time

```bash
export CONTROL_HOST=192.168.1.20      # the address the nodes reach this machine at
scripts/deploy/deploy-workers.sh check
scripts/deploy/deploy-workers.sh up
```

`check` goes through each node without changing it: SSH, Docker without sudo, the
Compose version, a writable `/srv/atelier`, the control plane's ports, and the NFS
server's port 2049 (or, without `ATELIER_MATERIALS_NFS`, whether materials are
present).

`up` deploys every node, or only the ones you name (`up gpu-b cpu-a`). For each it
copies the compose files and `experiments/`, writes a worker-only `.env`, gets the
image, copies the materials from NFS, starts the worker with the GPU or the CPU
compose file, and waits for the worker's `worker up on …` line — which says how many
GPUs it found. Last, it rewrites `deploy/prometheus/targets/gpu-nodes.yml` with every
node's address, so the dashboards follow.

**Materials.** The copy runs in a throwaway container of the worker image with the
share mounted by Docker itself (a `local` volume of type `nfs`), so the node's SSH
user needs no sudo and no NFS client; the files are written as that user.
`MATERIALS_NFS_OPTS` sets the mount options (default `nfsvers=4.1`).
`materials [node…]` copies without deploying or restarting anything — for a model
added to the share mid-term — and `MATERIALS_SKIP=1` leaves them out of an `up`.

A node that fails does not stop the others; the summary at the end lists it.

**Images.** With `ATELIER_REGISTRY` set in `.env` (Harbor, see [harbor.md](harbor.md))
each node pulls. Without one, point `IMAGES_DIR` at the tarballs from
`scripts/offline/export-images.sh` and the script streams the right one to each node,
once — it remembers the checksum:

```bash
CUDA_VARIANTS="cu128 cpu" scripts/offline/export-images.sh /media/usb/atelier-images
IMAGES_DIR=/media/usb/atelier-images scripts/deploy/deploy-workers.sh up
```

**Updating.** Run `up` again. `experiments/` is updated in place and a running worker
picks it up within a minute. The worker container is recreated only when its image or
settings changed — and a recreated worker fails the runs it had in progress, so update
between lessons. `sync` copies `experiments/` alone and never restarts anything.

**Other commands.** `status` shows each node's containers and the GPUs its worker
found; `logs gpu-a` follows one worker; `down` stops workers; `targets` rewrites the
Prometheus list without deploying. `DASHBOARDS=1` also runs node-exporter on each node
and writes `node-exporters.yml`.

On Windows, `scripts\deploy\deploy-workers.cmd` runs the same script in Git Bash.

## Kubernetes

### Once

Written for an on-premises, offline cluster — kubeadm or similar, Kubernetes 1.34
tested against the manifests' APIs — and just as fine on k3s. Everything it uses has
been stable for years: `apps/v1` DaemonSets, runtime classes, hostPath, hostPort, and
the NVIDIA device plugin's `nvidia.com/gpu` resource.

- **kubectl within one minor version of the cluster** — 1.33 to 1.35 for 1.34. The
  script warns when it is further apart.
- **NVIDIA's device plugin with GPU feature discovery on the GPU nodes.** GFD labels
  each one `nvidia.com/gpu.count=<n>`, and that label is all the script uses to tell
  GPU nodes from CPU ones. Two ways to have it offline:
  - the GPU Operator, which also brings the `nvidia` RuntimeClass, the container
    toolkit and optionally the driver. Mirror its images and Helm chart into Harbor
    first (NVIDIA documents the air-gapped install); with the driver and toolkit
    already on the nodes, install it with `driver.enabled=false` and
    `toolkit.enabled=false` and there is much less to mirror.
  - the standalone `nvidia-device-plugin` Helm chart with `gfd.enabled=true` — two
    images — on nodes where you installed the driver and `nvidia-container-toolkit`
    yourself. There is no RuntimeClass then; the script notices and leaves
    `runtimeClassName` out, so containerd's default runtime must be the nvidia one:
    `sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default`.
- **Images from Harbor.** Point containerd on each node at it (`config_path` in
  `/etc/containerd/config.toml`, and a `hosts.toml` under
  `/etc/containerd/certs.d/harbor.local/` — with the CA if the certificate is
  self-signed). If the project is private, create a pull secret and name it:
  ```bash
  kubectl create namespace atelier
  kubectl -n atelier create secret docker-registry harbor \
    --docker-server=harbor.local --docker-username=… --docker-password=…
  PULL_SECRET=harbor scripts/deploy/k8s-workers.sh apply
  ```
  Without a registry, import the tarballs into containerd's Kubernetes namespace on
  each node: `gunzip -c atelier-worker-cu128.tar.gz | sudo ctr -n k8s.io images import -`.
- **hostPort support in the CNI.** Metrics are published on each node's port 9101.
  Calico and Flannel support hostPort through the portmap plugin as installed; Cilium
  needs its hostPort feature (or kube-proxy replacement) enabled.
- **Pod Security.** The namespace is labelled `privileged`, because the workers need
  hostPath, hostPort and, on GPU nodes, the host IPC namespace. A cluster-wide
  admission policy (Kyverno, Gatekeeper) that forbids those needs an exception for
  the `atelier` namespace.
- **`/srv/atelier/experiments` on every node, identical** — the pods mount the node's
  folder. The SSH deploy's `sync` does exactly this, from the same `deploy/workers`
  list, and needs only SSH.
- **The NFS client on every node** (`nfs-common` on Ubuntu), when
  `ATELIER_MATERIALS_NFS` is set: the kubelet mounts the share for the copy.
- **Keep workers off the control plane** if its machines are schedulable:
  `kubectl label node <name> atelier/worker=false`.

### Every time

```bash
export CONTROL_HOST=192.168.1.20
scripts/deploy/deploy-workers.sh sync     # experiments/ to every node
scripts/deploy/k8s-workers.sh apply
```

`apply` reads `.env` and asks the cluster which GPU counts its nodes have, writes an
overlay to `deploy/k8s/local/` (ignored by git — it holds the secrets), applies it,
waits for the rollout and rewrites the Prometheus targets. What it creates:

- **`atelier-worker-cpu`**, a DaemonSet on every node without `nvidia.com/gpu.count`, running
  `atelier-worker-cpu` with `ATELIER_GPU_COUNT=0`.
- **`atelier-worker-gpu-<n>`**, one DaemonSet per GPU count, on the nodes with that
  many GPUs, asking the device plugin for all `n` of them. A pod asks for a fixed
  number of GPUs, which is why an 8-GPU node and a 4-GPU node need separate
  DaemonSets. Add a node with a new count and run `apply` again; a count no node has
  any more has its DaemonSet removed.

With `ATELIER_MATERIALS_NFS` set, every worker pod also has an init container,
`materials`, that copies from the share into the node's `/srv/atelier/materials`
before the worker container starts — the same script as the SSH deploy. So a node
that joins the cluster has its materials before its worker takes a job, and every
pod restart picks up what was added to the share. A pod stuck in `Init` is that copy:
`kubectl -n atelier logs <pod> -c materials` shows its progress or why it failed.

Each worker is named after its node, reports its metrics on the node's port 9101, and
reaches the control plane at `CONTROL_HOST`, exactly as under Compose. An image or
settings change rolls through the nodes one at a time, and each restarted worker fails
the runs it had in progress.

`status` lists the nodes with their GPU labels, the worker pods and each worker's
`worker up on …` line. `restart <node>` restarts one worker, `targets` rewrites the
Prometheus list, `delete` removes the workers, config and secrets. `render` writes
the overlay without applying it, if you would rather read it first or apply it from
elsewhere (`kubectl apply -k deploy/k8s/local`).

Settings that differ per node in the SSH deploy — `slots=`, `gpus=` — are one value
for the whole cluster here: `ATELIER_CPU_SLOTS` from `.env`, and all of a GPU node's
GPUs.
