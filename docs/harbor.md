# Serving the images from your own registry

With a Harbor at `harbor.local` there is no USB stick: the node pulls from the
registry like any other Docker host, and updating the class after a code change
is one push and one `docker compose pull`.

## Once, in Harbor

Create a project — Harbor does not create one on push. Called `atelier` here, so
images land at `harbor.local/atelier/atelier-api` and so on. A public project lets
the node pull without credentials; a private one needs `docker login harbor.local`
on the node as well.

## On your Windows machine

```powershell
docker login harbor.local
.\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier
```

It builds the three Atelier images, pulls the five third-party ones, retags
everything with the `harbor.local/atelier/` prefix and pushes. The worker image's
CUDA and PyTorch versions are switches on this script — `-TorchVersion 2.10.0`,
`-CudaWheel cu126` and so on; see [runtime.md](runtime.md). Expect 20–40
minutes the first time — the worker image carries PyTorch and CUDA. Afterwards,
when only the code changed:

```powershell
.\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier -Skip Third
```

If Harbor uses a self-signed certificate, either install its CA on the machine or
add it under Docker Desktop → Settings → Docker Engine:

```json
{ "insecure-registries": ["harbor.local"] }
```

## On the classroom node

```bash
# .env
ATELIER_REGISTRY=harbor.local/atelier/
ATELIER_TAG=latest
```

```bash
docker compose pull
docker compose up -d --no-build
```

Every service in `docker-compose.yml` — including postgres, redis, Prometheus and
Grafana — takes the prefix, so nothing reaches Docker Hub. Leave `ATELIER_REGISTRY`
empty and the same file builds locally instead, which is what the laptop stack does.

If the node distrusts the certificate, add the same `insecure-registries` entry to
`/etc/docker/daemon.json` and `systemctl restart docker`.

## Checking before you install

```powershell
.\scripts\offline\check-node.ps1 -Node you@gpu-node -Registry harbor.local/atelier
```

Over SSH it reports the GPUs the driver sees, the Docker and Compose versions,
whether a container can reach the GPUs, free space under `/srv`, and whether the
registry answers. Anything red is a problem to fix before the first lesson.

## Updating mid-term

Push a new tag rather than overwriting `latest`, so a lesson in progress is not
disturbed by a pull:

```powershell
.\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier -Tag 2026-04-15 -Skip Third
```

```bash
ATELIER_TAG=2026-04-15 docker compose pull && docker compose up -d --no-build
```

Experiments are not in the images — they are mounted from disk — so adding or
editing an experiment needs no push at all. `rsync` the folder and the registry
picks it up within five seconds. Only backend, frontend or shared-library changes
need a rebuild.
