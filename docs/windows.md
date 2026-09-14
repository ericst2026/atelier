# Running from Windows

Atelier's home is the Linux GPU node. Windows comes into it in three ways, and
they need different setups.

| What you want | Use |
|---|---|
| Try the platform on your laptop | `docker-compose.windows.yml` (no GPU, five minutes) |
| Develop the code on your laptop | WSL2, or `scripts\dev.ps1` for a native Python/Node run |
| Deploy to the classroom node | SSH from Windows Terminal; the node runs Linux |
| Just teach a lesson | Nothing — a browser is all a teacher or student needs |

Students and teachers never install anything. They open `http://<server>/` in
Edge or Chrome, and every computation happens on the node.

## 1. Trying it on a Windows laptop

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with
the WSL2 backend, then in PowerShell:

```powershell
cd path\to\atelier
docker compose -f docker-compose.windows.yml up -d --build
```

Open `http://localhost:8080` and sign in as `teacher` / `teacher`.

This stack is the real thing with the GPU parts removed: SQLite instead of
Postgres, no Grafana, no CUDA. **Tokenizer** runs end to end — all four steps and
the grader — because it ships its own corpus and is pure Python. The four GPU
experiments appear on the home page and their steps will start and then fail for
want of CUDA, which is what you want to see: the failure arrives in the browser
with the traceback, exactly as a student would get it.

```powershell
docker compose -f docker-compose.windows.yml logs -f worker    # watch the scheduler
docker compose -f docker-compose.windows.yml down -v           # remove it all
```

The whole repository is mounted, so editing an `experiment.yaml` or a step script
shows up within five seconds without rebuilding anything.

## 2. Developing on Windows

**WSL2 is the smoother path.** Install Ubuntu from the Microsoft Store, clone the
repository *inside* the WSL filesystem (`~/atelier`, not `/mnt/c/...` — the Windows
drive mount is slow enough to be painful with `node_modules`), and everything in
this repository works unchanged, including `./scripts/dev.sh`. Edit from Windows
with VS Code's WSL extension.

**Native Windows** works too, without bash. You need Python 3.11+, Node 20+, and a
Redis — the simplest being `docker run -d -p 6379:6379 redis:7-alpine`. Then:

```powershell
.\scripts\dev.ps1
```

It creates `.venv`, installs both dependency sets on the first run, and starts the
API, the worker and the Vite dev server: frontend on `http://localhost:5173`, API
docs on `http://localhost:8000/api/docs`.

Two things behave differently on native Windows. The worker kills a run's process
tree with `os.killpg`, which is POSIX-only, so **Stop** will not work — under WSL2
or Docker it does. And a few experiment libraries expect a Unix environment. For
anything beyond frontend work, prefer WSL2.

### GPU in WSL2

If your laptop has an NVIDIA card, WSL2 can reach it: install the normal Windows
NVIDIA driver (not a driver inside WSL), then the container toolkit in Ubuntu.
Check with `nvidia-smi` inside WSL and `docker run --rm --gpus all
nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`. With that working, the production
`docker-compose.yml` runs on your laptop too — set `ATELIER_GPU_COUNT=1`. It is
enough to shake out the SFT and Reasoning experiments on a small model before you
touch the classroom node.

## 3. Deploying to the node from Windows

The node runs Linux; Windows is only your terminal.

```powershell
ssh you@node
# on the node, per docs/offline-install.md
```

To copy the repository and the materials across, use `scp` or `robocopy` to a
share. If you prepare the offline materials on your Windows machine, run the
fetcher inside WSL — `huggingface_hub` and `datasets` behave better there, and the
folder you produce goes straight to `/srv/atelier/materials`.

## Things that catch people out

**Line endings.** Git on Windows may check out `.sh` files with CRLF, and then
`docker compose up` fails with `exec /bin/bash^M: no such file or directory`. The
repository ships a `.gitattributes` that forces LF on shell scripts; if you copied
the files by hand instead of cloning, run `dos2unix scripts/*.sh scripts/offline/*.sh`.

**Port 80.** The production compose file publishes port 80, which IIS or another
service may already hold on Windows. The laptop stack uses 8080 for that reason;
change `ATELIER_HTTP_PORT` if even that is taken.

**Paths in `.env`.** Use forward slashes (`C:/atelier/materials`) or WSL paths.
Backslashes get eaten by Docker's variable handling.

**Docker Desktop memory.** The default WSL2 VM gets a fraction of your RAM. For
anything real, create `%UserProfile%\.wslconfig` with `[wsl2]` and
`memory=16GB`, then `wsl --shutdown`.

## "running scripts is disabled on this system"

PowerShell refuses to run `.ps1` files by default, and separately marks anything that
came out of a downloaded zip. Either can produce:

```
.\scripts\dev.ps1 cannot be loaded because running scripts is disabled on this system.
```

Three ways past it, in the order worth trying.

**Use the `.cmd` wrapper.** Every PowerShell script here has one next to it, and it
changes nothing on the machine:

```powershell
.\scripts\dev.cmd
.\scripts\offline\push-to-harbor.cmd -Registry harbor.local -Project atelier
```

**Or bypass for one command:**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1
```

**Or allow local scripts for your user**, which is a real change but a modest one —
it still refuses unsigned scripts from the internet:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If that alone does not fix it, the files still carry the mark Windows puts on
anything extracted from a downloaded archive. Clear it:

```powershell
.\scripts\unblock.cmd
# or
Get-ChildItem -Recurse -Include *.ps1 | Unblock-File
```

Do not use `Set-ExecutionPolicy Unrestricted` machine-wide to get past this. The
wrapper does the same job for these scripts and nothing else.

A path with a space in it — `D:\magnolia_work\experiment project\atelier` — is fine
for all of these; the wrappers quote correctly. It does confuse some Python tooling
later on, so if something odd turns up during a build, that is the first thing worth
ruling out.
