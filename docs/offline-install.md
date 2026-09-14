# Installing on an offline node

What has to cross the air gap depends on which track you teach.

**The main track downloads nothing.** Its corpus is generated from a seed on the
node and its models are trained from scratch, so once the images are in place
there is nothing else to copy — no weights, no datasets, no licences.

**The advanced track** (the five experiments that use public models and data)
needs the materials below.

Either way the container images have to get there. If you run your own registry,
push to it instead of copying tarballs — see [harbor.md](harbor.md), and
`scripts/offline/push-to-harbor.ps1` if you work from Windows.

## On a machine with internet

```bash
# 1. images (~5 GB compressed on disk; the PyTorch-based worker image is most of it)
./scripts/offline/export-images.sh /media/usb/atelier-images

# 2. materials (~11 GB with the list as shipped; trim scripts/offline/materials.yaml first)
pip install huggingface_hub datasets pyyaml
python scripts/offline/fetch-materials.py --out /media/usb/materials
# just one thing:
python scripts/offline/fetch-materials.py --out /media/usb/materials --names gsm8k Qwen2.5-0.5B-Instruct
```

`fetch-materials.py` writes the layout the experiments expect:

```
materials/
  models/<name>/                       HF snapshot (config.json, tokenizer.*, *.safetensors)
  datasets/<group>/<name>/<split>.jsonl
  hf-cache/                            HF_HOME for the runs; stays empty offline
```

Groups in use: `tokenizer/{pools,wikitext-103-raw,code,multilingual}`,
`pretrain/{tinystories,wikitext-103}`, `sft/{alpaca-cleaned,dolly-15k,smoltalk}`,
`rl/{gsm8k,hh-rlhf}`, `reasoning/math-500`. A missing material is not fatal: the
Materials tab marks it "not installed" and the steps that need it say so clearly.

## The full download list

| What | Where it comes from | Roughly |
|---|---|---|
| Worker/runtime image (PyTorch 2.4.1 + CUDA 12.1 + transformers, trl, peft) | built by `export-images.sh` | 4 GB compressed |
| API image (python:3.12-slim + FastAPI) and web image (nginx + the built React app) | built by `export-images.sh` | 400 MB |
| postgres:16-alpine, redis:7-alpine, prometheus, node-exporter, grafana | pulled by `export-images.sh` | 700 MB |
| `gpt2` (the tokenizer for pretraining) | `fetch-materials.py` | 0.6 GB |
| `Qwen2.5-0.5B` and `Qwen2.5-0.5B-Instruct` | `fetch-materials.py` | 2 GB |
| `Qwen2.5-1.5B` and `Qwen2.5-1.5B-Instruct` (larger runs and the judge) | `fetch-materials.py` | 6 GB |
| TinyStories, WikiText-103, code and multilingual slices | `fetch-materials.py` | 1.5 GB |
| Alpaca, Dolly, SmolTalk, GSM8K, HH-RLHF, MATH-500 | `fetch-materials.py` | 250 MB |
| NVIDIA driver and container toolkit packages, if the node has no package mirror | your distribution | 500 MB |

Both scripts build as well as download: the web image runs `npm install` and the
worker image runs `pip install`, so the npm registry and PyPI are reached on the
online machine and never on the node. That is the reason to export images rather
than copying this repository and building on the node — a build there would fail
at the first `npm install`.

Trimming for a shorter course: the two 1.5B models are only used by the SFT judge,
the larger pretraining runs and the optional stronger solver in Reasoning, so
dropping them saves 6 GB. Cutting `limit:` on TinyStories and WikiText in
`materials.yaml` saves most of the rest. The Tokenizer experiment needs no
downloads at all.

## On the classroom node

```bash
sudo mkdir -p /srv/atelier/{materials,data}
sudo rsync -a /media/usb/materials/ /srv/atelier/materials/
git clone <this> /srv/atelier/app && cd /srv/atelier/app   # or copy the folder
./scripts/offline/import-images.sh /media/usb/atelier-images
cp .env.example .env && $EDITOR .env
docker compose up -d --no-build
```

Requirements on the node: Docker with Compose v2, the NVIDIA driver, and the
NVIDIA container toolkit (`nvidia-ctk runtime configure --runtime=docker`). Check
with `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.

## Verifying

```bash
curl -s localhost/api/system/health           # {"ok":true,"redis":true,"worker":true}
curl -s localhost/api/system/gpus | head -c 200
docker compose logs worker | tail
```

Then sign in, open **Tokenizer**, and run step 1 — it needs no materials at all, so
it proves the whole chain (API → Redis → worker → run directory → websocket) in
about ten seconds.

## Everything stays local

The worker sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `HF_DATASETS_OFFLINE`
for every run, so a forgotten `from_pretrained("gpt2")` fails with a clear error
instead of hanging on a download. Grafana's update check and news feed are off. The
frontend bundles its fonts (system stack), icons and the Monaco editor — no CDN.

## Backups

`docker compose exec postgres pg_dump -U atelier atelier > atelier.sql` plus a copy
of `/srv/atelier/data` (workspaces, submissions, run artifacts) is a full backup.
Run directories are the bulky part; deleting old ones from Teacher → Runs is safe.
