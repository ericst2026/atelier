# The runtime: CUDA, PyTorch and Python

Nothing here is baked into the image. Four build arguments decide it, and they have
to agree with one another.

| Argument | Default | What it is |
|---|---|---|
| `CUDA_IMAGE` | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` | the base image |
| `CUDA_WHEEL` | `cu128` | which PyTorch wheel index to install from |
| `CUDA_MM` | `12.8` | the same version, used to check what actually got installed |
| `TORCH_VERSION` | `2.8.0` | the PyTorch release |
| `PYTHON_VERSION` | `3.12` | must be what the base image ships |

## Setting them

In `.env`, for `docker compose build`:

```bash
ATELIER_CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ATELIER_CUDA_WHEEL=cu128
ATELIER_CUDA_MM=12.8
ATELIER_TORCH_VERSION=2.8.0
ATELIER_PYTHON_VERSION=3.12
```

From the Harbor script, which is where you will usually do it:

```powershell
.\scripts\offline\push-to-harbor.ps1 -Registry harbor.local -Project atelier -TorchVersion 2.10.0
```

Or directly:

```bash
docker build -f backend/Dockerfile.worker \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 \
  --build-arg CUDA_WHEEL=cu126 --build-arg CUDA_MM=12.6 \
  -t atelier-worker .
```

## Combinations that work

| Cards in the class | Base image | Wheel | PyTorch |
|---|---|---|---|
| Ampere and newer — **A6000**, A100, L40S, H100, RTX 40/50 | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` | `cu128` | 2.7 – current |
| Anything back to Maxwell — GTX 10-series, V100, Titan V | `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` | `cu126` | 2.6 – current |
| Newest toolkit, Turing and up only | `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` | `cu130` | recent releases only |

Ubuntu 24.04 ships Python 3.12. If you move the base image back to a 22.04 tag, set
`PYTHON_VERSION=3.10` to match what that image carries, or install a newer Python
yourself.

## What CUDA 12.8 does and does not drop

Worth being precise, because the two get confused.

**The CUDA 12.8 toolkit still supports Maxwell, Pascal and Volta.** NVIDIA has marked
them feature-complete and deprecated, and CUDA 13.0 is where offline compilation for
them goes away — but 12.x still builds for them.

**PyTorch's prebuilt `cu128` wheels do not include them.** That is a packaging
decision about binary size, not a CUDA limitation. From release 2.8 the `cu128`
wheels are compiled for compute capabilities 7.5, 8.0, 8.6, 9.0, 10.0 and 12.0,
while the `cu126` wheels still cover 5.0, 6.0, 7.0, 7.5, 8.0, 8.6 and 9.0.

So a Pascal card is not excluded by CUDA 12.8. It is excluded by the particular wheel,
and `CUDA_WHEEL=cu126` gets it back. Building PyTorch from source against 12.8 with
`TORCH_CUDA_ARCH_LIST` set would also work, and is not worth it for a classroom.

Your A6000s are compute capability 8.6, which is in every row of that table. The
defaults are right for them, and this page only matters if you add a different card
later.

The build prints what it compiled for, so you never have to guess:

```
python 3.12.3 · torch 2.8.0 · cuda 12.8
compiled for: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

## Choosing a PyTorch version

Every release from 2.7 onwards publishes CUDA 12.8 wheels, so 2.8 and 2.10 both sit
on the same base image. 2.8 is the default because it has had the longest run in the
wild. Newer releases exist; change `TORCH_VERSION` and let the image's own check
catch a mismatch.

## The check at the end of the build

The Dockerfile finishes by asserting that `torch.version.cuda` starts with `CUDA_MM`,
then printing the architecture list. This is deliberate. A wheel built against a
different CUDA minor version installs perfectly happily and then fails on the first
kernel launch — which, without this check, means half an hour into a lesson rather
than during a build on your own machine.

## Driver

CUDA 12.8 needs driver 525.60.13 or newer through minor-version compatibility;
NVIDIA ships the 570 series alongside it. Check before installing:

```bash
nvidia-smi --query-gpu=driver_version,name --format=csv
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi -L
```

`scripts/offline/check-node.ps1` runs both over SSH with the rest of the pre-flight
checks.

## Why torch is not in requirements.txt

`experiments/requirements.txt` deliberately omits torch. Installing it from PyPI gets
whichever CUDA build PyPI defaults to that month, which may not match your base
image. The Dockerfile installs it first, alone, from the index you chose — which also
means editing the other requirements does not re-download two gigabytes.

## What the course actually needs

The main track imports only `torch` and `numpy`. The transformer, both training
loops, GRPO, DPO, LoRA, quantisation, the KV cache and the contrastive embedder are
all in `experiments/_lib/atelier_mini/`, written out rather than imported.

`transformers`, `datasets`, `peft` and `trl` are needed only by the six experiments
that use downloaded models — data curation, retrieval on Wikipedia, evaluation,
safety, interpretability — and by the advanced track. Teaching the generated-data
course alone, those four lines can come out and the image gets considerably smaller.

## Two images, not three

`backend/Dockerfile` builds the API, and with `--build-arg WITH_EXPERIMENTS=1` it
also serves as the worker on a machine with no GPU — that one argument adds numpy and
tqdm, which is all the difference there was. The laptop stack in
`docker-compose.windows.yml` runs both services from it.

`backend/Dockerfile.worker` is the GPU image: CUDA base, torch from the wheel index,
the full experiment requirements.

There used to be a third, `Dockerfile.cpu`, which differed from the API image by
three lines. It is gone.
