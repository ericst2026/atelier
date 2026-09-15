#!/usr/bin/env bash
# Run on a machine WITH internet. Builds and saves every image Atelier needs.
#   ./scripts/offline/export-images.sh /media/usb/atelier-images
#
# The worker image is built once per CUDA variant, tagged atelier-worker:<variant>
# and saved to its own tarball, so a node only needs to load the one it uses.
# cu128 is also tagged atelier-worker:latest, which is what the compose files run.
#   CUDA_VARIANTS="cu126 cu128 cu130" ./scripts/offline/export-images.sh /media/usb/atelier-images
#
# "cpu" builds the image for worker nodes without GPUs: plain Ubuntu, CPU-only
# PyTorch, no CUDA. It is also tagged atelier-worker-cpu:latest, which is what
# docker-compose.worker-cpu.yml runs.
#   CUDA_VARIANTS="cu128 cpu" ./scripts/offline/export-images.sh /media/usb/atelier-images
#
# On Windows run it from Git Bash, not PowerShell: PowerShell 5.1 corrupts binary
# data piped between programs, and every tarball here is made through a pipe.
set -euo pipefail
OUT="${1:-./atelier-images}"
CUDA_VARIANTS="${CUDA_VARIANTS:-cu128}"
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

# variant -> base image, CUDA x.y ("-" for none), PyTorch. See docs/runtime.md.
# PyTorch 2.8 publishes no cu130 wheels; 2.9 is the first release that does.
variant_args() {
  case "$1" in
    cu126) echo "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04 12.6 ${TORCH_CU126:-2.8.0}" ;;
    cu128) echo "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 12.8 ${TORCH_CU128:-2.8.0}" ;;
    cu130) echo "nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04 13.0 ${TORCH_CU130:-2.9.1}" ;;
    cpu)   echo "ubuntu:24.04 - ${TORCH_CPU:-2.8.0}" ;;
    *) echo "unknown CUDA variant '$1' (expected cu126, cu128, cu130 or cpu)" >&2; return 1 ;;
  esac
}
for v in $CUDA_VARIANTS; do variant_args "$v" > /dev/null; done

echo "building atelier images…"
docker build -t atelier-api:latest ./backend
docker build -t atelier-web:latest ./frontend
for v in $CUDA_VARIANTS; do
  read -r image mm torch <<< "$(variant_args "$v")"
  echo "building atelier-worker:$v ($([[ $v == cpu ]] && echo "CPU only" || echo "CUDA $mm") · torch $torch)"
  docker build -f backend/Dockerfile.worker \
    --build-arg CUDA_IMAGE="$image" --build-arg CUDA_WHEEL="$v" \
    --build-arg CUDA_MM="$mm" --build-arg TORCH_VERSION="$torch" \
    -t "atelier-worker:$v" .
done
if [[ " $CUDA_VARIANTS " == *" cu128 "* ]]; then
  docker tag atelier-worker:cu128 atelier-worker:latest
fi
if [[ " $CUDA_VARIANTS " == *" cpu "* ]]; then
  docker tag atelier-worker:cpu atelier-worker-cpu:latest
fi

PULL=(
  postgres:16-alpine
  redis:7-alpine
  prom/prometheus:v2.54.1
  prom/node-exporter:v1.8.2
  grafana/grafana:11.2.0
)
for img in "${PULL[@]}"; do
  echo "pulling $img"
  docker pull "$img"
done

# The base images the builds started from, saved too so the images can be rebuilt
# offline. BuildKit keeps what it pulls in its build cache, not as images that
# docker save can see, so pull them by name; the layers are already local.
BASE=($(awk 'toupper($1) == "FROM" && $2 !~ /\$/ { print $2 }' backend/Dockerfile frontend/Dockerfile | sort -u))
for img in "${BASE[@]}"; do
  echo "pulling base $img"
  docker pull "$img"
done
for v in $CUDA_VARIANTS; do
  read -r image _ <<< "$(variant_args "$v")"
  echo "pulling base $image"
  docker pull "$image"
done

echo "saving to $OUT (this takes a while; the worker tarballs are several GB each)"
docker save atelier-api:latest atelier-web:latest | gzip > "$OUT/atelier-app.tar.gz"
for v in $CUDA_VARIANTS; do
  tags=("atelier-worker:$v")
  [[ "$v" == cu128 ]] && tags+=(atelier-worker:latest)
  [[ "$v" == cpu ]] && tags+=(atelier-worker-cpu:latest)
  echo "saving ${tags[*]}"
  docker save "${tags[@]}" | gzip > "$OUT/atelier-worker-$v.tar.gz"
done
docker save "${PULL[@]}" | gzip > "$OUT/atelier-infra.tar.gz"
# base images: the api/web ones together, each worker variant's on its own (the
# CUDA ones are several GB); only needed on a machine that rebuilds the images
echo "saving base ${BASE[*]}"
docker save "${BASE[@]}" | gzip > "$OUT/atelier-base.tar.gz"
for v in $CUDA_VARIANTS; do
  read -r image _ <<< "$(variant_args "$v")"
  echo "saving base $image"
  docker save "$image" | gzip > "$OUT/atelier-base-$v.tar.gz"
done
echo "done. Copy $OUT to the classroom node and run import-images.sh."
