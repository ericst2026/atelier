#!/usr/bin/env bash
# Run on a machine WITH internet. Builds and saves every image Atelier needs.
#   ./scripts/offline/export-images.sh /media/usb/atelier-images
set -euo pipefail
OUT="${1:-./atelier-images}"
mkdir -p "$OUT"
cd "$(dirname "$0")/../.."

echo "building atelier images…"
docker build -t atelier-api:latest ./backend
docker build -t atelier-worker:latest -f backend/Dockerfile.worker .
docker build -t atelier-web:latest ./frontend

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

echo "saving to $OUT (this takes a while; ~5 GB of tarballs, mostly the worker image)"
docker save atelier-api:latest atelier-web:latest | gzip > "$OUT/atelier-app.tar.gz"
docker save atelier-worker:latest | gzip > "$OUT/atelier-worker.tar.gz"
docker save "${PULL[@]}" | gzip > "$OUT/atelier-infra.tar.gz"
echo "done. Copy $OUT to the classroom node and run import-images.sh."
