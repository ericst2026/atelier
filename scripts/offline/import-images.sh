#!/usr/bin/env bash
# Run on the offline classroom node.
#   ./scripts/offline/import-images.sh /media/usb/atelier-images
set -euo pipefail
IN="${1:-./atelier-images}"
for f in "$IN"/*.tar.gz; do
  echo "loading $f"
  gunzip -c "$f" | docker load
done
docker image ls | grep -E "atelier|postgres|redis|grafana|prometheus|node-exporter" || true
echo
echo "Images loaded. Because the images are already present, start with:"
echo "  docker compose up -d --no-build"
