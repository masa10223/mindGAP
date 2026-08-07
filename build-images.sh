#!/usr/bin/env bash
# mindgap スリムイメージをビルド (ELA_analysis 専用)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MIN_GB="${MIN_ROOT_GB:-6}"
avail_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
if [ "$avail_gb" -lt "$MIN_GB" ]; then
  echo "error: root (/) free ${avail_gb}GB — need ~${MIN_GB}GB" >&2
  echo "  docker rmi mindgap:latest && docker builder prune -f" >&2
  exit 1
fi

echo "=== build mindgap:slim ==="
docker build -f docker/Dockerfile.mindgap -t mindgap:slim .
docker tag mindgap:slim mindgap:latest
docker images mindgap --format '{{.Repository}}:{{.Tag}}\t{{.Size}}'
echo
echo "Next: ./setup_mindgap.sh"
