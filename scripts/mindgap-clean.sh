#!/usr/bin/env bash
# mindGAP ディスク回収ヘルパー
#
#   ./scripts/mindgap-clean.sh cache          # pip-cache のみ削除 (~数 GB)
#   ./scripts/mindgap-clean.sh venv           # 旧 venv 削除 (次回 setup で再作成)
#   ./scripts/mindgap-clean.sh all            # cache + venv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cmd="${1:-help}"
shift || true

clean_cache() {
  if [ -d pip-cache ]; then
    local before
    before="$(du -sh pip-cache 2>/dev/null | cut -f1 || echo '?')"
    rm -rf pip-cache/*
    mkdir -p pip-cache
    echo "pip-cache cleared (was ${before})"
  else
    echo "pip-cache not found (nothing to do)"
  fi
}

clean_venv() {
  if [ -d venv/jaxenv ]; then
    local before
    before="$(du -sh venv/jaxenv 2>/dev/null | cut -f1 || echo '?')"
    rm -rf venv/jaxenv
    echo "venv/jaxenv removed (was ${before})"
    echo "Recreate with: FORCE_BOOTSTRAP=1 ./setup_mindgap.sh"
  else
    echo "venv/jaxenv not found (nothing to do)"
  fi
}

case "$cmd" in
  cache) clean_cache ;;
  venv)  clean_venv ;;
  all)
    clean_cache
    clean_venv
    ;;
  help|*)
    cat <<EOF
mindGAP disk cleanup

  cache   remove ./pip-cache (safe; packages stay in venv)
  venv    remove ./venv/jaxenv (rebuilt on next setup)
  all     cache + venv

Typical reclaim after switching to slim venv mode:
  ./scripts/mindgap-clean.sh all
  FORCE_BOOTSTRAP=1 ./setup_mindgap.sh
EOF
    ;;
esac
