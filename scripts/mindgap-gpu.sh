#!/usr/bin/env bash
# mindGAP GPU / コンテナ復旧ヘルパー
#
# NVML Unknown Error などで Jupyter/GPU が落ちたとき:
#   ./scripts/mindgap-gpu.sh check
#   ./scripts/mindgap-gpu.sh recover
#   ./scripts/mindgap-gpu.sh watch          # check + 必要なら recover (cron 向け)
#   ./scripts/mindgap-gpu.sh install-cron   # 5分おき監視を crontab に登録
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=mindgap-env.sh
source "$(cd "$(dirname "$0")" && pwd)/mindgap-env.sh"

TIMEOUT="${TIMEOUT:-15}"
WATCH_LOG="${WATCH_LOG:-$ROOT/mindgap-gpu-watch.log}"
CRON_EVERY_MIN="${CRON_EVERY_MIN:-5}"

log() {
  echo "[$(date -Iseconds)] [$NAME] $*"
}

log_file() {
  log "$*" | tee -a "$WATCH_LOG"
}

nvidia_smi_ok() {
  local where="$1"
  local cmd="$2"
  local out rc=0

  out="$(timeout "$TIMEOUT" bash -lc "$cmd" 2>&1)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "  NG  $where (exit=$rc)" >&2
    [ -n "$out" ] && echo "       ${out//$'\n'/ }" >&2
    return 1
  fi
  if echo "$out" | grep -qiE 'nvml.*error|unknown error|driver/library version mismatch|requires reset|not found|insufficient'; then
    echo "  NG  $where (NVML/driver error)" >&2
    echo "       ${out//$'\n'/ }" >&2
    return 1
  fi
  echo "  OK  $where"
  return 0
}

container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$NAME"
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -qx "$NAME"
}

jupyter_running() {
  docker exec "$NAME" bash -lc 'pgrep -f "[j]upyter-lab"' >/dev/null 2>&1
}

jax_ok() {
  docker exec "$NAME" bash -lc \
    'source /work/venv/jaxenv/bin/activate && python -c "import jax; assert len(jax.devices())>0"' \
    >/dev/null 2>&1
}

start_jupyter() {
  docker exec "$NAME" bash -lc '
    set -e
    export JUPYTER_DATA_DIR=/work/.jupyter
    source /work/venv/jaxenv/bin/activate
    python -m ipykernel install --name jaxenv --display-name "Python (jaxenv)" >/dev/null 2>&1 || true
    if pgrep -f "[j]upyter-lab" >/dev/null 2>&1; then
      pkill -f "[j]upyter-lab" || true
      sleep 1
    fi
    nohup jupyter lab \
      --port='"$PORT"' --ip=0.0.0.0 --no-browser --allow-root \
      --notebook-dir=/work \
      > /work/jupyter.log 2>&1 &
  '
}

show_jupyter_url() {
  docker exec "$NAME" bash -lc "python - <<'PY'
import re, time
p = '/work/jupyter.log'
pat = re.compile(r'http://127\\.0\\.0\\.1:\\d+(?:/lab)?\\?token=\\S+')
for _ in range(30):
    try:
        txt = open(p).read()
    except FileNotFoundError:
        time.sleep(1); continue
    m = pat.findall(txt)
    if m:
        print(m[-1].replace('127.0.0.1', 'localhost')); break
    time.sleep(1)
else:
    print('token URL not found yet')
PY"
}

run_check() {
  local host_ok=0 container_ok=0 jupyter_ok=0 jax_dev_ok=0

  echo "GPU health check (timeout=${TIMEOUT}s)"
  nvidia_smi_ok "host nvidia-smi" "nvidia-smi -L" && host_ok=1 || true

  if container_running; then
    nvidia_smi_ok "container nvidia-smi" "docker exec $NAME nvidia-smi -L" && container_ok=1 || true
    if jupyter_running; then
      echo "  OK  jupyter-lab"
      jupyter_ok=1
    else
      echo "  NG  jupyter-lab (not running)" >&2
    fi
    if jax_ok; then
      echo "  OK  jax devices"
      jax_dev_ok=1
    else
      echo "  NG  jax devices" >&2
    fi
  else
    echo "  NG  container '$NAME' not running" >&2
  fi

  if [ "$host_ok" = 1 ] && [ "$container_ok" = 1 ] && [ "$jupyter_ok" = 1 ] && [ "$jax_dev_ok" = 1 ]; then
    return 0
  fi
  return 1
}

recover_permissions() {
  docker exec -u 0 "$NAME" bash -lc '
    mkdir -p /work/notebooks/csvs_for_paper /work/notebooks/figs_for_paper
    chmod -R u+rwX,g+rwX /work/notebooks 2>/dev/null || true
  ' >/dev/null 2>&1 || true
}

recover_jupyter_only() {
  log_file "recover: restart jupyter-lab"
  start_jupyter
  sleep 3
}

recover_container() {
  log_file "recover: docker restart $NAME (NVML/GPU reset)"
  if container_exists; then
    docker restart "$NAME" >/dev/null
  else
    log_file "recover: container missing -> setup_mindgap.sh"
    (cd "$ROOT" && ./setup_mindgap.sh) >>"$WATCH_LOG" 2>&1
    return 0
  fi
  sleep 5
  recover_permissions
  recover_jupyter_only
}

recover_full() {
  log_file "recover: recreate container via setup_mindgap.sh"
  (cd "$ROOT" && ./setup_mindgap.sh) >>"$WATCH_LOG" 2>&1
}

run_recover() {
  local host_ok=0 container_ok=0

  mkdir -p "$(dirname "$WATCH_LOG")"
  touch "$WATCH_LOG"

  if ! container_exists; then
    recover_full
    return 0
  fi

  nvidia_smi_ok "host nvidia-smi" "nvidia-smi -L" && host_ok=1 || true
  if container_running; then
    nvidia_smi_ok "container nvidia-smi" "docker exec $NAME nvidia-smi -L" && container_ok=1 || true
  fi

  if [ "$host_ok" = 0 ]; then
    log_file "recover: host NVML broken (needs sudo on host)"
    log_file "  sudo nvidia-smi -pm 1"
    log_file "  sudo systemctl restart nvidia-persistenced"
    log_file "  sudo nvidia-smi --gpu-reset   # if still broken"
    if container_running; then
      recover_container || true
    fi
    return 1
  fi

  if [ "$container_ok" = 0 ]; then
    recover_container
    return 0
  fi

  if ! jupyter_running || ! jax_ok; then
    recover_jupyter_only
    return 0
  fi

  log_file "recover: nothing to do (already healthy)"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  check)
    if run_check; then
      echo "All checks passed."
      exit 0
    fi
    echo
    echo "Recovery: ./scripts/mindgap-gpu.sh recover"
    exit 1
    ;;
  recover|restart)
    run_recover
    echo
    if run_check; then
      echo "Recovery succeeded."
      echo "JupyterLab URL:"
      show_jupyter_url || true
      exit 0
    fi
    echo "Recovery attempted but checks still failing. See $WATCH_LOG" >&2
    exit 1
    ;;
  watch)
    if run_check >/dev/null 2>&1; then
      log_file "watch: ok"
      exit 0
    fi
    log_file "watch: unhealthy -> auto recover"
    run_recover
    ;;
  install-cron)
    cron_line="*/${CRON_EVERY_MIN} * * * * cd $ROOT && ./scripts/mindgap-gpu.sh watch >> $WATCH_LOG 2>&1"
    if crontab -l 2>/dev/null | grep -Fq "scripts/mindgap-gpu.sh watch"; then
      echo "cron entry already exists:"
      crontab -l | grep "mindgap-gpu.sh watch"
      exit 0
    fi
    (crontab -l 2>/dev/null; echo "$cron_line") | crontab -
    echo "Installed cron (every ${CRON_EVERY_MIN} min):"
    echo "  $cron_line"
    echo "Log: $WATCH_LOG"
    ;;
  uninstall-cron)
    crontab -l 2>/dev/null | grep -Fv "scripts/mindgap-gpu.sh watch" | crontab - || true
    echo "Removed mindgap-gpu watch cron entries."
    ;;
  status)
    docker exec "$NAME" nvidia-smi 2>/dev/null || echo "nvidia-smi failed"
    ;;
  help|*)
    cat <<EOF
mindGAP GPU helper

  check          test host/container NVML, jupyter-lab, jax
  recover        auto-fix (jupyter -> container restart -> setup)
  restart        alias of recover
  watch          check + recover if needed (for cron)
  install-cron   register watch every ${CRON_EVERY_MIN} min
  uninstall-cron remove watch cron
  status         nvidia-smi inside container

NVML Unknown Error 対策:
  1) 自動: ./scripts/mindgap-gpu.sh install-cron
  2) 手動: ./scripts/mindgap-gpu.sh recover
  3) 予防 (host, sudo): sudo nvidia-smi -pm 1
                        sudo systemctl enable --now nvidia-persistenced

Log: $WATCH_LOG
EOF
    ;;
esac
