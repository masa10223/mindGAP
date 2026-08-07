#!/usr/bin/env bash
# mindGAP — mindGAP 用 Jupyter コンテナ起動
set -euo pipefail

# shellcheck source=scripts/mindgap-env.sh
source "$(cd "$(dirname "$0")" && pwd)/scripts/mindgap-env.sh"

FORCE_BOOTSTRAP="${FORCE_BOOTSTRAP:-0}"

echo "project: mindGAP  container: $NAME  port: $PORT  image: $IMAGE"

echo "[1/4] remove old container (if exists)"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[2/4] start container"
docker run --name "$NAME" --gpus all --shm-size=50g --restart unless-stopped -dit \
  --health-cmd='nvidia-smi -L >/dev/null 2>&1 && pgrep -f "[j]upyter-lab" >/dev/null || exit 1' \
  --health-interval=60s \
  --health-timeout=15s \
  --health-retries=3 \
  --health-start-period=120s \
  -p "${PORT}:${PORT}" \
  -v "$PWD":/work \
  "$IMAGE" \
  bash -lc "sleep infinity"

echo "[3/4] bootstrap jaxenv (only if needed) + start jupyter"
docker exec -u 0 "$NAME" bash -lc '
set -e
mkdir -p /work/tmp /work/pip-cache /work/venv /work/.jupyter \
  /work/notebooks/csvs_for_paper /work/notebooks/figs_for_paper
chmod -R u+rwX,g+rwX /work/notebooks 2>/dev/null || true
'
docker exec "$NAME" bash -lc '
export TMPDIR=/work/tmp
export PIP_CACHE_DIR=/work/pip-cache
export PIP_NO_CACHE_DIR=1
export JUPYTER_DATA_DIR=/work/.jupyter

REQ="/work/'"$REQ_FILE"'"
STAMP="/work/venv/jaxenv/.requirements.sha256"
IMAGE_STAMP="/work/venv/jaxenv/.mindgap-image"
VENV_PY="/work/venv/jaxenv/bin/python"
VENV_CFG="/work/venv/jaxenv/pyvenv.cfg"

needs_bootstrap() {
  [ "'"$FORCE_BOOTSTRAP"'" = "1" ] && return 0
  [ ! -x "$VENV_PY" ] && return 0
  [ ! -f "$VENV_CFG" ] && return 0
  grep -q "include-system-site-packages = true" "$VENV_CFG" || return 0
  [ ! -f "$IMAGE_STAMP" ] && return 0
  grep -qx "'"$IMAGE"'" "$IMAGE_STAMP" || return 0
  [ ! -f "$REQ" ] && return 0
  [ ! -f "$STAMP" ] && return 0
  sha256sum -c "$STAMP" >/dev/null 2>&1 || return 0
  "$VENV_PY" - <<'"'"'PY'"'"' >/dev/null 2>&1 || return 0
import importlib
for name in ("jax", "cudf", "cugraph", "cupy"):
    importlib.import_module(name)
PY
  return 1
}

if needs_bootstrap; then
  echo "  -> bootstrap: slim venv (image JAX/Jupyter shared via --system-site-packages)"
  rm -rf /work/venv/jaxenv
  if ! python3 -m venv --system-site-packages /work/venv/jaxenv 2>/dev/null; then
    pip install -q virtualenv
    virtualenv --system-site-packages /work/venv/jaxenv
  fi
  source /work/venv/jaxenv/bin/activate
  pip install -U pip
  pip install -r "$REQ"
  sha256sum "$REQ" > "$STAMP"
  echo "'"$IMAGE"'" > "$IMAGE_STAMP"
  rm -rf /work/pip-cache/*
else
  echo "  -> reuse existing venv (skip pip install)"
fi

source /work/venv/jaxenv/bin/activate
python -m ipykernel install --name jaxenv --display-name "Python (jaxenv)"

if pgrep -f "[j]upyter-lab" >/dev/null 2>&1; then
  pkill -f "[j]upyter-lab" || true
  sleep 1
fi
nohup jupyter lab \
  --port='"$PORT"' --ip=0.0.0.0 --no-browser --allow-root \
  --notebook-dir=/work \
  > /work/jupyter.log 2>&1 &
'

echo "[4/4] verify and show URL"
docker exec "$NAME" bash -lc '
source /work/venv/jaxenv/bin/activate
python - <<'"'"'PY'"'"'
import jax, cudf, cugraph, cupy
print("jax", jax.__version__)
print(jax.devices())
print("cudf", cudf.__version__)
print("cugraph", cugraph.__version__)
print("cupy", cupy.__version__)
PY
'

echo "----- Jupyter URL (token) -----"
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
        url = m[-1].replace('127.0.0.1', 'localhost')
        if '/lab' not in url:
            url = url.replace('/?', '/lab?')
        print(url); break
    time.sleep(1)
else:
    print('token URL not found yet')
PY"
host_ip="$(hostname -I | awk '{print $1}')"
echo "JupyterLab: http://${host_ip}:${PORT}/lab"
echo
echo "See DOCKER_GUIDE.md for details."
