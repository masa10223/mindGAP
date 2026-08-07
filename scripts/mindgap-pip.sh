#!/usr/bin/env bash
# mindGAP 環境の pip 操作ヘルパー
#
# 使い方:
#   ./scripts/mindgap-pip.sh install scikit-learn        # 1つ追加（venv に永続保存）
#   ./scripts/mindgap-pip.sh upgrade jax                # 1つ更新
#   ./scripts/mindgap-pip.sh sync                       # requirements-mindgap.txt から再インストール
#   ./scripts/mindgap-pip.sh freeze                     # 現在の環境を lock ファイルに保存
#   ./scripts/mindgap-pip.sh list                       # 主要パッケージのバージョン表示
set -euo pipefail

# shellcheck source=mindgap-env.sh
source "$(cd "$(dirname "$0")" && pwd)/mindgap-env.sh"

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "error: container '$NAME' is not running. run ./setup_mindgap.sh first." >&2
  exit 1
fi

run_pip() {
  docker exec "$NAME" bash -lc "
    set -e
    export TMPDIR=/work/tmp
    export PIP_CACHE_DIR=/work/pip-cache
    source /work/venv/jaxenv/bin/activate
    pip \"\$@\"
  " bash "$@"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  install|upgrade|uninstall)
    run_pip "$cmd" "$@"
    echo
    echo "Installed in ./venv/jaxenv (persisted)."
    echo "To record for next bootstrap, add the package to $REQ_FILE"
    echo "  or run: ./scripts/mindgap-pip.sh freeze"
    ;;
  sync)
    echo "Installing from /work/$REQ_FILE ..."
    docker exec "$NAME" bash -lc "
      set -e
      export TMPDIR=/work/tmp
      export PIP_CACHE_DIR=/work/pip-cache
      source /work/venv/jaxenv/bin/activate
      pip install -U pip
      pip install -r /work/$REQ_FILE
      sha256sum /work/$REQ_FILE > /work/venv/jaxenv/.requirements.sha256
    "
    echo "Done. requirements hash updated."
    ;;
  freeze)
    run_pip freeze > "$LOCK_FILE"
    echo "Saved: $LOCK_FILE"
    echo "Tip: copy relevant lines into $REQ_FILE when formalizing updates."
    ;;
  list)
    docker exec "$NAME" bash -lc '
      source /work/venv/jaxenv/bin/activate
      python - <<'"'"'PY'"'"'
import importlib
pkgs = ["jax", "cudf", "cupy", "cugraph", "pyarrow", "pandas", "numpy", "sklearn"]
for name in pkgs:
    mod = "sklearn" if name == "sklearn" else name
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"{name:12} {ver}")
    except Exception as e:
        print(f"{name:12} NOT INSTALLED ({e})")
PY
    '
    ;;
  help|*)
    cat <<EOF
mindGAP pip helper

  install <pkg>     add package to persisted venv
  upgrade <pkg>     upgrade one package
  uninstall <pkg>   remove package
  sync              reinstall from $REQ_FILE
  freeze            save pip freeze to $LOCK_FILE
  list              show key package versions

Recommended update workflow:
  1) Try:  ./scripts/mindgap-pip.sh install <pkg>
  2) Test in Jupyter (restart kernel)
  3) Record: edit $REQ_FILE, or ./scripts/mindgap-pip.sh freeze
  4) Full sync after editing requirements:
           ./scripts/mindgap-pip.sh sync
EOF
    ;;
esac
