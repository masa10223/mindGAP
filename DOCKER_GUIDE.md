# mindGAP Docker 指南書 (ELA_analysis)

Energy Landscape Analysis 用 Jupyter 環境。**mindgap イメージはこのプロジェクト専用**です。  
jaxOPT / 最適輸送は [`OPT_morpho`](../../EvoDevoMorpho/OPT_morpho/DOCKER_GUIDE.md) を参照してください。

## 構成

| 項目 | 値 |
|------|-----|
| Docker イメージ | `mindgap:latest` (~6–8 GB, 旧版 ~23 GB) |
| コンテナ名 | `mindGAP` |
| ポート | `8881` |
| マウント | このディレクトリ → `/work` |
| venv | `./venv/jaxenv` (cudf/cugraph 等) |

イメージに含む: JAX + CuPy + JupyterLab  
venv に追加: cudf, cugraph, 解析用 Python パッケージ (`requirements-mindgap.txt`)

## 初回セットアップ（スリム版へ移行）

```bash
cd /inthdd/tsutsumi/free_energy/ELA_analysis

# 1. 旧コンテナ・旧イメージ削除（容量確保）
docker stop mindGAP 2>/dev/null; docker rm mindGAP 2>/dev/null
docker rmi mindgap:latest 2>/dev/null || true
docker builder prune -f

# 2. スリムイメージをビルド
./build-images.sh

# 3. コンテナ起動 + venv bootstrap + Jupyter
./setup_mindgap.sh
```

ブラウザ: `http://<ホストIP>:8881/lab`  
トークン: `docker exec mindGAP tail -20 /work/jupyter.log`

## 日常の使い方

```bash
# 起動（イメージ未ビルドなら先に build-images.sh）
./setup_mindgap.sh

# ヘルスチェック
./scripts/mindgap-gpu.sh check

# GPU/Jupyter 復旧
./scripts/mindgap-gpu.sh recover

# venv 再作成（requirements 変更後）
./scripts/mindgap-clean.sh all
FORCE_BOOTSTRAP=1 ./setup_mindgap.sh

# パッケージ追加
./scripts/mindgap-pip.sh install <pkg>
# → requirements-mindgap.txt に追記
```

## スクリプト一覧

| ファイル | 役割 |
|----------|------|
| `build-images.sh` | スリム `mindgap:latest` をビルド |
| `setup_mindgap.sh` | コンテナ作成・Jupyter 起動 |
| `scripts/mindgap-env.sh` | 名前/ポート/イメージのデフォルト |
| `scripts/mindgap-gpu.sh` | NVML / GPU / Jupyter 監視・復旧 |
| `scripts/mindgap-pip.sh` | venv 内 pip 操作 |
| `scripts/mindgap-clean.sh` | pip-cache / venv 削除 |
| `docker/Dockerfile.mindgap` | イメージ定義 |

## トラブルシュート

**NVML Unknown Error**

```bash
./scripts/mindgap-gpu.sh recover
# ホスト側 (sudo):
sudo nvidia-smi -pm 1
sudo systemctl enable --now nvidia-persistenced
```

**ルート (/) ディスク不足**

Docker イメージは `/` に保存されます。不要イメージ削除:

```bash
docker system df
docker builder prune -f
```

**カーネル `Python (jaxenv)` が見えない**

```bash
FORCE_BOOTSTRAP=1 ./setup_mindgap.sh
```
