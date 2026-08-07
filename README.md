# mindGAP

Energy Landscape Analysis (ELA) 用の **GPU Jupyter 解析環境** とデモです。

- Docker イメージ: `mindgap:latest`（JAX + CuPy + JupyterLab）
- コンテナ名: `mindGAP`（ポート `8881`）
- デモ: [`notebooks/00_mindGAP_demo.ipynb`](notebooks/00_mindGAP_demo.ipynb)（**ダミーデータのみ**）

> 実アンケート原データ（`data/original/`）や解析成果物は Git 管理外です。

## 必要環境

- NVIDIA GPU + Docker + NVIDIA Container Toolkit
- ディスク空き（イメージビルド時）おおよそ 6 GB 以上

## セットアップ

```bash
git clone git@github.com:masa10223/mindGAP.git
cd mindGAP

# 1) イメージビルド
./build-images.sh

# 2) コンテナ起動 + venv bootstrap + JupyterLab
./setup_mindgap.sh
```

ブラウザ: `http://<host>:8881/lab`  
トークン確認:

```bash
docker exec mindGAP tail -20 /work/jupyter.log
```

詳細は [`DOCKER_GUIDE.md`](DOCKER_GUIDE.md) を参照。

## デモの動かし方

1. JupyterLab で `notebooks/00_mindGAP_demo.ipynb` を開く
2. カーネルは `jaxenv`（コンテナ内）を選択
3. 上から実行

デモ内容:

1. 合成二値質問票データ生成
2. VEM で \((h,J)\) 推定
3. \(J\) の被験者 PCA
4. exact fixation（\(2^9\) 列挙）のヒートマップ

## リポジトリ構成（主なもの）

| パス | 内容 |
|------|------|
| `docker/Dockerfile.mindgap` | mindgap イメージ定義 |
| `build-images.sh` | イメージビルド |
| `setup_mindgap.sh` | コンテナ起動 |
| `scripts/mindgap-*.sh` | GPU 監視 / pip / クリーンアップ |
| `requirements-mindgap.txt` | コンテナ venv 追加パッケージ |
| `notebooks/VEM_MEM.py` | VEM 実装 |
| `notebooks/00_mindGAP_demo.ipynb` | ダミーデータ demo |
| `src/` | ELA パッケージ本体 |

## 日常操作

```bash
./setup_mindgap.sh                 # 起動
./scripts/mindgap-gpu.sh check     # ヘルスチェック
./scripts/mindgap-gpu.sh recover   # GPU/Jupyter 復旧
FORCE_BOOTSTRAP=1 ./setup_mindgap.sh  # requirements 変更後の再構築
```

## 注意

- `venv/` はローカル生成物です（コミットしません）
- 実データ解析ノートは各自の環境で `data/original/` を用意してください
- 旧 ELA パッケージ説明は [`docs/README_ELA_package.md`](docs/README_ELA_package.md) にあります
