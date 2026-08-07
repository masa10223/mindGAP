# ELA Analysis

Energy Landscape Analysis (ELA) は、心理学的質問票データのエネルギーランドスケープ解析を行うPythonパッケージです。

## 特徴

- **質問文付きデータ生成**: オリジナルExcelファイルから質問文を列名として持つCSVファイルを生成
- **閾値情報付きファイル名**: ファイル名に使用した閾値情報を含める（例：`emotional_2204_thr3.csv`）
- **フォルダ分け整理**: 質問票ごとにフォルダを分けて整理
- **柔軟な設定**: JSON/YAML設定ファイル、環境変数、コマンドライン引数による設定
- **CUDA対応**: GPU加速による高速計算
- **メンテナンス性**: クラスベースの設計とモジュラー構造
- **OSS対応**: 依存関係の柔軟な管理とインストールオプション

## インストール

### 基本インストール

```bash
pip install -e .
```

### CUDA対応インストール

```bash
pip install -e .[cuda]
```

### 開発環境インストール

```bash
pip install -e .[dev,testing,docs]
```

## データ前処理の使い方

### 1. オリジナルデータから処理済みデータを作成

`create_processed_data.py`を使用して、オリジナルExcelファイルから質問文付きのCSVファイルを生成します。

#### 基本的な使い方

```bash
# 特定の質問票を処理
python create_processed_data.py --questionnaire_type PHQ-9

# 全ての質問票を一括処理
python create_processed_data.py --all
```

#### 利用可能な質問票タイプ

- `HQ25+4`: HQ-25+4質問票（閾値3）
- `PHQ-9`: PHQ-9質問票（閾値2） ２は固定必須
- `IPS-22`: IPS-22質問票（閾値2）2 or 3
- `IAT`: IAT質問票（閾値3）3固定
- `TACS-22`: TACS-22質問票（閾値2） 2 or 3

#### コマンドライン引数

```bash
python create_processed_data.py [OPTIONS]

引数:
  --questionnaire_type TEXT  処理する質問票タイプ（HQ25+4, PHQ-9, IPS-22, IAT, TACS-22）
  --all                     全ての質問票を処理
  --combine Q1 Q2           2つの質問票を組み合わせて処理
  --threshold1 INT          第1の質問票の閾値
  --threshold2 INT          第2の質問票の閾値
  --threshold INT           閾値を指定（デフォルトは質問票ごとに異なる）
  --help                    ヘルプメッセージを表示
```

### 2. 生成されるファイル構造

```
data/processed/
├── HQ25+4/                    # HQ-25+4質問票フォルダ
│   ├── HQ25+4_2006_thr3.csv  # 2006年データ（閾値3）
│   ├── HQ25+4_2012_thr3.csv  # 2012年データ（閾値3）
│   ├── HQ25+4_2108_thr3.csv  # 2021年8月データ（閾値3）
│   ├── HQ25+4_2204_thr3.csv  # 2022年4月データ（閾値3）
│   ├── HQ25+4_all_thr3.csv   # 全年代統合データ（閾値3）
│   ├── social_2006_thr3.csv  # 社会性カテゴリ（2006年）
│   ├── social_2012_thr3.csv  # 社会性カテゴリ（2012年）
│   ├── social_2108_thr3.csv  # 社会性カテゴリ（2021年8月）
│   ├── social_2204_thr3.csv  # 社会性カテゴリ（2022年4月）
│   ├── isolation_2006_thr3.csv  # 孤立カテゴリ（2006年）
│   ├── isolation_2012_thr3.csv  # 孤立カテゴリ（2012年）
│   ├── isolation_2108_thr3.csv  # 孤立カテゴリ（2021年8月）
│   ├── isolation_2204_thr3.csv  # 孤立カテゴリ（2022年4月）
│   ├── emotional_2006_thr3.csv  # 感情的サポートカテゴリ（2006年）
│   ├── emotional_2012_thr3.csv  # 感情的サポートカテゴリ（2012年）
│   ├── emotional_2108_thr3.csv  # 感情的サポートカテゴリ（2021年8月）
│   └── emotional_2204_thr3.csv  # 感情的サポートカテゴリ（2022年4月）
├── PHQ-9/                     # PHQ-9質問票フォルダ
│   ├── PHQ-9_2006_thr3.csv   # 2006年データ（閾値3）
│   ├── PHQ-9_2012_thr3.csv   # 2012年データ（閾値3）
│   ├── PHQ-9_2108_thr3.csv   # 2021年8月データ（閾値3）
│   ├── PHQ-9_2204_thr3.csv   # 2022年4月データ（閾値3）
│   └── PHQ-9_all_thr3.csv    # 全年代統合データ（閾値3）
├── IPS-22/                    # IPS-22質問票フォルダ
│   ├── IPS-22_2006_thr3.csv  # 2006年データ（閾値3）
│   ├── IPS-22_2012_thr3.csv  # 2012年データ（閾値3）
│   ├── IPS-22_2108_thr3.csv  # 2021年8月データ（閾値3）
│   ├── IPS-22_2204_thr3.csv  # 2022年4月データ（閾値3）
│   └── IPS-22_all_thr3.csv   # 全年代統合データ（閾値3）
├── IAT/                       # IAT質問票フォルダ
│   ├── IAT_2006_thr3.csv     # 2006年データ（閾値3）
│   ├── IAT_2012_thr3.csv     # 2012年データ（閾値3）
│   ├── IAT_2108_thr3.csv     # 2021年8月データ（閾値3）
│   ├── IAT_2204_thr3.csv     # 2022年4月データ（閾値3）
│   └── IAT_all_thr3.csv      # 全年代統合データ（閾値3）
└── TACS-22/                   # TACS-22質問票フォルダ
    ├── TACS-22_2006_thr4.csv # 2006年データ（閾値4）
    ├── TACS-22_2012_thr4.csv # 2012年データ（閾値4）
    ├── TACS-22_2108_thr4.csv # 2021年8月データ（閾値4）
    ├── TACS-22_2204_thr4.csv # 2022年4月データ（閾値4）
    └── TACS-22_all_thr4.csv  # 全年代統合データ（閾値4）
└── combined/                  # 組み合わせ質問票フォルダ
    ├── emotional_PHQ-9_2006_thr2_thr3.csv  # emotional + PHQ-9（2006年）
    ├── emotional_PHQ-9_2012_thr2_thr3.csv  # emotional + PHQ-9（2012年）
    ├── emotional_PHQ-9_2108_thr2_thr3.csv  # emotional + PHQ-9（2021年8月）
    ├── emotional_PHQ-9_2204_thr2_thr3.csv  # emotional + PHQ-9（2022年4月）
    ├── emotional_PHQ-9_all_thr2_thr3.csv   # emotional + PHQ-9（全年代統合）
    ├── social_PHQ-9_2006_thr2_thr3.csv     # social + PHQ-9（2006年）
    └── ...                    # その他の組み合わせ
```

### 3. 組み合わせ質問票処理

複数の質問票を組み合わせて解析を行うことができます。心理学的に意味のある組み合わせを分析することで、より深い洞察を得ることができます。

#### 基本的な使い方

```bash
# emotional + PHQ-9 の組み合わせ（デフォルト閾値）
python create_processed_data.py --combine emotional PHQ-9

# 閾値を指定した組み合わせ
python create_processed_data.py --combine emotional PHQ-9 --threshold1 2 --threshold2 3

# social + PHQ-9 の組み合わせ
python create_processed_data.py --combine social PHQ-9 --threshold1 2 --threshold2 2

# isolation + PHQ-9 の組み合わせ
python create_processed_data.py --combine isolation PHQ-9 --threshold1 3 --threshold2 3
```

#### 利用可能な組み合わせ

- **HQ-25+4カテゴリ + その他の質問票**:
  - `emotional` + `PHQ-9`: 感情的孤立とうつ症状
  - `social` + `PHQ-9`: 社会性とうつ症状
  - `isolation` + `PHQ-9`: 孤立とうつ症状
  - `emotional` + `IPS-22`: 感情的孤立と対人関係
  - `social` + `IPS-22`: 社会性と対人関係

- **質問票同士の組み合わせ**:
  - `PHQ-9` + `IPS-22`: うつ症状と対人関係
  - `PHQ-9` + `IAT`: うつ症状と内集団バイアス
  - `IPS-22` + `IAT`: 対人関係と内集団バイアス

#### 組み合わせデータの特徴

- **列名のプレフィックス**: 各質問票の列名にプレフィックスが付きます
  - `emotional_` + 元の列名
  - `PHQ-9_` + 元の列名
- **行数の調整**: 2つのデータセットの最小行数に合わせて調整
- **年代別処理**: 各年代ごとに個別に統合し、全年代統合データも作成

#### 組み合わせファイル名の規則

```
{質問票1}_{質問票2}_{年代}_thr{閾値1}_thr{閾値2}.csv
```

**例**:
- `emotional_PHQ-9_2006_thr2_thr3.csv`: emotional（閾値2）+ PHQ-9（閾値3）、2006年データ
- `social_PHQ-9_all_thr2_thr2.csv`: social（閾値2）+ PHQ-9（閾値2）、全年代統合データ

### 4. ファイル名の規則

#### 基本形式
```
{質問票名}_{年代}_thr{閾値}.csv
```

#### 例
- `PHQ-9_2006_thr3.csv`: PHQ-9質問票、2006年データ、閾値3
- `social_2204_thr3.csv`: HQ-25+4社会性カテゴリ、2022年4月データ、閾値3
- `TACS-22_all_thr4.csv`: TACS-22質問票、全年代統合データ、閾値4

### 4. 列名（質問文）の特徴

生成されるCSVファイルの列名は実際の質問文になります：

#### PHQ-9の例
```csv
物事に対してほとんど興味がない、または楽しめない,気分が落ち込む、憂うつになる、または絶望的な気持ちになる,寝付きが悪い、途中で目がさめる、または逆に眠り過ぎる,...
```

#### HQ-25+4の例
```csv
集団に入るのは苦手だ。,人づきあいは楽しくない。,大切な事柄について話し合える人が本当に誰もいない。,知らない人に会うのが大好きだ。,...
```

### 5. 使用例

#### 基本的な処理

```bash
# 1. 全ての質問票を処理
python create_processed_data.py --all

# 2. 特定の質問票のみ処理
python create_processed_data.py --questionnaire_type PHQ-9
python create_processed_data.py --questionnaire_type HQ25+4

# 3. 組み合わせ質問票の処理
python create_processed_data.py --combine emotional PHQ-9
python create_processed_data.py --combine social PHQ-9 --threshold1 2 --threshold2 3
python create_processed_data.py --combine isolation PHQ-9 --threshold1 3 --threshold2 2
```

#### 処理結果の確認

```bash
# 生成されたファイル一覧を確認
find data/processed -name "*_thr*.csv" | sort

# 特定のファイルの内容を確認（質問文が列名になっている）
head -3 data/processed/PHQ-9/PHQ-9_2006_thr3.csv
head -3 data/processed/HQ25+4/social_2006_thr3.csv

# 組み合わせデータの確認
head -3 data/processed/combined/emotional_PHQ-9_2006_thr2_thr3.csv
head -3 data/processed/combined/social_PHQ-9_all_thr2_thr2.csv
```

## ELA解析の実行

前処理済みデータを使用してELA解析を実行します。

### 基本的な使い方

```bash
# 処理済みCSVファイルから解析
python -m src.ela_pipeline --func PHQ-9_2006_thr3 --data_source csv

# HQ-25+4の社会性カテゴリを解析
python -m src.ela_pipeline --func social_2006_thr3 --data_source csv

# 全年代統合データを解析
python -m src.ela_pipeline --func PHQ-9_all_thr3 --data_source csv

# 組み合わせデータの解析
python -m src.ela_pipeline --func emotional_PHQ-9_2006_thr2_thr3 --data_source csv
python -m src.ela_pipeline --func social_PHQ-9_all_thr2_thr2 --data_source csv
```

### コマンドライン引数の詳細

#### 必須引数
- `--func`: 解析するデータファイル名（拡張子なし）
  - 例: `--func PHQ-9_2006_thr3`, `--func social_2204_thr3`, `--func HQ25+4_all_thr3`

#### データ関連引数
- `--data_source`: データソース（`csv`/`excel`/`auto`、デフォルト: `auto`）
- `--data_dir`: データディレクトリパス（デフォルト: `data`）

#### 閾値設定引数
- `--hq25_threshold`: HQ-25の閾値（2または3、デフォルト: 3）
- `--ips22_threshold`: IPS-22の閾値（デフォルト: 3）
- `--iat_threshold`: IATの閾値（デフォルト: 3）
- `--tacs22_threshold`: TACS-22の閾値（デフォルト: 4）

#### 出力・ログ関連引数
- `--output_dir`: 出力ディレクトリパス（デフォルト: `outputs`）
- `--log_dir`: ログディレクトリパス（デフォルト: `logs`）
- `--verbose`: 詳細な出力を有効にする

#### システム関連引数
- `--device_id`: CUDAデバイスID（デフォルト: 0）
- `--config`: 設定ファイルのパス

### 使用例

```bash
# PHQ-9の2006年データを解析
python -m src.ela_pipeline --func PHQ-9_2006_thr3 --data_source csv

# HQ-25+4の社会性カテゴリ（2022年4月）を解析
python -m src.ela_pipeline --func social_2204_thr3 --data_source csv

# HQ-25+4の孤立カテゴリ（全年代統合）を解析
python -m src.ela_pipeline --func isolation_all_thr3 --data_source csv

# TACS-22の全年代統合データを解析（閾値4）
python -m src.ela_pipeline --func TACS-22_all_thr4 --data_source csv

# 詳細ログ付きで実行
python -m src.ela_pipeline --func PHQ-9_all_thr3 --data_source csv --verbose

# カスタム出力ディレクトリで実行
python -m src.ela_pipeline --func social_2006_thr3 --data_source csv --output_dir outputs/custom_analysis
```

## バッチ処理

複数の解析を一括で実行する方法を提供します。

### 設定ファイルの作成

```bash
# サンプル設定ファイルを作成
python -m src.batch_runner --create-sample-config
```

### 設定ファイルの例

```json
{
  "description": "質問文付きデータの一括解析",
  "parallel_execution": true,
  "max_workers": 2,
  "jobs": [
    {
      "name": "PHQ9_2006_analysis",
      "questionnaire_type": "PHQ-9_2006_thr3",
      "data_source": "csv",
      "notes": "PHQ-9の2006年データ解析"
    },
    {
      "name": "HQ25_social_2204_analysis",
      "questionnaire_type": "social_2204_thr3",
      "data_source": "csv",
      "notes": "HQ-25+4社会性カテゴリ（2022年4月）解析"
    },
    {
      "name": "TACS22_all_analysis",
      "questionnaire_type": "TACS-22_all_thr4",
      "data_source": "csv",
      "notes": "TACS-22全年代統合データ解析"
    }
  ]
}
```

### バッチ実行

```bash
# 設定ファイルから一括実行
python -m src.batch_runner --config batch_config_sample.json

# 並列実行（高速化）
python -m src.batch_runner --config batch_config_sample.json --parallel --max-workers 4

# 逐次実行（安定性重視）
python -m src.batch_runner --config batch_config_sample.json --sequential
```

## 実行履歴管理

ELA解析の実行履歴を自動的に記録し、検索・比較・分析が可能です。

### 実行履歴の表示・検索

```bash
# 最近の実行履歴を表示（最新10件）
python -m src.history_cli list

# 特定の質問票タイプの実行履歴を表示
python -m src.history_cli list --questionnaire-type PHQ-9

# 成功した実行のみを表示
python -m src.history_cli list --status success

# 詳細な実行記録を表示
python -m src.history_cli show <execution_id>

# 実行統計を表示
python -m src.history_cli stats
```

## ディレクトリ構造

```
ELA_analysis/
├── src/                       # ソースコード
│   ├── ela_pipeline.py       # メイン解析パイプライン
│   ├── data_processing.py    # データ処理モジュール
│   ├── batch_runner.py       # バッチ実行モジュール
│   └── history_cli.py        # 実行履歴管理CLI
├── data/                      # データディレクトリ
│   ├── original/             # オリジナルデータ（Excel形式）
│   │   ├── HQ25+4.xlsx      # HQ-25+4質問票データ
│   │   ├── PHQ-9.xlsx       # PHQ-9質問票データ
│   │   ├── IPS-22.xlsx      # IPS-22質問票データ
│   │   ├── IAT.xlsx         # IAT質問票データ
│   │   └── TACS-22.xlsx     # TACS-22質問票データ
│   └── processed/            # 処理済みデータ（CSV形式）
│       ├── HQ25+4/          # HQ-25+4フォルダ
│       ├── PHQ-9/           # PHQ-9フォルダ
│       ├── IPS-22/          # IPS-22フォルダ
│       ├── IAT/             # IATフォルダ
│       └── TACS-22/         # TACS-22フォルダ
├── outputs/                  # 出力ディレクトリ
│   ├── csvs/                # CSV出力
│   ├── figs/                # 図表出力
│   └── integrated/          # 統合解析結果
├── logs/                    # ログファイル
├── execution_history/       # 実行履歴
├── batch_outputs/          # バッチ実行結果
├── create_processed_data.py # データ前処理スクリプト
└── README.md               # このファイル
```

## データの種類と用途

### オリジナルデータ（data/original/）
- **各質問票のExcelファイル**: 元の生データファイル
- **構造**: 1行目に年代情報、2行目に質問文、3行目以降に回答データ
- **用途**: `create_processed_data.py`で処理済みCSVファイルを生成

### 処理済みデータ（data/processed/）
- **質問文付きCSVファイル**: 列名が実際の質問文になっている
- **閾値情報付きファイル名**: 使用した閾値がファイル名に含まれている
- **フォルダ分け**: 質問票ごとに整理されている
- **用途**: ELA解析の実行

## 設定

### 閾値パラメータの設定

各質問票の二値化における閾値を設定できます:

#### デフォルト閾値
- **HQ-25+4**: 3（1-3を陰性、4を陽性）
- **PHQ-9**: 3（1-3を陰性、4を陽性）
- **IPS-22**: 3（1-3を陰性、4を陽性）
- **IAT**: 3（1-3を陰性、4を陽性）
- **TACS-22**: 4（1-4を陰性、5を陽性）

#### 閾値の変更
```bash
# HQ-25+4の閾値を2に変更
python -m src.ela_pipeline --func HQ25+4_2006_thr2 --data_source csv --hq25_threshold 2

# 複数の閾値を指定
python -m src.ela_pipeline --func PHQ-9_2006_thr3 --data_source csv --hq25_threshold 2 --ips22_threshold 3
```

## 開発

### テストの実行

```bash
pytest tests/
```

### コードフォーマット

```bash
black src/
flake8 src/
mypy src/
```

## 依存関係

### 基本依存関係

- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.9.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- networkx >= 2.8.0
- tqdm >= 4.64.0
- openpyxl >= 3.0.0
- PyYAML >= 6.0

### オプション依存関係

- **CUDA**: cupy-cuda11x >= 10.0.0
- **可視化**: plotly >= 5.0.0
- **Jupyter**: jupyter >= 1.0.0
- **テスト**: pytest >= 7.0.0
- **開発**: black, flake8, mypy

## ライセンス

MIT License

## サポート

問題が発生した場合は、GitHubのIssuesページで報告してください。