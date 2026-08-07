# バッチ設定ファイルの説明

## 概要
このディレクトリには、様々な質問票タイプと年代別の一括実行設定ファイルが含まれています。

## 設定ファイル一覧

### 1. 基本設定ファイル
- **`batch_config_20250910.json`**: PHQ-9の年代別一括実行設定
- **`batch_config_sample.json`**: サンプル設定ファイル

### 2. HQ-25+4関連設定ファイル
- **`batch_config_hq25_socialization.json`**: HQ-25+4のSocialization項目（1,4,6,8,11,13,15,18,20,23,25）の年代別一括実行設定
- **`batch_config_hq25_isolation_corrected.json`**: HQ-25+4のIsolation項目（2,5,9,12,16,19,22,24）の年代別一括実行設定
- **`batch_config_hq25_emotional_support.json`**: HQ-25+4のEmotional support項目（3,7,10,14,17,21）の年代別一括実行設定

### 3. 統合設定ファイル
- **`batch_config_phq9_hq25_integrated.json`**: PHQ-9とHQ-25+4を統合した年代別一括実行設定
- **`batch_config_phq9_hq25_social_isolation.json`**: PHQ-9とHQ-25+4のSocial/Isolation項目を統合した年代別一括実行設定

## 列範囲の説明

### HQ-25+4の構造
- **全データ**: 152列（4年代 × 38列）
- **年代別分割**: 各行を4つの年代に分割し、各年代38列
- **列削除**: `"Unnamed: 21","Unnamed: 51","Unnamed: 81","Unnamed: 111","Unnamed: 141"`を削除
- **最終列数**: 各年代29列

### HQ-25+4の項目分割
- **Socialization項目**: 列1,4,6,8,11,13,15,18,20,23,25（11項目）- 社会的な項目
- **Isolation項目**: 列2,5,9,12,16,19,22,24（8項目）- 孤立感に関する項目
- **Emotional support項目**: 列3,7,10,14,17,21（6項目）- 情緒的サポートに関する項目

### PHQ-9の構造
- **全データ**: 42列（4年代 × 10.5列）
- **年代別分割**: 各行を4つの年代に分割し、各年代9列
- **列削除**: 列6, 11, 21, 31, 41を削除
- **最終列数**: 各年代9列

## 使用方法

### 1. 個別実行
```bash
# HQ-25+4 Social項目の2012年データを実行
python -m src.ela_pipeline --func HQ-25+4 --r1 29 --r2 44 --data_source excel --hq25_threshold 2

# HQ-25+4 Isolation項目の2012年データを実行
python -m src.ela_pipeline --func HQ-25+4 --r1 44 --r2 58 --data_source excel --hq25_threshold 2
```

### 2. バッチ実行
```bash
# HQ-25+4 Social項目の年代別一括実行
python -m src.batch_runner --config configs/batch_config_hq25_social.json

# HQ-25+4 Isolation項目の年代別一括実行
python -m src.batch_runner --config configs/batch_config_hq25_isolation.json

# PHQ-9とHQ-25+4統合の年代別一括実行
python -m src.batch_runner --config configs/batch_config_phq9_hq25_integrated.json
```

## 注意事項

1. **列範囲**: 各設定ファイルの列範囲は、元のスクリプトの処理に基づいて設定されています
2. **閾値**: HQ-25+4の閾値は2に設定されています（1-2: 0, 3-5: 1）
3. **並列実行**: 各設定ファイルは並列実行に対応しており、4個のGPUデバイスを使用します
4. **データソース**: 全ての設定ファイルは`excel`データソースを使用します

## 出力ファイル

各実行により、以下のファイルが`outputs/`ディレクトリに生成されます：
- `{questionnaire_type}_{column_range}_h.csv`: エネルギーランドスケープの局所最小値
- `{questionnaire_type}_{column_range}_W.csv`: 重み行列
- `{questionnaire_type}_{column_range}_graph.csv`: ベイシングラフ
- `{questionnaire_type}_{column_range}_D.csv`: 切断グラフ
