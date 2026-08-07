# MCMCシミュレーション機能統合ガイド

## 概要

ELAパイプラインにMCMC（Markov Chain Monte Carlo）シミュレーション機能が統合されました。これにより、エネルギーランドスケープの動的解析が可能になります。

## 新機能

### 1. MCMCシミュレーション
- **メトロポリス・ヘイスティングス法**による状態サンプリング
- エネルギーランドスケープ上の状態遷移軌跡の生成
- バーンイン期間の設定による初期状態の影響除去

### 2. 軌跡解析
- **主要状態の特定**: 出現頻度の高い状態を自動検出
- **滞在確率の計算**: 各主要状態での滞在時間の割合
- **遷移頻度の分析**: 状態間の遷移パターンの定量化

### 3. 可視化機能
- **滞在確率の棒グラフ**: 主要状態の滞在確率を可視化
- **遷移ネットワーク図**: 状態間の遷移関係をネットワーク図で表示
- **CSV出力**: 解析結果を構造化データとして保存

## 使用方法

### 基本的な使用方法

```bash
# MCMCシミュレーションを有効にして解析実行
python -m src.ela_pipeline --func PHQ-9_2204_thr2 --enable_mcmc

# より詳細な設定でMCMC解析
python -m src.ela_pipeline --func PHQ-9_2204_thr2 \
    --enable_mcmc \
    --mcmc_steps 20000 \
    --mcmc_burn_in 0.3 \
    --num_major_states 10
```

### コマンドラインオプション

| オプション | 説明 | デフォルト値 |
|-----------|------|-------------|
| `--enable_mcmc` | MCMCシミュレーションを有効にする | False |
| `--mcmc_steps` | MCMCシミュレーションのステップ数 | 10000 |
| `--mcmc_burn_in` | バーンイン期間の割合 (0.0-1.0) | 0.2 |
| `--num_major_states` | 主要状態の数 | 5 |

### 使用例

#### 1. 基本的なMCMC解析
```bash
python -m src.ela_pipeline --func PHQ-9_2204_thr2 --enable_mcmc
```

#### 2. 高精度MCMC解析
```bash
python -m src.ela_pipeline --func PHQ-9_2204_thr2 \
    --enable_mcmc \
    --mcmc_steps 50000 \
    --mcmc_burn_in 0.1 \
    --num_major_states 8
```

#### 3. 小規模テスト
```bash
python -m src.ela_pipeline --func PHQ-9_2204_thr2 \
    --enable_mcmc \
    --mcmc_steps 1000 \
    --mcmc_burn_in 0.1 \
    --num_major_states 3
```

## 出力ファイル

MCMC解析が有効な場合、以下のファイルが生成されます：

### CSVファイル
- `sojourn_probabilities.csv`: 主要状態の滞在確率
- `transition_matrix.csv`: 状態間の遷移頻度行列

### 可視化ファイル
- `sojourn_probabilities.png`: 滞在確率の棒グラフ
- `transition_network.png`: 遷移ネットワーク図

### 出力ディレクトリ構造
```
outputs/
└── csvs/
    └── [questionnaire_type]/
        ├── [filename]_h.csv          # 閾値パラメータ
        ├── [filename]_W.csv          # 相互作用パラメータ
        ├── [filename]_graph.csv      # Basin graph
        ├── [filename]_D.csv          # Disconnectivity graph
        └── mcmc_results/             # MCMC解析結果
            ├── sojourn_probabilities.csv
            ├── transition_matrix.csv
            ├── sojourn_probabilities.png
            └── transition_network.png
```

## パラメータの推奨設定

### ステップ数 (`--mcmc_steps`)
- **小規模テスト**: 1,000-5,000
- **通常解析**: 10,000-20,000
- **高精度解析**: 50,000以上

### バーンイン期間 (`--mcmc_burn_in`)
- **短時間**: 0.1 (10%)
- **標準**: 0.2 (20%)
- **長時間**: 0.3-0.5 (30-50%)

### 主要状態数 (`--num_major_states`)
- **小規模データ**: 3-5
- **中規模データ**: 5-10
- **大規模データ**: 10-20

## 注意事項

1. **計算時間**: MCMCシミュレーションは計算集約的な処理です。ステップ数を増やすと計算時間が大幅に増加します。

2. **メモリ使用量**: 軌跡データはメモリに保存されるため、大きなステップ数ではメモリ不足に注意してください。

3. **データサイズ**: 次元数が大きいデータセットでは、MCMCシミュレーションに時間がかかる可能性があります。

4. **収束性**: 十分なステップ数とバーンイン期間を設定して、MCMCチェーンの収束を確保してください。

## トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー
```bash
# ステップ数を減らす
--mcmc_steps 5000
```

#### 2. 計算時間が長すぎる
```bash
# バーンイン期間を短くする
--mcmc_burn_in 0.1
```

#### 3. 主要状態が見つからない
```bash
# 主要状態数を減らす
--num_major_states 3
```

## テスト実行

統合されたMCMC機能をテストするには：

```bash
# テストスクリプトを実行
python test_mcmc_pipeline.py

# 改良されたMCMC例（パラメータ推定機能付き）
python example_enhanced_mcmc.py
```

## 技術的詳細

### 実装された関数

#### 基本MCMC機能
- `run_mcmc_simulation()`: MCMCシミュレーションの実行（進捗バー付き）
- `calc_energy_mcmc()`: MCMC用エネルギー計算
- `calculate_model_moments_mcmc()`: モデルモーメントのMCMC近似
- `find_major_states()`: 主要状態の特定
- `analyze_trajectory_dynamics()`: 軌跡ダイナミクス解析
- `save_and_plot_dynamics()`: 結果保存と可視化

#### パラメータ推定機能（新規追加）
- `calculate_empirical_moments()`: データから経験的モーメントを計算
- `calculate_model_moments_exact_enhanced()`: 厳密計算によるモデルモーメント
- `calculate_model_moments_mcmc_enhanced()`: MCMC近似によるモデルモーメント
- `estimate_parameters_exact()`: 厳密計算によるパラメータ推定
- `estimate_parameters_mcmc()`: MCMC近似によるパラメータ推定
- `prepare_data()`: データの準備と整形

### アルゴリズム
- **メトロポリス・ヘイスティングス法**: 状態提案と受容/棄却
- **単一スピンフリップ**: 1つの変数を反転させる提案
- **ボルツマン分布**: 温度T=1での平衡分布

## 今後の拡張予定

1. **温度スケジューリング**: アニーリング法の実装
2. **並列MCMC**: 複数チェーンの並列実行
3. **収束診断**: Gelman-Rubin統計量などの収束判定
4. **適応的サンプリング**: 効率的なサンプリング戦略

---

**作成日**: 2024年12月
**バージョン**: 1.0
**対応パイプライン**: ELA Pipeline v2.0+
