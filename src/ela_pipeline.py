"""
ELA統合パイプライン

Energy Landscape Analysisの完全なパイプラインを実行します。
メンテナンス性とOSS対応を考慮した設計になっています。
"""

import os
import sys
import time
import argparse
import json
import pandas as pd
import numpy as np
import cupy as cp
import gc
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import pytz
from datetime import datetime
from contextlib import contextmanager

# パス設定（直接実行とモジュール実行の両方に対応）
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "energy_landscape"))

# インポート
from src.data_processing import DataProcessor
from src.logging_utils import ELALogger, create_logger_from_args
from src.execution_history import ExecutionHistoryManager, create_execution_summary
from energy_landscape.core import (
    calc_trans, calc_trans_fast, calc_basin_graph, calc_discon_graph
)
from energy_landscape.ela import (
    fit_approx, calc_accuracy, run_mcmc_simulation, calculate_energy,
    calculate_model_moments_mcmc, find_major_states, analyze_trajectory_dynamics,
    save_and_plot_dynamics
)
from energy_landscape.visualization import (
    plot_local_min, plot_full_basin_cugraph, plot_discon_graph
)


@contextmanager
def pandas_full_display():
    """pandasの表示設定を一時的に変更して完全なデータを表示/保存する"""
    # 現在の設定を保存
    old_max_rows = pd.get_option('display.max_rows')
    old_max_columns = pd.get_option('display.max_columns')
    old_width = pd.get_option('display.width')
    old_max_colwidth = pd.get_option('display.max_colwidth')
    
    try:
        # 完全表示のための設定
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        yield
    finally:
        # 元の設定に戻す
        pd.set_option('display.max_rows', old_max_rows)
        pd.set_option('display.max_columns', old_max_columns)
        pd.set_option('display.width', old_width)
        pd.set_option('display.max_colwidth', old_max_colwidth)


class ELAPipeline:
    """
    ELA統合パイプラインクラス
    
    設定可能なパラメータと柔軟な出力設定をサポートし、
    メンテナンス性と再利用性を向上させます。
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 output_dir: str = "outputs",
                 device_id: int = 0,
                 hq25_threshold: int = 3,
                 ips22_threshold: int = 4,
                 iat_threshold: int = 3,
                 tacs22_threshold: int = 4,
                 logger: Optional[ELALogger] = None):
        """
        ELAパイプラインの初期化
        
        Args:
            data_dir: データディレクトリパス（ELA_analysis内の相対パス）
            output_dir: 出力ディレクトリパス（ELA_analysis内の相対パス）
            device_id: CUDAデバイスID
            hq25_threshold: HQ-25の閾値（2または3）
            ips22_threshold: IPS-22の閾値
            iat_threshold: IATの閾値
            tacs22_threshold: TACS-22の閾値
            logger: ロガー（オプション）
        """
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.data_dir = ela_analysis_dir / data_dir
        self.output_dir = ela_analysis_dir / output_dir
        self.device_id = device_id
        self.hq25_threshold = hq25_threshold
        self.ips22_threshold = ips22_threshold
        self.iat_threshold = iat_threshold
        self.tacs22_threshold = tacs22_threshold
        self.logger = logger
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データプロセッサの初期化
        self.data_processor = DataProcessor(
            data_dir=str(self.data_dir),
            hq25_threshold=hq25_threshold,
            ips22_threshold=ips22_threshold,
            iat_threshold=iat_threshold,
            tacs22_threshold=tacs22_threshold
        )
        
        # 利用可能な質問票タイプ
        self.available_questionnaires = [
            "PHQ9+HQ25_social_integrated",
            "PHQ9+HQ25_isolation_integrated", 
            "PHQ9+HQ25_Isolation+social_integrated",
            "PHQ9_all",
            "HQ25_social_all",
            "HQ25_Isolation_all",
        ]
    
    def clear_memory(self):
        """CuPyメモリをクリア"""
        try:
            # CuPyメモリプールをクリア
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            # Pythonのガベージコレクション
            gc.collect()
            
            # より積極的なメモリクリア
            try:
                cp._default_memory_pool.free_all_blocks()
            except:
                pass
                
            print("Memory cleared successfully")
        except Exception as e:
            print(f"Warning: Memory clear failed: {e}")
    
    def load_and_preprocess_data(self, questionnaire_type: str, 
                               column_range: Optional[tuple] = None,
                               column_indices: Optional[List[int]] = None,
                               data_source: str = "auto",
                               time_period: Optional[str] = None) -> pd.DataFrame:
        """
        質問票データの読み込みと前処理
        
        Args:
            questionnaire_type: 質問票のタイプまたはファイル名（例：social_2006_thr2）
            column_range: 列範囲のタプル (start, end) または None
            column_indices: 特定の列インデックスのリスト または None
            data_source: データソース（"auto", "csv", "excel"）
            time_period: 年代（"2006", "2012", "2108", "2204"）- HQ-25+4の場合のみ
            
        Returns:
            処理済みのデータ
        """
        if self.logger:
            self.logger.log_step(f"Loading data for {questionnaire_type}")
            self.logger.log_step(f"Data source: {data_source}")
            if time_period:
                self.logger.log_step(f"Time period: {time_period}")
            if column_indices:
                self.logger.log_step(f"Column indices: {column_indices}")
            elif column_range:
                self.logger.log_step(f"Column range: {column_range[0]}-{column_range[1]}")
        else:
            print(f"Loading data for {questionnaire_type}...")
            print(f"Data source: {data_source}")
            if time_period:
                print(f"Time period: {time_period}")
            if column_indices:
                print(f"Column indices: {column_indices}")
            elif column_range:
                print(f"Column range: {column_range[0]}-{column_range[1]}")
        
        # ファイル名ベースの解析に対応
        if "_thr" in questionnaire_type:
            # ファイル名から直接CSVファイルを読み込み
            csv_file = self.data_dir / "processed" / questionnaire_type / f"{questionnaire_type}.csv"
            if not csv_file.exists():
                # 組み合わせデータの場合（例: emotional_PHQ-9_2006_thr2_thr3）
                if "_" in questionnaire_type and questionnaire_type.count("_") >= 3:
                    # 組み合わせデータは {questionnaire1}_{questionnaire2} フォルダに保存されている
                    # 例: emotional_PHQ-9_2006_thr2_thr3 -> emotional_PHQ-9 フォルダ
                    parts = questionnaire_type.split("_")
                    if len(parts) >= 2:
                        folder_name = f"{parts[0]}_{parts[1]}"  # emotional_PHQ-9
                        csv_file = self.data_dir / "processed" / folder_name / f"{questionnaire_type}.csv"
                # カテゴリ別ファイルの場合
                elif questionnaire_type.startswith(("social_", "isolation_", "emotional_")):
                    csv_file = self.data_dir / "processed" / "HQ25+4" / f"{questionnaire_type}.csv"
                else:
                    # 質問票別ファイルの場合
                    questionnaire_name = questionnaire_type.split("_")[0]
                    csv_file = self.data_dir / "processed" / questionnaire_name / f"{questionnaire_type}.csv"
            
            if not csv_file.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
            # CSVファイルを直接読み込み（質問文を列名として保持）
            data = pd.read_csv(csv_file)
            data = data.fillna(0).astype(int)
            
            if self.logger:
                self.logger.log_step(f"Loaded CSV file: {csv_file}")
                self.logger.log_step(f"Data shape: {data.shape}")
            else:
                print(f"Loaded CSV file: {csv_file}")
                print(f"Data shape: {data.shape}")
            
            return data
        
        # データの読み込みと前処理
        # PHQ-9の場合は年代別分割処理を使用
        if questionnaire_type == "PHQ-9" and data_source in ["auto", "excel"] and column_range:
            # 列範囲から年代を推定
            time_period_map = {
                (0, 9): "2006",
                (9, 18): "2012", 
                (18, 27): "2108",
                (27, 36): "2204"
            }
            time_period = time_period_map.get(column_range)
            if time_period:
                data = self.data_processor.process_questionnaire_data_by_time_period(
                    questionnaire_type=questionnaire_type,
                    time_period=time_period,
                    data_source=data_source
                )
            else:
                # 通常の処理
                data = self.data_processor.process_questionnaire_data(
                    questionnaire_type=questionnaire_type,
                    data_source=data_source,
                    column_range=column_range,
                    column_indices=column_indices,
                    time_period=time_period
                )
        else:
            # 通常の処理
            data = self.data_processor.process_questionnaire_data(
                questionnaire_type=questionnaire_type,
                data_source=data_source,
                column_range=column_range,
                column_indices=column_indices,
                time_period=time_period
            )
        
        # デバッグ: 元データの情報
        print(f"DEBUG: 元データ形状: {data.shape}")
        print(f"DEBUG: 元データの型: {type(data)}")
        print(f"DEBUG: 元データの列名: {list(data.columns)[:5]}...")  # 最初の5列のみ表示
        
        # 転置してデータを準備
        data_transposed = data.T.copy()
        print(f"DEBUG: 転置後形状: {data_transposed.shape}")
        
        # 列名をリセットしてからDataFrameに変換
        data_transposed.columns = range(len(data_transposed.columns))
        data_transposed = pd.DataFrame(np.array(data_transposed))
        print(f"DEBUG: DataFrame変換後形状: {data_transposed.shape}")
        
        # NaNの確認
        nan_count = data_transposed.isnull().sum().sum()
        print(f"DEBUG: NaNの数: {nan_count}")
        
        data_transposed = data_transposed.dropna(axis=1)
        print(f"DEBUG: NaN削除後形状: {data_transposed.shape}")
        
        data_transposed = data_transposed.astype(int)
        print(f"DEBUG: 最終データ形状: {data_transposed.shape}")
        print(f"DEBUG: 最終データの型: {data_transposed.dtypes.iloc[0] if len(data_transposed) > 0 else 'Empty'}")
        
        # データが空でないかチェック
        if len(data_transposed) == 0:
            print("ERROR: データが空です！")
            raise ValueError("データが空です。前処理を確認してください。")
        elif len(data_transposed) == 1:
            print("WARNING: データが1行のみです。クラスタリングができない可能性があります。")
        
        if self.logger:
            self.logger.log_step(f"Data loaded: {data_transposed.shape}")
        else:
            print(f"Data loaded: {data_transposed.shape}")
        return data_transposed
    
    def run_energy_landscape_analysis(self, 
                                    data: pd.DataFrame, 
                                    filename: str,
                                    output_dir: Optional[Path] = None,
                                    figs_dir: Optional[Path] = None,
                                    csvs_dir: Optional[Path] = None,
                                    enable_mcmc: bool = False,
                                    mcmc_steps: int = 100000,
                                    mcmc_burn_in: float = 0.001,
                                    num_major_states: int = 4,
                                    mcmc_temperature: float = 3.0) -> Dict[str, Any]:
        """
        エネルギーランドスケープ解析の実行
        
        Args:
            data: 処理済みデータ
            filename: 出力ファイル名
            output_dir: 出力ディレクトリ（指定しない場合はデフォルトのoutput_dirを使用）
            figs_dir: 図の出力ディレクトリ
            csvs_dir: CSVの出力ディレクトリ
            enable_mcmc: MCMCシミュレーションを有効にするかどうか
            mcmc_steps: MCMCシミュレーションのステップ数
            mcmc_burn_in: MCMCのバーンイン期間の割合
            num_major_states: 主要状態の数
            
        Returns:
            解析結果の辞書
        """
        # 出力ディレクトリの設定
        if output_dir is None:
            output_dir = self.output_dir
        print("Starting Energy Landscape Analysis...")
        start_time = time.time()
        
        # タイムスタンプの記録
        start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
        print(f"Start Approximation: {start_now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        with cp.cuda.Device(self.device_id):
            # デバッグ: 解析開始時のデータ情報
            print(f"DEBUG: 解析開始 - データ形状: {data.shape}")
            print(f"DEBUG: データの値の範囲: {data.min().min()} ～ {data.max().max()}")
            print(f"DEBUG: データの一意値数: {data.nunique().sum()}")
            
            # 1. 近似フィッティング
            print("Fitting approximation...")
            try:
                h, W = fit_approx(data)
                print(f"DEBUG: フィッティング成功 - h形状: {h.shape}, W形状: {W.shape}")
                print(f"Fit Approximation: {time.time() - start_time:.2f}s")
            except Exception as e:
                print(f"ERROR: フィッティング失敗: {e}")
                raise
            
            ## Accuracy計算
            print("=== Accuracy計算開始 ===")
            print(f"データ形状: {data.shape}")
            print(f"hパラメータ形状: {h.shape}")
            print(f"Wパラメータ形状: {W.shape}")
            print(f"hの範囲: {h.min():.6f} ～ {h.max():.6f}")
            print(f"Wの範囲: {W.min():.6f} ～ {W.max():.6f}")
            
            # ログファイルに詳細情報を出力
            if self.logger:
                accuracy_params = {
                    "データ形状": data.shape,
                    "hパラメータ形状": h.shape,
                    "Wパラメータ形状": W.shape,
                    "h最小値": h.min(),
                    "h最大値": h.max(),
                    "W最小値": W.min(),
                    "W最大値": W.max()
                }
                self.logger.log_calculation_details("Accuracy計算開始", accuracy_params)
            
            acc1, acc2 = calc_accuracy(h, W, data, self.logger)
            
            print(f"Calculate Accuracy: {time.time() - start_time:.2f}s")
            print(f"=== Accuracy結果 ===")
            print(f"Accuracy 1: {acc1:.6f}")
            print(f"Accuracy 2: {acc2:.6f}")
            print(f"Accuracy 1 (パーセント): {acc1 * 100:.2f}%")
            print(f"Accuracy 2 (パーセント): {acc2 * 100:.2f}%")
            
            # ログファイルに結果を出力
            if self.logger:
                accuracy_results = {
                    "Accuracy1": acc1,
                    "Accuracy2": acc2,
                    "Accuracy1_パーセント": acc1 * 100,
                    "Accuracy2_パーセント": acc2 * 100
                }
                self.logger.log_calculation_details("Accuracy結果", accuracy_results)
            
            results['accuracy1'] = acc1
            results['accuracy2'] = acc2
            
            # 結果の保存（NumPy配列をDataFrameに変換）
            h_df = pd.DataFrame(h, index=data.columns, columns=['h'])
            
            # Wが正方行列であることを確認
            if W.shape[0] != W.shape[1]:
                raise ValueError(f"W must be a square matrix, but got shape {W.shape}")
            
            W_df = pd.DataFrame(W, index=data.columns, columns=data.columns)
            
            # 出力ディレクトリの存在確認と作成
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 完全なデータを保存するためにpandasの表示設定を一時的に変更
            with pandas_full_display():
                h_df.to_csv(output_dir / f"{filename}_h.csv", encoding='utf-8-sig')
                W_df.to_csv(output_dir / f"{filename}_W.csv", encoding='utf-8-sig')
            results['h'] = h_df
            results['W'] = W_df
            
            # 2. Basin graphの計算
            print("=== Basin Graph計算開始 ===")
            print(f"データ形状: {data.shape}")
            print(f"hパラメータ: 平均={h.mean():.6f}, 標準偏差={h.std():.6f}")
            print(f"Wパラメータ: 平均={W.mean():.6f}, 標準偏差={W.std():.6f}")
            
            # ログファイルに詳細情報を出力
            if self.logger:
                basin_params = {
                    "データ形状": data.shape,
                    "h平均": h.mean(),
                    "h標準偏差": h.std(),
                    "W平均": W.mean(),
                    "W標準偏差": W.std()
                }
                self.logger.log_calculation_details("Basin Graph計算開始", basin_params)
            
            try:
                graph = calc_basin_graph(h, W, data)
                print(f"Basin graph計算成功 - 形状: {graph.shape}")
                print(f"Basin graph列: {list(graph.columns)}")
                if 'energy' in graph.columns:
                    print(f"Energy範囲: {graph['energy'].min():.6f} ～ {graph['energy'].max():.6f}")
                if 'state_no' in graph.columns:
                    print(f"状態数: {graph['state_no'].nunique()}")
                
                # ログファイルに結果を出力
                if self.logger:
                    basin_results = {
                        "Basin_graph形状": graph.shape,
                        "Basin_graph列": list(graph.columns)
                    }
                    if 'energy' in graph.columns:
                        basin_results["Energy最小値"] = graph['energy'].min()
                        basin_results["Energy最大値"] = graph['energy'].max()
                    if 'state_no' in graph.columns:
                        basin_results["状態数"] = graph['state_no'].nunique()
                    self.logger.log_calculation_details("Basin Graph結果", basin_results)
                
                # 出力ディレクトリの存在確認と作成
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with pandas_full_display():
                    graph.to_csv(output_dir / f"{filename}_graph.csv", encoding='utf-8-sig')
                print(f"Calculate Basin Graph: {time.time() - start_time:.2f}s")
                # results['basin_graph'] = graph
            except Exception as e:
                print(f"ERROR: Basin graph計算失敗: {e}")
                print(f"DEBUG: エラー時のデータ形状: {data.shape}")
                if self.logger:
                    self.logger.log_error(f"Basin graph計算失敗: {e}")
                raise
            
            # 3. Disconnectivity graphの計算
            print("=== Disconnectivity Graph計算開始 ===")
            print(f"Basin graph形状: {graph.shape}")
            
            # ログファイルに詳細情報を出力
            if self.logger:
                discon_params = {
                    "Basin_graph形状": graph.shape
                }
                self.logger.log_calculation_details("Disconnectivity Graph計算開始", discon_params)
            
            D = calc_discon_graph(h, W, data, graph)
            print(f"Disconnectivity graph計算成功 - 形状: {D.shape}")
            print(f"距離行列の範囲: {D.min().min():.6f} ～ {D.max().max():.6f}")
            print(f"非ゼロ要素数: {(D != 0).sum().sum()}")
            
            # ログファイルに結果を出力
            if self.logger:
                discon_results = {
                    "Disconnectivity_graph形状": D.shape,
                    "距離行列最小値": D.min().min(),
                    "距離行列最大値": D.max().max(),
                    "非ゼロ要素数": (D != 0).sum().sum()
                }
                self.logger.log_calculation_details("Disconnectivity Graph結果", discon_results)
            
            # 出力ディレクトリの存在確認と作成
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with pandas_full_display():
                D.to_csv(output_dir / f"{filename}_D.csv", encoding='utf-8-sig')
            print(f"Calculate Disconnection Graph: {time.time() - start_time:.2f}s")
            results['disconnectivity_graph'] = D
            print(f"Plot Disconnection Graph: {time.time() - start_time:.2f}s")
            plot_discon_graph(D, save_fig=True, filename=filename)
            
            # 4. 遷移の計算
            print("=== 遷移計算開始 ===")
            print(f"データ形状: {data.shape}")
            print(f"Basin graph形状: {graph.shape}")
            
            # ログファイルに詳細情報を出力
            if self.logger:
                trans_params = {
                    "データ形状": data.shape,
                    "Basin_graph形状": graph.shape
                }
                self.logger.log_calculation_details("遷移計算開始", trans_params)
            
            freq, trans, trans2 = calc_trans(data, graph)
            print(f"遷移計算成功:")
            print(f"  Frequency形状: {freq.shape}")
            print(f"  Transition形状: {trans.shape}")
            print(f"  Transition2形状: {trans2.shape}")
            if not freq.empty:
                print(f"  Frequency範囲: {freq.min():.6f} ～ {freq.max():.6f}")
            if not trans.empty:
                print(f"  Transition範囲: {trans.min().min():.6f} ～ {trans.max().max():.6f}")
            
            # ログファイルに結果を出力
            if self.logger:
                trans_results = {
                    "Frequency形状": freq.shape,
                    "Transition形状": trans.shape,
                    "Transition2形状": trans2.shape
                }
                if not freq.empty:
                    trans_results["Frequency最小値"] = freq.min()
                    trans_results["Frequency最大値"] = freq.max()
                if not trans.empty:
                    trans_results["Transition最小値"] = trans.min().min()
                    trans_results["Transition最大値"] = trans.max().max()
                self.logger.log_calculation_details("遷移計算結果", trans_results)
            
            print(f"Calculate Transition Graph: {time.time() - start_time:.2f}s")
            results['frequency'] = freq
            # results['transition'] = trans
            # results['transition2'] = trans2
            
            # 5. MCMCシミュレーション（オプション）
            if enable_mcmc:
                print("=== MCMCシミュレーション開始 ===")
                print(f"MCMCパラメータ: ステップ数={mcmc_steps}, バーンイン={mcmc_burn_in}, 主要状態数={num_major_states}")
                
                # ログファイルに詳細情報を出力
                if self.logger:
                    mcmc_params = {
                        "MCMCステップ数": mcmc_steps,
                        "バーンイン期間": mcmc_burn_in,
                        "主要状態数": num_major_states,
                        "データ形状": data.shape
                    }
                    self.logger.log_calculation_details("MCMCシミュレーション開始", mcmc_params)
                
                try:
                    # MCMCシミュレーションの実行
                    print("  - MCMCシミュレーション実行中...")
                    trajectory = run_mcmc_simulation(h, W, n_steps=mcmc_steps, burn_in=mcmc_burn_in, temperature=mcmc_temperature)
                    print(f"  - 軌跡生成完了: 形状={trajectory.shape}")
                    
                    # モデルモーメントの計算
                    print("  - モデルモーメント計算中...")
                    model_means, model_correlations = calculate_model_moments_mcmc(h, W, n_samples=mcmc_steps//2)
                    print(f"  - モデル平均: {model_means}")
                    
                    # 主要状態の特定
                    print("  - 主要状態特定中...")
                    major_states = find_major_states(trajectory, num_major_states=num_major_states)
                    
                    # 軌跡ダイナミクスの解析
                    print("  - 軌跡ダイナミクス解析中...")
                    sojourn_probabilities, transition_matrix = analyze_trajectory_dynamics(trajectory, major_states)
                    
                    # 結果の保存と可視化
                    print("  - MCMC結果保存・可視化中...")
                    mcmc_output_dir = output_dir / "mcmc_results"
                    save_and_plot_dynamics(sojourn_probabilities, transition_matrix, major_states, str(mcmc_output_dir))
                    
                    # 結果を辞書に追加
                    # results['mcmc_trajectory'] = trajectory
                    results['mcmc_model_means'] = model_means
                    results['mcmc_model_correlations'] = model_correlations
                    results['mcmc_major_states'] = major_states
                    results['mcmc_sojourn_probabilities'] = sojourn_probabilities
                    results['mcmc_transition_matrix'] = transition_matrix
                    
                    # ログファイルに結果を出力
                    if self.logger:
                        mcmc_results = {
                            "軌跡形状": trajectory.shape,
                            "主要状態数": len(major_states),
                            "滞在確率範囲": f"{sojourn_probabilities.min():.6f} ～ {sojourn_probabilities.max():.6f}",
                            "遷移行列非ゼロ要素数": (transition_matrix > 0).sum()
                        }
                        self.logger.log_calculation_details("MCMCシミュレーション結果", mcmc_results)
                    
                    print(f"MCMCシミュレーション完了: {time.time() - start_time:.2f}s")
                    
                except Exception as e:
                    print(f"ERROR: MCMCシミュレーション失敗: {e}")
                    if self.logger:
                        self.logger.log_error(f"MCMCシミュレーション失敗: {e}")
                    print("WARNING: MCMCシミュレーションをスキップして続行します。")
            
            ### 6. 可視化
            print("Generating visualizations...")
            try:
                print(f"DEBUG: 可視化開始 - データ形状: {data.shape}, graph形状: {graph.shape}")
                self._generate_visualizations(data, graph, D, freq, trans, trans2, filename, figs_dir, csvs_dir)
                print(f"Plot Local Min: {time.time() - start_time:.2f}s")
            except Exception as e:
                print(f"ERROR: 可視化失敗: {e}")
                print(f"DEBUG: エラー時のデータ形状: {data.shape}")
                # 可視化エラーは解析を停止させない
                print("WARNING: 可視化をスキップして続行します。")
        
        results['elapsed_time'] = time.time() - start_time
        return results
    
    def _generate_visualizations(self, 
                               data: pd.DataFrame,
                               graph: pd.DataFrame,
                               D: pd.DataFrame,
                               freq: pd.DataFrame,
                               trans: pd.DataFrame,
                               trans2: pd.DataFrame,
                               filename: str,
                               figs_dir: Optional[Path] = None,
                               csvs_dir: Optional[Path] = None):
        """
        可視化の生成
        
        Args:
            data: データ
            graph: Basin graph
            D: Disconnectivity graph
            freq: 頻度
            trans: 遷移
            trans2: 遷移2
            filename: ファイル名
            figs_dir: 図の出力ディレクトリ
            csvs_dir: CSVの出力ディレクトリ
        """
        try:
            print("Generating visualizations...")
            
            # Local minimumのプロット（binary_matrixの画像出力を含む）
            print("  - Plotting local minimum (binary matrix)...")
            plot_local_min(data, graph, save_fig=True, filename=filename, output_dir=figs_dir, csvs_dir=csvs_dir)
            
            # Full basin graphのプロット
            # print("  - Plotting full basin graph...")
            # plot_full_basin_cugraph(graph, save_fig=True, filename=filename, output_dir=figs_dir)
            
            print("Visualization generation completed.")
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def run_pipeline(self, questionnaire_type: str, 
                    column_range: Optional[tuple] = None,
                    column_indices: Optional[List[int]] = None,
                    data_source: str = "auto",
                    time_period: Optional[str] = None,
                    record_history: bool = True,
                    enable_mcmc: bool = False,
                    mcmc_steps: int = 100000,
                    mcmc_burn_in: float = 0.001,
                    mcmc_temperature: float = 3.0,
                    num_major_states: int = 4) -> Dict[str, Any]:
        """
        完全なELAパイプラインの実行
        
        Args:
            questionnaire_type: 質問票のタイプ
            column_range: 列範囲のタプル (start, end) または None
            column_indices: 特定の列インデックスのリスト または None
            data_source: データソース（"auto", "csv", "excel"）
            time_period: 年代（"2006", "2012", "2108", "2204"）- HQ-25+4の場合のみ
            record_history: 実行履歴を記録するかどうか
            enable_mcmc: MCMCシミュレーションを有効にするかどうか
            mcmc_steps: MCMCシミュレーションのステップ数
            mcmc_burn_in: MCMCのバーンイン期間の割合
            num_major_states: 主要状態の数
            
        Returns:
            解析結果の辞書
        """
        # ファイル名ベースの解析の場合はチェックをスキップ
        if "_thr" not in questionnaire_type:
            # 利用可能な質問票タイプを動的に取得
            available_types = list(self.data_processor.get_preprocessing_function.__code__.co_consts)
            if questionnaire_type not in self.data_processor.get_preprocessing_function.__code__.co_names:
                # より柔軟なチェック
                try:
                    self.data_processor.get_preprocessing_function(questionnaire_type)
                except ValueError as e:
                    raise ValueError(f"Unknown questionnaire type: {questionnaire_type}. {e}")
        
        # 実行履歴の記録開始
        execution_id = None
        history_manager = None
        if record_history:
            history_manager = ExecutionHistoryManager()
            command = f"python -m src.ela_pipeline --func {questionnaire_type} --data_source {data_source}"
            if column_range:
                command += f" --r1 {column_range[0]} --r2 {column_range[1]}"
            if self.hq25_threshold != 3:
                command += f" --hq25_threshold {self.hq25_threshold}"
            
            parameters = {
                'func': questionnaire_type,
                'data_source': data_source,
                'hq25_threshold': self.hq25_threshold,
                'ips22_threshold': self.ips22_threshold,
                'iat_threshold': self.iat_threshold,
                'tacs22_threshold': self.tacs22_threshold,
                'device_id': self.device_id,
                'output_dir': str(self.output_dir),
                'column_range': column_range
            }
            
            execution_id = history_manager.start_execution(command, parameters)
            print(f"実行ID: {execution_id}")
        
        # メモリクリア
        self.clear_memory()
        
        print(f"Starting ELA pipeline for {questionnaire_type}")
        print(f"Data source: {data_source}")
        print(f"HQ-25 threshold: {self.hq25_threshold}")
        print(f"Output directory: {self.output_dir}")
        if column_range:
            print(f"Column range: {column_range[0]}-{column_range[1]}")
        
        # データの読み込みと前処理
        data = self.load_and_preprocess_data(questionnaire_type, column_range, column_indices, data_source, time_period)
        
        # 出力ディレクトリの設定（figs/とcsvs/に分ける）
        figs_dir = self.output_dir / "figs"
        csvs_dir = self.output_dir / "csvs" / questionnaire_type
        figs_dir.mkdir(parents=True, exist_ok=True)
        csvs_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名の生成（列範囲または列インデックスを含む）
        filename = questionnaire_type
        if column_indices:
            filename += f"_cols_{'_'.join(map(str, column_indices))}"
        elif column_range:
            filename += f"_{column_range[0]}_{column_range[1]}"
        
        try:
            # エネルギーランドスケープ解析の実行（csvs/ディレクトリを使用）
            results = self.run_energy_landscape_analysis(
                data, filename, output_dir=csvs_dir, figs_dir=figs_dir, csvs_dir=csvs_dir,
                enable_mcmc=enable_mcmc, mcmc_steps=mcmc_steps, mcmc_burn_in=mcmc_burn_in, mcmc_temperature=mcmc_temperature, num_major_states=num_major_states
            )
            
            print(f"Pipeline completed in {results['elapsed_time']:.2f} seconds")
            
            # 実行履歴の記録完了
            if record_history and execution_id and history_manager:
                try:
                    results_summary = create_execution_summary(execution_id, results, self.output_dir)
                    
                    # 出力ファイルの検索
                    output_files = []
                    if csvs_dir.exists():
                        for file_path in csvs_dir.rglob("*"):
                            if file_path.is_file() and filename in file_path.name:
                                output_files.append(str(file_path.relative_to(self.output_dir.parent)))
                    
                    history_manager.complete_execution(
                        execution_id=execution_id,
                        status='success',
                        duration_seconds=results['elapsed_time'],
                        output_files=output_files,
                        results_summary=results_summary,
                        notes=f"正常完了: {questionnaire_type}解析"
                    )
                    print(f"実行履歴を記録しました (ID: {execution_id})")
                except Exception as e:
                    print(f"Warning: 実行履歴の記録に失敗しました: {e}")
            
            # 解析完了後のメモリクリア
            self.clear_memory()
            
            return results
            
        except Exception as e:
            # エラー時の実行履歴記録
            if record_history and execution_id and history_manager:
                try:
                    history_manager.complete_execution(
                        execution_id=execution_id,
                        status='failed',
                        duration_seconds=0.0,
                        output_files=[],
                        results_summary={},
                        error_message=str(e),
                        notes=f"エラーで終了: {questionnaire_type}解析"
                    )
                    print(f"エラーを実行履歴に記録しました (ID: {execution_id})")
                except Exception as history_error:
                    print(f"Warning: エラー履歴の記録に失敗しました: {history_error}")
            
            # エラー時のメモリクリア
            self.clear_memory()
            
            raise e


def create_argument_parser() -> argparse.ArgumentParser:
    """
    コマンドライン引数パーサーの作成
    
    Returns:
        設定済みのArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Energy Landscape Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ela_pipeline.py --func PHQ9_all --hq25_threshold 3
  python ela_pipeline.py --func HQ25_social_all --hq25_threshold 2 --device_id 1
  python ela_pipeline.py --func PHQ9+HQ25_social_integrated --output_dir ./results
  python ela_pipeline.py --config ela_config.json --func PHQ9_all
  python ela_pipeline.py --func PHQ9_all --hq25_threshold 2 --verbose
  python ela_pipeline.py --func PHQ9_all --enable_mcmc --mcmc_steps 20000
  python ela_pipeline.py --func PHQ9_all --enable_mcmc --mcmc_burn_in 0.3 --num_major_states 10
        """
    )
    
    # 設定ファイル
    parser.add_argument(
        "--config",
        type=str,
        help="設定ファイルのパス (JSONまたはYAML形式)"
    )
    
    # 必須引数
    parser.add_argument(
        "--func",
        required=True,
        help="質問票のタイプを選択（例: PHQ-9, IPS-22, IAT, HQ-25+4, TACS-22, PHQ9+HQ25_social_2308等）"
    )
    
    # 列範囲指定
    parser.add_argument(
        "--r1",
        type=int,
        default=0,
        help="データ列範囲の開始位置 (デフォルト: 0)"
    )
    
    parser.add_argument(
        "--r2",
        type=int,
        help="データ列範囲の終了位置 (指定しない場合は全列を使用)"
    )
    
    parser.add_argument(
        "--column_indices",
        type=int,
        nargs="+",
        help="特定の列インデックスのリスト (例: --column_indices 1 4 6 8)"
    )
    
    parser.add_argument(
        "--time_period",
        type=str,
        choices=["2006", "2012", "2108", "2204"],
        help="年代指定 (HQ-25+4の場合のみ有効)"
    )
    
    # オプション引数
    parser.add_argument(
        "--hq25_threshold",
        type=int,
        choices=[2, 3],
        help="HQ-25の閾値 (2または3, デフォルト: 3)"
    )
    
    parser.add_argument(
        "--ips22_threshold",
        type=int,
        help="IPS-22の閾値 (デフォルト: 4)"
    )
    
    parser.add_argument(
        "--iat_threshold",
        type=int,
        help="IATの閾値 (デフォルト: 3)"
    )
    
    parser.add_argument(
        "--tacs22_threshold",
        type=int,
        help="TACS-22の閾値 (デフォルト: 4)"
    )
    
    parser.add_argument(
        "--device_id",
        type=int,
        help="CUDAデバイスID (デフォルト: 0)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="データディレクトリパス (デフォルト: data)"
    )
    
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["auto", "csv", "excel"],
        default="auto",
        help="データソース (auto: 自動判別, csv: 処理済みCSV, excel: オリジナルExcel) (デフォルト: auto)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="出力ディレクトリパス (デフォルト: outputs)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細な出力を有効にする"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="ログディレクトリ (デフォルト: logs)"
    )
    
    parser.add_argument(
        "--create_config",
        type=str,
        metavar="CONFIG_FILE",
        help="例示用設定ファイルを作成して終了"
    )
    
    # MCMCオプション
    parser.add_argument(
        "--enable_mcmc",
        action="store_true",
        help="MCMCシミュレーションを有効にする"
    )
    
    parser.add_argument(
        "--mcmc_steps",
        type=int,
        default=100000,
        help="MCMCシミュレーションのステップ数 (デフォルト: 100000)"
    )
    
    parser.add_argument(
        "--mcmc_burn_in",
        type=float,
        default=0.001,
        help="MCMCのバーンイン期間の割合 (デフォルト: 0.001, 論文設定: 100/100000)"
    )
    
    parser.add_argument(
        "--mcmc_temperature",
        type=float,
        default=1.0,
        help="MCMCの温度パラメータ (デフォルト: 1.0, エネルギー差が大きい場合の推奨値)"
    )
    
    parser.add_argument(
        "--num_major_states",
        type=int,
        default=4,
        help="主要状態の数 (デフォルト: 4, 論文設定: 2つの主要状態と2つの副次状態)"
    )
    
    return parser


def main():
    """
    メイン関数
    
    コマンドライン引数を解析し、ELAパイプラインを実行します。
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # 設定ファイル作成モード
        if args.create_config:
            from .config import create_example_config_file
            create_example_config_file(args.create_config)
            print(f"Example config file created: {args.create_config}")
            return
        
        # 設定の読み込み
        config = None
        if args.config:
            from .config import ConfigManager
            config_manager = ConfigManager(args.config)
            config = config_manager.load_config()
        
        # パラメータの設定（コマンドライン引数が優先）
        data_dir = args.data_dir or (config.data.data_dir if config else "data")
        output_dir = args.output_dir or (config.data.output_dir if config else "outputs")
        data_source = args.data_source or "auto"
        device_id = args.device_id if args.device_id is not None else (config.processing.device_id if config else 0)
        hq25_threshold = args.hq25_threshold or (config.processing.hq25_threshold if config else 3)
        ips22_threshold = args.ips22_threshold or 4
        iat_threshold = args.iat_threshold or 3
        tacs22_threshold = args.tacs22_threshold or 4
        verbose = args.verbose or (config.processing.verbose if config else False)
        
        # ロガーの設定
        logger = create_logger_from_args(args, job_type="normal")
        logger.log_start()
        
        # 列選択の設定
        column_range = None
        column_indices = None
        time_period = args.time_period
        
        if args.column_indices:
            column_indices = args.column_indices
        elif args.r2 is not None:
            column_range = (args.r1, args.r2)
        elif args.r1 != 0:
            # r1が指定されているがr2が指定されていない場合は、r1から最後まで
            column_range = (args.r1, None)
        
        # ELAパイプラインの初期化
        pipeline = ELAPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            device_id=device_id,
            hq25_threshold=hq25_threshold,
            ips22_threshold=ips22_threshold,
            iat_threshold=iat_threshold,
            tacs22_threshold=tacs22_threshold,
            logger=logger
        )
        
        # MCMCパラメータの設定
        enable_mcmc = args.enable_mcmc
        mcmc_steps = args.mcmc_steps
        mcmc_burn_in = args.mcmc_burn_in
        mcmc_temperature = args.mcmc_temperature
        num_major_states = args.num_major_states
        
        # パイプラインの実行
        results = pipeline.run_pipeline(
            args.func, column_range, column_indices, data_source, time_period,
            enable_mcmc=enable_mcmc, mcmc_steps=mcmc_steps, 
            mcmc_burn_in=mcmc_burn_in, mcmc_temperature=mcmc_temperature, num_major_states=num_major_states
        )
        
        # 結果の記録
        logger.log_completion(results)
        logger.save_results_summary(results, args)
        
        # 結果をJSON形式で保存
        results_json = {}
        for key, value in results.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                # DataFrame/Seriesは辞書形式に変換
                results_json[key] = {
                    'type': 'DataFrame' if isinstance(value, pd.DataFrame) else 'Series',
                    'shape': value.shape,
                    'data': value.to_dict() if isinstance(value, pd.Series) else value.to_dict('index')
                }
            elif isinstance(value, np.ndarray):
                # NumPy配列はリストに変換
                results_json[key] = {
                    'type': 'ndarray',
                    'shape': value.shape,
                    'data': value.tolist()
                }
            else:
                # その他の型はそのまま
                results_json[key] = value
        
        # JSONファイルに保存
        output_path = Path(output_dir) / f"csvs/{args.func}/{args.func}_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)
        
        if verbose:
            print("\n=== Results Summary ===")
            print(f"Questionnaire: {args.func}")
            print(f"HQ-25 Threshold: {hq25_threshold}")
            print(f"Device ID: {device_id}")
            print(f"Data Directory: {data_dir}")
            print(f"Output Directory: {output_dir}")
            print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")
            
        
        print("Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
