"""
固定W行列を使用したELA解析パイプライン

事前に学習されたW行列を使用して、新しいデータに対してELA解析を実行します。
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import cupy as cp
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pytz
from datetime import datetime
from contextlib import contextmanager

# 相対インポート（パッケージ内での使用）
try:
    from .data_processing import DataProcessor
    from .config import ELAConfig, ConfigManager
    from ..energy_landscape.core import (
        calc_trans, calc_trans_fast, calc_basin_graph, calc_discon_graph
    )
    from ..energy_landscape.ela import fit_approx
    from ..energy_landscape.visualization import (
        plot_local_min, plot_full_basin_cugraph
    )
except ImportError:
    # 直接実行時のフォールバック
    sys.path.append(str(Path(__file__).parent))
    sys.path.append(str(Path(__file__).parent.parent / "energy_landscape"))
    from data_processing import DataProcessor
    from config import ELAConfig, ConfigManager
    from core import calc_trans, calc_trans_fast, calc_basin_graph, calc_discon_graph
    from ela import fit_approx
    from visualization import plot_local_min, plot_full_basin_cugraph


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


class FixedWPipeline:
    """
    固定W行列を使用したELA解析パイプラインクラス
    
    事前に学習されたW行列を使用して、新しいデータに対してELA解析を実行します。
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 output_dir: str = "outputs/integrated",
                 device_id: int = 0,
                 hq25_threshold: int = 3):
        """
        固定Wパイプラインの初期化
        
        Args:
            data_dir: データファイルのディレクトリパス（ELA_analysis内の相対パス）
            output_dir: 出力ディレクトリパス（ELA_analysis内の相対パス）
            device_id: CUDAデバイスID
            hq25_threshold: HQ-25の閾値（2または3）
        """
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.data_dir = ela_analysis_dir / data_dir
        self.output_dir = ela_analysis_dir / output_dir
        self.device_id = device_id
        self.hq25_threshold = hq25_threshold
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データプロセッサの初期化
        self.data_processor = DataProcessor(
            data_dir=str(self.data_dir),
            hq25_threshold=hq25_threshold
        )
        
        # 固定W行列のパス
        self.fixed_w_paths = {
            'social': self.output_dir / "PHQ9+HQ25_social_integrated_fixedW.csv",
            'isolation': self.output_dir / "PHQ9+HQ25_isolation_integrated_fixedW.csv"
        }
    
    def load_fixed_w_matrix(self, w_type: str) -> pd.DataFrame:
        """
        固定W行列を読み込みます
        
        Args:
            w_type: W行列のタイプ（'social' または 'isolation'）
            
        Returns:
            固定W行列
        """
        if w_type not in self.fixed_w_paths:
            raise ValueError(f"Unknown W type: {w_type}. Available types: {list(self.fixed_w_paths.keys())}")
        
        w_path = self.fixed_w_paths[w_type]
        if not w_path.exists():
            raise FileNotFoundError(f"Fixed W matrix not found: {w_path}")
        
        return pd.read_csv(w_path, index_col=0)
    
    def fit_approx_fixed_w(self, data: pd.DataFrame, W: pd.DataFrame, 
                          max_iter: int = 1000, alpha: float = 0.9) -> tuple:
        """
        固定W行列を使用して近似フィッティングを実行します
        
        Args:
            data: 入力データ
            W: 固定W行列
            max_iter: 最大反復回数
            alpha: 学習率
            
        Returns:
            (h, W) のタプル
        """
        X = 2 * data - 1  # Convert {0,1} to {-1,1}
        n, k = X.shape
        h = np.zeros(n)
        
        X_mean = X.mean(axis=1)
        X_corr = X.dot(X.T) / k
        np.fill_diagonal(X_corr.values, 0)
        
        for _ in range(max_iter):
            # Fix: W.dot(X) not W.dot(X).T, and h[:, np.newaxis] to broadcast
            Y = np.tanh(W.values.dot(X) + h[:, np.newaxis])  # shape: (n, k)
            
            h += alpha * (X_mean - Y.mean(axis=1))
            
            Z_arr = (X.dot(Y.T) / k + X.dot(Y.T).T / k) / 2
            Z = pd.DataFrame(Z_arr, index=X.index, columns=X.index)
            np.fill_diagonal(Z.values, 0)
            
            if np.allclose(X_mean, Y.mean(axis=1)) and np.allclose(X_corr, Z):
                break
        
        return h, W
    
    def load_and_preprocess_data(self, questionnaire_type: str, 
                               column_range: Optional[tuple] = None) -> pd.DataFrame:
        """
        質問票データの読み込みと前処理
        
        Args:
            questionnaire_type: 質問票のタイプ
            column_range: 列範囲のタプル (start, end) または None
            
        Returns:
            処理済みのデータ
        """
        print(f"Loading data for {questionnaire_type}...")
        if column_range:
            print(f"Column range: {column_range[0]}-{column_range[1]}")
        
        # データの読み込みと前処理
        data = self.data_processor.process_questionnaire_data(
            questionnaire_type=questionnaire_type,
            data_source="csv",
            column_range=column_range
        )
        
        # 転置してデータを準備
        data_transposed = data.T.copy()
        data_transposed = pd.DataFrame(np.array(data_transposed))
        data_transposed = data_transposed.dropna(axis=1)
        data_transposed = data_transposed.astype(int)
        
        print(f"Data loaded: {data_transposed.shape}")
        return data_transposed
    
    def run_fixed_w_analysis(self, 
                           data: pd.DataFrame, 
                           W: pd.DataFrame,
                           filename: str) -> Dict[str, Any]:
        """
        固定W行列を使用したエネルギーランドスケープ解析の実行
        
        Args:
            data: 処理済みデータ
            W: 固定W行列
            filename: 出力ファイル名
            
        Returns:
            解析結果の辞書
        """
        print("Starting Fixed W Energy Landscape Analysis...")
        start_time = time.time()
        
        # タイムスタンプの記録
        start_now = datetime.now(pytz.timezone("Asia/Tokyo"))
        print(f"Start Approximation: {start_now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        with cp.cuda.Device(self.device_id):
            # 1. 固定W行列を使用した近似フィッティング
            print("Fitting approximation with fixed W...")
            h, W = self.fit_approx_fixed_w(data, W)
            print(f"Fit Approximation: {time.time() - start_time:.2f}s")
            
            # 出力ディレクトリの存在確認と作成
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # 結果の保存
            with pandas_full_display():
                h.to_csv(self.output_dir / f"{filename}_h_fixedW.csv", encoding='utf-8-sig')
            results['h'] = h
            results['W'] = W
            
            # 2. Basin graphの計算
            print("Calculating basin graph...")
            graph = calc_basin_graph(h, W, data)
            # 出力ディレクトリの存在確認と作成
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with pandas_full_display():
                graph.to_csv(self.output_dir / f"{filename}_graph_fixedW.csv", encoding='utf-8-sig')
            print(f"Calculate Basin Graph: {time.time() - start_time:.2f}s")
            results['basin_graph'] = graph
            
            # 3. Disconnectivity graphの計算
            print("Calculating disconnectivity graph...")
            D = calc_discon_graph(h, W, data, graph)
            # 出力ディレクトリの存在確認と作成
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            with pandas_full_display():
                D.to_csv(self.output_dir / f"{filename}_D_fixedW.csv", encoding='utf-8-sig')
            print(f"Calculate Disconnection Graph: {time.time() - start_time:.2f}s")
            results['disconnectivity_graph'] = D
            
            # 4. 遷移の計算
            print("Calculating transitions...")
            freq, trans, trans2 = calc_trans(data, graph)
            print(f"Calculate Transition Graph: {time.time() - start_time:.2f}s")
            results['frequency'] = freq
            results['transition'] = trans
            results['transition2'] = trans2
            
            # 5. 遷移行列の計算
            print("Calculating transition matrix...")
            P = calc_trans_fast(h, W, data)
            print(f"Calculate Transition Matrix: {time.time() - start_time:.2f}s")
            results['transition_matrix'] = P
            
            # 6. 可視化
            print("Generating visualizations...")
            self._generate_visualizations(data, graph, filename)
            print(f"Plot Local Min: {time.time() - start_time:.2f}s")
        
        results['elapsed_time'] = time.time() - start_time
        return results
    
    def _generate_visualizations(self, 
                               data: pd.DataFrame,
                               graph: pd.DataFrame,
                               filename: str):
        """
        可視化の生成
        
        Args:
            data: データ
            graph: Basin graph
            filename: ファイル名
        """
        try:
            # Local minimumのプロット
            plot_local_min(data, graph, save_fig=True, filename=filename)
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    def run_pipeline(self, questionnaire_type: str, 
                    w_type: str,
                    column_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        固定W行列を使用した完全なELAパイプラインの実行
        
        Args:
            questionnaire_type: 質問票のタイプ
            w_type: 固定W行列のタイプ（'social' または 'isolation'）
            column_range: 列範囲のタプル (start, end) または None
            
        Returns:
            解析結果の辞書
        """
        print(f"Starting Fixed W ELA pipeline for {questionnaire_type}")
        print(f"Fixed W type: {w_type}")
        print(f"HQ-25 threshold: {self.hq25_threshold}")
        print(f"Output directory: {self.output_dir}")
        if column_range:
            print(f"Column range: {column_range[0]}-{column_range[1]}")
        
        # 固定W行列の読み込み
        W = self.load_fixed_w_matrix(w_type)
        print(f"Fixed W matrix loaded: {W.shape}")
        
        # データの読み込みと前処理
        data = self.load_and_preprocess_data(questionnaire_type, column_range)
        
        # ファイル名の生成（列範囲を含む）
        filename = questionnaire_type
        if column_range:
            filename += f"_{column_range[0]}_{column_range[1]}"
        
        # 固定W行列を使用したエネルギーランドスケープ解析の実行
        results = self.run_fixed_w_analysis(data, W, filename)
        
        print(f"Pipeline completed in {results['elapsed_time']:.2f} seconds")
        return results


def create_argument_parser() -> argparse.ArgumentParser:
    """
    コマンドライン引数パーサーの作成
    
    Returns:
        設定済みのArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Fixed W Energy Landscape Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fixed_w_pipeline.py --filename PHQ9+HQ25_social_2308 --fixedW social --r1 0 --r2 20
  python fixed_w_pipeline.py --filename PHQ9+HQ25_Isolation_2308 --fixedW isolation --r1 0 --r2 17
  python fixed_w_pipeline.py --filename PHQ9+HQ25_social_2012 --fixedW social --r1 0 --r2 20 --hq25_threshold 2
        """
    )
    
    # 必須引数
    parser.add_argument(
        "--filename",
        required=True,
        help="質問票のファイル名（例: PHQ9+HQ25_social_2308）"
    )
    
    parser.add_argument(
        "--fixedW",
        required=True,
        choices=['social', 'isolation'],
        help="固定W行列のタイプ（social または isolation）"
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
    
    # オプション引数
    parser.add_argument(
        "--hq25_threshold",
        type=int,
        choices=[2, 3],
        help="HQ-25の閾値 (2または3, デフォルト: 3)"
    )
    
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="CUDAデバイスID (デフォルト: 0)"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="データディレクトリパス (デフォルト: data)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/integrated",
        help="出力ディレクトリパス (デフォルト: outputs/integrated)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細な出力を有効にする"
    )
    
    return parser


def main():
    """
    メイン関数
    
    コマンドライン引数を解析し、固定W ELAパイプラインを実行します。
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # パラメータの設定
        data_dir = args.data_dir
        output_dir = args.output_dir
        device_id = args.device_id
        hq25_threshold = args.hq25_threshold or 3
        verbose = args.verbose
        
        # 列範囲の設定
        column_range = None
        if args.r2 is not None:
            column_range = (args.r1, args.r2)
        elif args.r1 != 0:
            # r1が指定されているがr2が指定されていない場合は、r1から最後まで
            column_range = (args.r1, None)
        
        # 固定Wパイプラインの初期化
        pipeline = FixedWPipeline(
            data_dir=data_dir,
            output_dir=output_dir,
            device_id=device_id,
            hq25_threshold=hq25_threshold
        )
        
        # パイプラインの実行
        results = pipeline.run_pipeline(
            questionnaire_type=args.filename,
            w_type=args.fixedW,
            column_range=column_range
        )
        
        if verbose:
            print("\n=== Results Summary ===")
            print(f"Questionnaire: {args.filename}")
            print(f"Fixed W Type: {args.fixedW}")
            print(f"HQ-25 Threshold: {hq25_threshold}")
            print(f"Device ID: {device_id}")
            print(f"Data Directory: {data_dir}")
            print(f"Output Directory: {output_dir}")
            print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")
        
        print("Fixed W Analysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
