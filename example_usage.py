#!/usr/bin/env python3
"""
ELA Analysis 使用例

このスクリプトは、ELA Analysisパッケージの使用方法を示します。
"""

import sys
from pathlib import Path

# パッケージのパスを追加
sys.path.append(str(Path(__file__).parent))

from src.data_processing import DataProcessor
from src.ela_pipeline import ELAPipeline
from src.config import ELAConfig, ConfigManager


def example_basic_usage():
    """基本的な使用例"""
    print("=== 基本的な使用例 ===")
    
    # データプロセッサの初期化
    processor = DataProcessor(
        data_dir="../inputfiles",
        hq25_threshold=3  # 閾値を3に設定
    )
    
    # データの読み込みと前処理
    try:
        data = processor.process_questionnaire_data(
            questionnaire_type="PHQ9_all",
            data_source="csv"
        )
        print(f"データの形状: {data.shape}")
        print("データの読み込みと前処理が完了しました。")
    except Exception as e:
        print(f"データの読み込みでエラーが発生しました: {e}")


def example_pipeline_usage():
    """パイプラインの使用例"""
    print("\n=== パイプラインの使用例 ===")
    
    # ELAパイプラインの初期化
    pipeline = ELAPipeline(
        data_dir="../inputfiles",
        output_dir="./example_output",
        device_id=0,
        hq25_threshold=2  # 閾値を2に設定
    )
    
    print("ELAパイプラインが初期化されました。")
    print(f"HQ-25閾値: {pipeline.hq25_threshold}")
    print(f"出力ディレクトリ: {pipeline.output_dir}")
    
    # 実際の解析は時間がかかるため、コメントアウト
    # results = pipeline.run_pipeline("PHQ9_all")
    # print(f"解析完了: {results['elapsed_time']:.2f}秒")


def example_config_usage():
    """設定ファイルの使用例"""
    print("\n=== 設定ファイルの使用例 ===")
    
    # 設定の作成
    config = ELAConfig()
    config.processing.hq25_threshold = 2
    config.processing.verbose = True
    config.data.output_dir = "./config_output"
    
    # 設定ファイルの保存
    config.to_json("example_config.json")
    print("設定ファイルを保存しました: example_config.json")
    
    # 設定ファイルの読み込み
    config_manager = ConfigManager("example_config.json")
    loaded_config = config_manager.load_config()
    print(f"読み込まれたHQ-25閾値: {loaded_config.processing.hq25_threshold}")
    print(f"読み込まれた出力ディレクトリ: {loaded_config.data.output_dir}")


def example_hq25_threshold_comparison():
    """HQ-25閾値の比較例"""
    print("\n=== HQ-25閾値の比較例 ===")
    
    # サンプルデータの作成
    import pandas as pd
    import numpy as np
    
    # サンプルHQ-25データ（1-4のスケール）
    sample_data = pd.DataFrame({
        'Q1': [1, 2, 3, 4, 1, 2, 3, 4],
        'Q2': [2, 3, 4, 1, 2, 3, 4, 1],
        'Q3': [3, 4, 1, 2, 3, 4, 1, 2],
    })
    
    print("元のデータ:")
    print(sample_data)
    
    # 閾値2での処理
    processor_threshold_2 = DataProcessor(hq25_threshold=2)
    processed_2 = processor_threshold_2.hq25_condition(sample_data)
    print("\n閾値2での処理結果:")
    print(processed_2)
    
    # 閾値3での処理
    processor_threshold_3 = DataProcessor(hq25_threshold=3)
    processed_3 = processor_threshold_3.hq25_condition(sample_data)
    print("\n閾値3での処理結果:")
    print(processed_3)
    
    # 違いの説明
    print("\n閾値の違い:")
    print("- 閾値2: 1-2を陰性(0)、3-4を陽性(1)として分類")
    print("- 閾値3: 1-3を陰性(0)、4を陽性(1)として分類")


def main():
    """メイン関数"""
    print("ELA Analysis 使用例")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_pipeline_usage()
        example_config_usage()
        example_hq25_threshold_comparison()
        
        print("\n" + "=" * 50)
        print("全ての使用例が完了しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


