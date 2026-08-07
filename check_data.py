#!/usr/bin/env python3
"""
データファイルの確認スクリプト

このスクリプトは、ELA_analysisディレクトリ内の
オリジナルデータと処理済みデータの状況を確認します。
"""

import sys
from pathlib import Path

# ELA_analysisディレクトリを基準としたパス設定
ela_analysis_dir = Path(__file__).parent
sys.path.insert(0, str(ela_analysis_dir / "src"))

from data_processing import DataProcessor

def main():
    """データファイルの確認を実行"""
    print("=== ELA Analysis データファイル確認 ===\n")
    
    # データプロセッサの初期化
    processor = DataProcessor()
    
    # オリジナルデータの確認
    print("📁 オリジナルデータ（data/original/）:")
    original_sheets = processor.get_available_original_sheets()
    if original_sheets:
        for file_name, sheets in original_sheets.items():
            print(f"  📄 {file_name}:")
            for sheet in sheets:
                print(f"    - {sheet}")
    else:
        print("  ❌ オリジナルデータが見つかりません")
    print()
    
    # 処理済みデータの確認
    print("📁 処理済みデータ（data/processed/）:")
    processed_files = processor.get_available_processed_files()
    if processed_files:
        for file_name in processed_files:
            print(f"  📄 {file_name}.csv")
    else:
        print("  ❌ 処理済みデータが見つかりません")
    print()
    
    # データソースの自動判別テスト
    print("🔍 データソース自動判別テスト:")
    test_questionnaires = ["PHQ-9", "IPS-22", "HQ-25", "PHQ9_all", "HQ25_social_all"]
    
    for questionnaire in test_questionnaires:
        csv_available = processor.is_processed_data_available(questionnaire)
        excel_available = processor.is_original_data_available(questionnaire)
        
        if csv_available:
            print(f"  ✅ {questionnaire}: 処理済みCSV利用可能")
        elif excel_available:
            print(f"  ✅ {questionnaire}: オリジナルExcel利用可能")
        else:
            print(f"  ❌ {questionnaire}: データが見つかりません")
    
    print("\n=== 確認完了 ===")

if __name__ == "__main__":
    main()
