#!/usr/bin/env python3
"""
出力ファイル整理スクリプト

既存の出力ファイルを質問票タイプごとのフォルダに整理します。
"""

import shutil
from pathlib import Path
import re


def organize_output_files(outputs_dir: Path = None):
    """
    出力ファイルを質問票タイプごとのフォルダに整理
    
    Args:
        outputs_dir: 出力ディレクトリのパス
    """
    if outputs_dir is None:
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent
        outputs_dir = ela_analysis_dir / "outputs"
    
    if not outputs_dir.exists():
        print(f"出力ディレクトリが見つかりません: {outputs_dir}")
        return
    
    print(f"出力ファイルを整理中: {outputs_dir}")
    
    # 質問票タイプのパターン
    questionnaire_patterns = {
        'PHQ9_all': r'^PHQ9_all',
        'PHQ-9': r'^PHQ-9',
        'HQ-25\+4': r'^HQ-25\+4',
        'IPS-22': r'^IPS-22',
        'IAT': r'^IAT',
        'TACS-22': r'^TACS-22'
    }
    
    # 各質問票タイプのフォルダを作成
    for questionnaire_type, pattern in questionnaire_patterns.items():
        folder_name = questionnaire_type.replace('+', '_').replace('-', '_')
        questionnaire_dir = outputs_dir / folder_name
        questionnaire_dir.mkdir(exist_ok=True)
        
        # 該当するファイルを移動
        moved_files = []
        for file_path in outputs_dir.glob("*.csv"):
            if re.match(pattern, file_path.name):
                dest_path = questionnaire_dir / file_path.name
                if not dest_path.exists():
                    shutil.move(str(file_path), str(dest_path))
                    moved_files.append(file_path.name)
                    print(f"移動: {file_path.name} → {folder_name}/")
                else:
                    print(f"スキップ: {file_path.name} (既に存在)")
        
        if moved_files:
            print(f"✅ {folder_name}: {len(moved_files)}ファイルを移動")
        else:
            print(f"📁 {folder_name}: 移動するファイルなし")
    
    # その他のファイル（figsフォルダなど）
    other_dirs = ['figs']
    for dir_name in other_dirs:
        other_dir = outputs_dir / dir_name
        if other_dir.exists():
            print(f"📁 {dir_name}: そのまま保持")
    
    print("整理完了！")


def main():
    """メイン関数"""
    organize_output_files()


if __name__ == "__main__":
    main()


