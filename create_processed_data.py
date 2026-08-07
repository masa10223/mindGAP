#!/usr/bin/env python3
"""
オリジナルxlsxファイルからprocessedデータを作成するスクリプト

使用方法:
    python create_processed_data.py [--questionnaire_type QUESTIONNAIRE_TYPE] [--all]

引数:
    --questionnaire_type: 処理する質問票タイプ（HQ25+4, PHQ-9, IPS-22, IAT, TACS-22）
    --all: 全ての質問票を処理
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

def get_time_period_ranges(questionnaire_type: str) -> dict:
    """各質問票の年代別列範囲を取得"""
    ranges = {
        'HQ25+4': {
            '2006': (1, 30),   # 列1-29 (29列)
            '2012': (30, 59),  # 列30-58 (29列)
            '2108': (59, 88),  # 列59-87 (29列)
            '2204': (88, 117),  # 列88-116 (29列)
            '2308': (117, 146)  # 列88-116 (29列)
        },
        'PHQ-9': {
            '2006': (1, 10),   # 列1-9 (9列)
            '2012': (10, 19),  # 列10-18 (9列)
            '2108': (19, 28),  # 列19-27 (9列)
            '2204': (28, 37),   # 列28-36 (9列)
            '2308': (37, 46)   # 列37-45 (9列)
        },
        'IPS-22': {
            '2006': (1, 23),   # 列1-22 (22列)
            '2012': (23, 45),  # 列23-44 (22列)
            '2108': (45, 67),  # 列45-66 (22列)
            '2204': (67, 89)   # 列67-88 (22列)
        },
        'IAT': {
            '2006': (1, 21),   # 列1-20 (20列)
            '2012': (21, 41),  # 列21-40 (20列)
            '2108': (41, 61),  # 列41-60 (20列)
            '2204': (61, 81)   # 列61-80 (20列)
        },
        'TACS-22': {
            '2006': (1, 23),   # 列1-22 (22列)
            '2012': (23, 45),  # 列23-44 (22列)
            '2108': (45, 67),  # 列45-66 (22列)
            '2204': (67, 89)   # 列67-88 (22列)
        }
    }
    return ranges.get(questionnaire_type, {})

def get_preprocessing_function(questionnaire_type: str, threshold: int = 3):
    """各質問票の前処理関数を取得"""
    
    def hq25_condition(data: pd.DataFrame) -> pd.DataFrame:
        """HQ-25+4の前処理"""
        def binarize(x):
            try:
                x = float(x)
                if x >= threshold:  # 動的閾値
                    return 1
                elif 1 <= x < threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return data.applymap(binarize)
    
    def phq9_condition(data: pd.DataFrame) -> pd.DataFrame:
        """PHQ-9の前処理"""
        def binarize(x):
            try:
                x = float(x)
                if x >= threshold:  # 動的閾値
                    return 1
                elif 0 <= x < threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return data.applymap(binarize)
    
    def ips22_condition(data: pd.DataFrame) -> pd.DataFrame:
        """IPS-22の前処理"""
        def binarize(x):
            try:
                x = float(x)
                if x >= threshold:  # 動的閾値
                    return 1
                elif 1 <= x < threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return data.applymap(binarize)
    
    def iat_condition(data: pd.DataFrame) -> pd.DataFrame:
        """IATの前処理"""
        def binarize(x):
            try:
                x = float(x)
                if x >= threshold:  # 動的閾値
                    return 1
                elif 1 <= x < threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return data.applymap(binarize)
    
    def tacs22_condition(data: pd.DataFrame) -> pd.DataFrame:
        """TACS-22の前処理"""
        def binarize(x):
            try:
                x = float(x)
                if x >= threshold:  # 動的閾値
                    return 1
                elif 1 <= x < threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return data.applymap(binarize)
    
    preprocessing_map = {
        "HQ25+4": hq25_condition,
        "PHQ-9": phq9_condition,
        "IPS-22": ips22_condition,
        "IAT": iat_condition,
        "TACS-22": tacs22_condition
    }
    
    return preprocessing_map.get(questionnaire_type)

def process_questionnaire_data_with_questions(questionnaire_type: str, threshold: int = 3) -> dict:
    """質問票データの処理（質問文を列名として取得、threshold情報をファイル名に含める）"""
    print(f"処理中: {questionnaire_type}")
    
    # ファイルパス
    original_file = Path(f"data/original/{questionnaire_type}.xlsx")
    # HQ-25+4 の場合はハイフン表記にも対応
    if questionnaire_type == 'HQ25+4' and not original_file.exists():
        alt = Path("data/original/HQ-25+4.xlsx")
        if alt.exists():
            original_file = alt
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # データ読み込み（ヘッダなしで読み込み、質問文を手動で設定）
    raw_df = pd.read_excel(original_file, sheet_name=0, header=None, engine="openpyxl")
    print(f"  生データ読み込み完了: {raw_df.shape}")

    # 質問文を列名として抽出（2行目、インデックス1）
    question_headers = raw_df.iloc[1].tolist()
    # データ部分を抽出（3行目から、ID列を除外）
    df_data = raw_df.iloc[2:].copy()
    # ID列（0番目）を除外
    df_data = df_data.iloc[:, 1:].copy()
    
    # 質問文をクリーンアップして列名として設定
    clean_headers = []
    for i, q in enumerate(question_headers[1:]):  # ID列を除外
        if pd.isna(q) or str(q).strip() == '' or str(q).strip() == 'nan':
            clean_headers.append(f"Question_{i+1}")
        else:
            clean_text = str(q).replace("\n", " ").replace("\r", " ").strip()
            # 日付部分を削除
            if "／" in clean_text:
                clean_text = clean_text.split("／")[0].strip()
            # 長すぎる場合は短縮
            if len(clean_text) > 50:
                clean_text = clean_text[:50] + "..."
            clean_headers.append(clean_text)
    
    
    df_data.columns = clean_headers
    df_data.reset_index(drop=True, inplace=True)
    print(f"  データ整形後: {df_data.shape}")
    
    # 前処理関数取得
    preprocess_func = get_preprocessing_function(questionnaire_type, threshold)
    if not preprocess_func:
        print(f"  エラー: 未知の質問票タイプ '{questionnaire_type}'")
        return {}
    
    # 年代別列範囲取得
    time_ranges = get_time_period_ranges(questionnaire_type)
    if not time_ranges:
        print(f"  エラー: 年代別列範囲が定義されていません")
        return {}
    
    results = {}
    
    # 各年代のデータを処理
    for period, (start_col, end_col) in time_ranges.items():
        print(f"  処理中: {period}年 (列{start_col}-{end_col})")
        
        # 列選択（1列目のnoは除外済み）
        #　一行目は選択肢なので、2行目からスタート、なおnanを削除
        period_data = df_data.iloc[1:, start_col-1:end_col-1].copy().dropna() # -1 because df_data already excluded 'no' column
        
        # 前処理（列名を一時的に数値に変更して処理）
        temp_data = period_data.copy()
        temp_data.columns = range(len(temp_data.columns))
        processed_data = preprocess_func(temp_data)
        
        # 列名を質問文に戻す
        processed_data.columns = period_data.columns
        
        # データクリーニング
        processed_data = processed_data.astype(float, errors="ignore")
        processed_data = processed_data.fillna(0)  # NaNを0で埋める
        processed_data = processed_data.astype(int)
        
        # 有効なデータが少ない行を削除
        processed_data = processed_data.dropna(axis=0, how='all')
        
        print(f"    処理後: {processed_data.shape}")
        
        # ファイル保存（フォルダ分け、threshold情報を含むファイル名）
        questionnaire_output_dir = processed_dir / questionnaire_type
        questionnaire_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = questionnaire_output_dir / f"{questionnaire_type}_{period}_thr{threshold}.csv"
        processed_data.to_csv(output_file, index=False)
        print(f"    保存完了: {output_file}")
        
        results[period] = processed_data.shape
    
    # 全年代統合データの作成
    print(f"  全年代統合データ作成中...")
    all_data = []
    for period, (start_col, end_col) in time_ranges.items():
        period_data = df_data.iloc[1:, start_col-1:end_col-1].copy().dropna()
        # 前処理（列名を一時的に数値に変更して処理）
        temp_data = period_data.copy()
        temp_data.columns = range(len(temp_data.columns))
        processed_data = preprocess_func(temp_data)
        # 列名を質問文に戻す
        processed_data.columns = period_data.columns
        # データクリーニング
        processed_data = processed_data.astype(float, errors="ignore")
        processed_data = processed_data.fillna(0)
        processed_data = processed_data.astype(int)
        all_data.append(processed_data)
    
    # 全データを縦に結合（同じ列名なので縦に繋がる）
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    combined_data = combined_data.dropna(axis=0, how='all')
    
    # 最初の9列のみを抽出（質問文の列）
    question_columns = combined_data.columns[:9]
    combined_data = combined_data[question_columns]
    
    # 統合ファイル保存（フォルダ分け、threshold情報を含む）
    questionnaire_output_dir = processed_dir / questionnaire_type
    questionnaire_output_dir.mkdir(parents=True, exist_ok=True)
    combined_file = questionnaire_output_dir / f"{questionnaire_type}_all_thr{threshold}.csv"
    combined_data.to_csv(combined_file, index=False)
    print(f"  統合データ保存完了: {combined_file} ({combined_data.shape})")
    
    results['all'] = combined_data.shape
    
    return results

def process_questionnaire_data(questionnaire_type: str) -> dict:
    """質問票データの処理（従来版、互換性のため残す）"""
    print(f"処理中: {questionnaire_type}")
    
    # ファイルパス
    original_file = Path(f"data/original/{questionnaire_type}.xlsx")
    # HQ-25+4 の場合はハイフン表記にも対応
    if questionnaire_type == 'HQ25+4' and not original_file.exists():
        alt = Path("data/original/HQ-25+4.xlsx")
        if alt.exists():
            original_file = alt
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # データ読み込み（最初の2行をスキップ）
    df = pd.read_excel(original_file, sheet_name=0, header=2)
    print(f"  読み込み完了: {df.shape}")
    
    # 前処理関数取得
    preprocess_func = get_preprocessing_function(questionnaire_type, threshold)
    if not preprocess_func:
        print(f"  エラー: 未知の質問票タイプ '{questionnaire_type}'")
        return {}
    
    # 年代別列範囲取得
    time_ranges = get_time_period_ranges(questionnaire_type)
    if not time_ranges:
        print(f"  エラー: 年代別列範囲が定義されていません")
        return {}
    
    results = {}
    
    # 各年代のデータを処理
    for period, (start_col, end_col) in time_ranges.items():
        print(f"  処理中: {period}年 (列{start_col}-{end_col-1})")
        
        # 列選択（1列目のnoは除外）
        period_data = df.iloc[:, start_col:end_col].copy()
        
        # ID列（0番目）を除外
        if len(period_data.columns) > 0 and period_data.columns[0] == 0:
            period_data = period_data.iloc[:, 1:].copy()
        
        # 列名を列番号に変更
        period_data.columns = range(len(period_data.columns))
        
        # 前処理
        processed_data = preprocess_func(period_data)
        
        # データクリーニング
        processed_data = processed_data.astype(float, errors="ignore")
        processed_data = processed_data.fillna(0)  # NaNを0で埋める
        processed_data = processed_data.astype(int)
        
        # 有効なデータが少ない行を削除
        processed_data = processed_data.dropna(axis=0, how='all')
        
        print(f"    処理後: {processed_data.shape}")
        
        # ファイル保存（フォルダ分け）
        questionnaire_output_dir = processed_dir / questionnaire_type
        questionnaire_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = questionnaire_output_dir / f"{questionnaire_type}_{period}.csv"
        processed_data.to_csv(output_file, index=False)
        print(f"    保存完了: {output_file}")
        
        results[period] = processed_data.shape
    
    # 全年代統合データの作成
    print(f"  全年代統合データ作成中...")
    all_data = []
    for period, (start_col, end_col) in time_ranges.items():
        period_data = df.iloc[:, start_col:end_col].copy()
        
        # ID列（0番目）を除外
        if len(period_data.columns) > 0 and period_data.columns[0] == 0:
            period_data = period_data.iloc[:, 1:].copy()
        
        # 列名を列番号に変更
        period_data.columns = range(len(period_data.columns))
        
        processed_data = preprocess_func(period_data)
        processed_data = processed_data.astype(float, errors="ignore")
        processed_data = processed_data.fillna(0)
        processed_data = processed_data.astype(int)
        processed_data = processed_data.dropna(axis=0, how='all')
        all_data.append(processed_data)
    
    # 全データを結合
    combined_data = pd.concat(all_data, axis=0, ignore_index=True)
    combined_data = combined_data.dropna(axis=0, how='all')
    
    # 統合ファイル保存（フォルダ分け）
    questionnaire_output_dir = processed_dir / questionnaire_type
    questionnaire_output_dir.mkdir(parents=True, exist_ok=True)
    combined_file = questionnaire_output_dir / f"{questionnaire_type}_all.csv"
    combined_data.to_csv(combined_file, index=False)
    print(f"  統合データ保存完了: {combined_file} ({combined_data.shape})")
    
    results['all'] = combined_data.shape
    
    # HQ-25+4 のカテゴリ別抽出（質問文列名・social_YYYY等で保存）
    if questionnaire_type == 'HQ25+4':
        print("  HQ-25+4 カテゴリ別抽出を実行します...")
        cat_results = process_hq25_categories(original_file, processed_dir)
        for k, v in cat_results.items():
            print(f"    保存完了: {k}.csv -> 形状 {v}")
        results.update(cat_results)

    return results

def _read_hq25_block_with_questions(original_file: Path, start_col: int, end_col: int) -> tuple[pd.DataFrame, list]:
    """
    HQ-25+4専用: ヘッダなしで読み込み、質問文を列名として付与したデータブロックを返す
    - 行0: 年代情報
    - 行1: 質問文
    - 行2以降: 回答データ
    """
    # ヘッダなしで読み込み
    df_raw = pd.read_excel(original_file, sheet_name=0, header=None, engine="openpyxl")
    
    # 質問文の行を取得（2行目、インデックス1）
    question_texts = df_raw.iloc[1, start_col:end_col].tolist()
    
    # データ本体を取得（3行目以降、インデックス2以降）
    values = df_raw.iloc[2:, start_col:end_col].copy()
    
    # 質問文をクリーンアップして列名として設定
    clean_names = []
    for i, q in enumerate(question_texts):
        if pd.isna(q) or str(q).strip() == '':
            clean_names.append(f"Question_{start_col + i}")
        else:
            clean_text = str(q).replace("\n", " ").replace("\r", " ").strip()
            # 日付部分を削除して短縮
            if "／" in clean_text:
                clean_text = clean_text.split("／")[0].strip()
            # 長すぎる場合は短縮
            if len(clean_text) > 30:
                clean_text = clean_text[:30] + "..."
            clean_names.append(clean_text)
    
    values.columns = clean_names
    # インデックスをリセット
    values.reset_index(drop=True, inplace=True)
    return values, clean_names


def process_hq25_categories(original_file: Path, output_dir: Path, threshold: int = 3) -> dict:
    """
    HQ-25+4のカテゴリ（social / isolation / emotional）を質問文の列名で抽出し、
    各年代別と全年代統合データを保存する。
    """
    category_to_indices = {
        "social": [0,1,3,5,7,10,12,14,17,19,24],
        "isolation": [4,8,11,15,18,21,22,23],
        "emotional": [6,9,13,16,20],
    }

    time_ranges = get_time_period_ranges('HQ25+4')
    preprocess_func = get_preprocessing_function('HQ25+4', threshold)
    results: dict = {}
    
    # 各カテゴリの全年代データを格納
    category_all_data = {cat: [] for cat in category_to_indices.keys()}

    for period, (start_col, end_col) in time_ranges.items():
        # ブロック読み込み（質問文列名付き）
        block_df, block_questions = _read_hq25_block_with_questions(original_file, start_col, end_col)
        # 前処理（二値化）
        processed_block = preprocess_func(block_df)
        # NaNを欠損値として扱い、適切に処理
        processed_block = processed_block.astype(float, errors="ignore")
        # 全列がNaNの行のみを削除（一部の列がNaNでも残す）
        processed_block = processed_block.dropna(how='all')
        # 残りのNaNは0で埋める
        processed_block = processed_block.fillna(0).astype(int)

        for cat, idx_list in category_to_indices.items():
            # 有効なインデックスのみ選択
            valid_idx = [i for i in idx_list if 0 <= i < processed_block.shape[1]]
            if not valid_idx:
                continue
            cat_df = processed_block.iloc[:, valid_idx].copy()
            # 列名は質問文（すでに設定済み）をスライスで保持
            cat_df.columns = [block_df.columns[i] for i in valid_idx]

            # 年代別保存（social_2006_thr{threshold}.csv のように）
            out_path = output_dir / 'HQ25+4' / f"{cat}_{period}_thr{threshold}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cat_df.to_csv(out_path, index=False)
            results[f"{cat}_{period}"] = cat_df.shape
            
            # 全年代統合用にデータを保存
            category_all_data[cat].append(cat_df)

    # 各カテゴリの全年代統合データを作成
    for cat, data_list in category_all_data.items():
        if data_list:
            # 列名を正規化（括弧の種類を統一）
            for i, df in enumerate(data_list):
                df.columns = df.columns.str.replace('（', '(').str.replace('）', ')')
            
            combined_df = pd.concat(data_list, ignore_index=True)
            # 統合データも適切に処理
            combined_df = combined_df.dropna(how='all')
            combined_df = combined_df.fillna(0)
            # 保存（social_all_thr{threshold}.csv のように）
            out_path = output_dir / 'HQ25+4' / f"{cat}_all_thr{threshold}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(out_path, index=False)
            results[f"{cat}_all"] = combined_df.shape
            print(f"    統合データ保存完了: {cat}_all_thr{threshold}.csv -> 形状 {combined_df.shape}")

    return results

def process_hq25_single_category(original_file: Path, output_dir: Path, category: str, threshold: int = 3) -> dict:
    """
    HQ-25+4の特定カテゴリ（emotional/social/isolation）のみを抽出し、
    {category}_all_thr{threshold}.csv として保存する。
    """
    category_to_indices = {
        "social": [0,1,3,5,7,10,12,14,17,19,24],
        "isolation": [4,8,11,15,18,21,22,23],
        "emotional": [6,9,13,16,20],
    }
    
    if category not in category_to_indices:
        raise ValueError(f"Unknown category: {category}. Available: {list(category_to_indices.keys())}")
    
    time_ranges = get_time_period_ranges('HQ25+4')
    preprocess_func = get_preprocessing_function('HQ25+4', threshold)
    results: dict = {}
    
    # 全年代のデータを結合
    all_data = []
    
    for period, (start_col, end_col) in time_ranges.items():
        # ブロック読み込み（質問文列名付き）
        block_df, block_questions = _read_hq25_block_with_questions(original_file, start_col, end_col)
        # 前処理（二値化）
        processed_block = preprocess_func(block_df)
        # NaNを欠損値として扱い、適切に処理
        processed_block = processed_block.astype(float, errors="ignore")
        # 全列がNaNの行のみを削除（一部の列がNaNでも残す）
        processed_block = processed_block.dropna(how='all')
        # 残りのNaNは0で埋める
        processed_block = processed_block.fillna(0).astype(int)
        
        # 指定カテゴリの列のみ抽出
        idx_list = category_to_indices[category]
        valid_idx = [i for i in idx_list if 0 <= i < processed_block.shape[1]]
        if valid_idx:
            cat_df = processed_block.iloc[:, valid_idx].copy()
            # 列名は質問文（すでに設定済み）をスライスで保持
            cat_df.columns = [block_df.columns[i] for i in valid_idx]
            all_data.append(cat_df)
            results[f"{category}_{period}"] = cat_df.shape
    
    # 全年代統合データを作成
    if all_data:
        # 列名を正規化（括弧の種類を統一）
        for i, df in enumerate(all_data):
            df.columns = df.columns.str.replace('（', '(').str.replace('）', ')')
        
        combined_df = pd.concat(all_data, ignore_index=True)
        # 統合データも適切に処理
        combined_df = combined_df.dropna(how='all')
        combined_df = combined_df.fillna(0)
        # 保存（emotional_all_thr{threshold}.csv のように）
        out_path = output_dir / 'HQ25+4' / f"{category}_all_thr{threshold}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_path, index=False)
        results[f"{category}_all"] = combined_df.shape
        print(f"    統合データ保存完了: {category}_all_thr{threshold}.csv -> 形状 {combined_df.shape}")
    
    return results

def process_combined_questionnaires(questionnaire1: str, questionnaire2: str, 
                                  threshold1: int = 3, threshold2: int = 3) -> dict:
    """
    2つの質問票を組み合わせて処理する
    
    Args:
        questionnaire1: 第1の質問票（例: 'emotional'）
        questionnaire2: 第2の質問票（例: 'PHQ-9'）
        threshold1: 第1の質問票の閾値
        threshold2: 第2の質問票の閾値
    
    Returns:
        処理結果の辞書
    """
    print(f"=== 組み合わせ質問票処理開始: {questionnaire1} + {questionnaire2} ===")
    print(f"閾値: {questionnaire1}={threshold1}, {questionnaire2}={threshold2}")
    
    # 出力ディレクトリの設定
    output_dir = Path(f"data/processed/{questionnaire1}_{questionnaire2}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 第1の質問票のデータを読み込み
    if questionnaire1 in ['emotional', 'social', 'isolation']:
        # HQ-25+4カテゴリの場合
        data1 = _load_hq25_category_data(questionnaire1, threshold1)
    else:
        # 通常の質問票の場合
        data1 = _load_questionnaire_data(questionnaire1, threshold1)
    
    # 第2の質問票のデータを読み込み
    data2 = _load_questionnaire_data(questionnaire2, threshold2)
    
    # 年代別にデータを統合
    for period in ['2006', '2012', '2108', '2204', '2308', 'all']:
        if period in data1 and period in data2:
            print(f"  {period}年データを統合中...")
            
            # データを取得
            df1 = data1[period].copy()
            df2 = data2[period].copy()
            
            # 列名にプレフィックスを追加
            df1.columns = [f"{questionnaire1}_{col}" for col in df1.columns]
            df2.columns = [f"{questionnaire2}_{col}" for col in df2.columns]
            
            # 行数を合わせる（最小の行数に合わせる）
            min_rows = min(len(df1), len(df2))
            df1 = df1.iloc[:min_rows]
            df2 = df2.iloc[:min_rows]
            
            # データを横に結合
            combined_df = pd.concat([df1, df2], axis=1)
            
            # ファイル名を生成
            filename = f"{questionnaire1}_{questionnaire2}_{period}_thr{threshold1}_thr{threshold2}"
            output_file = output_dir / f"{filename}.csv"
            
            # 保存
            combined_df.to_csv(output_file, index=False)
            results[period] = combined_df.shape
            print(f"    保存完了: {filename}.csv -> 形状 {combined_df.shape}")
    
    # 全年代統合データを作成
    if 'all' in results:
        print(f"✅ 組み合わせ処理完了: {questionnaire1} + {questionnaire2}")
        for period, shape in results.items():
            print(f"  {period}: {shape}")
    else:
        print("❌ 全年代統合データが見つかりません")
    
    return results

def _load_hq25_category_data(category: str, threshold: int) -> dict:
    """HQ-25+4カテゴリのデータを読み込み"""
    data = {}
    base_dir = Path("data/processed/HQ25+4")
    
    for period in ['2006', '2012', '2108', '2204', '2308', 'all']:
        file_path = base_dir / f"{category}_{period}_thr{threshold}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            data[period] = df
            print(f"    読み込み: {file_path} -> 形状 {df.shape}")
    
    return data

def _load_questionnaire_data(questionnaire: str, threshold: int) -> dict:
    """通常の質問票のデータを読み込み"""
    data = {}
    base_dir = Path(f"data/processed/{questionnaire}")
    
    for period in ['2006', '2012', '2108', '2204', '2308', 'all']:
        file_path = base_dir / f"{questionnaire}_{period}_thr{threshold}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            data[period] = df
            print(f"    読み込み: {file_path} -> 形状 {df.shape}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='オリジナルxlsxファイルからprocessedデータを作成')
    parser.add_argument('--questionnaire_type', type=str, help='処理する質問票タイプ')
    parser.add_argument('--all', action='store_true', help='全ての質問票を処理')
    parser.add_argument('--threshold', type=int, help='閾値を指定（デフォルトは質問票ごとに異なる）')
    parser.add_argument('--combine', nargs=2, metavar=('Q1', 'Q2'), help='2つの質問票を組み合わせて処理')
    parser.add_argument('--threshold1', type=int, help='第1の質問票の閾値')
    parser.add_argument('--threshold2', type=int, help='第2の質問票の閾値')
    
    args = parser.parse_args()
    
    if args.combine:
        # 組み合わせ処理
        questionnaire1, questionnaire2 = args.combine
        threshold1 = args.threshold1 if args.threshold1 is not None else 3
        threshold2 = args.threshold2 if args.threshold2 is not None else 3
        
        print("=== 組み合わせ質問票処理 ===")
        results = process_combined_questionnaires(questionnaire1, questionnaire2, threshold1, threshold2)
        print("=== 処理完了 ===")
        return
    elif args.all:
        questionnaire_types = ['HQ25+4', 'PHQ-9', 'IPS-22', 'IAT', 'TACS-22']
    elif args.questionnaire_type:
        questionnaire_types = [args.questionnaire_type]
    else:
        print("エラー: --questionnaire_type、--all、または --combine を指定してください")
        sys.exit(1)
    
    # HQ-25+4カテゴリのマッピング
    hq25_categories = {
        'emotional': 'emotional',
        'social': 'social', 
        'isolation': 'isolation'
    }
    
    print("=== オリジナルxlsxファイルからprocessedデータ作成開始 ===")
    print()
    
    for questionnaire_type in questionnaire_types:
        try:
            # 各質問票のデフォルトthreshold値を設定
            threshold_map = {
                'HQ25+4': 3,
                'PHQ-9': 3,
                'IPS-22': 3,
                'IAT': 3,
                'TACS-22': 4
            }
            # コマンドライン引数で指定された閾値を使用、なければデフォルト値
            threshold = args.threshold if args.threshold is not None else threshold_map.get(questionnaire_type, 3)
            
            # HQ-25+4カテゴリの場合は特別処理
            if questionnaire_type in hq25_categories:
                print(f"  HQ-25+4 {questionnaire_type} カテゴリ抽出を実行します...")
                original_file = Path("data/original/HQ25+4.xlsx")
                # HQ-25+4 の場合はハイフン表記にも対応
                if not original_file.exists():
                    alt = Path("data/original/HQ-25+4.xlsx")
                    if alt.exists():
                        original_file = alt
                
                # 特定カテゴリのみ処理
                cat_results = process_hq25_single_category(original_file, Path("data/processed"), questionnaire_type, threshold)
                for k, v in cat_results.items():
                    print(f"    保存完了: {k}_thr{threshold}.csv -> 形状 {v}")
                results = cat_results
            else:
                # 通常の質問票処理
                results = process_questionnaire_data_with_questions(questionnaire_type, threshold)
                
                if questionnaire_type == 'HQ25+4':
                    # HQ-25+4 の全カテゴリ抽出（質問文列名・social_YYYY_thr{threshold}等で保存）
                    print(f"  HQ-25+4 全カテゴリ抽出を実行します...")
                    original_file = Path(f"data/original/{questionnaire_type}.xlsx")
                    # HQ-25+4 の場合はハイフン表記にも対応
                    if not original_file.exists():
                        alt = Path("data/original/HQ-25+4.xlsx")
                        if alt.exists():
                            original_file = alt
                    cat_results = process_hq25_categories(original_file, Path("data/processed"), threshold)
                    for k, v in cat_results.items():
                        print(f"    保存完了: {k}_thr{threshold}.csv -> 形状 {v}")
                    results.update(cat_results)
            
            print(f"✅ {questionnaire_type} 処理完了")
            for period, shape in results.items():
                print(f"  {period}: {shape}")
            print()
        except Exception as e:
            print(f"❌ {questionnaire_type} 処理エラー: {e}")
            print()
    
    print("=== 処理完了 ===")

if __name__ == "__main__":
    main()
