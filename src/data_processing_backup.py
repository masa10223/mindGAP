"""
データ前処理モジュール

質問票データの読み込みと前処理を行う関数群を提供します。
OSS対応のため、設定可能なパラメータと柔軟なパス設定をサポートします。
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
from pathlib import Path


class DataProcessor:
    """
    データ前処理を行うクラス
    
    設定可能なパラメータと柔軟なパス設定をサポートし、
    メンテナンス性と再利用性を向上させます。
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 hq25_threshold: int = 3,
                 ips22_threshold: int = 4,
                 iat_threshold: int = 3,
                 tacs22_threshold: int = 4):
        """
        データプロセッサの初期化
        
        Args:
            data_dir: データファイルのディレクトリパス（ELA_analysis内の相対パス）
            hq25_threshold: HQ-25の閾値（2または3）
            ips22_threshold: IPS-22の閾値（デフォルト: 4）
            iat_threshold: IATの閾値（デフォルト: 3）
            tacs22_threshold: TACS-22の閾値（デフォルト: 4）
        """
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.data_dir = ela_analysis_dir / data_dir
        self.original_dir = self.data_dir / "original"
        self.processed_dir = self.data_dir / "processed"
        self.hq25_threshold = hq25_threshold
        self.ips22_threshold = ips22_threshold
        self.iat_threshold = iat_threshold
        self.tacs22_threshold = tacs22_threshold
        
        # 閾値の検証
        if hq25_threshold not in [2, 3]:
            raise ValueError("hq25_threshold must be 2 or 3")
    
    def load_excel_data(self, column: str = "IPS-22", 
                       filepath: Optional[str] = None) -> pd.DataFrame:
        """
        オリジナルExcelファイルから特定のシートを読み込みます
        
        Args:
            column: 読み込むシート名
            filepath: Excelファイルのパス（Noneの場合はデフォルトパスを使用）
            
        Returns:
            指定されたシートのデータを含むDataFrame
        """
        if filepath is None:
            # デフォルトでdata.xlsxを探し、なければdata2.xlsxを使用
            data_xlsx = self.original_dir / "data.xlsx"
            data2_xlsx = self.original_dir / "data2.xlsx"
            
            if data_xlsx.exists():
                filepath = data_xlsx
            elif data2_xlsx.exists():
                filepath = data2_xlsx
            else:
                raise FileNotFoundError(f"Neither data.xlsx nor data2.xlsx found in {self.original_dir}")
        
        # 全てのシートを辞書として読み込み
        # 最初の行は質問文、2行目は選択肢説明、3行目以降が回答データ
        df_sheet_all = pd.read_excel(
            filepath, sheet_name=None, header=2, engine="openpyxl"
        )
        
        if column not in df_sheet_all:
            available_sheets = list(df_sheet_all.keys())
            raise ValueError(f"Sheet '{column}' not found. Available sheets: {available_sheets}")
        
        return df_sheet_all[column]
    
    def load_csv_data(self, name: str, 
                     use_index: bool = False,
                     directory: Optional[str] = None) -> pd.DataFrame:
        """
        処理済みCSVファイルを読み込みます
        
        Args:
            name: ファイル名（拡張子なし）
            use_index: 最初の列をインデックスとして使用するか
            directory: 読み込みディレクトリ（Noneの場合は処理済みディレクトリを使用）
            
        Returns:
            読み込まれたデータのDataFrame
        """
        if directory is None:
            directory = self.processed_dir
        else:
            directory = Path(directory)
        
        filepath = directory / f"{name}.csv"
        
        if use_index:
            return pd.read_csv(filepath, index_col=0)
        else:
            return pd.read_csv(filepath)
    
    def phq9_condition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        PHQ-9質問票の回答を臨床的カットオフに基づいて二値化します
        
        ルール:
            - 2以上 → 1 (陽性反応)
            - 1 → 0 (陰性反応)
        
        元のスクリプトに従い、以下の列を削除します:
        - 'Unnamed: 6', 'Unnamed: 11', 'Unnamed: 21', 'Unnamed: 31', 'Unnamed: 41'
        
        Args:
            data: 元のPHQ-9回答データ
            
        Returns:
            二値化されたPHQ-9データ
        """
        def binarize(x):
            if pd.isna(x):
                return np.nan
            try:
                # 文字列の場合は数値に変換
                x = float(x)
                if x >= 2:
                    return 1
                elif x == 1:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        data_processed = data.applymap(binarize)
        
        # 元のスクリプトに従い、不要な列を削除（列インデックス6, 11, 21, 31, 41）
        columns_to_drop_indices = [6, 11, 21, 31, 41]
        existing_columns_to_drop = [data_processed.columns[i] for i in columns_to_drop_indices if i < len(data_processed.columns)]
        if existing_columns_to_drop:
            data_processed = data_processed.drop(columns=existing_columns_to_drop)
        
        # 最初の36列のみを使用（4×9=36）
        if data_processed.shape[1] > 36:
            data_processed = data_processed.iloc[:, :36]
        
        return data_processed
    
    def hq25_condition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        HQ-25質問票の回答を臨床的カットオフに基づいて二値化します
        
        ルール:
            - 4以上 → 1 (陽性反応)
            - 1以上閾値未満 → 0 (陰性反応)
        
        閾値は初期化時に設定された値を使用します（デフォルト: 3）
        
        Args:
            data: 元のHQ-25回答データ
            
        Returns:
            二値化されたHQ-25データ
        """
        def binarize(x):
            if pd.isna(x):
                return np.nan
            try:
                # 文字列の場合は数値に変換
                x = float(x)
                if x >= self.hq25_threshold:
                    return 1
                elif 1 <= x < self.hq25_threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        data_processed = data.applymap(binarize)
        
        # Excelエクスポートで生成される可能性のある不要な列を削除
        drop_columns = [
            "Unnamed: 21", "Unnamed: 51", "Unnamed: 81",
            "Unnamed: 111", "Unnamed: 141"
        ]
        for col in drop_columns:
            if col in data_processed.columns:
                data_processed = data_processed.drop(columns=[col])
        
        return data_processed
    
    def identity_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        恒等変換 - 入力DataFrameをそのまま返します
        
        一貫した処理パイプラインを維持するのに有用です
        
        Args:
            data: 入力データ
            
        Returns:
            変更されていない入力データ
        """
        return data.copy()
    
    def ips22_condition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        IPS-22質問票の回答を臨床的カットオフに基づいて二値化します
        
        ルール:
            - 閾値以上 → 1 (陽性反応)
            - 1以上閾値未満 → 0 (陰性反応)
        
        Args:
            data: 元のIPS-22回答データ
            
        Returns:
            二値化されたIPS-22データ
        """
        def binarize(x):
            if pd.isna(x):
                return np.nan
            try:
                # 文字列の場合は数値に変換
                x = float(x)
                if x >= self.ips22_threshold:
                    return 1
                elif 1 <= x < self.ips22_threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        data_processed = data.applymap(binarize)
        
        # 不要な列を削除
        drop_columns = ["Unnamed: 60", "Unnamed: 83"]
        for col in drop_columns:
            if col in data_processed.columns:
                data_processed = data_processed.drop(columns=[col])
        
        return data_processed
    
    def iat_condition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        IAT質問票の回答を臨床的カットオフに基づいて二値化します
        
        ルール:
            - 閾値以上 → 1 (陽性反応)
            - 1以上閾値未満 → 0 (陰性反応)
        
        Args:
            data: 元のIAT回答データ
            
        Returns:
            二値化されたIATデータ
        """
        def binarize(x):
            if pd.isna(x):
                return np.nan
            try:
                # 文字列の場合は数値に変換
                x = float(x)
                if x >= self.iat_threshold:
                    return 1
                elif 1 <= x < self.iat_threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        return data.applymap(binarize)
    
    def tacs22_condition(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        TACS-22質問票の回答を臨床的カットオフに基づいて二値化します
        
        ルール:
            - 閾値以上 → 1 (陽性反応)
            - 1以上閾値未満 → 0 (陰性反応)
        
        Args:
            data: 元のTACS-22回答データ
            
        Returns:
            二値化されたTACS-22データ
        """
        def binarize(x):
            if pd.isna(x):
                return np.nan
            try:
                # 文字列の場合は数値に変換
                x = float(x)
                if x >= self.tacs22_threshold:
                    return 1
                elif 1 <= x < self.tacs22_threshold:
                    return 0
                else:
                    return np.nan
            except (ValueError, TypeError):
                return np.nan
        
        data_processed = data.applymap(binarize)
        
        # 不要な列を削除
        drop_columns = ["Unnamed: 33", "Unnamed: 56", "Unnamed: 79"]
        for col in drop_columns:
            if col in data_processed.columns:
                data_processed = data_processed.drop(columns=[col])
        
        return data_processed
    
    def get_preprocessing_function(self, questionnaire_type: str):
        """
        質問票タイプに応じた前処理関数を返します
        
        Args:
            questionnaire_type: 質問票のタイプ
            
        Returns:
            前処理関数
        """
        preprocessing_map = {
            # 基本質問票
            "PHQ-9": self.phq9_condition,
            "PHQ9-alldata": self.phq9_condition,
            "PHQ9-alldata_9ans": self.phq9_condition,
            "IPS-22": self.ips22_condition,
            "IPS-22-alldata": self.ips22_condition,
            "IAT": self.iat_condition,
            "IAT-alldata": self.iat_condition,
            "HQ-25": self.hq25_condition,
            "HQ-25+4": self.hq25_condition,
            "HQ-25+4_alldata": self.hq25_condition,
            "TACS-22": self.tacs22_condition,
            "TACS-22-alldata": self.tacs22_condition,
            
            # 複合質問票（既に前処理済み）
            "PHQ-9_hq25_tacs22": self.identity_transform,
            "PHQ-9_hq25_iat": self.identity_transform,
            "PHQ-9_hq25_tacs22_iat": self.identity_transform,
            "PHQ-9_tacs22_iat": self.identity_transform,
            
            # PHQ9+HQ25シリーズ
            "PHQ9+HQ25_social_prev2204": self.identity_transform,
            "PHQ9+HQ25_social_2308": self.identity_transform,
            "PHQ9+HQ25_2204": self.identity_transform,
            "PHQ9+HQ25_2108": self.identity_transform,
            "PHQ9+HQ25_2012": self.identity_transform,
            "PHQ9+HQ25_2006": self.identity_transform,
            "PHQ9+HQ25_social_2204": self.identity_transform,
            "PHQ9+HQ25_social_2108": self.identity_transform,
            "PHQ9+HQ25_social_2012": self.identity_transform,
            "PHQ9+HQ25_social_2006": self.identity_transform,
            "PHQ9+HQ25_Isolation_prev2204": self.identity_transform,
            "PHQ9+HQ25_Isolation_2308": self.identity_transform,
            "PHQ9+HQ25_Isolation_2204": self.identity_transform,
            "PHQ9+HQ25_Isolation_2108": self.identity_transform,
            "PHQ9+HQ25_Isolation_2012": self.identity_transform,
            "PHQ9+HQ25_Isolation_2006": self.identity_transform,
            "PHQ9+HQ25_Isolation+": self.identity_transform,
            
            # 統合データ
            "PHQ9+HQ25_social_integrated": self.identity_transform,
            "PHQ9+HQ25_isolation_integrated": self.identity_transform,
            "PHQ9+HQ25_Isolation+social_integrated": self.identity_transform,
            "PHQ9_all": self.identity_transform,
            "HQ25_social_all": self.identity_transform,
            "HQ25_Isolation_all": self.identity_transform,
        }
        
        if questionnaire_type not in preprocessing_map:
            available_types = list(preprocessing_map.keys())
            raise ValueError(f"Unknown questionnaire type '{questionnaire_type}'. "
                           f"Available types: {available_types}")
        
        return preprocessing_map[questionnaire_type]
    
    def process_questionnaire_data(self, 
                                 questionnaire_type: str,
                                 data_source: str = "auto",
                                 column_range: Optional[tuple] = None,
                                 **kwargs) -> pd.DataFrame:
        """
        質問票データの完全な処理パイプラインを実行します
        
        Args:
            questionnaire_type: 質問票のタイプ
            data_source: データソース（"auto", "csv", "excel"）
            column_range: 列範囲のタプル (start, end) または None
            **kwargs: データ読み込み関数への追加引数
            
        Returns:
            処理済みのデータ
        """
        # データの読み込み
        if data_source == "auto":
            # 自動判別：まず処理済みCSVを探し、なければオリジナルExcelを探す
            if self.is_processed_data_available(questionnaire_type):
                raw_data = self.load_csv_data(questionnaire_type, use_index=True, **kwargs)
            elif self.is_original_data_available(questionnaire_type):
                raw_data = self.load_excel_data(questionnaire_type, **kwargs)
            else:
                raise FileNotFoundError(f"Data for '{questionnaire_type}' not found in either processed or original data")
        elif data_source == "csv":
            raw_data = self.load_csv_data(questionnaire_type, use_index=True, **kwargs)
        elif data_source == "excel":
            raw_data = self.load_excel_data(questionnaire_type, **kwargs)
        else:
            raise ValueError("data_source must be 'auto', 'csv', or 'excel'")
        
        # 前処理関数の取得と適用
        preprocess_func = self.get_preprocessing_function(questionnaire_type)
        processed_data = preprocess_func(raw_data)
        
        # 列範囲の適用
        if column_range is not None:
            start_col, end_col = column_range
            if end_col > len(processed_data.columns):
                end_col = len(processed_data.columns)
            if start_col < 0:
                start_col = 0
            processed_data = processed_data.iloc[:, start_col:end_col]
        
        # データ型の変換とクリーニング
        processed_data = processed_data.astype(int, errors="ignore")
        processed_data = processed_data.dropna(axis=1)
        processed_data = processed_data.astype(int)
        
        return processed_data
    
    def get_available_original_sheets(self) -> Dict[str, list[str]]:
        """
        利用可能なオリジナルExcelファイルとシートを取得します
        
        Returns:
            ファイル名をキー、シート名のリストを値とする辞書
        """
        available_files = {}
        
        for excel_file in ["data.xlsx", "data2.xlsx"]:
            filepath = self.original_dir / excel_file
            if filepath.exists():
                try:
                    df_sheet_all = pd.read_excel(filepath, sheet_name=None, index_col=0, engine="openpyxl")
                    available_files[excel_file] = list(df_sheet_all.keys())
                except Exception as e:
                    print(f"Warning: Could not read {excel_file}: {e}")
        
        return available_files
    
    def get_available_processed_files(self) -> list[str]:
        """
        利用可能な処理済みCSVファイルを取得します
        
        Returns:
            ファイル名（拡張子なし）のリスト
        """
        if not self.processed_dir.exists():
            return []
        
        csv_files = []
        for file_path in self.processed_dir.glob("*.csv"):
            csv_files.append(file_path.stem)
        
        return sorted(csv_files)
    
    def is_original_data_available(self, sheet_name: str) -> bool:
        """
        指定されたシート名のオリジナルデータが利用可能かチェックします
        
        Args:
            sheet_name: シート名
            
        Returns:
            利用可能な場合True
        """
        available_files = self.get_available_original_sheets()
        for sheets in available_files.values():
            if sheet_name in sheets:
                return True
        return False
    
    def is_processed_data_available(self, file_name: str) -> bool:
        """
        指定されたファイル名の処理済みデータが利用可能かチェックします
        
        Args:
            file_name: ファイル名（拡張子なし）
            
        Returns:
            利用可能な場合True
        """
        available_files = self.get_available_processed_files()
        return file_name in available_files
    
    def get_time_period_ranges(self, questionnaire_type: str) -> Dict[str, tuple]:
        """
        質問票タイプに応じた年代別列範囲を取得
        
        Args:
            questionnaire_type: 質問票のタイプ
            
        Returns:
            年代別の列範囲辞書 {年代: (start_col, end_col)}
        """
        time_period_ranges = {
            "HQ-25+4": {
                "2006": (0, 29),
                "2012": (29, 58), 
                "2108": (58, 87),
                "2204": (87, 116)
            },
            "PHQ-9": {
                "2006": (0, 9),
                "2012": (9, 18),
                "2108": (18, 27), 
                "2204": (27, 36)
            },
            "IPS-22": {
                "2006": (0, 22),
                "2012": (22, 44),
                "2108": (44, 66),
                "2204": (66, 88)
            },
            "IAT": {
                "2006": (0, 20),
                "2012": (20, 40),
                "2108": (40, 60),
                "2204": (60, 80)
            },
            "TACS-22": {
                "2006": (0, 22),
                "2012": (22, 44),
                "2108": (44, 66),
                "2204": (66, 88)
            }
        }
        
        return time_period_ranges.get(questionnaire_type, {})
    
    def process_phq9_with_reshape(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        PHQ-9データを元のスクリプトに従って処理し、年代別に分割します
        
        元のスクリプトの処理:
        1. 不要な列を削除
        2. 全データを前処理
        3. reshape(rows * 4, 9) で年代別に分割
        
        Args:
            data: 元のPHQ-9データ
            
        Returns:
            年代別に分割されたデータの辞書
        """
        # 前処理（二値化）- phq9_condition内で列削除も行われる
        processed_data = self.phq9_condition(data)
        
        # リシェイプ: 各行を4つの年代に分割
        reshaped_data = np.array(processed_data).reshape(processed_data.shape[0] * 4, 9)
        reshaped_df = pd.DataFrame(reshaped_data)
        
        # 年代別に分割
        time_periods = ["2006", "2012", "2108", "2204"]
        results = {}
        
        for i, period in enumerate(time_periods):
            start_row = i * processed_data.shape[0]
            end_row = (i + 1) * processed_data.shape[0]
            period_data = reshaped_df.iloc[start_row:end_row].copy()
            
            # 有効データが少ない行を削除（元のスクリプトの処理）
            period_data = period_data.dropna(axis=0, how='all')
            
            results[period] = period_data
        
        return results
    
    def process_questionnaire_data_by_time_period(self, questionnaire_type: str,
                                                time_period: str,
                                                data_source: str = "auto") -> pd.DataFrame:
        """
        年代別に質問票データを処理
        
        Args:
            questionnaire_type: 質問票のタイプ
            time_period: 年代（"2006", "2012", "2108", "2204"）
            data_source: データソース（"auto", "csv", "excel"）
            
        Returns:
            指定された年代の処理済みデータ
        """
        # PHQ-9の場合は特別な処理
        if questionnaire_type == "PHQ-9" and data_source in ["auto", "excel"]:
            # 全データを読み込んでリシェイプ処理
            data = self.load_excel_data(questionnaire_type)
            reshaped_results = self.process_phq9_with_reshape(data)
            return reshaped_results[time_period]
        
        # 年代別列範囲を取得
        time_ranges = self.get_time_period_ranges(questionnaire_type)
        if time_period not in time_ranges:
            available_periods = list(time_ranges.keys())
            raise ValueError(f"Time period '{time_period}' not available for {questionnaire_type}. Available: {available_periods}")
        
        column_range = time_ranges[time_period]
        
        # 通常の処理を実行
        return self.process_questionnaire_data(
            questionnaire_type=questionnaire_type,
            data_source=data_source,
            column_range=column_range
        )
    
    def process_all_time_periods(self, questionnaire_type: str,
                               data_source: str = "auto") -> Dict[str, pd.DataFrame]:
        """
        全年代の質問票データを処理
        
        Args:
            questionnaire_type: 質問票のタイプ
            data_source: データソース（"auto", "csv", "excel"）
            
        Returns:
            年代別の処理済みデータ辞書 {年代: DataFrame}
        """
        time_ranges = self.get_time_period_ranges(questionnaire_type)
        if not time_ranges:
            raise ValueError(f"No time period ranges defined for {questionnaire_type}")
        
        results = {}
        for time_period in time_ranges.keys():
            try:
                data = self.process_questionnaire_data_by_time_period(
                    questionnaire_type=questionnaire_type,
                    time_period=time_period,
                    data_source=data_source
                )
                results[time_period] = data
                print(f"✅ {questionnaire_type} {time_period}: {data.shape}")
            except Exception as e:
                print(f"❌ {questionnaire_type} {time_period}: {e}")
                results[time_period] = None
        
        return results


# 後方互換性のための関数（非推奨）
def load_data(column: str = "IPS-22", filepath: str = "../inputfiles/data.xlsx") -> pd.DataFrame:
    """後方互換性のための関数（非推奨）"""
    processor = DataProcessor()
    return processor.load_excel_data(column, filepath)


def load_data2(name: str, csv_dir: str = "../Data_csv") -> pd.DataFrame:
    """後方互換性のための関数（非推奨）"""
    processor = DataProcessor(csv_dir=csv_dir)
    return processor.load_csv_data(name)


def load_data3(name: str, data_dir: str = "../inputfiles") -> pd.DataFrame:
    """後方互換性のための関数（非推奨）"""
    processor = DataProcessor(data_dir=data_dir)
    return processor.load_csv_data(name, use_index=True)


def phq9_cond(data: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数（非推奨）"""
    processor = DataProcessor()
    return processor.phq9_condition(data)


def hq25_cond(data: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """後方互換性のための関数（非推奨）"""
    processor = DataProcessor(hq25_threshold=threshold)
    return processor.hq25_condition(data)


def stay(data: pd.DataFrame) -> pd.DataFrame:
    """後方互換性のための関数（非推奨）"""
    processor = DataProcessor()
    return processor.identity_transform(data)


# 年代別分割機能をDataProcessorクラスに追加
def add_time_period_methods():
    """DataProcessorクラスに年代別分割メソッドを動的に追加"""
    
    def get_time_period_ranges(self, questionnaire_type: str) -> Dict[str, tuple]:
        """
        質問票タイプに応じた年代別列範囲を取得
        
        Args:
            questionnaire_type: 質問票のタイプ
            
        Returns:
            年代別の列範囲辞書 {年代: (start_col, end_col)}
        """
        time_period_ranges = {
            "HQ-25+4": {
                "2006": (0, 29),
                "2012": (29, 58), 
                "2108": (58, 87),
                "2204": (87, 116)
            },
            "PHQ-9": {
                "2006": (0, 9),
                "2012": (9, 18),
                "2108": (18, 27), 
                "2204": (27, 36)
            },
            "IPS-22": {
                "2006": (0, 22),
                "2012": (22, 44),
                "2108": (44, 66),
                "2204": (66, 88)
            },
            "IAT": {
                "2006": (0, 20),
                "2012": (20, 40),
                "2108": (40, 60),
                "2204": (60, 80)
            },
            "TACS-22": {
                "2006": (0, 22),
                "2012": (22, 44),
                "2108": (44, 66),
                "2204": (66, 88)
            }
        }
        
        return time_period_ranges.get(questionnaire_type, {})
    
    def process_phq9_with_reshape(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        PHQ-9データを元のスクリプトに従って処理し、年代別に分割します
        
        元のスクリプトの処理:
        1. 不要な列を削除
        2. 全データを前処理
        3. reshape(rows * 4, 9) で年代別に分割
        
        Args:
            data: 元のPHQ-9データ
            
        Returns:
            年代別に分割されたデータの辞書
        """
        # 前処理（二値化）- phq9_condition内で列削除も行われる
        processed_data = self.phq9_condition(data)
        
        # リシェイプ: 各行を4つの年代に分割
        reshaped_data = np.array(processed_data).reshape(processed_data.shape[0] * 4, 9)
        reshaped_df = pd.DataFrame(reshaped_data)
        
        # 年代別に分割
        time_periods = ["2006", "2012", "2108", "2204"]
        results = {}
        
        for i, period in enumerate(time_periods):
            start_row = i * processed_data.shape[0]
            end_row = (i + 1) * processed_data.shape[0]
            period_data = reshaped_df.iloc[start_row:end_row].copy()
            
            # 有効データが少ない行を削除（元のスクリプトの処理）
            period_data = period_data.dropna(axis=0, how='all')
            
            results[period] = period_data
        
        return results
    
    def process_questionnaire_data_by_time_period(self, questionnaire_type: str,
                                                time_period: str,
                                                data_source: str = "auto") -> pd.DataFrame:
        """
        年代別に質問票データを処理
        
        Args:
            questionnaire_type: 質問票のタイプ
            time_period: 年代（"2006", "2012", "2108", "2204"）
            data_source: データソース（"auto", "csv", "excel"）
            
        Returns:
            指定された年代の処理済みデータ
        """
        # PHQ-9の場合は特別な処理
        if questionnaire_type == "PHQ-9" and data_source in ["auto", "excel"]:
            # 全データを読み込んでリシェイプ処理
            data = self.load_excel_data(questionnaire_type)
            reshaped_results = self.process_phq9_with_reshape(data)
            return reshaped_results[time_period]
        
        # 年代別列範囲を取得
        time_ranges = self.get_time_period_ranges(questionnaire_type)
        if time_period not in time_ranges:
            available_periods = list(time_ranges.keys())
            raise ValueError(f"Time period '{time_period}' not available for {questionnaire_type}. Available: {available_periods}")
        
        column_range = time_ranges[time_period]
        
        # 通常の処理を実行
        return self.process_questionnaire_data(
            questionnaire_type=questionnaire_type,
            data_source=data_source,
            column_range=column_range
        )
    
    def process_all_time_periods(self, questionnaire_type: str,
                               data_source: str = "auto") -> Dict[str, pd.DataFrame]:
        """
        全年代の質問票データを処理
        
        Args:
            questionnaire_type: 質問票のタイプ
            data_source: データソース（"auto", "csv", "excel"）
            
        Returns:
            年代別の処理済みデータ辞書 {年代: DataFrame}
        """
        time_ranges = self.get_time_period_ranges(questionnaire_type)
        if not time_ranges:
            raise ValueError(f"No time period ranges defined for {questionnaire_type}")
        
        results = {}
        for time_period in time_ranges.keys():
            try:
                data = self.process_questionnaire_data_by_time_period(
                    questionnaire_type=questionnaire_type,
                    time_period=time_period,
                    data_source=data_source
                )
                results[time_period] = data
                print(f"✅ {questionnaire_type} {time_period}: {data.shape}")
            except Exception as e:
                print(f"❌ {questionnaire_type} {time_period}: {e}")
                results[time_period] = None
        
        return results


