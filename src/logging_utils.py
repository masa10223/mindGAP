"""
ログ管理ユーティリティ

引数に基づいてログファイルを適切に出力し、パラメータと結果を記録します。
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json
import argparse
import pytz


def get_jst_time():
    """日本時間を取得"""
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst)


class JSTFormatter(logging.Formatter):
    """日本時間を使用するログフォーマッター"""
    
    def formatTime(self, record, datefmt=None):
        """ログの時刻を日本時間でフォーマット"""
        jst_time = get_jst_time()
        if datefmt:
            return jst_time.strftime(datefmt)
        else:
            return jst_time.strftime('%Y-%m-%d %H:%M:%S')


class ELALogger:
    """
    ELA解析用のログ管理クラス
    
    引数に基づいてログファイルを生成し、パラメータと結果を記録します。
    """
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 log_level: str = "INFO",
                 enable_console: bool = True):
        """
        ログ管理の初期化
        
        Args:
            log_dir: ログディレクトリ
            log_level: ログレベル
            enable_console: コンソール出力を有効にするか
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_level = log_level
        self.enable_console = enable_console
        
        # ログファイル名（後で設定）
        self.log_file = None
        self.logger = None
    
    def setup_logger(self, 
                    questionnaire_type: str,
                    args: argparse.Namespace,
                    job_type: str = "normal") -> str:
        """
        ログファイルとロガーを設定します
        
        Args:
            questionnaire_type: 質問票のタイプ
            args: コマンドライン引数
            job_type: ジョブタイプ（normal, fixed_w等）
            
        Returns:
            ログファイルのパス
        """
        # ログファイル名の生成（日本時間）
        timestamp = get_jst_time().strftime("%Y%m%d_%H%M%S")
        
        # パラメータを含むファイル名
        filename_parts = [questionnaire_type]
        
        # 列範囲の追加
        if hasattr(args, 'r1') and hasattr(args, 'r2'):
            if args.r2 is not None:
                filename_parts.append(f"r{args.r1}_{args.r2}")
            elif args.r1 != 0:
                filename_parts.append(f"r{args.r1}_end")
        
        # 閾値の追加
        if hasattr(args, 'hq25_threshold') and args.hq25_threshold:
            filename_parts.append(f"hq25_{args.hq25_threshold}")
        
        if hasattr(args, 'ips22_threshold') and args.ips22_threshold:
            filename_parts.append(f"ips22_{args.ips22_threshold}")
        
        if hasattr(args, 'iat_threshold') and args.iat_threshold:
            filename_parts.append(f"iat_{args.iat_threshold}")
        
        if hasattr(args, 'tacs22_threshold') and args.tacs22_threshold:
            filename_parts.append(f"tacs22_{args.tacs22_threshold}")
        
        # 固定Wの追加
        if job_type == "fixed_w" and hasattr(args, 'fixedW'):
            filename_parts.append(f"fixedW_{args.fixedW}")
        
        # デバイスIDの追加
        if hasattr(args, 'device_id') and args.device_id != 0:
            filename_parts.append(f"dev{args.device_id}")
        
        # ファイル名の構築
        base_filename = "_".join(filename_parts)
        self.log_file = self.log_dir / f"{base_filename}_{timestamp}.log"
        
        # ロガーの設定
        self.logger = logging.getLogger(f"ELA_{base_filename}")
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # 既存のハンドラーをクリア
        self.logger.handlers.clear()
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_formatter = JSTFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # コンソールハンドラー
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = JSTFormatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # パラメータの記録
        self.log_parameters(args, job_type)
        
        return str(self.log_file)
    
    def log_parameters(self, args: argparse.Namespace, job_type: str = "normal"):
        """
        パラメータをログに記録します
        
        Args:
            args: コマンドライン引数
            job_type: ジョブタイプ
        """
        if not self.logger:
            return
        
        self.logger.info("=" * 60)
        self.logger.info("ELA Analysis Parameters")
        self.logger.info("=" * 60)
        
        # 基本パラメータ
        self.logger.info(f"Job Type: {job_type}")
        self.logger.info(f"Questionnaire Type: {getattr(args, 'func', getattr(args, 'filename', 'Unknown'))}")
        
        # 列範囲
        if hasattr(args, 'r1') and hasattr(args, 'r2'):
            if args.r2 is not None:
                self.logger.info(f"Column Range: {args.r1} to {args.r2}")
            elif args.r1 != 0:
                self.logger.info(f"Column Range: {args.r1} to end")
            else:
                self.logger.info("Column Range: All columns")
        
        # 閾値パラメータ
        thresholds = []
        if hasattr(args, 'hq25_threshold') and args.hq25_threshold:
            thresholds.append(f"HQ-25: {args.hq25_threshold}")
        if hasattr(args, 'ips22_threshold') and args.ips22_threshold:
            thresholds.append(f"IPS-22: {args.ips22_threshold}")
        if hasattr(args, 'iat_threshold') and args.iat_threshold:
            thresholds.append(f"IAT: {args.iat_threshold}")
        if hasattr(args, 'tacs22_threshold') and args.tacs22_threshold:
            thresholds.append(f"TACS-22: {args.tacs22_threshold}")
        
        if thresholds:
            self.logger.info(f"Thresholds: {', '.join(thresholds)}")
        
        # 固定Wパラメータ
        if job_type == "fixed_w" and hasattr(args, 'fixedW'):
            self.logger.info(f"Fixed W Type: {args.fixedW}")
        
        # システムパラメータ
        if hasattr(args, 'device_id'):
            self.logger.info(f"CUDA Device ID: {args.device_id}")
        
        if hasattr(args, 'data_dir'):
            self.logger.info(f"Data Directory: {args.data_dir}")
        
        if hasattr(args, 'output_dir'):
            self.logger.info(f"Output Directory: {args.output_dir}")
        
        # その他のパラメータ
        other_params = []
        for attr in dir(args):
            if not attr.startswith('_') and attr not in [
                'func', 'filename', 'r1', 'r2', 'hq25_threshold', 'ips22_threshold',
                'iat_threshold', 'tacs22_threshold', 'fixedW', 'device_id',
                'data_dir', 'output_dir', 'verbose', 'config', 'create_config'
            ]:
                value = getattr(args, attr)
                if value is not None and value != False:
                    other_params.append(f"{attr}: {value}")
        
        if other_params:
            self.logger.info(f"Other Parameters: {', '.join(other_params)}")
        
        self.logger.info("=" * 60)
    
    def log_start(self, message: str = "Starting ELA Analysis"):
        """解析開始をログに記録"""
        if self.logger:
            self.logger.info(f"🚀 {message}")
    
    def log_step(self, step: str, elapsed_time: Optional[float] = None):
        """解析ステップをログに記録"""
        if self.logger:
            if elapsed_time:
                self.logger.info(f"✅ {step} (Completed in {elapsed_time:.2f}s)")
            else:
                self.logger.info(f"⏳ {step}")
    
    def log_debug(self, message: str):
        """デバッグ情報をログに記録"""
        if self.logger:
            self.logger.debug(f"🔍 {message}")
    
    def log_info(self, message: str):
        """情報をログに記録"""
        if self.logger:
            self.logger.info(f"ℹ️ {message}")
    
    def log_accuracy_details(self, accuracy_data: Dict[str, Any]):
        """Accuracy計算の詳細情報をログに記録"""
        if self.logger:
            self.logger.info("=== Accuracy計算の詳細情報 ===")
            for key, value in accuracy_data.items():
                if isinstance(value, float):
                    self.logger.info(f"{key}: {value:.6f}")
                else:
                    self.logger.info(f"{key}: {value}")
    
    def log_calculation_details(self, calc_type: str, details: Dict[str, Any]):
        """計算の詳細情報をログに記録"""
        if self.logger:
            self.logger.info(f"=== {calc_type}計算の詳細情報 ===")
            for key, value in details.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        self.logger.info(f"{key}: {value:.6f}")
                    else:
                        self.logger.info(f"{key}: {value}")
                elif hasattr(value, 'shape'):
                    self.logger.info(f"{key}: 形状={value.shape}")
                else:
                    self.logger.info(f"{key}: {value}")
    
    def log_error(self, error: str, exception: Optional[Exception] = None):
        """エラーをログに記録"""
        if self.logger:
            self.logger.error(f"❌ {error}")
            if exception:
                self.logger.error(f"Exception: {str(exception)}")
    
    def log_warning(self, warning: str):
        """警告をログに記録"""
        if self.logger:
            self.logger.warning(f"⚠️ {warning}")
    
    def log_completion(self, results: Dict[str, Any]):
        """解析完了をログに記録"""
        if not self.logger:
            return
        
        self.logger.info("=" * 60)
        self.logger.info("ELA Analysis Completed Successfully")
        self.logger.info("=" * 60)
        
        # 結果のサマリー
        if 'elapsed_time' in results:
            self.logger.info(f"Total Elapsed Time: {results['elapsed_time']:.2f} seconds")
        
        # データの形状
        if 'processed_data' in results:
            data_shape = results['processed_data'].shape
            self.logger.info(f"Data Shape: {data_shape[0]} variables, {data_shape[1]} samples")
        
        # その他の結果
        for key, value in results.items():
            if key not in ['elapsed_time', 'processed_data'] and value is not None:
                if hasattr(value, 'shape'):
                    self.logger.info(f"{key}: {value.shape}")
                else:
                    self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 60)
    
    def save_results_summary(self, results: Dict[str, Any], args: argparse.Namespace):
        """
        結果のサマリーをJSONファイルに保存
        
        Args:
            results: 解析結果
            args: コマンドライン引数
        """
        if not self.log_file:
            return
        
        # サマリーファイルのパス
        if isinstance(self.log_file, Path):
            summary_file = self.log_file.parent / f"{self.log_file.stem}_summary.json"
        else:
            summary_file = str(self.log_file).replace('.log', '_summary.json')
        
        # サマリーの構築
        summary = {
            'timestamp': get_jst_time().isoformat(),
            'parameters': vars(args),
            'results': {
                'elapsed_time': results.get('elapsed_time'),
                'data_shape': results.get('processed_data', {}).shape if 'processed_data' in results else None,
                'status': 'success' if 'elapsed_time' in results else 'failed'
            }
        }
        
        # 結果の詳細（形状情報のみ）
        for key, value in results.items():
            if hasattr(value, 'shape'):
                summary['results'][f'{key}_shape'] = value.shape
        
        # JSONファイルに保存
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            if self.logger:
                self.logger.info(f"Results summary saved to: {summary_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save results summary: {e}")


def create_logger_from_args(args: argparse.Namespace, 
                          job_type: str = "normal") -> ELALogger:
    """
    引数からロガーを作成します
    
    Args:
        args: コマンドライン引数
        job_type: ジョブタイプ
        
    Returns:
        設定済みのロガー
    """
    log_dir = getattr(args, 'log_dir', './logs')
    log_level = getattr(args, 'log_level', 'INFO')
    verbose = getattr(args, 'verbose', False)
    
    logger = ELALogger(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=verbose
    )
    
    # 質問票タイプの取得
    questionnaire_type = getattr(args, 'func', getattr(args, 'filename', 'unknown'))
    
    # ロガーの設定
    logger.setup_logger(questionnaire_type, args, job_type)
    
    return logger
