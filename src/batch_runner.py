#!/usr/bin/env python3
"""
一括実行スクリプト

複数のELA解析を一括で実行し、詳細なログを取得します。
既存の実行履歴システムと連携して、実行結果を追跡可能です。
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import concurrent.futures
import threading
from dataclasses import dataclass
import pytz
import psutil
import os

from .execution_history import ExecutionHistoryManager


def get_jst_time():
    """日本時間を取得"""
    jst = pytz.timezone('Asia/Tokyo')
    return datetime.now(jst)


def get_memory_usage():
    """メモリ使用量を取得"""
    try:
        # システム全体のメモリ使用量
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent
        }
    except Exception as e:
        return {'error': str(e)}


def get_gpu_memory_usage():
    """GPUメモリ使用量を取得"""
    try:
        import cupy as cp
        # デバイスを明示的に設定
        device_id = cp.cuda.Device().id
        mempool = cp.get_default_memory_pool()
        
        # GPUメモリ情報を取得
        meminfo = cp.cuda.runtime.memGetInfo()
        total_memory = meminfo[1]  # 総メモリ
        free_memory = meminfo[0]   # 空きメモリ
        used_memory = total_memory - free_memory
        
        return {
            'device_id': device_id,
            'used_bytes': used_memory,
            'total_bytes': total_memory,
            'free_bytes': free_memory,
            'used_gb': used_memory / (1024**3),
            'total_gb': total_memory / (1024**3),
            'free_gb': free_memory / (1024**3),
            'mempool_used_gb': mempool.used_bytes() / (1024**3)
        }
    except Exception as e:
        return {'error': str(e)}


@dataclass
class BatchJob:
    """一括実行ジョブの定義"""
    name: str
    questionnaire_type: str
    data_source: str = "auto"
    hq25_threshold: int = 3
    ips22_threshold: int = 4
    iat_threshold: int = 3
    tacs22_threshold: int = 4
    column_range: Optional[tuple] = None
    column_indices: Optional[List[int]] = None
    time_period: Optional[str] = None
    device_id: int = 0
    notes: Optional[str] = None


class BatchRunner:
    """一括実行管理クラス"""
    
    def __init__(self, output_dir: str = "batch_outputs", max_workers: int = 1):
        """
        一括実行の初期化
        
        Args:
            output_dir: 一括実行の出力ディレクトリ
            max_workers: 並列実行の最大ワーカー数
        """
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.output_dir = ela_analysis_dir / output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_workers = max_workers
        self.history_manager = ExecutionHistoryManager()
        
        # 一括実行ログファイル（JST）
        jst_time = get_jst_time()
        self.batch_log_file = self.output_dir / f"batch_log_{jst_time.strftime('%Y%m%d_%H%M%S')}.json"
        self.batch_results = []
        
        # ロック（並列実行時用）
        self.lock = threading.Lock()
    
    def create_job_from_config(self, config: Dict[str, Any]) -> BatchJob:
        """設定辞書からBatchJobを作成"""
        return BatchJob(
            name=config.get('name', config.get('questionnaire_type', 'unknown')),
            questionnaire_type=config['questionnaire_type'],
            data_source=config.get('data_source', 'auto'),
            hq25_threshold=config.get('hq25_threshold', 3),
            ips22_threshold=config.get('ips22_threshold', 4),
            iat_threshold=config.get('iat_threshold', 3),
            tacs22_threshold=config.get('tacs22_threshold', 4),
            column_range=tuple(config['column_range']) if config.get('column_range') else None,
            column_indices=config.get('column_indices'),
            time_period=config.get('time_period'),
            device_id=config.get('device_id', 0),
            notes=config.get('notes')
        )
    
    def run_single_job(self, job: BatchJob) -> Dict[str, Any]:
        """単一ジョブの実行"""
        start_time = time.time()
        
        # 実行前のメモリ使用量を記録
        memory_before = get_memory_usage()
        gpu_memory_before = get_gpu_memory_usage()
        
        print(f"実行開始: {job.name} ({job.questionnaire_type})")
        if 'error' not in memory_before:
            print(f"  メモリ使用量: {memory_before['used_gb']:.1f}GB / {memory_before['total_gb']:.1f}GB ({memory_before['percent']:.1f}%)")
        if 'error' not in gpu_memory_before:
            print(f"  GPUメモリ使用量: {gpu_memory_before['used_gb']:.1f}GB / {gpu_memory_before['total_gb']:.1f}GB (デバイス{gpu_memory_before.get('device_id', '?')})")
        
        # コマンドの構築
        cmd = [
            sys.executable, "-m", "src.ela_pipeline",
            "--func", job.questionnaire_type,
            "--data_source", job.data_source,
            "--hq25_threshold", str(job.hq25_threshold),
            "--ips22_threshold", str(job.ips22_threshold),
            "--iat_threshold", str(job.iat_threshold),
            "--tacs22_threshold", str(job.tacs22_threshold),
            "--device_id", str(job.device_id)
        ]
        
        # 列選択の追加
        if job.column_indices:
            cmd.extend(["--column_indices"] + [str(i) for i in job.column_indices])
        elif job.column_range:
            cmd.extend(["--r1", str(job.column_range[0]), "--r2", str(job.column_range[1])])
        
        # 年代指定の追加
        if job.time_period:
            cmd.extend(["--time_period", job.time_period])
        
        # 実行
        try:
            print(f"実行開始: {job.name} ({job.questionnaire_type})")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 結果の記録
            job_result = {
                'job_name': job.name,
                'questionnaire_type': job.questionnaire_type,
                'data_source': job.data_source,
                'status': 'success' if result.returncode == 0 else 'failed',
                'duration_seconds': duration,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd),
                'timestamp': datetime.now().isoformat(),
                'notes': job.notes
            }
            
            # 実行後のメモリ使用量を記録
            memory_after = get_memory_usage()
            gpu_memory_after = get_gpu_memory_usage()
            
            if result.returncode == 0:
                print(f"✅ 完了: {job.name} ({duration:.1f}秒)")
            else:
                print(f"❌ 失敗: {job.name} ({duration:.1f}秒)")
                print(f"   エラー: {result.stderr.strip()}")
            
            # メモリ使用量の変化を表示
            if 'error' not in memory_after and 'error' not in memory_before:
                memory_diff = memory_after['used_gb'] - memory_before['used_gb']
                print(f"  メモリ変化: {memory_diff:+.1f}GB (現在: {memory_after['used_gb']:.1f}GB)")
            
            if 'error' not in gpu_memory_after and 'error' not in gpu_memory_before:
                gpu_memory_diff = gpu_memory_after['used_gb'] - gpu_memory_before['used_gb']
                print(f"  GPUメモリ変化: {gpu_memory_diff:+.1f}GB (現在: {gpu_memory_after['used_gb']:.1f}GB)")
            
            return job_result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            job_result = {
                'job_name': job.name,
                'questionnaire_type': job.questionnaire_type,
                'data_source': job.data_source,
                'status': 'error',
                'duration_seconds': duration,
                'return_code': -1,
                'stdout': '',
                'stderr': str(e),
                'command': ' '.join(cmd),
                'timestamp': datetime.now().isoformat(),
                'notes': job.notes
            }
            
            print(f"❌ エラー: {job.name} ({duration:.1f}秒)")
            print(f"   エラー: {str(e)}")
            
            return job_result
    
    def run_batch(self, jobs: List[BatchJob], parallel: bool = False) -> Dict[str, Any]:
        """
        一括実行の実行
        
        Args:
            jobs: 実行するジョブのリスト
            parallel: 並列実行するかどうか
            
        Returns:
            一括実行結果の辞書
        """
        start_time = time.time()
        start_time_jst = get_jst_time()
        print(f"一括実行開始: {len(jobs)}件のジョブ")
        print(f"開始時刻: {start_time_jst.strftime('%Y-%m-%d %H:%M:%S JST')}")
        print(f"並列実行: {'有効' if parallel else '無効'}")
        
        # GPU使用時の並列実行設定を自動調整
        if parallel:
            # 使用されるdevice_idを確認
            device_ids = set(getattr(job, 'device_id', 0) for job in jobs)
            max_device_id = max(device_ids) if device_ids else 0
            
            # GPU使用時は最大ワーカー数を調整
            if max_device_id > 0:
                # GPU使用時は各デバイスごとに1ワーカーを推奨
                recommended_workers = min(len(device_ids), self.max_workers)
                print(f"GPU使用検出: {len(device_ids)}個のデバイス (ID: {sorted(device_ids)})")
                print(f"推奨ワーカー数: {recommended_workers}")
                self.max_workers = recommended_workers
            else:
                print(f"CPU実行: 最大ワーカー数 {self.max_workers}")
        
        # 実行
        if parallel and self.max_workers > 1:
            results = self._run_parallel(jobs)
        else:
            results = self._run_sequential(jobs)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 結果の集計
        successful_jobs = [r for r in results if r['status'] == 'success']
        failed_jobs = [r for r in results if r['status'] in ['failed', 'error']]
        
        batch_summary = {
            'batch_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'total_duration_seconds': total_duration,
            'total_jobs': len(jobs),
            'successful_jobs': len(successful_jobs),
            'failed_jobs': len(failed_jobs),
            'success_rate': len(successful_jobs) / len(jobs) if jobs else 0,
            'parallel_execution': parallel,
            'max_workers': self.max_workers,
            'results': results
        }
        
        # ログファイルに保存
        with open(self.batch_log_file, 'w', encoding='utf-8') as f:
            json.dump(batch_summary, f, ensure_ascii=False, indent=2)
        
        # 結果の表示
        end_time_jst = get_jst_time()
        print(f"\n{'='*60}")
        print(f"一括実行完了")
        print(f"終了時刻: {end_time_jst.strftime('%Y-%m-%d %H:%M:%S JST')}")
        print(f"総実行時間: {total_duration:.1f}秒")
        print(f"成功: {len(successful_jobs)}件")
        print(f"失敗: {len(failed_jobs)}件")
        print(f"成功率: {batch_summary['success_rate']:.1%}")
        print(f"ログファイル: {self.batch_log_file}")
        
        if failed_jobs:
            print(f"\n失敗したジョブ:")
            for job in failed_jobs:
                print(f"  - {job['job_name']}: {job['stderr'].strip()}")
        
        return batch_summary
    
    def _run_sequential(self, jobs: List[BatchJob]) -> List[Dict[str, Any]]:
        """逐次実行"""
        results = []
        for i, job in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] {job.name}")
            result = self.run_single_job(job)
            results.append(result)
            
            with self.lock:
                self.batch_results.append(result)
        return results
    
    def _run_parallel(self, jobs: List[BatchJob]) -> List[Dict[str, Any]]:
        """並列実行（GPU使用時はdevice_idを考慮）"""
        results = []
        
        # GPU使用時の並列実行を考慮
        # 同じdevice_idのジョブは同時実行しないように調整
        device_jobs = {}
        for job in jobs:
            device_id = getattr(job, 'device_id', 0)
            if device_id not in device_jobs:
                device_jobs[device_id] = []
            device_jobs[device_id].append(job)
        
        print(f"並列実行: {len(device_jobs)}個のGPUデバイスを使用")
        for device_id, device_job_list in device_jobs.items():
            print(f"  GPU {device_id}: {len(device_job_list)}ジョブ")
        
        # 各デバイスごとに並列実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # ジョブの実行
            future_to_job = {
                executor.submit(self.run_single_job, job): job 
                for job in jobs
            }
            
            # 結果の収集
            for i, future in enumerate(concurrent.futures.as_completed(future_to_job), 1):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self.lock:
                        self.batch_results.append(result)
                    
                    device_id = getattr(job, 'device_id', 0)
                    print(f"\n[{i}/{len(jobs)}] {job.name} 完了 (GPU {device_id})")
                    
                except Exception as e:
                    device_id = getattr(job, 'device_id', 0)
                    print(f"\n[{i}/{len(jobs)}] {job.name} エラー (GPU {device_id}): {e}")
        
        return results
    
    def load_jobs_from_file(self, config_file: str) -> List[BatchJob]:
        """設定ファイルからジョブを読み込み"""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_file}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        if 'jobs' not in config_data:
            raise ValueError("設定ファイルに'jobs'キーがありません")
        
        jobs = []
        for job_config in config_data['jobs']:
            job = self.create_job_from_config(job_config)
            jobs.append(job)
        
        return jobs
    
    def create_sample_config(self, output_file: str = "batch_config_sample.json"):
        """サンプル設定ファイルを作成"""
        sample_config = {
            "description": "一括実行設定ファイルのサンプル",
            "parallel_execution": True,
            "max_workers": 2,
            "jobs": [
                {
                    "name": "PHQ9_all_analysis",
                    "questionnaire_type": "PHQ9_all",
                    "data_source": "csv",
                    "hq25_threshold": 3,
                    "notes": "PHQ-9の全データ解析"
                },
                {
                    "name": "HQ25_threshold2",
                    "questionnaire_type": "HQ-25+4",
                    "data_source": "excel",
                    "hq25_threshold": 2,
                    "notes": "HQ-25閾値2での解析"
                },
                {
                    "name": "HQ25_threshold3",
                    "questionnaire_type": "HQ-25+4",
                    "data_source": "excel",
                    "hq25_threshold": 3,
                    "notes": "HQ-25閾値3での解析"
                }
            ]
        }
        
        output_path = Path(__file__).parent.parent / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, ensure_ascii=False, indent=2)
        
        print(f"サンプル設定ファイルを作成しました: {output_path}")
        return output_path


def create_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーの作成"""
    parser = argparse.ArgumentParser(
        description="ELA解析一括実行ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # サンプル設定ファイルを作成
  python -m src.batch_runner --create-sample-config

  # 設定ファイルから一括実行
  python -m src.batch_runner --config batch_config.json

  # 逐次実行（並列無効）
  python -m src.batch_runner --config batch_config.json --sequential

  # 並列実行（ワーカー数指定）
  python -m src.batch_runner --config batch_config.json --parallel --max-workers 4

  # 個別ジョブの実行
  python -m src.batch_runner --job PHQ9_all --data-source csv
        """
    )
    
    # 設定ファイル関連
    parser.add_argument('--config', help='一括実行設定ファイル（JSON）')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='サンプル設定ファイルを作成')
    
    # 実行方式
    parser.add_argument('--parallel', action='store_true', 
                       help='並列実行を有効にする')
    parser.add_argument('--sequential', action='store_true', 
                       help='逐次実行を強制する')
    parser.add_argument('--max-workers', type=int, default=2, 
                       help='並列実行の最大ワーカー数（デフォルト: 2）')
    
    # 個別ジョブ実行
    parser.add_argument('--job', help='実行する質問票タイプ')
    parser.add_argument('--data-source', choices=['auto', 'csv', 'excel'], 
                       default='auto', help='データソース')
    parser.add_argument('--hq25-threshold', type=int, default=3, 
                       help='HQ-25閾値')
    
    # 出力設定
    parser.add_argument('--output-dir', default='batch_outputs', 
                       help='一括実行の出力ディレクトリ')
    
    return parser


def main():
    """メイン関数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # サンプル設定ファイルの作成
        if args.create_sample_config:
            runner = BatchRunner()
            runner.create_sample_config()
            return
        
        # 実行方式の決定
        parallel = args.parallel and not args.sequential
        max_workers = args.max_workers if parallel else 1
        
        runner = BatchRunner(
            output_dir=args.output_dir,
            max_workers=max_workers
        )
        
        # ジョブの準備
        jobs = []
        
        if args.config:
            # 設定ファイルから読み込み
            jobs = runner.load_jobs_from_file(args.config)
            print(f"設定ファイルから {len(jobs)} 件のジョブを読み込みました")
            
            # 設定ファイルの並列実行設定を読み込み
            config_path = Path(args.config)
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 設定ファイルの並列実行設定を優先（コマンドライン引数で上書き可能）
            if 'parallel_execution' in config_data:
                config_parallel = config_data['parallel_execution']
                if not args.sequential:  # --sequentialが指定されていない場合のみ
                    parallel = config_parallel
                    print(f"設定ファイルの並列実行設定: {parallel}")
            
            if 'max_workers' in config_data:
                config_max_workers = config_data['max_workers']
                if parallel:  # 並列実行の場合のみ
                    max_workers = config_max_workers
                    print(f"設定ファイルの最大ワーカー数: {max_workers}")
                    runner.max_workers = max_workers
            
        elif args.job:
            # 個別ジョブの実行
            job = BatchJob(
                name=args.job,
                questionnaire_type=args.job,
                data_source=args.data_source,
                hq25_threshold=args.hq25_threshold
            )
            jobs = [job]
            
        else:
            print("エラー: --config または --job を指定してください")
            parser.print_help()
            return
        
        # 一括実行
        if jobs:
            runner.run_batch(jobs, parallel=parallel)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()