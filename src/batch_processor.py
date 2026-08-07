"""
バッチ処理モジュール

複数のELA解析を並列または順次実行するためのバッチ処理システムを提供します。
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import yaml
from dataclasses import dataclass, asdict

try:
    from .ela_pipeline import ELAPipeline
    from .fixed_w_pipeline import FixedWPipeline
    from .config import ELAConfig, ConfigManager
except ImportError:
    # 直接実行時のフォールバック
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from ela_pipeline import ELAPipeline
    from fixed_w_pipeline import FixedWPipeline
    from config import ELAConfig, ConfigManager


@dataclass
class BatchJob:
    """バッチジョブの定義"""
    job_type: str  # 'normal' または 'fixed_w'
    questionnaire_type: str
    column_range: Optional[tuple] = None
    w_type: Optional[str] = None  # 'social' または 'isolation' (fixed_wの場合)
    hq25_threshold: int = 3
    device_id: int = 0
    output_dir: Optional[str] = None
    data_dir: Optional[str] = None


@dataclass
class BatchConfig:
    """バッチ処理の設定"""
    jobs: List[BatchJob]
    max_workers: int = 1
    parallel: bool = False
    log_dir: str = "logs"
    output_base_dir: str = "outputs"
    data_dir: str = "data"


class BatchProcessor:
    """
    バッチ処理クラス
    
    複数のELA解析ジョブを管理し、並列または順次実行します。
    """
    
    def __init__(self, config: BatchConfig):
        """
        バッチプロセッサの初期化
        
        Args:
            config: バッチ処理の設定
        """
        self.config = config
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.log_dir = ela_analysis_dir / config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 結果の記録
        self.results = []
        self.failed_jobs = []
    
    def create_job_from_land_do_style(self, 
                                    questionnaire_type: str,
                                    r1: int = 0,
                                    r2: Optional[int] = None,
                                    device_id: int = 0,
                                    job_type: str = 'normal') -> BatchJob:
        """
        land_do.shスタイルのジョブを作成します
        
        Args:
            questionnaire_type: 質問票のタイプ
            r1: 列範囲の開始
            r2: 列範囲の終了
            device_id: CUDAデバイスID
            job_type: ジョブタイプ（'normal' または 'fixed_w'）
            
        Returns:
            バッチジョブ
        """
        column_range = None
        if r2 is not None:
            column_range = (r1, r2)
        elif r1 != 0:
            column_range = (r1, None)
        
        return BatchJob(
            job_type=job_type,
            questionnaire_type=questionnaire_type,
            column_range=column_range,
            device_id=device_id
        )
    
    def create_fixed_w_job(self,
                          questionnaire_type: str,
                          w_type: str,
                          r1: int = 0,
                          r2: Optional[int] = None,
                          device_id: int = 0) -> BatchJob:
        """
        固定Wジョブを作成します
        
        Args:
            questionnaire_type: 質問票のタイプ
            w_type: 固定W行列のタイプ
            r1: 列範囲の開始
            r2: 列範囲の終了
            device_id: CUDAデバイスID
            
        Returns:
            バッチジョブ
        """
        column_range = None
        if r2 is not None:
            column_range = (r1, r2)
        elif r1 != 0:
            column_range = (r1, None)
        
        return BatchJob(
            job_type='fixed_w',
            questionnaire_type=questionnaire_type,
            column_range=column_range,
            w_type=w_type,
            device_id=device_id
        )
    
    def execute_single_job(self, job: BatchJob) -> Dict[str, Any]:
        """
        単一ジョブを実行します
        
        Args:
            job: 実行するジョブ
            
        Returns:
            実行結果
        """
        start_time = time.time()
        job_id = f"{job.questionnaire_type}_{job.job_type}"
        if job.column_range:
            job_id += f"_{job.column_range[0]}_{job.column_range[1]}"
        
        print(f"Starting job: {job_id}")
        
        try:
            if job.job_type == 'normal':
                # 通常のELAパイプライン
                pipeline = ELAPipeline(
                    data_dir=job.data_dir or self.config.data_dir,
                    output_dir=job.output_dir or f"{self.config.output_base_dir}/meta_analysis",
                    device_id=job.device_id,
                    hq25_threshold=job.hq25_threshold
                )
                result = pipeline.run_pipeline(
                    questionnaire_type=job.questionnaire_type,
                    column_range=job.column_range
                )
                
            elif job.job_type == 'fixed_w':
                # 固定Wパイプライン
                pipeline = FixedWPipeline(
                    data_dir=job.data_dir or self.config.data_dir,
                    output_dir=job.output_dir or f"{self.config.output_base_dir}/integrated",
                    device_id=job.device_id,
                    hq25_threshold=job.hq25_threshold
                )
                result = pipeline.run_pipeline(
                    questionnaire_type=job.questionnaire_type,
                    w_type=job.w_type,
                    column_range=job.column_range
                )
            
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            elapsed_time = time.time() - start_time
            result['job_id'] = job_id
            result['elapsed_time'] = elapsed_time
            result['status'] = 'success'
            
            print(f"Job completed: {job_id} ({elapsed_time:.2f}s)")
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_result = {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'elapsed_time': elapsed_time
            }
            print(f"Job failed: {job_id} - {e}")
            return error_result
    
    def run_batch(self) -> Dict[str, Any]:
        """
        バッチ処理を実行します
        
        Returns:
            バッチ処理の結果
        """
        print(f"Starting batch processing with {len(self.config.jobs)} jobs")
        print(f"Parallel execution: {self.config.parallel}")
        print(f"Max workers: {self.config.max_workers}")
        
        start_time = time.time()
        
        if self.config.parallel and self.config.max_workers > 1:
            # 並列実行
            self._run_parallel()
        else:
            # 順次実行
            self._run_sequential()
        
        total_time = time.time() - start_time
        
        # 結果の集計
        successful_jobs = [r for r in self.results if r['status'] == 'success']
        failed_jobs = [r for r in self.results if r['status'] == 'failed']
        
        summary = {
            'total_jobs': len(self.config.jobs),
            'successful_jobs': len(successful_jobs),
            'failed_jobs': len(failed_jobs),
            'total_time': total_time,
            'results': self.results,
            'failed_job_details': failed_jobs
        }
        
        # 結果の保存
        self._save_results(summary)
        
        print(f"\nBatch processing completed:")
        print(f"  Total jobs: {summary['total_jobs']}")
        print(f"  Successful: {summary['successful_jobs']}")
        print(f"  Failed: {summary['failed_jobs']}")
        print(f"  Total time: {total_time:.2f}s")
        
        return summary
    
    def _run_sequential(self):
        """順次実行"""
        for i, job in enumerate(self.config.jobs):
            print(f"\nExecuting job {i+1}/{len(self.config.jobs)}")
            result = self.execute_single_job(job)
            self.results.append(result)
    
    def _run_parallel(self):
        """並列実行"""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # ジョブの送信
            future_to_job = {
                executor.submit(self.execute_single_job, job): job 
                for job in self.config.jobs
            }
            
            # 結果の収集
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    self.results.append(result)
                except Exception as e:
                    error_result = {
                        'job_id': f"{job.questionnaire_type}_{job.job_type}",
                        'status': 'failed',
                        'error': str(e),
                        'elapsed_time': 0
                    }
                    self.results.append(error_result)
    
    def _save_results(self, summary: Dict[str, Any]):
        """結果をファイルに保存"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # JSON形式で保存
        json_path = self.log_dir / f"batch_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        # サマリーをテキスト形式で保存
        summary_path = self.log_dir / f"batch_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Batch Processing Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Total jobs: {summary['total_jobs']}\n")
            f.write(f"Successful jobs: {summary['successful_jobs']}\n")
            f.write(f"Failed jobs: {summary['failed_jobs']}\n")
            f.write(f"Total time: {summary['total_time']:.2f}s\n\n")
            
            if summary['failed_jobs'] > 0:
                f.write("Failed Jobs:\n")
                f.write("------------\n")
                for failed in summary['failed_job_details']:
                    f.write(f"- {failed['job_id']}: {failed['error']}\n")
        
        print(f"Results saved to: {json_path}")
        print(f"Summary saved to: {summary_path}")


def create_land_do_style_batch() -> BatchConfig:
    """
    land_do.shスタイルのバッチ設定を作成します
    
    Returns:
        バッチ設定
    """
    processor = BatchProcessor(BatchConfig(jobs=[]))
    
    jobs = [
        # PHQ9+HQ25_social シリーズ
        processor.create_job_from_land_do_style("PHQ9+HQ25_social_2012", 0, 20),
        processor.create_job_from_land_do_style("PHQ9+HQ25_social_2006", 0, 20),
        processor.create_job_from_land_do_style("PHQ9+HQ25_social_prev2204", 0, 20),
        
        # PHQ9+HQ25_Isolation シリーズ
        processor.create_job_from_land_do_style("PHQ9+HQ25_Isolation_2308", 0, 17),
        processor.create_job_from_land_do_style("PHQ9+HQ25_Isolation_2204", 0, 17),
        processor.create_job_from_land_do_style("PHQ9+HQ25_Isolation_2108", 0, 17),
        processor.create_job_from_land_do_style("PHQ9+HQ25_Isolation_2012", 0, 17),
        processor.create_job_from_land_do_style("PHQ9+HQ25_Isolation_2006", 0, 17),
        processor.create_job_from_land_do_style("PHQ9+HQ25_Isolation_prev2204", 0, 17),
    ]
    
    return BatchConfig(
        jobs=jobs,
        max_workers=2,
        parallel=True,
        log_dir="logs",
        output_base_dir="outputs",
        data_dir="data"
    )


def create_fixed_w_style_batch() -> BatchConfig:
    """
    fixed_W.shスタイルのバッチ設定を作成します
    
    Returns:
        バッチ設定
    """
    processor = BatchProcessor(BatchConfig(jobs=[]))
    
    jobs = [
        # PHQ9+HQ25_Isolation 固定Wシリーズ
        processor.create_fixed_w_job("PHQ9+HQ25_Isolation_2006", "isolation", 0, 17),
        processor.create_fixed_w_job("PHQ9+HQ25_Isolation_2012", "isolation", 0, 17),
        processor.create_fixed_w_job("PHQ9+HQ25_Isolation_2108", "isolation", 0, 17),
        processor.create_fixed_w_job("PHQ9+HQ25_Isolation_2204", "isolation", 0, 17),
        processor.create_fixed_w_job("PHQ9+HQ25_Isolation_2308", "isolation", 0, 17),
        
        # PHQ9+HQ25_social 固定Wシリーズ
        processor.create_fixed_w_job("PHQ9+HQ25_social_2006", "social", 0, 20),
        processor.create_fixed_w_job("PHQ9+HQ25_social_2012", "social", 0, 20),
        processor.create_fixed_w_job("PHQ9+HQ25_social_2108", "social", 0, 20),
        processor.create_fixed_w_job("PHQ9+HQ25_social_2204", "social", 0, 20),
        processor.create_fixed_w_job("PHQ9+HQ25_social_2308", "social", 0, 20),
    ]
    
    return BatchConfig(
        jobs=jobs,
        max_workers=2,
        parallel=True,
        log_dir="logs",
        output_base_dir="outputs",
        data_dir="data"
    )


def create_argument_parser() -> argparse.ArgumentParser:
    """
    コマンドライン引数パーサーの作成
    
    Returns:
        設定済みのArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Batch Processing for ELA Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_processor.py --style land_do
  python batch_processor.py --style fixed_w
  python batch_processor.py --config batch_config.json
  python batch_processor.py --style land_do --parallel --max_workers 4
        """
    )
    
    # バッチスタイルの選択
    parser.add_argument(
        "--style",
        choices=['land_do', 'fixed_w'],
        help="事前定義されたバッチスタイルを選択"
    )
    
    # 設定ファイル
    parser.add_argument(
        "--config",
        type=str,
        help="バッチ設定ファイルのパス (JSONまたはYAML形式)"
    )
    
    # 並列処理オプション
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="並列実行を有効にする"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="並列実行時の最大ワーカー数 (デフォルト: 2)"
    )
    
    # ログ設定
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="ログディレクトリ (デフォルト: logs)"
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
    
    コマンドライン引数を解析し、バッチ処理を実行します。
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # バッチ設定の作成
        if args.config:
            # 設定ファイルから読み込み
            config_path = Path(args.config)
            if config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # 設定の構築
            jobs = [BatchJob(**job_dict) for job_dict in config_dict['jobs']]
            batch_config = BatchConfig(
                jobs=jobs,
                max_workers=config_dict.get('max_workers', 2),
                parallel=config_dict.get('parallel', False),
                log_dir=config_dict.get('log_dir', './logs'),
                output_base_dir=config_dict.get('output_base_dir', '../Data_csv'),
                data_dir=config_dict.get('data_dir', '../inputfiles')
            )
            
        elif args.style == 'land_do':
            batch_config = create_land_do_style_batch()
        elif args.style == 'fixed_w':
            batch_config = create_fixed_w_style_batch()
        else:
            raise ValueError("Either --style or --config must be specified")
        
        # コマンドライン引数でオーバーライド
        if args.parallel:
            batch_config.parallel = True
        if args.max_workers:
            batch_config.max_workers = args.max_workers
        if args.log_dir:
            batch_config.log_dir = args.log_dir
        
        # バッチプロセッサの実行
        processor = BatchProcessor(batch_config)
        summary = processor.run_batch()
        
        if args.verbose:
            print("\n=== Detailed Results ===")
            for result in summary['results']:
                print(f"Job: {result['job_id']}")
                print(f"  Status: {result['status']}")
                print(f"  Time: {result['elapsed_time']:.2f}s")
                if result['status'] == 'failed':
                    print(f"  Error: {result['error']}")
                print()
        
        print("Batch processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
