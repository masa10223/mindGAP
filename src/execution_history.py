#!/usr/bin/env python3
"""
実行履歴管理システム

ELA解析の実行履歴をJSON形式で記録し、検索・比較・分析機能を提供します。
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ExecutionRecord:
    """実行記録のデータクラス"""
    execution_id: str
    timestamp: str
    command: str
    parameters: Dict[str, Any]
    status: str  # 'success', 'failed', 'running'
    duration_seconds: float
    input_data_info: Dict[str, Any]
    output_files: List[str]
    results_summary: Dict[str, Any]
    error_message: Optional[str] = None
    notes: Optional[str] = None


class ExecutionHistoryManager:
    """実行履歴管理クラス"""
    
    def __init__(self, history_dir: str = "execution_history"):
        """
        実行履歴管理の初期化
        
        Args:
            history_dir: 履歴ファイルの保存ディレクトリ
        """
        # ELA_analysisディレクトリを基準としたパス設定
        ela_analysis_dir = Path(__file__).parent.parent
        self.history_dir = ela_analysis_dir / history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # 履歴ファイルのパス
        self.history_file = self.history_dir / "execution_history.json"
        self.summary_file = self.history_dir / "execution_summary.csv"
        
        # 履歴の読み込み
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """履歴ファイルから実行記録を読み込み"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def _save_history(self):
        """履歴をファイルに保存"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        
        # CSVサマリーも更新
        self._update_summary_csv()
    
    def _update_summary_csv(self):
        """CSVサマリーファイルを更新"""
        if not self.history:
            return
        
        summary_data = []
        for record in self.history:
            summary_data.append({
                'execution_id': record['execution_id'],
                'timestamp': record['timestamp'],
                'questionnaire_type': record['parameters'].get('func', ''),
                'data_source': record['parameters'].get('data_source', ''),
                'hq25_threshold': record['parameters'].get('hq25_threshold', ''),
                'status': record['status'],
                'duration_seconds': record['duration_seconds'],
                'output_files_count': len(record['output_files']),
                'notes': record.get('notes', '')
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.summary_file, index=False, encoding='utf-8-sig')
    
    def start_execution(self, command: str, parameters: Dict[str, Any]) -> str:
        """
        新しい実行を開始し、実行IDを返す
        
        Args:
            command: 実行されたコマンド
            parameters: 実行パラメータ
            
        Returns:
            実行ID
        """
        execution_id = str(uuid.uuid4())[:8]  # 短いID
        
        record = ExecutionRecord(
            execution_id=execution_id,
            timestamp=datetime.now().isoformat(),
            command=command,
            parameters=parameters,
            status='running',
            duration_seconds=0.0,
            input_data_info={},
            output_files=[],
            results_summary={},
            error_message=None,
            notes=None
        )
        
        # 履歴に追加
        self.history.append(asdict(record))
        self._save_history()
        
        return execution_id
    
    def update_execution(self, execution_id: str, **kwargs):
        """
        実行中の記録を更新
        
        Args:
            execution_id: 実行ID
            **kwargs: 更新するフィールド
        """
        for record in self.history:
            if record['execution_id'] == execution_id:
                for key, value in kwargs.items():
                    # 有効なフィールドかチェック
                    valid_fields = ['status', 'duration_seconds', 'output_files', 
                                  'results_summary', 'error_message', 'notes', 
                                  'input_data_info']
                    if key in valid_fields:
                        record[key] = value
                break
        
        self._save_history()
    
    def complete_execution(self, execution_id: str, status: str = 'success', 
                          duration_seconds: float = 0.0, output_files: List[str] = None,
                          results_summary: Dict[str, Any] = None, 
                          error_message: str = None, notes: str = None):
        """
        実行を完了として記録
        
        Args:
            execution_id: 実行ID
            status: 実行ステータス
            duration_seconds: 実行時間
            output_files: 出力ファイルリスト
            results_summary: 結果サマリー
            error_message: エラーメッセージ
            notes: 備考
        """
        self.update_execution(
            execution_id,
            status=status,
            duration_seconds=duration_seconds,
            output_files=output_files or [],
            results_summary=results_summary or {},
            error_message=error_message,
            notes=notes
        )
    
    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """指定された実行IDの記録を取得"""
        for record in self.history:
            if record['execution_id'] == execution_id:
                return record
        return None
    
    def search_executions(self, **filters) -> List[Dict[str, Any]]:
        """
        実行記録を検索
        
        Args:
            **filters: 検索条件（例: questionnaire_type='PHQ-9', status='success'）
            
        Returns:
            マッチした実行記録のリスト
        """
        results = []
        for record in self.history:
            match = True
            for key, value in filters.items():
                if key == 'questionnaire_type':
                    if record['parameters'].get('func') != value:
                        match = False
                        break
                elif key == 'data_source':
                    if record['parameters'].get('data_source') != value:
                        match = False
                        break
                elif key == 'status':
                    if record['status'] != value:
                        match = False
                        break
                elif key in record['parameters']:
                    if record['parameters'][key] != value:
                        match = False
                        break
                elif key in record:
                    if record[key] != value:
                        match = False
                        break
            
            if match:
                results.append(record)
        
        return results
    
    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の実行記録を取得"""
        return sorted(self.history, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """実行統計を取得"""
        if not self.history:
            return {}
        
        total_executions = len(self.history)
        successful_executions = len([r for r in self.history if r['status'] == 'success'])
        failed_executions = len([r for r in self.history if r['status'] == 'failed'])
        
        # 質問票タイプ別統計
        questionnaire_stats = {}
        for record in self.history:
            qtype = record['parameters'].get('func', 'unknown')
            if qtype not in questionnaire_stats:
                questionnaire_stats[qtype] = {'total': 0, 'success': 0, 'failed': 0}
            questionnaire_stats[qtype]['total'] += 1
            if record['status'] == 'success':
                questionnaire_stats[qtype]['success'] += 1
            elif record['status'] == 'failed':
                questionnaire_stats[qtype]['failed'] += 1
        
        # 平均実行時間
        completed_executions = [r for r in self.history if r['status'] in ['success', 'failed'] and r['duration_seconds'] > 0]
        avg_duration = sum(r['duration_seconds'] for r in completed_executions) / len(completed_executions) if completed_executions else 0
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': failed_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'average_duration_seconds': avg_duration,
            'questionnaire_statistics': questionnaire_stats
        }
    
    def export_history(self, output_file: str, format: str = 'json'):
        """
        実行履歴をエクスポート
        
        Args:
            output_file: 出力ファイルパス
            format: 出力形式（'json' または 'csv'）
        """
        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(self.history)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    def clear_history(self, confirm: bool = False):
        """実行履歴をクリア"""
        if not confirm:
            print("履歴をクリアするには confirm=True を指定してください")
            return
        
        self.history = []
        self._save_history()
        print("実行履歴をクリアしました")


def create_execution_summary(execution_id: str, results: Dict[str, Any], 
                           output_dir: Path) -> Dict[str, Any]:
    """
    実行結果からサマリーを作成
    
    Args:
        execution_id: 実行ID
        results: 実行結果
        output_dir: 出力ディレクトリ
        
    Returns:
        サマリーデータ
    """
    # 出力ファイルの検索
    output_files = []
    if output_dir.exists():
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and execution_id in file_path.name:
                output_files.append(str(file_path.relative_to(output_dir.parent)))
    
    return {
        'execution_id': execution_id,
        'total_states': results.get('total_states', 0),
        'local_minima_count': results.get('local_minima_count', 0),
        'basin_count': results.get('basin_count', 0),
        'transition_count': results.get('transition_count', 0),
        'elapsed_time': results.get('elapsed_time', 0),
        'output_files': output_files,
        'memory_usage_mb': results.get('memory_usage_mb', 0),
        'gpu_used': results.get('gpu_used', False)
    }
