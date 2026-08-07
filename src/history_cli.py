#!/usr/bin/env python3
"""
実行履歴CLIツール

ELA解析の実行履歴を表示・検索・分析するためのコマンドラインツールです。
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Dict, Any

from execution_history import ExecutionHistoryManager, create_execution_summary


def format_duration(seconds: float) -> str:
    """実行時間をフォーマット"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分"
    else:
        return f"{seconds/3600:.1f}時間"


def format_timestamp(timestamp: str) -> str:
    """タイムスタンプをフォーマット"""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def print_execution_record(record: Dict[str, Any], detailed: bool = False):
    """実行記録を表示"""
    print(f"\n{'='*60}")
    print(f"実行ID: {record['execution_id']}")
    print(f"実行時刻: {format_timestamp(record['timestamp'])}")
    print(f"ステータス: {record['status']}")
    print(f"実行時間: {format_duration(record['duration_seconds'])}")
    
    params = record['parameters']
    print(f"質問票タイプ: {params.get('func', 'N/A')}")
    print(f"データソース: {params.get('data_source', 'N/A')}")
    print(f"HQ-25閾値: {params.get('hq25_threshold', 'N/A')}")
    
    if detailed:
        print(f"\nコマンド: {record['command']}")
        print(f"全パラメータ:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        if record['output_files']:
            print(f"\n出力ファイル:")
            for file_path in record['output_files']:
                print(f"  - {file_path}")
        
        if record['results_summary']:
            print(f"\n結果サマリー:")
            for key, value in record['results_summary'].items():
                print(f"  {key}: {value}")
        
        if record.get('error_message'):
            print(f"\nエラーメッセージ: {record['error_message']}")
        
        if record.get('notes'):
            print(f"\n備考: {record['notes']}")


def list_executions(args):
    """実行履歴一覧を表示"""
    manager = ExecutionHistoryManager()
    
    # 検索条件
    filters = {}
    if args.questionnaire_type:
        filters['questionnaire_type'] = args.questionnaire_type
    if args.data_source:
        filters['data_source'] = args.data_source
    if args.status:
        filters['status'] = args.status
    
    # 実行記録を取得
    if filters:
        records = manager.search_executions(**filters)
    else:
        records = manager.get_recent_executions(args.limit)
    
    if not records:
        print("実行履歴が見つかりませんでした。")
        return
    
    print(f"実行履歴 ({len(records)}件):")
    
    for record in records:
        print_execution_record(record, detailed=args.detailed)
    
    if not args.detailed:
        print(f"\n詳細表示: python -m src.history_cli show <execution_id>")


def show_execution(args):
    """特定の実行記録を詳細表示"""
    manager = ExecutionHistoryManager()
    record = manager.get_execution(args.execution_id)
    
    if not record:
        print(f"実行ID '{args.execution_id}' の記録が見つかりません。")
        return
    
    print_execution_record(record, detailed=True)


def show_statistics(args):
    """実行統計を表示"""
    manager = ExecutionHistoryManager()
    stats = manager.get_statistics()
    
    if not stats:
        print("実行履歴がありません。")
        return
    
    print("=== 実行統計 ===")
    print(f"総実行回数: {stats['total_executions']}")
    print(f"成功回数: {stats['successful_executions']}")
    print(f"失敗回数: {stats['failed_executions']}")
    print(f"成功率: {stats['success_rate']:.1%}")
    print(f"平均実行時間: {format_duration(stats['average_duration_seconds'])}")
    
    if stats['questionnaire_statistics']:
        print(f"\n=== 質問票タイプ別統計 ===")
        for qtype, qstats in stats['questionnaire_statistics'].items():
            success_rate = qstats['success'] / qstats['total'] if qstats['total'] > 0 else 0
            print(f"{qtype}: {qstats['total']}回実行 (成功率: {success_rate:.1%})")


def export_history(args):
    """実行履歴をエクスポート"""
    manager = ExecutionHistoryManager()
    
    if args.format == 'json':
        output_file = args.output or "execution_history_export.json"
    else:
        output_file = args.output or "execution_history_export.csv"
    
    manager.export_history(output_file, args.format)
    print(f"実行履歴を {output_file} にエクスポートしました。")


def compare_executions(args):
    """実行記録を比較"""
    manager = ExecutionHistoryManager()
    
    records = []
    for execution_id in args.execution_ids:
        record = manager.get_execution(execution_id)
        if record:
            records.append(record)
        else:
            print(f"警告: 実行ID '{execution_id}' の記録が見つかりません。")
    
    if len(records) < 2:
        print("比較するには最低2つの実行記録が必要です。")
        return
    
    print("=== 実行記録比較 ===")
    print(f"{'項目':<20} {'実行1':<15} {'実行2':<15} {'実行3':<15}")
    print("-" * 70)
    
    # 比較項目
    comparison_items = [
        ('実行ID', 'execution_id'),
        ('質問票タイプ', 'parameters.func'),
        ('データソース', 'parameters.data_source'),
        ('HQ-25閾値', 'parameters.hq25_threshold'),
        ('ステータス', 'status'),
        ('実行時間', 'duration_seconds'),
        ('出力ファイル数', 'output_files'),
    ]
    
    for item_name, item_path in comparison_items:
        values = []
        for record in records:
            if '.' in item_path:
                # ネストした属性
                parts = item_path.split('.')
                value = record
                for part in parts:
                    value = value.get(part, 'N/A')
            else:
                value = record.get(item_path, 'N/A')
            
            if item_name == '実行時間':
                value = format_duration(value) if isinstance(value, (int, float)) else str(value)
            elif item_name == '出力ファイル数':
                value = len(value) if isinstance(value, list) else str(value)
            else:
                value = str(value)
            
            values.append(value)
        
        # 表示用に調整
        while len(values) < 3:
            values.append('N/A')
        
        print(f"{item_name:<20} {values[0]:<15} {values[1]:<15} {values[2]:<15}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="ELA解析実行履歴管理ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 最近の実行履歴を表示
  python -m src.history_cli list

  # 特定の質問票タイプの実行履歴を表示
  python -m src.history_cli list --questionnaire-type PHQ-9

  # 詳細な実行記録を表示
  python -m src.history_cli show abc12345

  # 実行統計を表示
  python -m src.history_cli stats

  # 実行記録を比較
  python -m src.history_cli compare abc12345 def67890

  # 実行履歴をエクスポート
  python -m src.history_cli export --format json --output my_history.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='利用可能なコマンド')
    
    # list コマンド
    list_parser = subparsers.add_parser('list', help='実行履歴一覧を表示')
    list_parser.add_argument('--questionnaire-type', help='質問票タイプでフィルタ')
    list_parser.add_argument('--data-source', help='データソースでフィルタ')
    list_parser.add_argument('--status', help='ステータスでフィルタ')
    list_parser.add_argument('--limit', type=int, default=10, help='表示件数制限')
    list_parser.add_argument('--detailed', action='store_true', help='詳細表示')
    
    # show コマンド
    show_parser = subparsers.add_parser('show', help='特定の実行記録を詳細表示')
    show_parser.add_argument('execution_id', help='実行ID')
    
    # stats コマンド
    stats_parser = subparsers.add_parser('stats', help='実行統計を表示')
    
    # compare コマンド
    compare_parser = subparsers.add_parser('compare', help='実行記録を比較')
    compare_parser.add_argument('execution_ids', nargs='+', help='比較する実行ID')
    
    # export コマンド
    export_parser = subparsers.add_parser('export', help='実行履歴をエクスポート')
    export_parser.add_argument('--format', choices=['json', 'csv'], default='json', help='出力形式')
    export_parser.add_argument('--output', help='出力ファイル名')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_executions(args)
        elif args.command == 'show':
            show_execution(args)
        elif args.command == 'stats':
            show_statistics(args)
        elif args.command == 'compare':
            compare_executions(args)
        elif args.command == 'export':
            export_history(args)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
