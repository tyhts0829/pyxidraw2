#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyxiDraw ベンチマークシステム コマンドライン界面

統一ベンチマークシステムのコマンドライン実行環境。
プラグインシステムと設定管理を統合した使いやすいインターフェース。

使用例:
    python -m benchmarks run                    # 全ベンチマーク実行
    python -m benchmarks run --config custom.yaml
    python -m benchmarks run --parallel        # 並列実行
    python -m benchmarks list                  # 利用可能ターゲット一覧
    python -m benchmarks run --target effects.noise.high_intensity
    python -m benchmarks validate results.json # 結果検証
    python -m benchmarks compare baseline.json current.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from benchmarks.core.config import BenchmarkConfigManager, get_config
from benchmarks.core.runner import UnifiedBenchmarkRunner
from benchmarks.core.validator import BenchmarkValidator, BenchmarkResultAnalyzer
from benchmarks.benchmark_result_manager import BenchmarkResultManager


def create_parser() -> argparse.ArgumentParser:
    """コマンドラインパーサーを作成"""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="PyxiDraw Unified Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python -m benchmarks run                           # 全ベンチマーク実行
  python -m benchmarks run --config custom.yaml     # カスタム設定で実行
  python -m benchmarks run --parallel --workers 4   # 並列実行
  python -m benchmarks run --target effects.noise   # 特定ターゲットのみ
  python -m benchmarks list                         # 利用可能ターゲット一覧
  python -m benchmarks validate results.json        # 結果検証
  python -m benchmarks compare old.json new.json    # 結果比較
  python -m benchmarks config template config.yaml  # 設定テンプレート作成
        """)
    
    # グローバルオプション
    parser.add_argument("--config", "-c", type=Path, 
                       help="設定ファイルパス (YAML/JSON)")
    parser.add_argument("--output-dir", "-o", type=Path,
                       help="出力ディレクトリ")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="詳細出力")
    
    # サブコマンド
    subparsers = parser.add_subparsers(dest="command", help="利用可能なコマンド")
    
    # run サブコマンド
    run_parser = subparsers.add_parser("run", help="ベンチマークを実行")
    run_parser.add_argument("--target", "-t", action="append",
                           help="特定のターゲットを実行 (複数指定可)")
    run_parser.add_argument("--plugin", "-p", action="append",
                           help="特定のプラグインのみ実行")
    run_parser.add_argument("--parallel", action="store_true",
                           help="並列実行を有効化")
    run_parser.add_argument("--workers", type=int,
                           help="並列実行のワーカー数")
    run_parser.add_argument("--warmup", type=int,
                           help="ウォームアップ実行回数")
    run_parser.add_argument("--runs", type=int,
                           help="測定実行回数")
    run_parser.add_argument("--timeout", type=float,
                           help="タイムアウト時間（秒）")
    run_parser.add_argument("--no-save", action="store_true",
                           help="結果を保存しない")
    run_parser.add_argument("--no-charts", action="store_true",
                           help="チャートを生成しない")
    
    # list サブコマンド
    list_parser = subparsers.add_parser("list", help="利用可能なターゲットを表示")
    list_parser.add_argument("--plugin", "-p",
                            help="特定プラグインのターゲットのみ表示")
    list_parser.add_argument("--format", choices=["table", "json", "yaml"],
                            default="table", help="出力フォーマット")
    
    # validate サブコマンド
    validate_parser = subparsers.add_parser("validate", help="ベンチマーク結果を検証")
    validate_parser.add_argument("results_file", type=Path,
                                help="検証する結果ファイル")
    validate_parser.add_argument("--report", type=Path,
                                help="検証レポートの出力先")
    
    # compare サブコマンド
    compare_parser = subparsers.add_parser("compare", help="ベンチマーク結果を比較")
    compare_parser.add_argument("baseline", type=Path,
                               help="ベースライン結果ファイル")
    compare_parser.add_argument("current", type=Path,
                               help="現在の結果ファイル")
    compare_parser.add_argument("--regression-threshold", type=float, default=-0.1,
                               help="回帰検出の閾値 (デフォルト: -10%)")
    
    # config サブコマンド
    config_parser = subparsers.add_parser("config", help="設定管理")
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    template_parser = config_subparsers.add_parser("template", help="設定テンプレートを作成")
    template_parser.add_argument("output_file", type=Path,
                                help="出力ファイルパス")
    
    show_parser = config_subparsers.add_parser("show", help="現在の設定を表示")
    
    return parser


def load_config(args) -> 'BenchmarkConfig':
    """コマンドライン引数から設定を読み込み"""
    config_manager = BenchmarkConfigManager(args.config)
    config = config_manager.load_config()
    
    # コマンドライン引数で設定を上書き
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if hasattr(args, 'parallel') and args.parallel:
        config.parallel = True
    
    if hasattr(args, 'workers') and args.workers:
        config.max_workers = args.workers
    
    if hasattr(args, 'warmup') and args.warmup:
        config.warmup_runs = args.warmup
    
    if hasattr(args, 'runs') and args.runs:
        config.measurement_runs = args.runs
    
    if hasattr(args, 'timeout') and args.timeout:
        config.timeout_seconds = args.timeout
    
    if hasattr(args, 'no_charts') and args.no_charts:
        config.generate_charts = False
    
    # 設定の妥当性をチェック
    config_manager.validate_config(config)
    
    return config


def cmd_run(args) -> int:
    """runコマンドの実行"""
    config = load_config(args)
    
    if args.verbose:
        print(f"設定: {config}")
    
    # ランナーを作成
    runner = UnifiedBenchmarkRunner(config)
    
    try:
        # ベンチマーク実行
        if hasattr(args, 'target') and args.target:
            # 特定ターゲットのみ実行
            results = runner.run_specific_targets(args.target)
        else:
            # 全ベンチマーク実行
            results = runner.run_all_benchmarks()
        
        if not results:
            print("No benchmarks were executed")
            return 1
        
        # 結果の保存
        if not args.no_save:
            result_manager = BenchmarkResultManager(str(config.output_dir))
            saved_file = result_manager.save_results(results)
            print(f"\nResults saved to: {saved_file}")
        
        # 結果の検証と分析
        analyzer = BenchmarkResultAnalyzer()
        analysis = analyzer.analyze_results(results)
        
        # 要約を表示
        print(f"\n=== BENCHMARK SUMMARY ===")
        summary = analysis["summary"]
        print(f"Total modules: {summary['total_modules']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        if summary['successful'] > 0:
            print(f"Fastest time: {summary['fastest_time']*1000:.3f}ms")
            print(f"Slowest time: {summary['slowest_time']*1000:.3f}ms")
            print(f"Average time: {summary['average_time']*1000:.3f}ms")
        
        # 検証結果を表示
        validation = analysis["validation"]
        if validation["errors"]:
            print(f"\n⚠️  Validation errors: {len(validation['errors'])}")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        if validation["warnings"]:
            print(f"\n⚠️  Warnings: {len(validation['warnings'])}")
            for warning in validation["warnings"][:5]:  # 最初の5つのみ表示
                print(f"  - {warning}")
        
        return 0 if validation["is_valid"] else 1
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during benchmark execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_list(args) -> int:
    """listコマンドの実行"""
    config = load_config(args)
    runner = UnifiedBenchmarkRunner(config)
    
    try:
        available_targets = runner.get_available_targets()
        
        if args.plugin:
            # 特定プラグインのみ表示
            if args.plugin in available_targets:
                targets = {args.plugin: available_targets[args.plugin]}
            else:
                print(f"Plugin '{args.plugin}' not found")
                return 1
        else:
            targets = available_targets
        
        if args.format == "json":
            print(json.dumps(targets, indent=2, ensure_ascii=False))
        elif args.format == "yaml":
            try:
                import yaml
                print(yaml.dump(targets, default_flow_style=False, allow_unicode=True))
            except ImportError:
                print("PyYAML not installed, falling back to table format")
                _print_targets_table(targets)
        else:  # table
            _print_targets_table(targets)
        
        return 0
        
    except Exception as e:
        print(f"Error listing targets: {e}")
        return 1


def _print_targets_table(targets: dict) -> None:
    """ターゲットをテーブル形式で表示"""
    total_targets = 0
    
    for plugin_name, target_list in targets.items():
        print(f"\n=== {plugin_name.upper()} ===")
        if target_list:
            for target in target_list:
                print(f"  {target}")
            print(f"  ({len(target_list)} targets)")
            total_targets += len(target_list)
        else:
            print("  (no targets)")
    
    print(f"\nTotal: {total_targets} targets")


def cmd_validate(args) -> int:
    """validateコマンドの実行"""
    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        return 1
    
    try:
        # 結果ファイルを読み込み
        result_manager = BenchmarkResultManager()
        results = result_manager.load_results(str(args.results_file))
        
        # 結果を検証
        analyzer = BenchmarkResultAnalyzer()
        analysis = analyzer.analyze_results(results)
        
        validation = analysis["validation"]
        
        print(f"=== VALIDATION REPORT ===")
        print(f"File: {args.results_file}")
        print(f"Valid: {'✓' if validation['is_valid'] else '✗'}")
        
        if validation["errors"]:
            print(f"\nErrors ({len(validation['errors'])}):")
            for error in validation["errors"]:
                print(f"  - {error}")
        
        if validation["warnings"]:
            print(f"\nWarnings ({len(validation['warnings'])}):")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        # 統計情報
        summary = analysis["summary"]
        print(f"\nSummary:")
        print(f"  Modules: {summary['total_modules']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
        # レポート保存
        if args.report:
            report_data = {
                "validation": validation,
                "analysis": analysis,
                "timestamp": analysis.get("timestamp", ""),
            }
            
            with open(args.report, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nReport saved to: {args.report}")
        
        return 0 if validation["is_valid"] else 1
        
    except Exception as e:
        print(f"Error validating results: {e}")
        return 1


def cmd_compare(args) -> int:
    """compareコマンドの実行"""
    if not args.baseline.exists():
        print(f"Baseline file not found: {args.baseline}")
        return 1
    
    if not args.current.exists():
        print(f"Current file not found: {args.current}")
        return 1
    
    try:
        # 結果ファイルを読み込み
        result_manager = BenchmarkResultManager()
        baseline_results = result_manager.load_results(str(args.baseline))
        current_results = result_manager.load_results(str(args.current))
        
        # 比較実行
        validator = BenchmarkValidator()
        regressions = validator.detect_performance_regression(
            current_results, baseline_results, args.regression_threshold
        )
        
        print(f"=== PERFORMANCE COMPARISON ===")
        print(f"Baseline: {args.baseline}")
        print(f"Current: {args.current}")
        print(f"Regression threshold: {args.regression_threshold:.1%}")
        
        if regressions:
            print(f"\n⚠️  Performance regressions detected ({len(regressions)}):")
            for regression in regressions:
                print(f"  - {regression}")
            return 1
        else:
            print(f"\n✓ No significant performance regressions detected")
            return 0
        
    except Exception as e:
        print(f"Error comparing results: {e}")
        return 1


def cmd_config(args) -> int:
    """configコマンドの実行"""
    if args.config_action == "template":
        # 設定テンプレートを作成
        config_manager = BenchmarkConfigManager()
        config_manager.create_template_config(args.output_file)
        print(f"Configuration template created: {args.output_file}")
        return 0
    
    elif args.config_action == "show":
        # 現在の設定を表示
        config = get_config()
        print("=== CURRENT CONFIGURATION ===")
        print(f"Warmup runs: {config.warmup_runs}")
        print(f"Measurement runs: {config.measurement_runs}")
        print(f"Timeout: {config.timeout_seconds}s")
        print(f"Output directory: {config.output_dir}")
        print(f"Parallel execution: {config.parallel}")
        print(f"Max workers: {config.max_workers}")
        print(f"Continue on error: {config.continue_on_error}")
        print(f"Max errors: {config.max_errors}")
        print(f"Generate charts: {config.generate_charts}")
        return 0
    
    else:
        print("Invalid config action")
        return 1


def main() -> int:
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # コマンド実行
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "config":
        return cmd_config(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())