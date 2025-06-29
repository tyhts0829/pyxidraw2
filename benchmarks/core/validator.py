#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果検証モジュール

ベンチマーク結果の妥当性検証、パフォーマンス回帰検出、
統計的な分析を行います。
"""

import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from benchmarks.core.types import (
    BenchmarkResult,
    ValidationResult,
    ComparisonResult,
    PerformanceStats,
    BenchmarkMetrics,
)
from benchmarks.core.exceptions import ValidationError


class BenchmarkValidator:
    """ベンチマーク結果の妥当性検証クラス"""
    
    def __init__(self, tolerance: float = 0.1, confidence_level: float = 0.95):
        self.tolerance = tolerance  # パフォーマンス許容値（10%）
        self.confidence_level = confidence_level  # 信頼度（95%）
    
    def validate_result(self, result: BenchmarkResult) -> ValidationResult:
        """単一のベンチマーク結果を検証"""
        errors = []
        warnings = []
        metrics = {}

        # 1. 構造検証
        structure_errors = self._validate_structure(result)
        if structure_errors:
            return ValidationResult(
                is_valid=False,
                errors=structure_errors,
                warnings=warnings,
                metrics=metrics
            )

        # 2. 成功ステータスの検証
        if not result["success"]:
            error_msg = result.get("error", "Unknown error")
            errors.append(f"Benchmark failed: {error_msg}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )

        # 3. タイミングデータとメトリクスの検証 (成功した場合のみ)
        timing_errors, timing_warnings, timing_metrics = self._validate_timings(result)
        errors.extend(timing_errors)
        warnings.extend(timing_warnings)
        metrics.update(timing_metrics)

        metrics_errors, metrics_warnings = self._validate_metrics(result)
        errors.extend(metrics_errors)
        warnings.extend(metrics_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_multiple_results(self, results: Dict[str, BenchmarkResult]) -> ValidationResult:
        """複数のベンチマーク結果を検証"""
        all_errors = []
        all_warnings = []
        all_metrics = {}
        
        valid_count = 0
        
        for module_name, result in results.items():
            validation = self.validate_result(result)
            
            if validation["is_valid"]:
                valid_count += 1
            
            # エラーと警告にモジュール名を付加
            module_errors = [f"{module_name}: {error}" for error in validation["errors"]]
            module_warnings = [f"{module_name}: {warning}" for warning in validation["warnings"]]
            
            all_errors.extend(module_errors)
            all_warnings.extend(module_warnings)
            all_metrics[module_name] = validation["metrics"]
        
        # 全体的なメトリクスを追加
        total_modules = len(results)
        success_rate = valid_count / total_modules if total_modules > 0 else 0
        
        overall_metrics = {
            "total_modules": total_modules,
            "valid_modules": valid_count,
            "success_rate": success_rate,
            "failed_modules": total_modules - valid_count,
        }
        
        # 成功率が低い場合は警告
        if success_rate < 0.8:
            all_warnings.append(f"Low success rate: {success_rate:.1%} ({valid_count}/{total_modules})")
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
            metrics={"overall": overall_metrics, "modules": all_metrics}
        )
    
    def compare_results(self, current: BenchmarkResult, baseline: BenchmarkResult) -> ComparisonResult:
        """2つのベンチマーク結果を比較"""
        if not (current["success"] and baseline["success"]):
            raise ValidationError("Both results must be successful for comparison")
        
        # 平均実行時間を比較
        current_avg = self._calculate_overall_average(current)
        baseline_avg = self._calculate_overall_average(baseline)
        
        if baseline_avg == 0:
            improvement_ratio = float('inf') if current_avg == 0 else -float('inf')
        else:
            improvement_ratio = (baseline_avg - current_avg) / baseline_avg
        
        # 統計的有意性をテスト
        is_significant, p_value = self._statistical_significance_test(current, baseline)
        
        return ComparisonResult(
            baseline=baseline,
            current=current,
            improvement_ratio=improvement_ratio,
            is_significant=is_significant,
            p_value=p_value
        )
    
    def detect_performance_regression(self, 
                                    current: Dict[str, BenchmarkResult],
                                    baseline: Dict[str, BenchmarkResult],
                                    regression_threshold: float = -0.1) -> List[str]:
        """パフォーマンス回帰を検出"""
        regressions = []
        
        for module_name in current.keys():
            if module_name not in baseline:
                continue
            
            try:
                comparison = self.compare_results(current[module_name], baseline[module_name])
                
                # 改善率が閾値を下回る（つまり悪化）かつ統計的に有意な場合
                if (comparison["improvement_ratio"] < regression_threshold and 
                    comparison["is_significant"]):
                    
                    degradation = abs(comparison["improvement_ratio"]) * 100
                    regressions.append(
                        f"{module_name}: {degradation:.1f}% performance degradation "
                        f"(p-value: {comparison['p_value']:.3f})"
                    )
            
            except Exception as e:
                regressions.append(f"{module_name}: comparison failed ({e})")
        
        return regressions
    
    def calculate_performance_stats(self, result: BenchmarkResult) -> PerformanceStats:
        """パフォーマンス統計を計算"""
        if not result["success"] or not result["timings"]:
            raise ValidationError("Cannot calculate stats for unsuccessful result")
        
        # 全タイミングデータを収集
        all_times = []
        for times in result["timings"].values():
            all_times.extend(times)
        
        if not all_times:
            raise ValidationError("No timing data available")
        
        times_array = np.array(all_times)
        
        return PerformanceStats(
            mean=float(np.mean(times_array)),
            median=float(np.median(times_array)),
            std=float(np.std(times_array)),
            min=float(np.min(times_array)),
            max=float(np.max(times_array)),
            percentile_95=float(np.percentile(times_array, 95)),
            percentile_99=float(np.percentile(times_array, 99))
        )
    
    def _validate_structure(self, result: BenchmarkResult) -> List[str]:
        """ベンチマーク結果の構造を検証"""
        errors = []
        
        required_fields = ["module", "timestamp", "success", "status", "timings", "average_times", "metrics"]
        
        for field in required_fields:
            if field not in result:
                errors.append(f"Missing required field: {field}")
        
        # タイムスタンプの形式チェック
        if "timestamp" in result:
            try:
                datetime.fromisoformat(result["timestamp"])
            except ValueError:
                errors.append("Invalid timestamp format")
        
        # ステータスの整合性チェック
        if "success" in result and "status" in result:
            if result["success"] and result["status"] != "success":
                errors.append("Inconsistent success status")
            elif not result["success"] and result["status"] == "success":
                errors.append("Inconsistent failure status")
        
        return errors
    
    def _validate_timings(self, result: BenchmarkResult) -> Tuple[List[str], List[str], BenchmarkMetrics]:
        """タイミングデータを検証"""
        errors = []
        warnings = []
        metrics = {}
        
        timings = result["timings"]
        average_times = result["average_times"]
        
        if not timings:
            errors.append("No timing data available")
            return errors, warnings, metrics
        
        # 各サイズのタイミングデータを検証
        for size_name, times in timings.items():
            if not times:
                warnings.append(f"No timing data for {size_name}")
                continue
            
            times_array = np.array(times)
            
            # 負の値チェック
            if np.any(times_array < 0):
                errors.append(f"Negative timing values in {size_name}")
            
            # 異常値検出（平均から3標準偏差を超える値）
            if len(times_array) > 3:
                mean_time = np.mean(times_array)
                std_time = np.std(times_array)
                outliers = np.abs(times_array - mean_time) > 3 * std_time
                
                if np.any(outliers):
                    outlier_count = np.sum(outliers)
                    warnings.append(f"{size_name}: {outlier_count} outlier(s) detected")
            
            # 変動係数（CV）チェック
            if len(times_array) > 1:
                cv = np.std(times_array) / np.mean(times_array)
                metrics[f"{size_name}_cv"] = float(cv)
                
                if cv > 0.5:  # CV > 50%
                    warnings.append(f"{size_name}: high variability (CV: {cv:.2%})")
            
            # 平均時間の整合性チェック
            if size_name in average_times:
                calculated_avg = np.mean(times_array)
                reported_avg = average_times[size_name]
                
                relative_error = abs(calculated_avg - reported_avg) / calculated_avg
                if relative_error > 0.01:  # 1%以上の誤差
                    errors.append(f"{size_name}: average time mismatch")
        
        return errors, warnings, metrics
    
    def _validate_metrics(self, result: BenchmarkResult) -> Tuple[List[str], List[str]]:
        """メトリクスを検証"""
        errors = []
        warnings = []
        
        metrics = result.get("metrics", {})
        
        # メトリクスの範囲チェック
        if "overall_avg_time" in metrics:
            avg_time = metrics["overall_avg_time"]
            if avg_time < 0:
                errors.append("Negative overall average time")
            elif avg_time > 10.0:  # 10秒以上
                warnings.append(f"Very slow execution: {avg_time:.2f}s")
        
        if "overall_std_time" in metrics:
            std_time = metrics["overall_std_time"]
            if std_time < 0:
                errors.append("Negative standard deviation")
        
        # 測定回数チェック
        if "total_measurements" in metrics:
            total_measurements = metrics["total_measurements"]
            if total_measurements < 10:
                warnings.append(f"Low measurement count: {total_measurements}")
        
        return errors, warnings
    
    def _calculate_overall_average(self, result: BenchmarkResult) -> float:
        """全体の平均実行時間を計算"""
        if not result["average_times"]:
            return 0.0
        
        return statistics.mean(result["average_times"].values())
    
    def _statistical_significance_test(self, 
                                     current: BenchmarkResult, 
                                     baseline: BenchmarkResult) -> Tuple[bool, Optional[float]]:
        """統計的有意性テスト（t検定）"""
        try:
            # 全タイミングデータを収集
            current_times = []
            baseline_times = []
            
            for times in current["timings"].values():
                current_times.extend(times)
            
            for times in baseline["timings"].values():
                baseline_times.extend(times)
            
            if len(current_times) < 3 or len(baseline_times) < 3:
                return False, None
            
            # Welch's t-test（等分散を仮定しない）
            t_statistic, p_value = stats.ttest_ind(current_times, baseline_times, equal_var=False)
            
            # 有意水準での判定
            alpha = 1 - self.confidence_level
            is_significant = p_value < alpha
            
            return is_significant, float(p_value)
        
        except Exception:
            return False, None


class BenchmarkResultAnalyzer:
    """ベンチマーク結果分析クラス"""
    
    def __init__(self):
        self.validator = BenchmarkValidator()
    
    def analyze_results(self, results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
        """ベンチマーク結果を包括的に分析"""
        analysis = {
            "validation": self.validator.validate_multiple_results(results),
            "summary": self._generate_summary(results),
            "performance_ranking": self._rank_by_performance(results),
            "statistics": self._calculate_statistics(results),
        }
        
        return analysis
    
    def _generate_summary(self, results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
        """結果の要約を生成"""
        total = len(results)
        successful = sum(1 for r in results.values() if r["success"])
        failed = total - successful
        
        if successful > 0:
            all_times = []
            for result in results.values():
                if result["success"] and result["average_times"]:
                    all_times.extend(result["average_times"].values())
            
            if all_times:
                fastest = min(all_times)
                slowest = max(all_times)
                avg_time = statistics.mean(all_times)
            else:
                fastest = slowest = avg_time = 0.0
        else:
            fastest = slowest = avg_time = 0.0
        
        return {
            "total_modules": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "fastest_time": fastest,
            "slowest_time": slowest,
            "average_time": avg_time,
        }
    
    def _rank_by_performance(self, results: Dict[str, BenchmarkResult]) -> List[Tuple[str, float]]:
        """パフォーマンス順にランキング"""
        performance_data = []
        
        for module_name, result in results.items():
            if result["success"] and result["average_times"]:
                avg_time = statistics.mean(result["average_times"].values())
                performance_data.append((module_name, avg_time))
        
        # 実行時間でソート（昇順）
        performance_data.sort(key=lambda x: x[1])
        
        return performance_data
    
    def _calculate_statistics(self, results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
        """統計情報を計算"""
        stats_data = {}
        
        # 成功したモジュールの統計
        successful_results = [r for r in results.values() if r["success"]]
        
        if successful_results:
            # njit使用率
            njit_count = sum(1 for r in successful_results 
                           if r.get("metrics", {}).get("has_njit", False))
            njit_rate = njit_count / len(successful_results)
            
            # キャッシュ使用率
            cache_count = sum(1 for r in successful_results 
                            if r.get("metrics", {}).get("has_cache", False))
            cache_rate = cache_count / len(successful_results)
            
            # 測定回数の統計
            measurement_counts = [r.get("metrics", {}).get("total_measurements", 0) 
                                for r in successful_results]
            measurement_counts = [c for c in measurement_counts if c > 0]
            
            stats_data = {
                "njit_usage_rate": njit_rate,
                "cache_usage_rate": cache_rate,
                "avg_measurements": statistics.mean(measurement_counts) if measurement_counts else 0,
                "min_measurements": min(measurement_counts) if measurement_counts else 0,
                "max_measurements": max(measurement_counts) if measurement_counts else 0,
            }
        
        return stats_data


# 便利関数
def validate_results(results: Dict[str, BenchmarkResult]) -> ValidationResult:
    """ベンチマーク結果を検証する便利関数"""
    validator = BenchmarkValidator()
    return validator.validate_multiple_results(results)


def analyze_benchmark_results(results: Dict[str, BenchmarkResult]) -> Dict[str, any]:
    """ベンチマーク結果を分析する便利関数"""
    analyzer = BenchmarkResultAnalyzer()
    return analyzer.analyze_results(results)