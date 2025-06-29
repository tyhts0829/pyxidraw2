#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマーク結果チャート生成モジュール

様々な種類のチャートを生成し、ベンチマーク結果を視覚化します。
既存のBenchmarkVisualizerの機能を拡張・改良しています。
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    import japanize_matplotlib
    japanize_matplotlib.japanize()
    HAS_JAPANESE = True
except ImportError:
    HAS_JAPANESE = False

from benchmarks.core.types import BenchmarkResult, ChartConfig, ChartType


class ChartGenerator:
    """ベンチマーク結果のチャート生成クラス"""
    
    def __init__(self, output_dir: Optional[Path] = None, style: str = "seaborn-v0_8"):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # スタイル設定
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Seabornテーマ設定
        sns.set_theme()
        
        # 日本語フォント設定
        if HAS_JAPANESE:
            japanize_matplotlib.japanize()
    
    def create_performance_chart(self, 
                                results: Dict[str, BenchmarkResult],
                                chart_type: ChartType = "bar",
                                save_path: Optional[Path] = None) -> Path:
        """パフォーマンスチャートを作成"""
        
        if chart_type == "bar":
            return self._create_bar_chart(results, save_path)
        elif chart_type == "box":
            return self._create_box_plot(results, save_path)
        elif chart_type == "heatmap":
            return self._create_heatmap(results, save_path)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
    
    def create_comparison_chart(self,
                              baseline: Dict[str, BenchmarkResult],
                              current: Dict[str, BenchmarkResult],
                              save_path: Optional[Path] = None) -> Path:
        """比較チャートを作成"""
        return self._create_comparison_chart(baseline, current, save_path)
    
    def create_historical_chart(self,
                               historical_data: Dict[str, Dict[str, BenchmarkResult]],
                               save_path: Optional[Path] = None) -> Path:
        """履歴チャートを作成"""
        return self._create_historical_chart(historical_data, save_path)
    
    def _create_bar_chart(self, results: Dict[str, BenchmarkResult], save_path: Optional[Path]) -> Path:
        """横棒グラフを作成"""
        # データを抽出
        viz_data = self._extract_visualization_data(results)
        
        if not viz_data["modules"]:
            raise ValueError("No data to visualize")
        
        # フィギュア作成
        n_modules = len(viz_data["modules"])
        fig_height = max(8, n_modules * 0.4)
        fig, axes = plt.subplots(1, 3, figsize=(18, fig_height))
        
        y_pos = np.arange(n_modules)
        
        # チャート設定
        chart_configs = [
            ("small_times", "lightblue", "Small Data"),
            ("medium_times", "lightgreen", "Medium Data"), 
            ("large_times", "lightcoral", "Large Data"),
        ]
        
        # チャート作成
        for ax, (data_key, color, title) in zip(axes, chart_configs):
            if data_key not in viz_data or not viz_data[data_key]:
                continue
                
            bars = ax.barh(y_pos, viz_data[data_key], color=color, alpha=0.8)
            
            # Y軸ラベル
            labels = [
                f"{m} {s} {n}"
                for m, s, n in zip(viz_data["modules"], viz_data["success_status"], viz_data["optimization_status"])
            ]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=10)
            
            # 軸ラベルとタイトル
            ax.set_xlabel("実行時間 (ms)", fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # 対数スケール
            ax.set_xscale("log")
            
            # グリッド
            ax.grid(True, which="major", axis="x", alpha=0.6, linestyle="-", linewidth=0.8)
            ax.grid(True, which="minor", axis="x", alpha=0.3, linestyle=":", linewidth=0.5)
            ax.grid(True, which="major", axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
            
            # 値ラベル
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(width, bar.get_y() + bar.get_height() / 2, 
                           f"{width:.3f}", ha="left", va="center", fontsize=8)
        
        # 全体タイトル
        plt.suptitle(f'ベンチマーク結果 - {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        # 凡例
        legend_text = "◯ = 成功, × = 失敗, ◆ = 最適化済み, ◇ = 未最適化"
        fig.text(0.5, 0.02, legend_text, ha="center", fontsize=10)
        
        # 保存
        save_path = save_path or (self.output_dir / f"performance_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def _create_box_plot(self, results: Dict[str, BenchmarkResult], save_path: Optional[Path]) -> Path:
        """ボックスプロットを作成"""
        # 成功した結果のみ抽出
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            raise ValueError("No successful results to visualize")
        
        # データ準備
        plot_data = []
        labels = []
        
        for module_name, result in successful_results.items():
            for size_name, times in result["timings"].items():
                for time_val in times:
                    plot_data.append({
                        "module": module_name,
                        "size": size_name,
                        "time_ms": time_val * 1000
                    })
        
        if not plot_data:
            raise ValueError("No timing data to visualize")
        
        # データフレーム作成（pandasが利用可能な場合）
        try:
            import pandas as pd
            df = pd.DataFrame(plot_data)
            
            # ボックスプロット作成
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=df, x="module", y="time_ms", hue="size", ax=ax)
            
            ax.set_yscale("log")
            ax.set_ylabel("実行時間 (ms)")
            ax.set_xlabel("モジュール")
            ax.set_title("ベンチマーク結果分布")
            plt.xticks(rotation=45, ha="right")
            
        except ImportError:
            # pandasが利用できない場合の代替実装
            fig, ax = plt.subplots(figsize=(12, 8))
            
            modules = list(successful_results.keys())
            size_names = ["small", "medium", "large"]
            
            box_data = []
            box_labels = []
            
            for module in modules:
                result = successful_results[module]
                for size_name in size_names:
                    if size_name in result["timings"]:
                        times_ms = [t * 1000 for t in result["timings"][size_name]]
                        box_data.append(times_ms)
                        box_labels.append(f"{module}\n{size_name}")
            
            ax.boxplot(box_data, labels=box_labels)
            ax.set_yscale("log")
            ax.set_ylabel("実行時間 (ms)")
            ax.set_title("ベンチマーク結果分布")
            plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        
        # 保存
        save_path = save_path or (self.output_dir / f"boxplot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def _create_heatmap(self, results: Dict[str, BenchmarkResult], save_path: Optional[Path]) -> Path:
        """ヒートマップを作成"""
        # 成功した結果のみ抽出
        successful_results = {k: v for k, v in results.items() if v["success"]}
        
        if not successful_results:
            raise ValueError("No successful results to visualize")
        
        # データマトリックス作成
        modules = list(successful_results.keys())
        sizes = ["small", "medium", "large"]
        
        data_matrix = np.zeros((len(modules), len(sizes)))
        
        for i, module in enumerate(modules):
            result = successful_results[module]
            for j, size in enumerate(sizes):
                if size in result["average_times"]:
                    data_matrix[i, j] = result["average_times"][size] * 1000  # ミリ秒に変換
        
        # ヒートマップ作成
        fig, ax = plt.subplots(figsize=(8, max(6, len(modules) * 0.3)))
        
        im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")
        
        # 軸設定
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels(sizes)
        ax.set_yticks(range(len(modules)))
        ax.set_yticklabels(modules)
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("実行時間 (ms)", rotation=270, labelpad=15)
        
        # 値を表示
        for i in range(len(modules)):
            for j in range(len(sizes)):
                if data_matrix[i, j] > 0:
                    ax.text(j, i, f"{data_matrix[i, j]:.2f}", 
                           ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title("ベンチマーク結果ヒートマップ")
        plt.tight_layout()
        
        # 保存
        save_path = save_path or (self.output_dir / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def _create_comparison_chart(self, 
                               baseline: Dict[str, BenchmarkResult],
                               current: Dict[str, BenchmarkResult], 
                               save_path: Optional[Path]) -> Path:
        """比較チャートを作成"""
        # 共通モジュールを抽出
        common_modules = set(baseline.keys()) & set(current.keys())
        common_modules = {m for m in common_modules 
                         if baseline[m]["success"] and current[m]["success"]}
        
        if not common_modules:
            raise ValueError("No common successful results to compare")
        
        # 改善率を計算
        modules = list(common_modules)
        improvements = []
        
        for module in modules:
            baseline_avg = np.mean(list(baseline[module]["average_times"].values()))
            current_avg = np.mean(list(current[module]["average_times"].values()))
            
            if baseline_avg > 0:
                improvement = (baseline_avg - current_avg) / baseline_avg * 100
            else:
                improvement = 0
            
            improvements.append(improvement)
        
        # チャート作成
        fig, ax = plt.subplots(figsize=(12, max(6, len(modules) * 0.4)))
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(range(len(modules)), improvements, color=colors, alpha=0.7)
        
        # 軸設定
        ax.set_yticks(range(len(modules)))
        ax.set_yticklabels(modules)
        ax.set_xlabel("改善率 (%)")
        ax.set_title("ベンチマーク結果比較")
        
        # ゼロライン
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 値ラベル
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            width = bar.get_width()
            ax.text(width + (5 if width >= 0 else -5), bar.get_y() + bar.get_height() / 2,
                   f"{imp:+.1f}%", ha="left" if width >= 0 else "right", va="center")
        
        # グリッド
        ax.grid(True, axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_path or (self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def _create_historical_chart(self, 
                               historical_data: Dict[str, Dict[str, BenchmarkResult]],
                               save_path: Optional[Path]) -> Path:
        """履歴チャートを作成"""
        if len(historical_data) < 2:
            raise ValueError("Need at least 2 data points for historical chart")
        
        # タイムスタンプでソート
        sorted_timestamps = sorted(historical_data.keys())
        
        # 共通モジュールを見つける
        all_modules = set()
        for data in historical_data.values():
            all_modules.update(data.keys())
        
        # すべてのタイムスタンプで成功しているモジュールのみ
        stable_modules = []
        for module in all_modules:
            if all(module in historical_data[ts] and historical_data[ts][module]["success"] 
                  for ts in sorted_timestamps):
                stable_modules.append(module)
        
        if not stable_modules:
            raise ValueError("No stable modules found across all timestamps")
        
        # チャート作成
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = range(len(sorted_timestamps))
        
        for module in stable_modules[:10]:  # 最大10モジュール
            y_values = []
            for ts in sorted_timestamps:
                result = historical_data[ts][module]
                avg_time = np.mean(list(result["average_times"].values())) * 1000  # ミリ秒
                y_values.append(avg_time)
            
            ax.plot(x, y_values, marker='o', label=module, linewidth=2, markersize=4)
        
        # 軸設定
        ax.set_xticks(x)
        ax.set_xticklabels([ts[:8] + "\n" + ts[9:].replace("_", ":") for ts in sorted_timestamps], 
                          rotation=45, ha="right")
        ax.set_ylabel("実行時間 (ms)")
        ax.set_title(f"ベンチマーク履歴 (最新{len(sorted_timestamps)}回)")
        ax.set_yscale("log")
        
        # 凡例
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        
        # グリッド
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        save_path = save_path or (self.output_dir / f"historical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        return save_path
    
    def _extract_visualization_data(self, results: Dict[str, BenchmarkResult]) -> Dict[str, List[Any]]:
        """可視化用データを抽出"""
        modules = []
        small_times = []
        medium_times = []
        large_times = []
        success_status = []
        optimization_status = []
        
        for module, data in sorted(results.items()):
            modules.append(module)
            
            if data["success"]:
                # 時間データ（ミリ秒に変換）
                small_time = data["average_times"].get("small", 0) * 1000
                medium_time = data["average_times"].get("medium", 0) * 1000
                large_time = data["average_times"].get("large", 0) * 1000
                
                small_times.append(small_time if small_time > 0 else 0.001)
                medium_times.append(medium_time if medium_time > 0 else 0.001)
                large_times.append(large_time if large_time > 0 else 0.001)
                success_status.append("◯")
            else:
                small_times.append(0.001)
                medium_times.append(0.001)
                large_times.append(0.001)
                success_status.append("×")
            
            # 最適化状況（njitまたはキャッシュの使用）
            metrics = data.get("metrics", {})
            has_optimization = metrics.get("has_njit", False) or metrics.get("has_cache", False)
            optimization_status.append("◆" if has_optimization else "◇")
        
        return {
            "modules": modules,
            "small_times": small_times,
            "medium_times": medium_times,
            "large_times": large_times,
            "success_status": success_status,
            "optimization_status": optimization_status,
        }


# 便利関数
def create_performance_chart(results: Dict[str, BenchmarkResult], 
                           chart_type: ChartType = "bar",
                           output_dir: Optional[Path] = None) -> Path:
    """パフォーマンスチャートを作成する便利関数"""
    generator = ChartGenerator(output_dir)
    return generator.create_performance_chart(results, chart_type)


def create_comparison_chart(baseline: Dict[str, BenchmarkResult],
                          current: Dict[str, BenchmarkResult],
                          output_dir: Optional[Path] = None) -> Path:
    """比較チャートを作成する便利関数"""
    generator = ChartGenerator(output_dir)
    return generator.create_comparison_chart(baseline, current)