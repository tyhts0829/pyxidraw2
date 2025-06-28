#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry対応エフェクトモジュールベンチマークスイート

このスクリプトは、Geometry対応のエフェクト（transform, rotate, scale, translate）をベンチマークし、
実行時間の測定と結果の可視化を行います。
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Set seaborn theme
sns.set_theme()
# 日本語フォント設定
japanize_matplotlib.japanize()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.benchmark_result_manager import BenchmarkResultManager
from benchmarks.benchmark_visualizer import BenchmarkVisualizer
from engine.core.geometry import Geometry
import inspect
import importlib

# Type aliases
BenchmarkResult = Dict[str, Any]
TimingData = Dict[str, List[float]]


class GeometryEffectVisualizer(BenchmarkVisualizer):
    """Geometry effects専用のベンチマーク可視化クラス（ミリ秒表示、対数スケール対応）"""

    def _extract_visualization_data(self, results: Dict[str, BenchmarkResult]) -> Dict[str, List[Any]]:
        """可視化用のデータを抽出（ミリ秒単位）"""
        modules: List[str] = []
        small_times: List[float] = []
        medium_times: List[float] = []
        large_times: List[float] = []
        has_njit: List[str] = []
        success_status: List[str] = []

        for module, data in sorted(results.items()):
            modules.append(module)

            if data["success"]:
                small_time = data["average_times"].get("small", 0)
                medium_time = data["average_times"].get("medium", 0)
                large_time = data["average_times"].get("large", 0)
                # Convert to milliseconds
                small_times.append(small_time * 1000 if small_time > 0 else 0.001)
                medium_times.append(medium_time * 1000 if medium_time > 0 else 0.001)
                large_times.append(large_time * 1000 if large_time > 0 else 0.001)
                success_status.append("◯")
            else:
                small_times.append(0.001)
                medium_times.append(0.001)
                large_times.append(0.001)
                success_status.append("×")

            njit_funcs = data.get("njit_functions", {})
            has_njit.append("◆" if any(njit_funcs.values()) else "◇")

        return {
            "modules": modules,
            "small_times": small_times,
            "medium_times": medium_times,
            "large_times": large_times,
            "has_njit": has_njit,
            "success_status": success_status,
        }

    def _create_benchmark_charts(self, viz_data: Dict[str, List[Any]]) -> Tuple[Figure, List[Axes]]:
        """ベンチマークチャートを作成（対数スケール、ミリ秒表示）"""
        modules = viz_data["modules"]
        fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(modules) * 0.4)))
        y_pos = np.arange(len(modules))

        # Chart configurations
        chart_configs = [
            ("small_times", "lightblue", "Small Data Size"),
            ("medium_times", "lightgreen", "Medium Data Size"),
            ("large_times", "lightcoral", "Large Data Size"),
        ]

        # Create charts
        for ax, (data_key, color, title) in zip(axes, chart_configs):
            bars = ax.barh(y_pos, viz_data[data_key], color=color)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [
                    f"{m} {s} {n}"
                    for m, s, n in zip(viz_data["modules"], viz_data["success_status"], viz_data["has_njit"])
                ]
            )
            ax.set_xlabel("Time (ms)")
            ax.set_title(title)

            # Set log scale for x-axis
            ax.set_xscale("log")

            # Add vertical grid lines for better comparison
            ax.grid(True, which="major", axis="x", alpha=0.6, linestyle="-", linewidth=0.8)
            ax.grid(True, which="minor", axis="x", alpha=0.3, linestyle=":", linewidth=0.5)

            # Add horizontal grid lines as well for better readability
            ax.grid(True, which="major", axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

            # Add value labels with 3 decimal places
            for bar in bars:
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", ha="left", va="center", fontsize=8
                    )

        plt.suptitle(f'Geometry Effect Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        fig.text(0.5, 0.02, "◯ = Success, × = Failed, ◆ = Uses njit, ◇ = No njit", ha="center", fontsize=10)

        return fig, axes


class GeometryEffectBenchmark:
    """Geometry対応エフェクト用ベンチマークシステム"""

    def __init__(self, output_dir: str = "benchmark_results", warmup_runs: int = 5, benchmark_runs: int = 20) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.effects_dir = self.output_dir / "geometry_effects"
        self.effects_dir.mkdir(exist_ok=True)

        # Benchmark parameters
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

        # Initialize test shapes lazily
        self._test_geometries: Optional[Dict[str, Geometry]] = None

        # Initialize result manager and custom visualizer
        self.result_manager = BenchmarkResultManager(str(self.output_dir))
        self.visualizer = GeometryEffectVisualizer(str(self.output_dir))

    @property
    def test_geometries(self) -> Dict[str, Geometry]:
        """テストデータの遅延初期化"""
        if self._test_geometries is None:
            self._test_geometries = {
                "small": self._create_rectangle(1, 1),
                "medium": self._create_polygon(20),
                "large": self._create_complex_shape(),
            }
        return self._test_geometries

    @staticmethod
    def _create_rectangle(width: float, height: float) -> Geometry:
        """シンプルな長方形を作成"""
        hw, hh = width / 2, height / 2
        vertices = np.array(
            [[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0], [-hw, -hh, 0.0]], dtype=np.float32
        )
        return Geometry.from_lines([vertices])

    @staticmethod
    def _create_polygon(n: int) -> Geometry:
        """n辺の正多角形を作成"""
        angles = np.linspace(0, 2 * np.pi, n + 1)
        vertices = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)]).astype(np.float32)
        return Geometry.from_lines([vertices])

    @staticmethod
    def _create_circle(radius: float, segments: int = 64) -> Geometry:
        """円を作成"""
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        vertices = np.column_stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]).astype(
            np.float32
        )
        return Geometry.from_lines([vertices])

    def _create_complex_shape(self) -> Geometry:
        """ベンチマーク用の複雑な形状を作成（複数の円を結合）"""
        circles = []
        for i in range(10):
            radius = 1.0 + i * 0.1
            angles = np.linspace(0, 2 * np.pi, 64 + 1)
            vertices = np.column_stack(
                [radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)]
            ).astype(np.float32)
            circles.append(vertices)
        return Geometry.from_lines(circles)

    def get_geometry_effects(self) -> Dict[str, List[Tuple[str, Any]]]:
        """ベンチマーク対象のGeometry対応エフェクトを取得"""
        # api.effectsからnoise関数をインポート
        from api.effects import noise

        return {
            "transform": [
                ("identity", lambda g: g.transform()),
                ("scale_translate", lambda g: g.transform(center=(0, 0, 0), scale=(2, 2, 2), rotate=(0, 0, 0))),
                ("rotate_scale", lambda g: g.transform(center=(0, 0, 0), scale=(1.5, 1.5, 1.5), rotate=(45, 0, 0))),
            ],
            "rotate": [
                ("x_90", lambda g: g.rotate(90, 0, 0)),
                ("y_45", lambda g: g.rotate(0, 45, 0)),
                ("z_180", lambda g: g.rotate(0, 0, 180)),
                ("xyz", lambda g: g.rotate(30, 45, 60)),
            ],
            "scale": [
                ("uniform_2x", lambda g: g.scale(2, 2, 2)),
                ("non_uniform", lambda g: g.scale(2, 0.5, 1)),
                ("tiny", lambda g: g.scale(0.1, 0.1, 0.1)),
            ],
            "translate": [
                ("x_10", lambda g: g.translate(10, 0, 0)),
                ("y_20", lambda g: g.translate(0, 20, 0)),
                ("xyz", lambda g: g.translate(10, 20, 30)),
            ],
            "noise": [
                ("low_intensity", lambda g: noise(g, intensity=0.1, frequency=1.0)),
                ("medium_intensity", lambda g: noise(g, intensity=0.5, frequency=1.0)),
                ("high_intensity", lambda g: noise(g, intensity=1.0, frequency=1.0)),
                ("high_frequency", lambda g: noise(g, intensity=0.5, frequency=3.0)),
            ],
        }

    def benchmark_effect(self, effect_name: str, variations: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """単一のエフェクトタイプをベンチマーク"""
        results = {
            "effect": effect_name,
            "timestamp": datetime.now().isoformat(),
            "variations": {},
        }

        for var_name, effect_func in variations:
            var_results = {
                "success": False,
                "error": None,
                "timings": {},
                "average_times": {},
            }

            try:
                # Benchmark for different data sizes
                for size_name, test_geom in self.test_geometries.items():
                    times = self._benchmark_single_size(effect_func, test_geom, size_name)

                    if times is None:
                        var_results["error"] = f"Error during benchmark for {size_name} size"
                        break

                    if times:
                        var_results["timings"][size_name] = times
                        var_results["average_times"][size_name] = np.mean(times)

                var_results["success"] = bool(var_results["timings"])

            except Exception as e:
                var_results["error"] = f"Failed to benchmark {effect_name}.{var_name}: {str(e)}"

            results["variations"][var_name] = var_results

        return results

    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """すべてのGeometry対応エフェクトのベンチマークを実行"""
        effects = self.get_geometry_effects()
        all_results = {}

        print(f"Found {len(effects)} geometry effects to benchmark")
        print("-" * 50)

        for effect_name, variations in effects.items():
            print(f"Benchmarking {effect_name}...", end=" ")
            result = self.benchmark_effect(effect_name, variations)

            # Calculate overall average
            total_times = []
            success_count = 0
            for var_result in result["variations"].values():
                if var_result["success"]:
                    success_count += 1
                    total_times.extend(var_result["average_times"].values())

            if total_times:
                avg_time = np.mean(total_times)
                print(f"◯ (avg: {avg_time*1000:.2f}ms, {success_count}/{len(variations)} variations)")
            else:
                print(f"× (all variations failed)")

            all_results[effect_name] = result

        return all_results

    def save_results(self, results: Dict[str, Dict]) -> str:
        """ベンチマーク結果を保存"""
        # Convert to the expected format for BenchmarkResultManager
        formatted_results = self._format_results_for_manager(results)
        return self.result_manager.save_results(formatted_results)

    def _check_njit_usage(self, effect_name: str) -> bool:
        """エフェクトでnjitが使用されているかを検出"""
        try:
            # 各エフェクトに対応するモジュールを検査
            effect_modules = {
                "noise": "effects.noise",
                "transform": "effects.transform",
                "rotate": "effects.rotation",
                "scale": "effects.scaling", 
                "translate": "effects.translation"
            }
            
            module_name = effect_modules.get(effect_name)
            if module_name:
                try:
                    effect_module = importlib.import_module(module_name)
                    
                    # モジュール内でnjitデコレータが使用されているかチェック
                    module_source = inspect.getsource(effect_module)
                    if "@njit" in module_source or "from numba import njit" in module_source:
                        return True
                        
                    # 関数レベルでもチェック
                    for name, obj in inspect.getmembers(effect_module):
                        if inspect.isfunction(obj):
                            # numbaでラップされた関数を検出
                            if hasattr(obj, '__wrapped__') or hasattr(obj, 'py_func'):
                                return True
                            # ソースコードからnjitデコレータを検出
                            try:
                                func_source = inspect.getsource(obj)
                                if "@njit" in func_source:
                                    return True
                            except (OSError, TypeError):
                                continue
                except ImportError:
                    # モジュールが見つからない場合は False
                    return False
                            
            return False
        except Exception as e:
            print(f"Debug: njit detection error for {effect_name}: {e}")
            return False

    def _format_results_for_manager(self, results: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """GeometryエフェクトベンチマークのresultsをBenchmarkResultManager形式に変換"""
        formatted = {}

        for effect_name, effect_data in results.items():
            # Calculate overall averages across all variations and sizes
            all_times = []
            has_njit = self._check_njit_usage(effect_name)

            for var_data in effect_data["variations"].values():
                if var_data["success"]:
                    all_times.extend(var_data["average_times"].values())

            if all_times:
                formatted[effect_name] = {
                    "module": effect_name,
                    "timestamp": effect_data["timestamp"],
                    "success": True,
                    "error": None,
                    "njit_functions": {"apply": has_njit},
                    "timings": {
                        "small": [
                            var_data["average_times"].get("small", 0)
                            for var_data in effect_data["variations"].values()
                            if var_data["success"] and "small" in var_data["average_times"]
                        ][:1],
                        "medium": [
                            var_data["average_times"].get("medium", 0)
                            for var_data in effect_data["variations"].values()
                            if var_data["success"] and "medium" in var_data["average_times"]
                        ][:1],
                        "large": [
                            var_data["average_times"].get("large", 0)
                            for var_data in effect_data["variations"].values()
                            if var_data["success"] and "large" in var_data["average_times"]
                        ][:1],
                    },
                    "average_times": {
                        "small": np.mean(
                            [
                                var_data["average_times"].get("small", 0)
                                for var_data in effect_data["variations"].values()
                                if var_data["success"] and "small" in var_data["average_times"]
                            ]
                        ),
                        "medium": np.mean(
                            [
                                var_data["average_times"].get("medium", 0)
                                for var_data in effect_data["variations"].values()
                                if var_data["success"] and "medium" in var_data["average_times"]
                            ]
                        ),
                        "large": np.mean(
                            [
                                var_data["average_times"].get("large", 0)
                                for var_data in effect_data["variations"].values()
                                if var_data["success"] and "large" in var_data["average_times"]
                            ]
                        ),
                    },
                }
            else:
                formatted[effect_name] = {
                    "module": effect_name,
                    "timestamp": effect_data["timestamp"],
                    "success": False,
                    "error": "No successful variations",
                    "njit_functions": {"apply": has_njit},
                    "timings": {},
                    "average_times": {},
                }

        return formatted

    def visualize_results(self, results: BenchmarkResult, save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の可視化"""
        # Convert to the expected format for BenchmarkVisualizer
        formatted_results = self._format_results_for_manager(results)
        self.visualizer.visualize_results(formatted_results, save_path)

    def _benchmark_single_size(
        self, effect_func, test_geom: Geometry, size_name: str
    ) -> Optional[List[float]]:  # noqa: ARG002
        """単一サイズのデータでベンチマークを実行"""
        # Warmup
        for _ in range(self.warmup_runs):
            try:
                effect_func(test_geom)
            except Exception:
                return None

        # Benchmark
        times: List[float] = []
        for _ in range(self.benchmark_runs):
            try:
                start_time = time.perf_counter()
                effect_func(test_geom)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception:
                return None

        return times

    def compare_historical(self, num_recent: int = 5) -> None:
        """最近のベンチマーク結果を比較"""
        historical_data = self.result_manager.get_historical_results(num_recent)
        if historical_data:
            self.visualizer.compare_historical(historical_data, num_recent)
        else:
            print("Not enough historical data for comparison")


def main() -> None:
    """メインのベンチマーク実行"""
    benchmark = GeometryEffectBenchmark()

    print("Starting Geometry Effect Benchmark Suite")
    print("=" * 50)

    # Run benchmarks
    results = benchmark.run_benchmarks()

    # Save results
    saved_file = benchmark.save_results(results)
    print(f"\nResults saved to: {saved_file}")

    # Visualize results
    benchmark.visualize_results(results)
    print(f"Visualization saved to: {benchmark.effects_dir / 'latest_chart.png'}")

    # Compare historical data
    print("\nGenerating historical comparison...")
    benchmark.compare_historical()

    # Summary
    print("\nSummary:")
    print("-" * 30)
    total_variations = sum(len(r["variations"]) for r in results.values())
    successful_variations = sum(1 for r in results.values() for v in r["variations"].values() if v["success"])

    print(f"Total effects: {len(results)}")
    print(f"Total variations: {total_variations}")
    print(f"Successful: {successful_variations}")
    print(f"Failed: {total_variations - successful_variations}")


if __name__ == "__main__":
    main()
