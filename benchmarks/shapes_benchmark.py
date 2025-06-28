#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry対応シェイプモジュールベンチマークスイート

このスクリプトは、Geometry対応のshapes APIをベンチマークし、
実行時間の測定、キャッシュ使用状況の確認、失敗の追跡を行います。
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import inspect
import importlib

# Set seaborn theme
sns.set_theme()
# 日本語フォント設定
japanize_matplotlib.japanize()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.core.geometry import Geometry
from api import shapes
from benchmarks.benchmark_result_manager import BenchmarkResultManager
from benchmarks.benchmark_visualizer import BenchmarkVisualizer

# Type aliases
BenchmarkResult = Dict[str, Any]
TimingData = Dict[str, List[float]]
ShapeFunction = Callable[..., Geometry]


class GeometryShapeVisualizer(BenchmarkVisualizer):
    """Geometry shapes専用のベンチマーク可視化クラス（ミリ秒表示、対数スケール対応）"""

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
                # For shapes, we only have one timing measurement per shape
                time_ms = data["average_times"].get("default", 0) * 1000 if data["average_times"] else 0
                # Use the same time for all three categories
                small_times.append(time_ms if time_ms > 0 else 0.001)
                medium_times.append(time_ms if time_ms > 0 else 0.001)
                large_times.append(time_ms if time_ms > 0 else 0.001)
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
        # For shapes, we'll show all three columns but they'll have the same data
        fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(modules) * 0.4)))
        y_pos = np.arange(len(modules))

        # Chart configurations
        chart_configs = [
            ("small_times", "lightblue", "Simple Shapes"),
            ("medium_times", "lightgreen", "Medium Complexity"), 
            ("large_times", "lightcoral", "Complex Shapes"),
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

        plt.suptitle(f'Geometry Shape Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        fig.text(0.5, 0.02, "◯ = Success, × = Failed, ◆ = Uses cache, ◇ = No cache", ha="center", fontsize=10)

        return fig, axes


class GeometryShapeBenchmark:
    """Geometry対応シェイプ用ベンチマークシステム"""

    def __init__(self, output_dir: str = "benchmark_results", warmup_runs: int = 5, benchmark_runs: int = 20) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.shapes_dir = self.output_dir / "geometry_shapes"
        self.shapes_dir.mkdir(exist_ok=True)

        # Benchmark parameters
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        
        # Initialize result manager and custom visualizer for shapes
        # Create a custom result manager that uses 'shapes' directory instead of 'effects'
        class ShapeResultManager(BenchmarkResultManager):
            def __init__(self, output_dir: str):
                super().__init__(output_dir)
                self.effects_dir = self.output_dir / "shapes"
                self.effects_dir.mkdir(exist_ok=True)
        
        self.result_manager = ShapeResultManager(str(self.output_dir))
        self.visualizer = GeometryShapeVisualizer(str(self.output_dir))

    def get_shape_functions(self) -> Dict[str, List[Tuple[str, Callable, Dict[str, Any]]]]:
        """ベンチマーク対象のGeometry対応シェイプ関数を取得"""
        return {
            # 2D基本形状
            "polygon": [
                ("triangle", shapes.polygon, {"n_sides": 3}),
                ("hexagon", shapes.polygon, {"n_sides": 6}),
                ("circle_20", shapes.polygon, {"n_sides": 20}),
                ("circle_50", shapes.polygon, {"n_sides": 50}),
            ],
            "grid": [
                ("small_5x5", shapes.grid, {"rows": 5, "cols": 5}),
                ("medium_10x10", shapes.grid, {"rows": 10, "cols": 10}),
                ("large_20x20", shapes.grid, {"rows": 20, "cols": 20}),
            ],
            # 3D基本形状
            "sphere": [
                ("low_res", shapes.sphere, {"subdivisions": 0}),
                ("medium_res", shapes.sphere, {"subdivisions": 0.5}),
                ("high_res", shapes.sphere, {"subdivisions": 1}),
            ],
            "cylinder": [
                ("default", shapes.cylinder, {}),
                ("tall", shapes.cylinder, {"height": 2}),
                ("wide", shapes.cylinder, {"radius": 1.5}),
            ],
            "cone": [
                ("default", shapes.cone, {}),
                ("sharp", shapes.cone, {"height": 2, "radius": 0.5}),
                ("flat", shapes.cone, {"height": 0.5, "radius": 1.5}),
            ],
            "torus": [
                ("default", shapes.torus, {}),
                ("thick", shapes.torus, {"major_radius": 1, "minor_radius": 0.5}),
                ("thin", shapes.torus, {"major_radius": 1, "minor_radius": 0.1}),
            ],
            "capsule": [
                ("default", shapes.capsule, {}),
                ("tall", shapes.capsule, {"height": 2}),
                ("wide", shapes.capsule, {"radius": 1}),
            ],
            # 正多面体
            "polyhedron": [
                ("tetrahedron", shapes.polyhedron, {"polygon_type": "tetrahedron"}),
                ("hexahedron", shapes.polyhedron, {"polygon_type": "hexahedron"}),
                ("octahedron", shapes.polyhedron, {"polygon_type": "octahedron"}),
                ("dodecahedron", shapes.polyhedron, {"polygon_type": "dodecahedron"}),
                ("icosahedron", shapes.polyhedron, {"polygon_type": "icosahedron"}),
            ],
            # パラメトリック曲線
            "lissajous": [
                ("simple", shapes.lissajous, {"a": 3, "b": 2}),
                ("medium", shapes.lissajous, {"a": 5, "b": 4}),
                ("complex", shapes.lissajous, {"a": 7, "b": 6}),
            ],
            "attractor": [
                ("lorenz_short", shapes.attractor, {"attractor_type": "lorenz", "points": 1000}),
                ("lorenz_medium", shapes.attractor, {"attractor_type": "lorenz", "points": 5000}),
                ("rossler", shapes.attractor, {"attractor_type": "rossler", "points": 5000}),
            ],
            # テキスト・グリフ
            "text": [
                ("single_char", shapes.text, {"text": "A"}),
                ("word", shapes.text, {"text": "HELLO"}),
                ("sentence", shapes.text, {"text": "HELLO WORLD"}),
            ],
            "asemic_glyph": [
                ("default", shapes.asemic_glyph, {}),
                ("small", shapes.asemic_glyph, {"region": (-0.25, -0.25, 0.25, 0.25)}),
                ("large", shapes.asemic_glyph, {"region": (-1, -1, 1, 1)}),
            ],
        }

    def benchmark_shape(self, shape_name: str, variations: List[Tuple[str, Callable, Dict[str, Any]]]) -> Dict[str, Any]:
        """単一のシェイプタイプをベンチマーク"""
        results = {
            "shape": shape_name,
            "timestamp": datetime.now().isoformat(),
            "variations": {},
        }

        for var_name, shape_func, params in variations:
            var_results = {
                "success": False,
                "error": None,
                "timings": [],
                "average_time": 0,
                "vertex_count": 0,
                "params": params,
            }

            try:
                # Warmup
                for _ in range(self.warmup_runs):
                    shape_func(**params)

                # Benchmark
                times: List[float] = []
                for _ in range(self.benchmark_runs):
                    start_time = time.perf_counter()
                    geom = shape_func(**params)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)

                if times:
                    var_results["timings"] = times
                    var_results["average_time"] = np.mean(times)
                    var_results["vertex_count"] = len(geom.coords)
                    var_results["success"] = True

            except Exception as e:
                var_results["error"] = f"Failed to benchmark {shape_name}.{var_name}: {str(e)}"

            results["variations"][var_name] = var_results

        return results

    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """すべてのGeometry対応シェイプのベンチマークを実行"""
        shape_functions = self.get_shape_functions()
        all_results = {}

        print(f"Found {len(shape_functions)} shape types to benchmark")
        print("-" * 50)

        for shape_name, variations in shape_functions.items():
            print(f"Benchmarking {shape_name}...", end=" ")
            result = self.benchmark_shape(shape_name, variations)
            
            # Calculate overall average
            total_times = []
            success_count = 0
            for var_result in result["variations"].values():
                if var_result["success"]:
                    success_count += 1
                    total_times.append(var_result["average_time"])
            
            if total_times:
                avg_time = np.mean(total_times)
                print(f"◯ (avg: {avg_time*1000:.2f}ms, {success_count}/{len(variations)} variations)")
            else:
                print(f"× (all variations failed)")

            all_results[shape_name] = result

        return all_results

    def _check_cache_usage(self, shape_name: str) -> bool:
        """シェイプでlru_cacheが使用されているかを検出"""
        try:
            # shapes モジュールを検査
            shapes_module = importlib.import_module("api.shapes")
            
            # 特定のshape関数を取得
            if hasattr(shapes_module, shape_name):
                shape_func = getattr(shapes_module, shape_name)
                
                # 関数のソースコードを検査
                try:
                    func_source = inspect.getsource(shape_func)
                    if "@lru_cache" in func_source or "lru_cache" in func_source:
                        return True
                except (OSError, TypeError):
                    pass
                    
            # shapes モジュール全体のソースも検査
            try:
                module_source = inspect.getsource(shapes_module)
                if "@lru_cache" in module_source or "from functools import lru_cache" in module_source:
                    return True
            except (OSError, TypeError):
                pass
                
            return False
        except Exception:
            return False
    
    def save_results(self, results: Dict[str, Dict]) -> str:
        """ベンチマーク結果を保存"""
        # Convert to the expected format for BenchmarkResultManager
        formatted_results = self._format_results_for_manager(results)
        return self.result_manager.save_results(formatted_results)
    
    def _format_results_for_manager(self, results: Dict[str, Dict]) -> Dict[str, Dict[str, Any]]:
        """GeometryシェイプベンチマークのresultsをBenchmarkResultManager形式に変換"""
        formatted = {}
        
        for shape_name, shape_data in results.items():
            # Calculate overall average across all variations
            all_times = []
            has_cache = self._check_cache_usage(shape_name)
            
            for var_data in shape_data["variations"].values():
                if var_data["success"]:
                    all_times.append(var_data["average_time"])
            
            if all_times:
                avg_time = np.mean(all_times)
                formatted[shape_name] = {
                    "module": shape_name,
                    "timestamp": shape_data["timestamp"],
                    "success": True,
                    "error": None,
                    "cache_functions": {"apply": has_cache},
                    "timings": {"default": [avg_time]},
                    "average_times": {"default": avg_time},
                }
            else:
                formatted[shape_name] = {
                    "module": shape_name,
                    "timestamp": shape_data["timestamp"],
                    "success": False,
                    "error": "No successful variations",
                    "cache_functions": {"apply": has_cache},
                    "timings": {},
                    "average_times": {},
                }
        
        return formatted

    def visualize_results(self, results: BenchmarkResult, save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の可視化"""
        # Convert to the expected format for BenchmarkVisualizer
        formatted_results = self._format_results_for_manager(results)
        self.visualizer.visualize_results(formatted_results, save_path)
        
    def compare_historical(self, num_recent: int = 5) -> None:
        """最近のベンチマーク結果を比較"""
        historical_data = self.result_manager.get_historical_results(num_recent)
        if historical_data:
            self.visualizer.compare_historical(historical_data, num_recent)
        else:
            print("Not enough historical data for comparison")


    def print_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """結果のサマリーを表示"""
        print("\nDetailed Summary:")
        print("-" * 50)
        
        # 形状タイプ別の統計
        shape_types = {
            "2D": ["polygon", "grid"],
            "3D": ["sphere", "cylinder", "cone", "torus", "capsule"],
            "Polyhedra": ["polyhedron"],
            "Parametric": ["lissajous", "attractor"],
            "Text": ["text", "asemic_glyph"]
        }
        
        for type_name, shapes in shape_types.items():
            times = []
            for shape in shapes:
                if shape in results:
                    for var_data in results[shape]["variations"].values():
                        if var_data["success"]:
                            times.append(var_data["average_time"])
            
            if times:
                print(f"\n{type_name} Shapes:")
                print(f"  Count: {len(times)}")
                print(f"  Avg time: {np.mean(times)*1000:.2f}ms")
                print(f"  Min time: {np.min(times)*1000:.2f}ms")
                print(f"  Max time: {np.max(times)*1000:.2f}ms")
        
        # 最も遅い5つの形状
        print("\nSlowest 5 shapes:")
        all_shapes = []
        for shape_name, shape_data in results.items():
            for var_name, var_data in shape_data["variations"].items():
                if var_data["success"]:
                    all_shapes.append((
                        f"{shape_name}.{var_name}",
                        var_data["average_time"] * 1000,
                        var_data["vertex_count"]
                    ))
        
        all_shapes.sort(key=lambda x: x[1], reverse=True)
        for name, time_ms, vertices in all_shapes[:5]:
            print(f"  {name}: {time_ms:.2f}ms ({vertices} vertices)")


def main() -> None:
    """メインのベンチマーク実行"""
    benchmark = GeometryShapeBenchmark()

    print("Starting Geometry Shape Benchmark Suite")
    print("=" * 50)

    # Run benchmarks
    results = benchmark.run_benchmarks()

    # Save results
    saved_file = benchmark.save_results(results)
    print(f"\nResults saved to: {saved_file}")

    # Visualize results
    benchmark.visualize_results(results)
    print(f"Visualization saved to: {benchmark.shapes_dir / 'latest_chart.png'}")
    
    # Generate historical comparison
    print("\nGenerating historical comparison...")
    benchmark.compare_historical()

    # Print summary
    benchmark.print_summary(results)

    # Summary
    print("\nOverall Summary:")
    print("-" * 30)
    total_variations = sum(len(r["variations"]) for r in results.values())
    successful_variations = sum(1 for r in results.values() for v in r["variations"].values() if v["success"])
    
    print(f"Total shape types: {len(results)}")
    print(f"Total variations: {total_variations}")
    print(f"Successful: {successful_variations}")
    print(f"Failed: {total_variations - successful_variations}")


if __name__ == "__main__":
    main()