#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geometry対応シェイプモジュールベンチマークスイート

このスクリプトは、Geometry対応のshapes APIをベンチマークし、
実行時間の測定、キャッシュ使用状況の確認、失敗の追跡を行います。
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.core.geometry import Geometry
from api import shapes

# Type aliases
BenchmarkResult = Dict[str, Any]
TimingData = Dict[str, List[float]]
ShapeFunction = Callable[..., Geometry]


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

    def save_results(self, results: Dict[str, Dict]) -> str:
        """タイムスタンプ付きでベンチマーク結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.shapes_dir / f"benchmark_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        # Also save as latest
        latest_file = self.shapes_dir / "latest.json"
        with open(latest_file, "w") as f:
            json.dump(results, f, indent=2)

        return str(filename)

    def visualize_results(self, results: BenchmarkResult, save_path: Optional[str] = None) -> None:
        """ベンチマーク結果の可視化"""
        # 結果を処理時間でソート（最も遅いものから）
        sorted_shapes = []
        for shape_name, shape_data in results.items():
            avg_times = [v["average_time"] for v in shape_data["variations"].values() if v["success"]]
            if avg_times:
                sorted_shapes.append((shape_name, np.mean(avg_times)))
        sorted_shapes.sort(key=lambda x: x[1], reverse=True)
        
        # 上位15個の形状を選択
        top_shapes = [name for name, _ in sorted_shapes[:15]]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        
        # Chart 1: Top 15 shapes by average time
        self._plot_top_shapes(ax1, results, top_shapes)
        
        # Chart 2: Time vs vertex count scatter plot
        self._plot_time_vs_vertices(ax2, results)

        plt.suptitle(f'Geometry Shape Benchmarks - {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save chart
        self._save_chart(fig, save_path)

    def _plot_top_shapes(self, ax: Axes, results: Dict[str, Any], top_shapes: List[str]) -> None:
        """上位シェイプの実行時間をプロット"""
        shapes = []
        times = []
        colors = []
        
        for shape_name in top_shapes:
            if shape_name in results:
                shape_data = results[shape_name]
                for var_name, var_data in shape_data["variations"].items():
                    if var_data["success"]:
                        shapes.append(f"{shape_name}\n{var_name}")
                        times.append(var_data["average_time"] * 1000)
                        # Color by shape type
                        if shape_name in ["sphere", "cylinder", "cone", "torus", "capsule"]:
                            colors.append("lightcoral")  # 3D shapes
                        elif shape_name == "polyhedron":
                            colors.append("lightgreen")  # Polyhedra
                        elif shape_name in ["lissajous", "attractor"]:
                            colors.append("lightsalmon")  # Parametric
                        elif shape_name in ["text", "asemic_glyph"]:
                            colors.append("lightgoldenrodyellow")  # Text
                        else:
                            colors.append("lightblue")  # 2D shapes
        
        y_pos = np.arange(len(shapes))
        bars = ax.barh(y_pos, times, color=colors)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(shapes, fontsize=8)
        ax.set_xlabel('Time (ms)')
        ax.set_title('Top 15 Shapes by Execution Time')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}',
                   ha='left', va='center', fontsize=7)

    def _plot_time_vs_vertices(self, ax: Axes, results: Dict[str, Any]) -> None:
        """実行時間と頂点数の相関をプロット"""
        vertex_counts = []
        times = []
        labels = []
        colors = []
        
        for shape_name, shape_data in results.items():
            for var_name, var_data in shape_data["variations"].items():
                if var_data["success"]:
                    vertex_counts.append(var_data["vertex_count"])
                    times.append(var_data["average_time"] * 1000)
                    labels.append(f"{shape_name}_{var_name}")
                    
                    # Color by shape type
                    if shape_name in ["sphere", "cylinder", "cone", "torus", "capsule"]:
                        colors.append("red")  # 3D shapes
                    elif shape_name == "polyhedron":
                        colors.append("green")  # Polyhedra
                    elif shape_name in ["lissajous", "attractor"]:
                        colors.append("orange")  # Parametric
                    elif shape_name in ["text", "asemic_glyph"]:
                        colors.append("purple")  # Text
                    else:
                        colors.append("blue")  # 2D shapes
        
        ax.scatter(vertex_counts, times, c=colors, alpha=0.6, s=50)
        ax.set_xlabel('Vertex Count')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Execution Time vs Vertex Count')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='2D Shapes'),
            Patch(facecolor='red', label='3D Shapes'),
            Patch(facecolor='green', label='Polyhedra'),
            Patch(facecolor='orange', label='Parametric'),
            Patch(facecolor='purple', label='Text')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

    def _save_chart(self, fig: Figure, save_path: Optional[str] = None) -> None:
        """チャートを保存"""
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = self.shapes_dir / f"benchmark_chart_{timestamp}.png"
            plt.savefig(save_file, dpi=150, bbox_inches="tight")

            # Also save as latest
            latest_chart = self.shapes_dir / "latest_chart.png"
            plt.savefig(latest_chart, dpi=150, bbox_inches="tight")

        plt.close()

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