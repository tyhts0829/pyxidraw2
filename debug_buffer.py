#!/usr/bin/env python3
"""Buffer effectのデバッグ"""

import numpy as np
from api.shapes import polygon
from effects.buffer import Buffer


def debug_buffer():
    """バッファー処理のデバッグ"""
    print("=== Buffer Effect デバッグ ===\n")
    
    # テスト用の正方形を作成
    square = polygon(n_sides=4).size(50).at(100, 100, 0)
    print(f"元の正方形: {square.coords.shape}")
    print(f"座標範囲: X[{square.coords[:, 0].min():.1f}, {square.coords[:, 0].max():.1f}] "
          f"Y[{square.coords[:, 1].min():.1f}, {square.coords[:, 1].max():.1f}]")
    
    buffer_effect = Buffer()
    
    # 小さなバッファー（成功する）
    print("\n--- 小さなバッファー (distance=0.2) ---")
    buffered_small = buffer_effect.apply(square, distance=0.2, join_style=0.1, resolution=0.5)
    print(f"結果: {buffered_small.coords.shape}")
    print(f"座標範囲: X[{buffered_small.coords[:, 0].min():.1f}, {buffered_small.coords[:, 0].max():.1f}] "
          f"Y[{buffered_small.coords[:, 1].min():.1f}, {buffered_small.coords[:, 1].max():.1f}]")
    
    # 大きなバッファー（失敗する）
    print("\n--- 大きなバッファー (distance=0.8) ---")
    buffered_large = buffer_effect.apply(square, distance=0.8, join_style=0.5, resolution=0.8)
    print(f"結果: {buffered_large.coords.shape}")
    print(f"座標範囲: X[{buffered_large.coords[:, 0].min():.1f}, {buffered_large.coords[:, 0].max():.1f}] "
          f"Y[{buffered_large.coords[:, 1].min():.1f}, {buffered_large.coords[:, 1].max():.1f}]")
    
    # 元の座標と比較
    print(f"\n元の座標と大きなバッファーの比較:")
    print(f"座標が同じ: {np.allclose(square.coords, buffered_large.coords)}")
    
    # actual_distanceを確認
    actual_distance_small = 0.2 * 25.0
    actual_distance_large = 0.8 * 25.0
    print(f"\n実際の距離: 小={actual_distance_small}, 大={actual_distance_large}")


if __name__ == "__main__":
    debug_buffer()