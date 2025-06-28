#!/usr/bin/env python3
"""Buffer effectの動作確認テスト"""

import numpy as np
from api.shapes import polygon
from effects.buffer import Buffer


def test_buffer_basic():
    """基本的なバッファーエフェクトのテスト"""
    print("基本的なバッファーエフェクトのテストを開始...")
    
    # テスト用の正方形を作成
    square = polygon(n_sides=4).size(50).at(100, 100, 0)
    print(f"元の正方形頂点数: {len(square.coords)}")
    
    # バッファーエフェクトを適用
    buffer_effect = Buffer()
    
    # 小さなバッファー
    buffered_small = buffer_effect.apply(square, distance=0.2, join_style=0.1, resolution=0.5)
    print(f"小バッファー後の頂点数: {len(buffered_small.coords)}")
    
    # 大きなバッファー
    buffered_large = buffer_effect.apply(square, distance=0.8, join_style=0.5, resolution=0.8)
    print(f"大バッファー後の頂点数: {len(buffered_large.coords)}")
    
    # バッファー処理の結果を確認（頂点数またはgeometry自体の変化をチェック）
    small_changed = len(buffered_small.coords) > len(square.coords) or not np.allclose(square.coords, buffered_small.coords)
    large_changed = len(buffered_large.coords) > len(square.coords) or not np.allclose(square.coords, buffered_large.coords)
    
    print(f"小バッファー変化: {small_changed}, 大バッファー変化: {large_changed}")
    
    if small_changed and large_changed:
        print("✅ 基本バッファーテスト成功")
        return True
    else:
        print("❌ 基本バッファーテスト失敗")
        return False


def test_buffer_join_styles():
    """異なる接合スタイルのテスト"""
    print("\n異なる接合スタイルのテストを開始...")
    
    # テスト用の三角形を作成
    triangle = polygon(n_sides=3).size(40).at(100, 100, 0)
    print(f"元の三角形頂点数: {len(triangle.coords)}")
    
    buffer_effect = Buffer()
    
    # Round join style (0.0-0.33)
    buffered_round = buffer_effect.apply(triangle, distance=0.5, join_style=0.1)
    print(f"Round join後の頂点数: {len(buffered_round.coords)}")
    
    # Mitre join style (0.33-0.67)
    buffered_mitre = buffer_effect.apply(triangle, distance=0.5, join_style=0.5)
    print(f"Mitre join後の頂点数: {len(buffered_mitre.coords)}")
    
    # Bevel join style (0.67-1.0)
    buffered_bevel = buffer_effect.apply(triangle, distance=0.5, join_style=0.8)
    print(f"Bevel join後の頂点数: {len(buffered_bevel.coords)}")
    
    # 全ての接合スタイルでバッファーが適用されることを確認（座標変化もチェック）
    round_changed = len(buffered_round.coords) > len(triangle.coords) or not np.allclose(triangle.coords, buffered_round.coords)
    mitre_changed = len(buffered_mitre.coords) > len(triangle.coords) or not np.allclose(triangle.coords, buffered_mitre.coords)
    bevel_changed = len(buffered_bevel.coords) > len(triangle.coords) or not np.allclose(triangle.coords, buffered_bevel.coords)
    
    print(f"Round変化: {round_changed}, Mitre変化: {mitre_changed}, Bevel変化: {bevel_changed}")
    
    if round_changed and mitre_changed and bevel_changed:
        print("✅ 接合スタイルテスト成功")
        return True
    else:
        print("❌ 接合スタイルテスト失敗")
        return False


def test_buffer_line():
    """線に対するバッファーのテスト"""
    print("\n線に対するバッファーのテストを開始...")
    
    # テスト用の直線を作成（Geometryクラスを直接使用）
    from engine.core.geometry import Geometry
    points = np.array([[50, 50, 0], [150, 100, 0], [250, 50, 0]], dtype=np.float32)
    test_line = Geometry.from_lines([points])
    print(f"元の線頂点数: {len(test_line.coords)}")
    
    buffer_effect = Buffer()
    
    # バッファー適用
    buffered_line = buffer_effect.apply(test_line, distance=0.3, resolution=0.7)
    print(f"バッファー後の頂点数: {len(buffered_line.coords)}")
    
    # バッファーによって頂点数が増加することを確認
    if len(buffered_line.coords) > len(test_line.coords):
        print("✅ 線バッファーテスト成功")
        return True
    else:
        print("❌ 線バッファーテスト失敗")
        return False


def test_buffer_zero_distance():
    """距離0でのバッファーテスト"""
    print("\n距離0でのバッファーテストを開始...")
    
    # テスト用の円を作成
    circle = polygon(n_sides=8).size(30).at(100, 100, 0)
    original_coords = circle.coords.copy()
    
    buffer_effect = Buffer()
    
    # 距離0でバッファー適用
    buffered_zero = buffer_effect.apply(circle, distance=0.0)
    
    # 元の形状と同じであることを確認
    coords_equal = np.allclose(original_coords, buffered_zero.coords)
    print(f"距離0バッファー: 元の座標と同じ = {coords_equal}")
    
    if coords_equal:
        print("✅ 距離0バッファーテスト成功")
        return True
    else:
        print("❌ 距離0バッファーテスト失敗")
        return False


def main():
    """メインテスト実行"""
    print("=== Buffer Effect動作確認テスト ===\n")
    
    test_results = []
    
    try:
        test_results.append(test_buffer_basic())
        test_results.append(test_buffer_join_styles())
        test_results.append(test_buffer_line())
        test_results.append(test_buffer_zero_distance())
        
        print(f"\n=== テスト結果 ===")
        print(f"成功: {sum(test_results)}/{len(test_results)}")
        
        if all(test_results):
            print("✅ 全てのテストが成功しました！")
        else:
            print("❌ 一部のテストが失敗しました。")
            
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()