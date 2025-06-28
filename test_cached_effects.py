#!/usr/bin/env python3
"""
キャッシュ付きエフェクトチェーンのデモとテスト
"""

import arc
import time
from api.shapes import polygon, sphere
from api.runner import run_sketch
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    """
    キャッシュ効果を確認するためのテスト
    同じ操作を繰り返し実行してキャッシュヒット率をテスト
    """
    
    # キャッシュ統計を定期的に表示
    if int(t * 2) % 4 == 0:  # 2秒ごと
        stats = Geometry.get_cache_stats()
        print(f"Cache stats: {stats}")
    
    # 同じパラメータでの操作（キャッシュされるはず）
    base_subdivisions = 0.5
    
    # パターン1: 同じ球体に同じ変換（完全キャッシュヒット）
    sph1 = (sphere(subdivisions=base_subdivisions)
            .size(50)
            .at(100, 100)
            .spin(0.5))  # 固定角度
    
    # パターン2: 同じ球体に異なる変換（部分キャッシュヒット）
    sph2 = (sphere(subdivisions=base_subdivisions)
            .size(30)
            .at(150, 50)
            .spin(t * 0.1))  # 動的角度
    
    # パターン3: 多角形の繰り返しパターン（キャッシュ効果確認）
    poly1 = (polygon(n_sides=6)
             .size(20)
             .at(50, 50)
             .spin(0.0))  # 固定
    
    poly2 = (polygon(n_sides=6)
             .size(20)  # 同じサイズ
             .at(50, 150)  # 異なる位置
             .spin(0.0))  # 同じ角度
    
    # パターン4: 複雑なチェーン（段階的キャッシュ）
    complex_shape = (polygon(n_sides=8)
                     .scale(15, 15, 1)
                     .translate(150, 150, 0)
                     .rotate(0, 0, t * 0.05))
    
    # 全て組み合わせ
    scene = sph1 + sph2 + poly1 + poly2 + complex_shape
    return scene


if __name__ == "__main__":
    print("🚀 Cached Effects Demo - エフェクトチェーンキャッシュテスト")
    print("同じ操作の繰り返しでキャッシュ効果を確認します")
    
    # 初期キャッシュクリア
    Geometry.clear_effect_cache()
    
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()
    
    # 最終キャッシュ統計
    final_stats = Geometry.get_cache_stats()
    print(f"Final cache stats: {final_stats}")