#!/usr/bin/env python3
"""
キャッシュ効果を実感できる実用的なデモ
同じ形状を何度も使い回すシーンでキャッシュの恩恵を受ける
"""

import arc
from api.shapes import polygon, sphere
from api.runner import run_sketch
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    """
    同じ基本形状から多数のバリエーションを作成
    キャッシュにより高速化される
    """
    
    # 基本的な建物パーツ（キャッシュされる）
    building_base = polygon(n_sides=4).size(20)
    window = polygon(n_sides=4).size(3)
    
    # 建物1（オフィスビル風）
    building1 = building_base.at(50, 50)
    windows1 = []
    for i in range(3):
        for j in range(2):
            windows1.append(window.at(45 + j*10, 45 + i*5))
    
    # 建物2（同じ基本形状、異なる位置）
    building2 = building_base.at(150, 50)
    windows2 = []
    for i in range(3):
        for j in range(2):
            windows2.append(window.at(145 + j*10, 45 + i*5))
    
    # 円形の建物（球体利用）
    dome = sphere(subdivisions=0.3).size(25).at(100, 150)
    
    # 装飾的な要素（多角形バリエーション）
    decoration1 = polygon(n_sides=6).size(8).at(75, 150).spin(t * 0.1)
    decoration2 = polygon(n_sides=6).size(8).at(125, 150).spin(-t * 0.1)
    decoration3 = polygon(n_sides=8).size(6).at(100, 100).spin(t * 0.2)
    
    # 動的な雲（同じ基本円形から）
    cloud_base = sphere(subdivisions=0.2).size(15)
    cloud1 = cloud_base.at(30 + t*5, 180)
    cloud2 = cloud_base.at(170 - t*3, 175)
    
    # すべてを組み合わせ
    scene = building1 + building2 + dome + decoration1 + decoration2 + decoration3
    scene += cloud1 + cloud2
    
    # 窓を追加
    for w in windows1 + windows2:
        scene += w
    
    # 定期的にキャッシュ統計を表示
    if int(t) % 5 == 0 and t > 0:  # 5秒ごと
        stats = Geometry.get_cache_stats()
        print(f"シーン描画中 - キャッシュ統計: サイズ={stats['size']}")
    
    return scene


if __name__ == "__main__":
    print("🏙️  Cached Scene Demo - キャッシュによる高速化デモ")
    print("同じ基本形状を多数使い回してキャッシュ効果を実感")
    
    # キャッシュをクリア
    Geometry.clear_effect_cache()
    
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()
    
    final_stats = Geometry.get_cache_stats()
    print(f"デモ終了 - 最終キャッシュ統計: {final_stats}")