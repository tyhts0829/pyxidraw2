#!/usr/bin/env python3
"""
直感的なメソッドチェーンAPIのデモ
"""

import arc
from api.shapes import polygon, sphere
from api.runner import run_sketch
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    """
    様々なメソッドチェーン記法のデモ
    """
    
    # ユーザーの要求した記法
    sph = sphere(subdivisions=0.5).scale(100, 100, 100).translate(100, 100, 0)
    sph2 = sphere(subdivisions=0.3).scale(50, 50, 50).translate(50, 100, 0)
    
    # より簡潔な記法
    poly1 = polygon(n_sides=6).size(30).at(100, 50).spin(t * 0.1)
    
    # 従来の詳細記法
    poly2 = (polygon(n_sides=4)
             .scale_uniform(20)
             .center_at(150, 150)
             .rotate_z(t * 0.05, center=(150, 150, 0)))
    
    # 混在記法
    poly3 = (polygon(n_sides=8)
             .size(15)
             .move_to(50, 150)
             .spin(t * -0.05))
    
    # 複合変換記法
    poly4 = polygon(n_sides=3).transform(
        center=(175, 50, 0),
        scale=(25, 25, 1),
        rotate=(0, 0, t * 0.2)
    )
    
    # 全て組み合わせて返す
    scene = sph + sph2 + poly1 + poly2 + poly3 + poly4
    return scene


if __name__ == "__main__":
    print("🎨 Fluent API Demo - 直感的なメソッドチェーン記法")
    print("例: sphere().size(100).at(100, 100).spin(angle)")
    print("例: polygon(6).scale(50, 50, 50).translate(x, y, z)")
    
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()