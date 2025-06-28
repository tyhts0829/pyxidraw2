#!/usr/bin/env python3

import arc
from api.shapes import polygon, sphere
from api.runner import run_sketch
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    # ユーザーの希望する直感的なチェーン記法
    sph = sphere(subdivisions=0.5).scale(100, 100, 100).translate(100, 100, 0)
    sph2 = sphere(subdivisions=0.3).scale(50, 50, 50).translate(50, 100, 0)
    
    # 便利なショートカットメソッドの使用例
    poly = (polygon(n_sides=6)
            .scale_uniform(30)
            .center_at(100, 50)
            .rotate_z(t * 0.1, center=(100, 50, 0)))
    
    # 複合変換
    poly2 = polygon(n_sides=4).transform(center=(150, 150, 0), 
                                        scale=(20, 20, 1), 
                                        rotate=(t * 0.05, 0, 0))
    
    # さらなる例
    poly3 = (polygon(n_sides=8)
             .scale_uniform(15)
             .move_to(50, 150, 0)
             .rotate_z(t * -0.05, center=(50, 150, 0)))
    
    # すべて組み合わせ
    scene = sph + sph2 + poly + poly2 + poly3
    return scene


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()