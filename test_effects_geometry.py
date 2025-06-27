import arc
import numpy as np

from api.effects import rotation, scaling, transform, translation
from api.runner import run_sketch
from api.shapes import polygon, sphere
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    # 時間に基づいた動的変換のテスト

    # # 回転する多角形
    # poly_rotating = polygon(n_sides=6, center=(50, 50, 0), scale=(20, 20, 1))
    # poly_rotating = rotation(poly_rotating, center=(50, 50, 0), rotate=(0, 0, t * 0.1))

    # 回転する多角形
    poly_rotating = polygon(n_sides=6, center=(50, 50, 0), scale=(20, 20, 1))
    poly_rotating = rotation(poly_rotating, center=(50, 50, 0), rotate=(0, 0, t * 0.1))

    # 動的スケーリングする球体
    sph_scaling = sphere(subdivisions=0.5, center=(150, 50, 0), scale=(15, 15, 15))
    scale_factor = 1 + cc[1] * 2  # 1-3倍の範囲でスケール
    sph_scaling = scaling(sph_scaling, center=(150, 50, 0), scale=(scale_factor, scale_factor, scale_factor))

    # 移動する多角形
    poly2_moving = polygon(n_sides=8, center=(50, 150, 0), scale=(15, 15, 1))
    offset_x = cc[2] * 50  # CC値で移動
    poly2_moving = translation(poly2_moving, offset_x=offset_x, offset_y=0, offset_z=0)

    # 複合変換（回転+スケール+移動）
    poly3_complex = polygon(n_sides=4, center=(150, 150, 0), scale=(10, 10, 1))
    poly3_complex = transform(poly3_complex, center=(150, 150, 0), 
                             scale=(1 + cc[3], 1 + cc[3], 1), 
                             rotate=(t * 0.05, 0, 0))

    # 全て組み合わせ
    return poly_rotating + sph_scaling + poly2_moving + poly3_complex


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()
