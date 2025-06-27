import arc
import numpy as np

from api.runner import run_sketch
from api.shapes import grid, lissajous, polygon, sphere
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    # 複数の形状を組み合わせてテスト

    # 多角形（適切なスケール）
    poly = polygon(n_sides=6, center=(50, 50, 0), scale=(30, 30, 1))

    # 球体（適切なスケール）
    sph = sphere(subdivisions=cc[1], center=(150, 50, 0), scale=(25, 25, 25))

    # グリッド（適切なスケール）
    gr = grid(n_divisions=(0.3, 0.3), center=(50, 150, 0), scale=(60, 60, 1))

    # リサージュ曲線（適切なスケール）
    liss = lissajous(freq_x=3, freq_y=2, phase=t, points=100, center=(150, 150, 0), scale=(40, 40, 1))

    # 全て組み合わせ
    combined = poly + sph + gr + liss

    return combined


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()
