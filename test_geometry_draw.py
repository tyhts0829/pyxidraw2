import arc
import numpy as np

from api.runner import run_sketch
from geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    # 複数の線を含むGeometryを作成
    
    # 四角形
    rect = np.array([
        [50, 50, 0],
        [150, 50, 0],
        [150, 150, 0],
        [50, 150, 0],
        [50, 50, 0],  # Close
    ], dtype=np.float32)
    
    # 円（近似）
    n_points = 20
    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    radius = 30
    center_x, center_y = 100, 100
    circle = np.array([
        [center_x + radius * np.cos(angle), center_y + radius * np.sin(angle), 0]
        for angle in angles
    ], dtype=np.float32)
    
    # 対角線
    diagonal1 = np.array([
        [50, 50, 0],
        [150, 150, 0]
    ], dtype=np.float32)
    
    diagonal2 = np.array([
        [150, 50, 0],
        [50, 150, 0]
    ], dtype=np.float32)
    
    return Geometry.from_lines([rect, circle, diagonal1, diagonal2])


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()