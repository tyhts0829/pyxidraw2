import arc
import numpy as np

from api.runner import run_sketch
from api.shapes import polygon
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    # 簡単なテスト：キャンバス中央に小さな多角形
    canvas_width, canvas_height = CANVAS_SIZES["SQUARE_200"]  # 200x200mm
    center_x = canvas_width / 2  # 100mm
    center_y = canvas_height / 2  # 100mm
    
    if t < 1:  # 最初の1秒だけprintする
        print(f"Canvas size: {canvas_width}x{canvas_height}mm")
        print(f"Center: ({center_x}, {center_y})")
    
    # まず基本polygonを確認
    poly = polygon(n_sides=6)
    
    # 手動でスケールとセンターを適用
    from api.effects import scaling, translation
    poly = scaling(poly, scale=(50, 50, 1))
    poly = translation(poly, offset_x=center_x, offset_y=center_y, offset_z=0)
    
    if t < 1:  # 最初の1秒だけprintする
        print(f"Final polygon coords min: {poly.coords.min(axis=0)}")
        print(f"Final polygon coords max: {poly.coords.max(axis=0)}")
        print(f"Final polygon coords:\n{poly.coords}")
    
    return poly


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()