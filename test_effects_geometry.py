import arc
import numpy as np

from api.effects import (
    buffer,
    filling,
    noise,
    scaling,
    subdivision,
    transform,
    translation,
)
from api.runner import run_sketch
from api.shapes import polygon, polyhedron, sphere
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t, cc) -> Geometry:
    sph = polyhedron(12).transform(center=(100, 100, 0), scale=(100, 100, 100), rotate=(cc[1], cc[2], cc[3]))
    sph = buffer(sph, distance=cc[4])
    sph = filling(sph, density=cc[5])
    sph = subdivision(sph, n_divisions=cc[6])
    sph = noise(sph, intensity=cc[7])
    return sph


if __name__ == "__main__":
    arc.start(midi=False)  # MIDIを無効化してテスト
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=8, background=(1, 1, 1, 1))
    arc.stop()

"""
sph = sphere(subdivisions=0.5).scale(100, 100, 0).translate(100, 100, 0)
sph2 = sphere(subdivisions=0.3).scale(50, 50, 0).translate(50, 100, 0)
scene = sph + sph2
return scene
"""
