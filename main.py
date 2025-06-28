import arc
import numpy as np

from api.effects import (
    array,
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
    sph = sphere(subdivisions=cc[1]).transform(
        center=(100, 100, 0),
        scale=(100, 100, 100),
        rotate=(cc[2], cc[2], cc[2]),
    )
    # sph = buffer(sph, distance=cc[4])
    # sph = filling(sph, density=cc[5])
    # sph = subdivision(sph, n_divisions=cc[3])
    sph = noise(sph, intensity=cc[4])
    return sph


if __name__ == "__main__":
    arc.start(midi=False)  # MIDIを無効化してテスト
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=8, background=(1, 1, 1, 1))
    arc.stop()
