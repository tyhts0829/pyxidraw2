import arc
import numpy as np

from api.runner import run_sketch
from geometry import Geometry
from util.constants import CANVAS_SIZES

cw, ch = CANVAS_SIZES["SQUARE_200"]


def draw(t, cc) -> Geometry:

    size = 100  # Use MIDI CC to control size

    rect = np.array(
        [
            [-size / 2 + 100, -size / 2 + 100, 0],
            [size / 2 + 100, -size / 2 + 100, 0],
            [size / 2 + 100, size / 2 + 100, 0],
            [-size / 2 + 100, size / 2 + 100, 0],
            [-size / 2 + 100, -size / 2 + 100, 0],  # Close
        ],
        dtype=np.float32,
    )
    rect = Geometry.from_lines([rect])
    triangle = np.array(
        [
            [-size / 2 + 100, -size / 2 + 100, 0],
            [size / 2 + 100, -size / 2 + 100, 0],
            [0 + 100, size / 2 + 100, 0],
            [-size / 2 + 100, -size / 2 + 100, 0],  # Close
        ],
        dtype=np.float32,
    )
    triangle = Geometry.from_lines([triangle])
    return rect + triangle


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=6, background=(1, 1, 1, 1))
    arc.stop()
