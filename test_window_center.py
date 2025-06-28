#!/usr/bin/env python3
"""Test window centering functionality."""

from api.runner import run_sketch
from api.shapes import polygon
from engine.core.geometry import Geometry
from util.constants import CANVAS_SIZES


def draw(t: float, cc: dict[int, float]) -> Geometry:
    """Draw a simple shape for testing window position."""
    # Draw a square (4-sided polygon) at the center
    square = polygon(n_sides=4).scale(50, 50, 1).translate(100, 100, 0)
    return square


if __name__ == "__main__":
    # Run the sketch with a square canvas
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=4, background=(1, 1, 1, 1))