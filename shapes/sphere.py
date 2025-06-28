from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .base import BaseShape


@lru_cache(maxsize=128)
def _sphere_cached(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere vertices using latitude/longitude lines.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere triangles
    """
    # Number of segments based on subdivision level
    n_segments = 8 * (2**subdivisions)
    n_rings = n_segments // 2

    vertices_list = []

    # Generate sphere using latitude/longitude lines
    for i in range(n_rings):
        lat1 = np.pi * i / n_rings
        lat2 = np.pi * (i + 1) / n_rings

        ring = []
        for j in range(n_segments + 1):
            lon = 2 * np.pi * j / n_segments

            # Calculate vertices for this segment
            x1 = np.sin(lat1) * np.cos(lon) * 0.5
            y1 = np.sin(lat1) * np.sin(lon) * 0.5
            z1 = np.cos(lat1) * 0.5

            x2 = np.sin(lat2) * np.cos(lon) * 0.5
            y2 = np.sin(lat2) * np.sin(lon) * 0.5
            z2 = np.cos(lat2) * 0.5

            ring.extend([[x1, y1, z1], [x2, y2, z2]])

        vertices_list.append(np.array(ring, dtype=np.float32))

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_wireframe(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere wireframe using longitude/latitude lines only.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere wireframe
    """
    n_segments = 8 * (2**subdivisions)
    n_rings = n_segments // 2

    vertices_list = []

    # Longitude lines
    for j in range(n_segments):
        lon = 2 * np.pi * j / n_segments
        line = []
        for i in range(n_rings + 1):
            lat = np.pi * i / n_rings
            x = np.sin(lat) * np.cos(lon) * 0.5
            y = np.sin(lat) * np.sin(lon) * 0.5
            z = np.cos(lat) * 0.5
            line.append([x, y, z])
        vertices_list.append(np.array(line, dtype=np.float32))

    # Latitude lines
    for i in range(1, n_rings):  # Skip poles
        lat = np.pi * i / n_rings
        line = []
        for j in range(n_segments + 1):
            lon = 2 * np.pi * j / n_segments
            x = np.sin(lat) * np.cos(lon) * 0.5
            y = np.sin(lat) * np.sin(lon) * 0.5
            z = np.cos(lat) * 0.5
            line.append([x, y, z])
        vertices_list.append(np.array(line, dtype=np.float32))

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_zigzag(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere using zigzag pattern.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere spiral
    """
    n_points = 200 * (2**subdivisions)

    vertices = []
    # Golden angle spiral
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n_points):
        # Parametric sphere using golden angle
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)

        theta = golden_angle * i

        x = np.cos(theta) * radius * 0.5
        z = np.sin(theta) * radius * 0.5
        y = y * 0.5

        vertices.append([x, y, z])

    # Convert to line segments for smooth drawing
    vertices_list = []
    vertices_array = np.array(vertices, dtype=np.float32)

    # Create short line segments between consecutive points
    for i in range(len(vertices_array) - 1):
        line_segment = np.array([vertices_array[i], vertices_array[i + 1]], dtype=np.float32)
        vertices_list.append(line_segment)

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_icosphere(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere using icosphere pattern with hierarchical subdivision.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere icosphere
    """
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # 12 vertices of an icosahedron
    base_vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )

    # Normalize vertices to unit sphere
    norms = np.linalg.norm(base_vertices, axis=1, keepdims=True)
    base_vertices = base_vertices / norms * 0.5

    # Base triangular faces for icosahedron
    base_faces = [
        # Top cap triangles
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        # Bottom cap triangles
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        # Middle band triangles
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (9, 5, 4), (4, 11, 2), (2, 10, 6), (6, 7, 8), (8, 1, 9)
    ]

    def subdivide_triangle(v1, v2, v3, level):
        """Recursively subdivide a triangle into smaller triangles."""
        if level <= 0:
            # Base case: return the triangle edges
            return [
                (v1, v2), (v2, v3), (v3, v1)
            ]
        
        # Calculate midpoints and project to sphere surface
        def midpoint_on_sphere(p1, p2):
            mid = (p1 + p2) / 2
            norm = np.linalg.norm(mid)
            return mid / norm * 0.5  # Project to sphere radius 0.5
        
        m1 = midpoint_on_sphere(v1, v2)
        m2 = midpoint_on_sphere(v2, v3)
        m3 = midpoint_on_sphere(v3, v1)
        
        # Recursively subdivide the 4 new triangles
        edges = []
        edges.extend(subdivide_triangle(v1, m1, m3, level - 1))
        edges.extend(subdivide_triangle(m1, v2, m2, level - 1))
        edges.extend(subdivide_triangle(m3, m2, v3, level - 1))
        edges.extend(subdivide_triangle(m1, m2, m3, level - 1))
        
        return edges

    # Generate all edges with subdivision
    all_edges = []
    for face in base_faces:
        v1, v2, v3 = base_vertices[face[0]], base_vertices[face[1]], base_vertices[face[2]]
        edges = subdivide_triangle(v1, v2, v3, subdivisions)
        all_edges.extend(edges)

    # Convert edges to vertex arrays, removing duplicates
    vertices_list = []
    seen_edges = set()
    
    for edge in all_edges:
        # Create a hashable representation of the edge
        edge_key = tuple(sorted([tuple(edge[0]), tuple(edge[1])]))
        
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            line = np.array([edge[0], edge[1]], dtype=np.float32)
            vertices_list.append(line)

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_rings(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere using horizontal ring slices.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere rings
    """
    n_rings = 5 + 12 * subdivisions  # Number of slices per axis
    n_segments = 64  # Points per ring

    vertices_list = []

    # Create horizontal rings at different heights
    for i in range(n_rings):
        # Height from -0.5 to 0.5
        y = -0.5 + (i / (n_rings - 1))

        # Radius at this height (sphere equation: x² + y² + z² = r²)
        if abs(y) <= 0.5:
            radius = np.sqrt(0.25 - y * y)  # radius of circle at height y

            # Generate circle points
            ring_points = []
            for j in range(n_segments + 1):  # +1 to close the circle
                angle = 2 * np.pi * j / n_segments
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                ring_points.append([x, y, z])

            vertices_list.append(np.array(ring_points, dtype=np.float32))

    # Create rings perpendicular to X-axis (slicing along YZ plane)
    for i in range(n_rings):
        # X position from -0.5 to 0.5
        x = -0.5 + (i / (n_rings - 1))

        # Radius at this X position
        if abs(x) <= 0.5:
            radius = np.sqrt(0.25 - x * x)

            # Generate circle points in YZ plane
            ring_points = []
            for j in range(n_segments + 1):
                angle = 2 * np.pi * j / n_segments
                y = radius * np.cos(angle)
                z = radius * np.sin(angle)
                ring_points.append([x, y, z])

            vertices_list.append(np.array(ring_points, dtype=np.float32))

    # Create rings perpendicular to Z-axis (slicing along XY plane)
    for i in range(n_rings):
        # Z position from -0.5 to 0.5
        z = -0.5 + (i / (n_rings - 1))

        # Radius at this Z position
        if abs(z) <= 0.5:
            radius = np.sqrt(0.25 - z * z)

            # Generate circle points in XY plane
            ring_points = []
            for j in range(n_segments + 1):
                angle = 2 * np.pi * j / n_segments
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                ring_points.append([x, y, z])

            vertices_list.append(np.array(ring_points, dtype=np.float32))

    return vertices_list




class Sphere(BaseShape):
    """Sphere shape generator with multiple drawing styles."""

    def generate(self, subdivisions: float = 0.5, sphere_type: float = 0.5, **_params: Any) -> Geometry:
        """Generate a sphere with radius 1.

        Args:
            subdivisions: Subdivision level (0.0-1.0, mapped to 0-5)
            sphere_type: Drawing style (0.0-1.0):
                        0.0-0.2: Lat-Lon (default style)
                        0.2-0.4: Wireframe
                        0.4-0.6: Zigzag
                        0.6-0.8: Icosphere
                        0.8-1.0: Rings
            **params: Additional parameters (ignored)

        Returns:
            Geometry object containing sphere geometry
        """
        MIN_SUBDIVISIONS = 0
        MAX_SUBDIVISIONS = 5
        subdivisions_int = int(subdivisions * MAX_SUBDIVISIONS)
        if subdivisions_int < MIN_SUBDIVISIONS:
            subdivisions_int = MIN_SUBDIVISIONS

        # Select sphere generation method based on sphere_type
        if sphere_type < 0.2:
            # Default lat-lon style
            vertices_list = _sphere_cached(subdivisions_int)
        elif sphere_type < 0.4:
            # Wireframe style
            vertices_list = _sphere_wireframe(subdivisions_int)
        elif sphere_type < 0.6:
            # Zigzag style
            vertices_list = _sphere_zigzag(subdivisions_int)
        elif sphere_type < 0.8:
            # Icosphere style
            vertices_list = _sphere_icosphere(subdivisions_int)
        else:
            # Rings style
            vertices_list = _sphere_rings(subdivisions_int)

        return Geometry.from_lines(vertices_list)
