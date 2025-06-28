from typing import List

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from effects.effects_utils import transform_back, transform_to_xy_plane
from effects.scaling import _scaling


def _buffer(
    vertices_list: List[np.ndarray], distance: float = 0.25, join_style=0, resolution: float = 0.5
) -> List[np.ndarray]:
    if distance == 0:
        return vertices_list
    join_style = _determine_join_style(join_style)
    MAX_RESOLUTION = 10
    resolution = int(resolution * MAX_RESOLUTION)
    new_vertices_list = []
    for vertices in vertices_list:
        vertices = _close_curve(vertices, 1e-3)
        vertices_on_xy, R, z = transform_to_xy_plane(vertices)
        line = LineString(vertices_on_xy[:, :2])
        buffered_line = line.buffer(distance, join_style=join_style, resolution=resolution)  # type: ignore
        if buffered_line.is_empty:
            continue
        if isinstance(buffered_line, (LineString, MultiLineString)):
            new_vertices_list = _extract_vertices_from_line(new_vertices_list, buffered_line, R, z)
        elif isinstance(buffered_line, (Polygon, MultiPolygon)):
            new_vertices_list = _extract_vertices_from_polygon(new_vertices_list, buffered_line, R, z)
    # もとのスケールに戻す
    new_vertices_list = _scaling(new_vertices_list, 1 / (1 + distance * 2))
    return new_vertices_list


def _extract_vertices_from_polygon(new_vertices_list: list, buffered_line: BaseGeometry, R: np.ndarray, z: float):
    if isinstance(buffered_line, Polygon):
        polygons = [buffered_line]
    else:  # MultiPolygon
        polygons = buffered_line.geoms  # type: ignore
    for poly in polygons:
        coords = np.array(poly.exterior.coords)
        new_vertices = np.hstack([coords, np.zeros((len(coords), 1))])
        new_vertices = transform_back(new_vertices, R, z)
        new_vertices_list.append(new_vertices)
    return new_vertices_list


def _extract_vertices_from_line(new_vertices_list: list, buffered_line: BaseGeometry, R: np.ndarray, z: float):
    if isinstance(buffered_line, LineString):
        lines = [buffered_line]
    else:  # MultiLineString
        lines = buffered_line.geoms  # type: ignore
    for line in lines:
        coords = np.array(line.coords)
        new_vertices = np.hstack([coords, np.zeros((len(coords), 1))])
        new_vertices = transform_back(new_vertices, R, z)
        new_vertices_list.append(new_vertices)
    return new_vertices_list


def _determine_join_style(join_style):
    """
    join_styleがfloatの場合、0.0〜0.3なら"round"、0.3〜0.7なら"mitre"、0.7〜1.0なら"bevel"として扱う
    """
    if isinstance(join_style, (float, int)):
        if 0.0 <= join_style < 0.3:
            join_style = "round"
        elif 0.3 <= join_style < 0.7:
            join_style = "mitre"
        elif 0.7 <= join_style <= 1.0:
            join_style = "bevel"
        else:
            raise ValueError(f"join_style(float)の値は0.0〜1.0の範囲で指定してください: {join_style}")
    elif isinstance(join_style, str):
        if join_style not in ["round", "mitre", "bevel"]:
            raise ValueError(
                f"join_style(str)の値は'round', 'mitre', 'bevel'のいずれかで指定してください: {join_style}"
            )
    else:
        raise ValueError(f"join_styleの型が不正です: {join_style}")
    return join_style


def _close_curve(points, threshold):
    # 始点と終点の座標を取得
    start = points[0]
    end = points[-1]

    # 始点と終点の距離を計算
    dist = np.linalg.norm(start - end)

    # 距離がthreshold以下なら、終点を削除し、始点を終点として追加
    if dist <= threshold:
        points_copy = points[:-1]  # 終点を削除
        points = np.vstack([points_copy, start])  # 始点を終点として追加

    return points
