from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from engine.core.geometry import Geometry
from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_combined_transform(
    vertices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rotate: np.ndarray,
) -> np.ndarray:
    """頂点に組み合わせ変換を適用します。"""
    # 回転行列を一度だけ計算
    sx, sy, sz = np.sin(rotate)
    cx, cy, cz = np.cos(rotate)
    
    # Z * Y * X の結合行列を直接計算
    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = cy * cz
    R[0, 1] = sx * sy * cz - cx * sz
    R[0, 2] = cx * sy * cz + sx * sz
    R[1, 0] = cy * sz
    R[1, 1] = sx * sy * sz + cx * cz
    R[1, 2] = cx * sy * sz - sx * cz
    R[2, 0] = -sy
    R[2, 1] = sx * cy
    R[2, 2] = cx * cy
    
    # 全頂点に変換を一度に適用（スケール -> 回転 -> 移動）
    scaled = vertices * scale
    rotated = scaled @ R.T
    transformed = rotated + center
    
    return transformed


class Transform(BaseEffect):
    """任意の変換行列を適用します。"""

    TAU = math.tau  # 全回転角度（2 * pi）

    def apply(
        self,
        geometry: Geometry,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params: Any,
    ) -> Geometry:
        """変換エフェクトを適用します。

        Args:
            geometry: 入力Geometry
            center: 変換の中心点 (x, y, z)
            scale: スケール係数 (x, y, z)
            rotate: 回転角度（ラジアン） (x, y, z) 入力は0.0-1.0の範囲を想定。内部でmath.tauを掛けてラジアンに変換される。
            **params: 追加パラメータ（無視される）

        Returns:
            変換されたGeometry
        """

        # エッジケース: 空の座標配列
        if len(geometry.coords) == 0:
            return geometry

        # エッジケース: 変換がない場合
        if (center == (0, 0, 0) and scale == (1, 1, 1) and 
            abs(rotate[0]) < 1e-10 and abs(rotate[1]) < 1e-10 and abs(rotate[2]) < 1e-10):
            return geometry

        # NumPy配列に変換
        center_np = np.array(center, dtype=np.float32)
        scale_np = np.array(scale, dtype=np.float32)
        rotate_radians = np.array([
            rotate[0] * self.TAU,
            rotate[1] * self.TAU,
            rotate[2] * self.TAU,
        ], dtype=np.float32)

        # 全頂点に組み合わせ変換を一度に適用
        transformed_coords = _apply_combined_transform(geometry.coords, center_np, scale_np, rotate_radians)
        
        # 新しいGeometryを作成（offsetsは変更なし）
        return Geometry(transformed_coords, geometry.offsets)
