from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from engine.core.geometry import Geometry
from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_scaling(vertices: np.ndarray, scale_array: np.ndarray, center: np.ndarray) -> np.ndarray:
    """頂点にスケーリングを適用します。"""
    # 中心点に対してスケーリング
    centered = vertices - center
    scaled = centered * scale_array
    result = scaled + center
    return result


class Scaling(BaseEffect):
    """指定された軸に沿って頂点をスケールします。"""
    
    def apply(self, geometry: Geometry,
             center: tuple[float, float, float] = (0, 0, 0),
             scale: tuple[float, float, float] = (1, 1, 1),
             **params: Any) -> Geometry:
        """スケールエフェクトを適用します。
        
        Args:
            geometry: 入力Geometry
            center: スケーリングの中心点 (x, y, z)
            scale: 各軸のスケール率 (x, y, z)
            **params: 追加パラメータ（無視される）
            
        Returns:
            スケールされたGeometry
        """
        # スケール値がすべて1の場合は元のデータをそのまま返す
        if scale == (1, 1, 1):
            return geometry
        
        # エッジケース: 空の座標配列
        if len(geometry.coords) == 0:
            return geometry
        
        # NumPy配列に変換
        scale_np = np.array(scale, dtype=np.float32)
        center_np = np.array(center, dtype=np.float32)
        
        # 全頂点にスケーリングを一度に適用
        scaled_coords = _apply_scaling(geometry.coords, scale_np, center_np)
        
        # 新しいGeometryを作成（offsetsは変更なし）
        return Geometry(scaled_coords, geometry.offsets)