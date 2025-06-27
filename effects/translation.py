from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from engine.core.geometry import Geometry
from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_translation(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """頂点に移動を適用します。"""
    # Apply translation
    translated = vertices + offset
    return translated.astype(np.float32)


class Translation(BaseEffect):
    """指定されたオフセットで頂点を移動します。"""
    
    def apply(self, geometry: Geometry,
             offset_x: float = 0.0,
             offset_y: float = 0.0,
             offset_z: float = 0.0,
             **params: Any) -> Geometry:
        """移動エフェクトを適用します。
        
        Args:
            geometry: 入力Geometry
            offset_x: X軸の移動オフセット
            offset_y: Y軸の移動オフセット
            offset_z: Z軸の移動オフセット
            **params: 追加パラメータ（無視される）
            
        Returns:
            移動されたGeometry
        """
        # オフセットがゼロの場合は元のデータをそのまま返す
        if offset_x == 0.0 and offset_y == 0.0 and offset_z == 0.0:
            return geometry
        
        # エッジケース: 空の座標配列
        if len(geometry.coords) == 0:
            return geometry
        
        # Create offset vector
        offset = np.array([offset_x, offset_y, offset_z], dtype=np.float32)
        
        # 全頂点に移動を一度に適用
        translated_coords = _apply_translation(geometry.coords, offset)
        
        # 新しいGeometryを作成（offsetsは変更なし）
        return Geometry(translated_coords, geometry.offsets)