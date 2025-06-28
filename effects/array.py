from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from engine.core.geometry import Geometry
from .base import BaseEffect


class Array(BaseEffect):
    """入力のコピーを配列状に生成します。"""

    MAX_DUPLICATES = 10

    def apply(
        self,
        geometry: Geometry,
        n_duplicates: float = 0.5,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float] = (0.5, 0.5, 0.5),
        scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        **params: Any,
    ) -> Geometry:
        """配列エフェクトを適用します。

        Args:
            geometry: 入力Geometry
            n_duplicates: 複製数の係数（0.0-1.0、最大10個まで）
            offset: 各複製間のオフセット（x, y, z）
            rotate: 各複製における回転増分（0.0-1.0、0.5が中立）
            scale: 各複製におけるスケール係数（0.0-1.0、0.5が中立）
            center: 配列の中心点（x, y, z）
            **params: 追加パラメータ

        Returns:
            配列化されたGeometry

        Note:
            n_duplicatesが0の場合、元のgeometryをそのまま返します。
            各複製では前の複製に対して累積的にtransformが適用されます。
        """
        from api.effects import transform, translation

        n_duplicates_int = int(n_duplicates * self.MAX_DUPLICATES)
        if not n_duplicates_int:
            return geometry

        # 一旦centerを0, 0, 0に移動
        translated = translation(geometry, offset_x=-center[0], offset_y=-center[1], offset_z=-center[2])

        # 複製されたGeometryを収集するリスト
        geometries = []
        current_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        scale_arr = np.array(scale, dtype=np.float32)

        for n in range(n_duplicates_int):
            # transformed = transform(vertices_list, center=center, scale=current_scale, rotate=rotate)
            translated = transform(translated, center=offset, scale=tuple(current_scale), rotate=rotate)
            # 等差的にスケールを適用
            current_scale = _update_scale(current_scale, scale_arr)
            # centerに戻す
            result = translation(translated, offset_x=center[0], offset_y=center[1], offset_z=center[2])
            geometries.append(result)

        # すべてのGeometryを結合
        combined = geometries[0]
        for g in geometries[1:]:
            combined = combined + g

        return combined


@njit
def _update_scale(current_scale: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """スケール値を更新します（JIT高速化）。"""
    return current_scale * scale
