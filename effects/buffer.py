from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .base import BaseEffect


class Buffer(BaseEffect):
    """平行線を作成してパスをバッファー/オフセットします。"""
    
    def apply(self, geometry: Geometry, **params: Any) -> Geometry:
        """バッファーエフェクトを適用します。
        
        入力パスから指定された距離に平行線を作成します。
        
        Args:
            geometry: 入力Geometryオブジェクト
            distance: バッファー距離（正の値=外向き、負の値=内向き） - デフォルト 0.5 (0.0-1.0レンジで10倍される)
            join_style: 角の接合スタイル（"round", "miter", "bevel"） - デフォルト "round"
            **params: 追加パラメータ
            
        Returns:
            元のパスとオフセットパスを含むバッファー化されたGeometry
        """
        distance = params.get('distance', 0.5)
        join_style = params.get('join_style', 'round')
        
        # 0.0-1.0 レンジを実際の距離値にスケール（10倍）
        actual_distance = distance * 10.0
        
        if actual_distance == 0:
            return geometry
        
        buffered_results = []
        
        # 既存の線を取得
        coords, offsets = geometry.as_arrays()
        for i in range(len(offsets) - 1):
            vertices = coords[offsets[i]:offsets[i+1]]
            if len(vertices) < 2:
                buffered_results.append(vertices)
                continue
            
            # 元のパスを追加
            buffered_results.append(vertices)
            
            # オフセットパスを作成
            offset_paths = self._create_offset_paths(vertices, actual_distance, join_style)
            buffered_results.extend(offset_paths)
        
        return Geometry.from_lines(buffered_results)
    
    def _create_offset_paths(self, vertices: np.ndarray, distance: float, join_style: str) -> list[np.ndarray]:
        """入力パスの両側にオフセットパスを作成します。"""
        if len(vertices) < 2:
            return []
        
        # Create left and right offset paths
        left_path = self._offset_path(vertices, distance)
        right_path = self._offset_path(vertices, -distance)
        
        paths = []
        if left_path is not None:
            paths.append(left_path)
        if right_path is not None:
            paths.append(right_path)
        
        return paths
    
    def _offset_path(self, vertices: np.ndarray, distance: float) -> np.ndarray | None:
        """指定された距離に単一のオフセットパスを作成します。"""
        if len(vertices) < 2:
            return None
        
        offset_points = []
        
        for i in range(len(vertices) - 1):
            p1 = vertices[i]
            p2 = vertices[i + 1]
            
            # Calculate perpendicular vector
            direction = p2 - p1
            if np.linalg.norm(direction) == 0:
                continue
            
            # Get perpendicular vector (2D projection for now)
            if len(direction) >= 2:
                perp = np.array([-direction[1], direction[0], 0.0])
                if len(direction) == 3:
                    perp[2] = 0.0
            else:
                continue
            
            # Normalize and scale by distance
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 0:
                perp = perp / perp_norm * distance
                
                # Create offset points
                offset_p1 = p1 + perp
                offset_p2 = p2 + perp
                
                if i == 0:
                    offset_points.append(offset_p1)
                offset_points.append(offset_p2)
        
        return np.array(offset_points) if len(offset_points) >= 2 else None