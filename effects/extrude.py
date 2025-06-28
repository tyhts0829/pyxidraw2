from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect
from engine.core.geometry import Geometry


class Extrude(BaseEffect):
    """2D形状を3Dに押し出します。"""
    
    # クラス定数（0.0-1.0のレンジから実際の値へのスケーリング）
    MAX_DISTANCE = 200.0  # 最大押し出し距離
    MAX_SCALE = 3.0      # 最大スケール率
    MAX_SUBDIVISIONS = 5  # 最大細分化ステップ数
    
    def apply(
        self, 
        geometry: Geometry, 
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        distance: float = 0.5,
        scale: float = 0.5,
        subdivisions: float = 0.5,
        **params: Any
    ) -> Geometry:
        """押し出しエフェクトを適用します。
        
        2D形状を指定された方向に押し出して3D構造を作成します。
        
        Args:
            geometry: 入力Geometryオブジェクト
            direction: 押し出し方向ベクトル (x, y, z) - デフォルト (0, 0, 1)
            distance: 押し出し距離 (0.0-1.0) - デフォルト 0.5
            scale: 押し出したジオメトリのスケール率 (0.0-1.0) - デフォルト 0.5
            subdivisions: 細分化ステップ数 (0.0-1.0) - デフォルト 0.5
            **params: 追加パラメータ
            
        Returns:
            元の形状、押し出し形状、接続エッジを含む押し出しGeometry
        """
        # パラメータをスケーリング
        distance_scaled = distance * self.MAX_DISTANCE
        scale_scaled = scale * self.MAX_SCALE
        subdivisions_int = int(subdivisions * self.MAX_SUBDIVISIONS)
        
        # Apply subdivisions if requested
        if subdivisions_int > 0:
            geometry = self._subdivide_geometry(geometry, subdivisions_int)
        
        # Normalize direction vector
        direction_array = np.array(direction, dtype=np.float32)
        direction_norm = np.linalg.norm(direction_array)
        if direction_norm == 0:
            return geometry  # Can't extrude with zero direction
        
        direction_normalized = direction_array / direction_norm
        extrude_vector = direction_normalized * distance_scaled
        
        # 元のジオメトリの線を取得
        lines = []
        for i in range(len(geometry.offsets) - 1):
            start_idx = geometry.offsets[i]
            end_idx = geometry.offsets[i + 1]
            line = geometry.coords[start_idx:end_idx]
            lines.append(line)
        
        extruded_lines = []
        
        # 元の形状を追加
        extruded_lines.extend(lines)
        
        # 押し出した形状を作成
        for line in lines:
            # Create extruded version
            extruded_line = (line + extrude_vector) * scale_scaled
            extruded_lines.append(extruded_line)
            
            # Create connecting edges between original and extruded vertices
            for i in range(len(line)):
                segment = np.array([line[i], extruded_line[i]], dtype=np.float32)
                extruded_lines.append(segment)
        
        return Geometry.from_lines(extruded_lines)
    
    def _subdivide_geometry(self, geometry: Geometry, subdivisions: int) -> Geometry:
        """頂点密度を増やすための簡単な細分化を適用します。
        
        Args:
            geometry: 入力Geometryオブジェクト
            subdivisions: 細分化反復回数
            
        Returns:
            細分化されたGeometryオブジェクト
        """
        # 元のジオメトリの線を取得
        lines = []
        for i in range(len(geometry.offsets) - 1):
            start_idx = geometry.offsets[i]
            end_idx = geometry.offsets[i + 1]
            line = geometry.coords[start_idx:end_idx]
            lines.append(line)
        
        # 各線を細分化
        result_lines = []
        for line in lines:
            current = line.copy()
            
            for _ in range(subdivisions):
                if len(current) < 2:
                    break
                
                # Linear subdivision - insert midpoints
                new_vertices = [current[0]]
                for i in range(len(current) - 1):
                    midpoint = (current[i] + current[i + 1]) / 2
                    new_vertices.append(midpoint)
                    new_vertices.append(current[i + 1])
                
                current = np.array(new_vertices, dtype=np.float32)
            
            result_lines.append(current)
        
        return Geometry.from_lines(result_lines)