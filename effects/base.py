from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

from engine.core.geometry import Geometry


class BaseEffect(ABC):
    """キャッシュ機能を内蔵した、すべてのエフェクトのベースクラスです。"""
    
    def __init__(self):
        self._cache_enabled = True
    
    @abstractmethod
    def apply(self, geometry: Geometry, **params: Any) -> Geometry:
        """Geometryにエフェクトを適用します。
        
        Args:
            geometry: 変換するGeometryオブジェクト
            **params: エフェクト固有のパラメータ
            
        Returns:
            変換されたGeometryオブジェクト
        """
        pass
    
    def __call__(self, geometry: Geometry, **params: Any) -> Geometry:
        """自動キャッシュ機能でエフェクトを適用します。"""
        if self._cache_enabled:
            # Convert to hashable format
            hashable_geometry = self._geometry_to_hashable(geometry)
            hashable_params = self._params_to_hashable(params)
            return self._cached_apply(hashable_geometry, hashable_params)
        return self.apply(geometry, **params)
    
    @lru_cache(maxsize=128)
    def _cached_apply(self, hashable_geometry: tuple, hashable_params: tuple) -> Geometry:
        """applyメソッドのキャッシュバージョンです。"""
        geometry = self._hashable_to_geometry(hashable_geometry)
        params = self._hashable_to_params(hashable_params)
        return self.apply(geometry, **params)
    
    def _geometry_to_hashable(self, geometry: Geometry) -> tuple:
        """Geometryをハッシュ化可能な形式に変換します。"""
        coords_tuple = tuple(map(tuple, geometry.coords.tolist()))
        offsets_tuple = tuple(geometry.offsets.tolist())
        return (coords_tuple, offsets_tuple)
    
    def _hashable_to_geometry(self, hashable: tuple) -> Geometry:
        """ハッシュ化可能な形式をGeometryに戻します。"""
        coords_tuple, offsets_tuple = hashable
        coords = np.array(coords_tuple, dtype=np.float32)
        offsets = np.array(offsets_tuple, dtype=np.int32)
        return Geometry(coords, offsets)
    
    def _params_to_hashable(self, params: dict[str, Any]) -> tuple:
        """パラメータをハッシュ化可能な形式に変換します。"""
        items = []
        for key, value in sorted(params.items()):
            if isinstance(value, (list, tuple)):
                items.append((key, ('list_tuple', tuple(value))))
            elif isinstance(value, np.ndarray):
                items.append((key, ('numpy_array', tuple(value.flatten().tolist()), value.shape)))
            elif callable(value):
                # Skip callables
                continue
            else:
                items.append((key, ('primitive', value)))
        return tuple(items)
    
    def _hashable_to_params(self, hashable: tuple) -> dict[str, Any]:
        """ハッシュ化可能なパラメータを辞書に戻します。"""
        params = {}
        for key, value_info in hashable:
            if isinstance(value_info, tuple) and len(value_info) >= 2:
                value_type = value_info[0]
                if value_type == 'numpy_array' and len(value_info) == 3:
                    flat_data, shape = value_info[1], value_info[2]
                    params[key] = np.array(flat_data).reshape(shape)
                elif value_type == 'list_tuple':
                    params[key] = value_info[1]
                elif value_type == 'primitive':
                    params[key] = value_info[1]
                else:
                    params[key] = value_info
            else:
                params[key] = value_info
        return params
    
    def clear_cache(self):
        """LRUキャッシュをクリアします。"""
        if hasattr(self._cached_apply, 'cache_clear'):
            self._cached_apply.cache_clear()
    
    def disable_cache(self):
        """このエフェクトのキャッシュ機能を無効化します。"""
        self._cache_enabled = False
    
    def enable_cache(self):
        """このエフェクトのキャッシュ機能を有効化します。"""
        self._cache_enabled = True