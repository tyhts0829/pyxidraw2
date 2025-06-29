#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シリアライズ可能なベンチマークターゲット

pickle問題を解決するため、クロージャを使わないベンチマークターゲット実装。
"""

from typing import Any, Dict, Optional

# グローバル変数として各モジュールをキャッシュ
_cached_modules: Dict[str, Any] = {}


def _get_cached_module(module_name: str, function_name: Optional[str] = None):
    """モジュールまたは関数を遅延インポートしてキャッシュ"""
    cache_key = f"{module_name}.{function_name}" if function_name else module_name
    
    if cache_key not in _cached_modules:
        if module_name == "api.effects":
            import api.effects
            if function_name:
                _cached_modules[cache_key] = getattr(api.effects, function_name)
            else:
                _cached_modules[cache_key] = api.effects
        elif module_name == "api.shapes":
            import api.shapes
            if function_name:
                _cached_modules[cache_key] = getattr(api.shapes, function_name)
            else:
                _cached_modules[cache_key] = api.shapes
        else:
            raise ValueError(f"Unknown module: {module_name}")
    
    return _cached_modules[cache_key]


def init_worker():
    """ProcessPoolExecutorのワーカー初期化時に呼ばれる関数"""
    # よく使われるモジュールを事前にインポート
    try:
        import api.effects
        import api.shapes
        _cached_modules['api.effects'] = api.effects
        _cached_modules['api.shapes'] = api.shapes
        
        # 個別の関数もキャッシュ
        for func_name in ['noise', 'subdivision', 'extrude', 'filling', 'buffer', 'array']:
            _cached_modules[f'api.effects.{func_name}'] = getattr(api.effects, func_name)
        
        for func_name in ['polygon', 'grid', 'sphere', 'cylinder', 'cone', 'torus']:
            _cached_modules[f'api.shapes.{func_name}'] = getattr(api.shapes, func_name)
            
    except ImportError:
        pass  # インポートエラーは後で処理


class SerializableEffectTarget:
    """シリアライズ可能なエフェクトターゲット"""
    
    def __init__(self, effect_type: str, params: Dict[str, Any]):
        self.effect_type = effect_type
        self.params = params
    
    def __call__(self, geom):
        """エフェクトを適用"""
        if self.effect_type == "transform":
            return geom.transform(**self.params)
        elif self.effect_type == "scale":
            scale = self.params.get("scale", (1, 1, 1))
            center = self.params.get("center", (0, 0, 0))
            return geom.scale(scale[0], scale[1], scale[2], center=center)
        elif self.effect_type == "translate":
            translate = self.params.get("translate", (0, 0, 0))
            return geom.translate(translate[0], translate[1], translate[2])
        elif self.effect_type == "rotate":
            rotate = self.params.get("rotate", (0, 0, 0))
            return geom.rotate(rotate[0], rotate[1], rotate[2])
        elif self.effect_type == "noise":
            noise = _get_cached_module("api.effects", "noise")
            intensity = self.params.get("intensity", 0.5)
            frequency = self.params.get("frequency", 1.0)
            return noise(geom, intensity=intensity, frequency=frequency)
        elif self.effect_type == "subdivision":
            subdivision = _get_cached_module("api.effects", "subdivision")
            level = self.params.get("level", 1)
            return subdivision(geom, level=level)
        elif self.effect_type == "extrude":
            extrude = _get_cached_module("api.effects", "extrude")
            depth = self.params.get("depth", 10.0)
            return extrude(geom, depth=depth)
        elif self.effect_type == "filling":
            filling = _get_cached_module("api.effects", "filling")
            spacing = self.params.get("spacing", 10.0)
            angle = self.params.get("angle", 0.0)
            return filling(geom, spacing=spacing, angle=angle)
        elif self.effect_type == "buffer":
            buffer = _get_cached_module("api.effects", "buffer")
            distance = self.params.get("distance", 5.0)
            return buffer(geom, distance=distance)
        elif self.effect_type == "array":
            array = _get_cached_module("api.effects", "array")
            count_x = self.params.get("count_x", 2)
            count_y = self.params.get("count_y", 2)
            spacing_x = self.params.get("spacing_x", 10.0)
            spacing_y = self.params.get("spacing_y", 10.0)
            return array(geom, count_x=count_x, count_y=count_y, spacing_x=spacing_x, spacing_y=spacing_y)
        else:
            raise ValueError(f"Unknown effect type: {self.effect_type}")


class SerializableShapeTarget:
    """シリアライズ可能な形状ターゲット"""
    
    def __init__(self, shape_type: str, params: Dict[str, Any]):
        self.shape_type = shape_type
        self.params = params
    
    def __call__(self):
        """形状を生成"""
        if self.shape_type == "polygon":
            polygon = _get_cached_module("api.shapes", "polygon")
            return polygon(**self.params)
        elif self.shape_type == "grid":
            grid = _get_cached_module("api.shapes", "grid")
            return grid(**self.params)
        elif self.shape_type == "sphere":
            sphere = _get_cached_module("api.shapes", "sphere")
            return sphere(**self.params)
        elif self.shape_type == "cylinder":
            cylinder = _get_cached_module("api.shapes", "cylinder")
            return cylinder(**self.params)
        elif self.shape_type == "cone":
            cone = _get_cached_module("api.shapes", "cone")
            return cone(**self.params)
        elif self.shape_type == "torus":
            torus = _get_cached_module("api.shapes", "torus")
            return torus(**self.params)
        elif self.shape_type == "capsule":
            capsule = _get_cached_module("api.shapes", "capsule")
            return capsule(**self.params)
        elif self.shape_type == "polyhedron":
            polyhedron = _get_cached_module("api.shapes", "polyhedron")
            return polyhedron(**self.params)
        elif self.shape_type == "lissajous":
            lissajous_curve = _get_cached_module("api.shapes", "lissajous_curve")
            return lissajous_curve(**self.params)
        elif self.shape_type == "attractor":
            if self.params.get("attractor_type") == "lorenz":
                lorenz_attractor = _get_cached_module("api.shapes", "lorenz_attractor")
                return lorenz_attractor(**{k: v for k, v in self.params.items() if k != "attractor_type"})
            elif self.params.get("attractor_type") == "rossler":
                rossler_attractor = _get_cached_module("api.shapes", "rossler_attractor")
                return rossler_attractor(**{k: v for k, v in self.params.items() if k != "attractor_type"})
            else:
                raise ValueError(f"Unknown attractor type: {self.params.get('attractor_type')}")
        elif self.shape_type == "text":
            text = _get_cached_module("api.shapes", "text")
            return text(**self.params)
        elif self.shape_type == "asemic_glyph":
            asemic_glyph = _get_cached_module("api.shapes", "asemic_glyph")
            return asemic_glyph(**self.params)
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")