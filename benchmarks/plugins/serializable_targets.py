#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
シリアライズ可能なベンチマークターゲット

pickle問題を解決するため、クロージャを使わないベンチマークターゲット実装。
"""

from typing import Any, Dict


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
            from api.effects import noise
            intensity = self.params.get("intensity", 0.5)
            frequency = self.params.get("frequency", 1.0)
            return noise(geom, intensity=intensity, frequency=frequency)
        elif self.effect_type == "subdivision":
            from api.effects import subdivision
            level = self.params.get("level", 1)
            return subdivision(geom, level=level)
        elif self.effect_type == "extrude":
            from api.effects import extrude
            depth = self.params.get("depth", 10.0)
            return extrude(geom, depth=depth)
        elif self.effect_type == "filling":
            from api.effects import filling
            spacing = self.params.get("spacing", 10.0)
            angle = self.params.get("angle", 0.0)
            return filling(geom, spacing=spacing, angle=angle)
        elif self.effect_type == "buffer":
            from api.effects import buffer
            distance = self.params.get("distance", 5.0)
            return buffer(geom, distance=distance)
        elif self.effect_type == "array":
            from api.effects import array
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
            from api.shapes import polygon
            return polygon(**self.params)
        elif self.shape_type == "grid":
            from api.shapes import grid
            return grid(**self.params)
        elif self.shape_type == "sphere":
            from api.shapes import sphere
            return sphere(**self.params)
        elif self.shape_type == "cylinder":
            from api.shapes import cylinder
            return cylinder(**self.params)
        elif self.shape_type == "cone":
            from api.shapes import cone
            return cone(**self.params)
        elif self.shape_type == "torus":
            from api.shapes import torus
            return torus(**self.params)
        elif self.shape_type == "capsule":
            from api.shapes import capsule
            return capsule(**self.params)
        elif self.shape_type == "polyhedron":
            from api.shapes import polyhedron
            return polyhedron(**self.params)
        elif self.shape_type == "lissajous":
            from api.shapes import lissajous_curve
            return lissajous_curve(**self.params)
        elif self.shape_type == "attractor":
            if self.params.get("attractor_type") == "lorenz":
                from api.shapes import lorenz_attractor
                return lorenz_attractor(**{k: v for k, v in self.params.items() if k != "attractor_type"})
            elif self.params.get("attractor_type") == "rossler":
                from api.shapes import rossler_attractor
                return rossler_attractor(**{k: v for k, v in self.params.items() if k != "attractor_type"})
            else:
                raise ValueError(f"Unknown attractor type: {self.params.get('attractor_type')}")
        elif self.shape_type == "text":
            from api.shapes import text
            return text(**self.params)
        elif self.shape_type == "asemic_glyph":
            from api.shapes import asemic_glyph
            return asemic_glyph(**self.params)
        else:
            raise ValueError(f"Unknown shape type: {self.shape_type}")