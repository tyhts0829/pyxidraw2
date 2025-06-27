#!/usr/bin/env python3

import numpy as np
from api.shapes import polygon, sphere
from api.effects import scaling
from engine.core.geometry import Geometry

def test_scaling():
    print("=== Scaling Debug Test ===")
    
    # 1. 基本polygonのテスト
    print("\n1. Basic polygon test:")
    poly = polygon(n_sides=6)
    print(f"Basic polygon coords shape: {poly.coords.shape}")
    print(f"Basic polygon coords min: {poly.coords.min(axis=0)}")
    print(f"Basic polygon coords max: {poly.coords.max(axis=0)}")
    print(f"First 3 coords:\n{poly.coords[:3]}")
    
    # 2. Transform適用後
    print("\n2. After transform in polygon creation:")
    
    # 手動でtransformを適用してテスト
    from api.effects import transform
    poly_manual = transform(poly, center=(150, 50, 0), scale=(100, 100, 100), rotate=(0, 0, 0))
    print(f"Manual transform coords min: {poly_manual.coords.min(axis=0)}")
    print(f"Manual transform coords max: {poly_manual.coords.max(axis=0)}")
    print(f"Manual transform first 3 coords:\n{poly_manual.coords[:3]}")
    
    # polygon()関数で生成されたもの
    poly_transformed = polygon(n_sides=6, center=(150, 50, 0), scale=(100, 100, 100))
    print(f"Polygon function coords min: {poly_transformed.coords.min(axis=0)}")
    print(f"Polygon function coords max: {poly_transformed.coords.max(axis=0)}")
    print(f"Polygon function first 3 coords:\n{poly_transformed.coords[:3]}")
    
    # 3. スケーリングエフェクト適用
    print("\n3. After scaling effect:")
    scaled = scaling(poly_transformed, center=(150, 50, 0), scale=(2.0, 2.0, 2.0))
    print(f"Scaled coords shape: {scaled.coords.shape}")
    print(f"Scaled coords min: {scaled.coords.min(axis=0)}")
    print(f"Scaled coords max: {scaled.coords.max(axis=0)}")
    print(f"First 3 coords:\n{scaled.coords[:3]}")
    
    # 4. 基本sphere
    print("\n4. Basic sphere test:")
    sph = sphere(subdivisions=0.2)  # 小さいsubdivision
    print(f"Basic sphere coords shape: {sph.coords.shape}")
    print(f"Basic sphere coords min: {sph.coords.min(axis=0)}")
    print(f"Basic sphere coords max: {sph.coords.max(axis=0)}")
    print(f"First 3 coords:\n{sph.coords[:3]}")

if __name__ == "__main__":
    test_scaling()