#!/usr/bin/env python3
"""Geometry対応関数の動作確認テスト"""

import numpy as np
from api.shapes import polygon, sphere
from util.geometry import geometry_transform_to_xy_plane, geometry_transform_back


def test_basic_transformation():
    """基本的な変換テスト"""
    print("基本的な変換テストを開始...")
    
    # テスト用のシンプルなポリゴンを作成
    poly = polygon(n_sides=6).size(50).at(100, 100, 0)
    print(f"元のポリゴン頂点数: {len(poly.coords)}")
    print(f"元の座標範囲 - X: [{poly.coords[:, 0].min():.2f}, {poly.coords[:, 0].max():.2f}]")
    print(f"元の座標範囲 - Y: [{poly.coords[:, 1].min():.2f}, {poly.coords[:, 1].max():.2f}]")
    print(f"元の座標範囲 - Z: [{poly.coords[:, 2].min():.2f}, {poly.coords[:, 2].max():.2f}]")
    
    # XY平面に変換
    transformed_poly, rotation_matrix, z_offset = geometry_transform_to_xy_plane(poly)
    print(f"\n変換後の座標範囲 - X: [{transformed_poly.coords[:, 0].min():.2f}, {transformed_poly.coords[:, 0].max():.2f}]")
    print(f"変換後の座標範囲 - Y: [{transformed_poly.coords[:, 1].min():.2f}, {transformed_poly.coords[:, 1].max():.2f}]")
    print(f"変換後の座標範囲 - Z: [{transformed_poly.coords[:, 2].min():.2f}, {transformed_poly.coords[:, 2].max():.2f}]")
    print(f"Z軸オフセット: {z_offset:.2f}")
    
    # 元に戻す
    restored_poly = geometry_transform_back(transformed_poly, rotation_matrix, z_offset)
    print(f"\n復元後の座標範囲 - X: [{restored_poly.coords[:, 0].min():.2f}, {restored_poly.coords[:, 0].max():.2f}]")
    print(f"復元後の座標範囲 - Y: [{restored_poly.coords[:, 1].min():.2f}, {restored_poly.coords[:, 1].max():.2f}]")
    print(f"復元後の座標範囲 - Z: [{restored_poly.coords[:, 2].min():.2f}, {restored_poly.coords[:, 2].max():.2f}]")
    
    # 精度チェック
    diff = np.abs(poly.coords - restored_poly.coords).max()
    print(f"元の座標との最大差分: {diff:.6f}")
    
    if diff < 1e-5:
        print("✅ 基本変換テスト成功")
    else:
        print("❌ 基本変換テスト失敗")
    
    return diff < 1e-5


def test_3d_transformation():
    """3D形状での変換テスト"""
    print("\n3D形状での変換テストを開始...")
    
    # 球体を作成（Z軸方向にも広がりがある）
    sph = sphere(subdivisions=0.3).scale(30, 30, 30).translate(50, 50, 50)
    print(f"元の球体頂点数: {len(sph.coords)}")
    print(f"元の座標範囲 - X: [{sph.coords[:, 0].min():.2f}, {sph.coords[:, 0].max():.2f}]")
    print(f"元の座標範囲 - Y: [{sph.coords[:, 1].min():.2f}, {sph.coords[:, 1].max():.2f}]")
    print(f"元の座標範囲 - Z: [{sph.coords[:, 2].min():.2f}, {sph.coords[:, 2].max():.2f}]")
    
    # XY平面に変換
    transformed_sph, rotation_matrix, z_offset = geometry_transform_to_xy_plane(sph)
    print(f"\n変換後の座標範囲 - Z: [{transformed_sph.coords[:, 2].min():.2f}, {transformed_sph.coords[:, 2].max():.2f}]")
    print(f"Z軸オフセット: {z_offset:.2f}")
    
    # Z座標がほぼ0に平坦化されているかチェック
    z_range = transformed_sph.coords[:, 2].max() - transformed_sph.coords[:, 2].min()
    print(f"変換後のZ座標の範囲: {z_range:.6f}")
    
    # 元に戻す
    restored_sph = geometry_transform_back(transformed_sph, rotation_matrix, z_offset)
    
    # Z座標の復元をチェック（元のZ値の重心が正しく復元されているか）
    original_z_center = np.mean(sph.coords[:, 2])
    restored_z_center = np.mean(restored_sph.coords[:, 2])
    z_center_diff = abs(original_z_center - restored_z_center)
    print(f"Z重心の差分: {z_center_diff:.6f}")
    
    # XY座標は変更されていないはず
    xy_diff = np.abs(sph.coords[:, :2] - restored_sph.coords[:, :2]).max()
    print(f"XY座標の最大差分: {xy_diff:.6f}")
    
    if z_center_diff < 1e-4 and xy_diff < 1e-4 and z_range < 1e-4:
        print("✅ 3D変換テスト成功")
    else:
        print("❌ 3D変換テスト失敗")
    
    return z_center_diff < 1e-4 and xy_diff < 1e-4 and z_range < 1e-4


def test_empty_geometry():
    """空のGeometryでのテスト"""
    print("\n空のGeometryでのテストを開始...")
    
    from engine.core.geometry import Geometry
    
    # 空のGeometry
    empty_geom = Geometry(coords=np.array([], dtype=np.float32).reshape(0, 3), 
                         offsets=np.array([0], dtype=np.int32))
    
    # 変換を試行
    transformed, rotation_matrix, z_offset = geometry_transform_to_xy_plane(empty_geom)
    restored = geometry_transform_back(transformed, rotation_matrix, z_offset)
    
    print(f"空のGeometry変換: 頂点数 {len(transformed.coords)}")
    print(f"回転行列は単位行列: {np.allclose(rotation_matrix, np.eye(3))}")
    print(f"Zオフセット: {z_offset}")
    
    if len(transformed.coords) == 0 and np.allclose(rotation_matrix, np.eye(3)) and z_offset == 0.0:
        print("✅ 空のGeometryテスト成功")
        return True
    else:
        print("❌ 空のGeometryテスト失敗")
        return False


def main():
    """メインテスト実行"""
    print("=== Geometry対応関数の動作確認テスト ===\n")
    
    test_results = []
    
    try:
        test_results.append(test_basic_transformation())
        test_results.append(test_3d_transformation())
        test_results.append(test_empty_geometry())
        
        print(f"\n=== テスト結果 ===")
        print(f"成功: {sum(test_results)}/{len(test_results)}")
        
        if all(test_results):
            print("✅ 全てのテストが成功しました！")
        else:
            print("❌ 一部のテストが失敗しました。")
            
    except Exception as e:
        print(f"❌ テスト実行中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()