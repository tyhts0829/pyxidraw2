"""
api.effects モジュールのテスト

各エフェクト関数の基本的な動作を検証する。
効率性と保守性を重視した最小限のテスト実装。
"""
import math

import numpy as np
import pytest

from api import effects
from engine.core.geometry import Geometry


@pytest.fixture
def simple_geometry():
    """テスト用の簡単なGeometryオブジェクト"""
    vertices_list = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 1]], dtype=np.float32),
    ]
    return Geometry.from_lines(vertices_list)


@pytest.fixture
def square_geometry():
    """テスト用の正方形Geometry"""
    vertices_list = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    ]
    return Geometry.from_lines(vertices_list)


class TestRotation:
    def test_rotation_basic(self, simple_geometry):
        """基本的な回転テスト"""
        result = effects.rotation(simple_geometry, rotate=(0.25, 0.0, 0.0))
        assert isinstance(result, Geometry)
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_rotation_no_change(self, simple_geometry):
        """回転なし時のテスト"""
        result = effects.rotation(simple_geometry, rotate=(0.0, 0.0, 0.0))
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-5)

    def test_rotation_with_center(self, simple_geometry):
        """中心点指定回転のテスト"""
        result = effects.rotation(simple_geometry, center=(0.5, 0.5, 0.0), rotate=(0.0, 0.0, 0.25))
        assert isinstance(result, Geometry)


class TestScaling:
    def test_scaling_basic(self, simple_geometry):
        """基本スケーリングテスト"""
        result = effects.scaling(simple_geometry, scale=(2.0, 2.0, 2.0))
        assert isinstance(result, Geometry)
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_scaling_uniform(self, simple_geometry):
        """均一スケーリングテスト"""
        result = effects.scaling(simple_geometry, scale=(1.5, 1.5, 1.5))
        assert isinstance(result, Geometry)

    def test_scaling_no_change(self, simple_geometry):
        """スケーリングなし時のテスト"""
        result = effects.scaling(simple_geometry, scale=(1.0, 1.0, 1.0))
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-5)


class TestTranslation:
    def test_translation_basic(self, simple_geometry):
        """基本移動テスト"""
        result = effects.translation(simple_geometry, offset_x=1.0, offset_y=2.0, offset_z=3.0)
        assert isinstance(result, Geometry)
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_translation_no_change(self, simple_geometry):
        """移動なし時のテスト"""
        result = effects.translation(simple_geometry, offset_x=0.0, offset_y=0.0, offset_z=0.0)
        assert isinstance(result, Geometry)
        np.testing.assert_allclose(result.coords, simple_geometry.coords, rtol=1e-5)

    def test_translation_expected_offset(self, simple_geometry):
        """期待される移動量のテスト"""
        offset = [5.0, -3.0, 2.0]
        result = effects.translation(simple_geometry, offset_x=offset[0], offset_y=offset[1], offset_z=offset[2])
        expected = simple_geometry.coords + np.array(offset)
        np.testing.assert_allclose(result.coords, expected, rtol=1e-5)


class TestSubdivision:
    def test_subdivision_basic(self, simple_geometry):
        """基本細分化テスト"""
        result = effects.subdivision(simple_geometry, n_divisions=0.5)
        assert isinstance(result, Geometry)

    def test_subdivision_no_change(self, simple_geometry):
        """細分化なし時のテスト"""
        result = effects.subdivision(simple_geometry, n_divisions=0.0)
        assert isinstance(result, Geometry)

    def test_subdivision_max(self, simple_geometry):
        """最大細分化テスト"""
        result = effects.subdivision(simple_geometry, n_divisions=1.0)
        assert isinstance(result, Geometry)


class TestExtrude:
    def test_extrude_basic(self, square_geometry):
        """基本押し出しテスト"""
        result = effects.extrude(square_geometry, distance=0.5, direction=(0, 0, 1))
        assert isinstance(result, Geometry)

    def test_extrude_no_distance(self, square_geometry):
        """距離0での押し出しテスト"""
        result = effects.extrude(square_geometry, distance=0.0)
        assert isinstance(result, Geometry)

    def test_extrude_custom_direction(self, square_geometry):
        """カスタム方向での押し出しテスト"""
        result = effects.extrude(square_geometry, direction=(1, 0, 0), distance=0.8)
        assert isinstance(result, Geometry)


class TestFilling:
    def test_filling_lines(self, square_geometry):
        """線パターンでの塗りつぶしテスト"""
        result = effects.filling(square_geometry, pattern="lines", density=0.5)
        assert isinstance(result, Geometry)

    def test_filling_cross(self, square_geometry):
        """クロスパターンでの塗りつぶしテスト"""
        result = effects.filling(square_geometry, pattern="cross", density=0.3)
        assert isinstance(result, Geometry)

    def test_filling_dots(self, square_geometry):
        """ドットパターンでの塗りつぶしテスト"""
        result = effects.filling(square_geometry, pattern="dots", density=0.7)
        assert isinstance(result, Geometry)


class TestTransform:
    def test_transform_combined(self, simple_geometry):
        """複合変換テスト"""
        result = effects.transform(
            simple_geometry,
            center=(0.5, 0.5, 0.0),
            scale=(2.0, 1.5, 1.0),
            rotate=(0.0, 0.0, 0.25)
        )
        assert isinstance(result, Geometry)

    def test_transform_identity(self, simple_geometry):
        """恒等変換テスト"""
        result = effects.transform(
            simple_geometry,
            center=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            rotate=(0.0, 0.0, 0.0)
        )
        assert isinstance(result, Geometry)


class TestNoise:
    def test_noise_basic(self, simple_geometry):
        """基本ノイズテスト"""
        result = effects.noise(simple_geometry, intensity=0.1, frequency=0.5)
        assert isinstance(result, Geometry)
        assert len(result.offsets) == len(simple_geometry.offsets)

    def test_noise_no_intensity(self, simple_geometry):
        """強度0でのノイズテスト"""
        result = effects.noise(simple_geometry, intensity=0.0)
        assert isinstance(result, Geometry)

    def test_noise_with_time(self, simple_geometry):
        """時間パラメータ付きノイズテスト"""
        result = effects.noise(simple_geometry, intensity=0.3, time=1.5)
        assert isinstance(result, Geometry)


class TestBuffer:
    def test_buffer_basic(self, simple_geometry):
        """基本バッファテスト"""
        result = effects.buffer(simple_geometry, distance=0.5)
        assert isinstance(result, Geometry)

    def test_buffer_no_distance(self, simple_geometry):
        """距離0でのバッファテスト"""
        result = effects.buffer(simple_geometry, distance=0.0)
        assert isinstance(result, Geometry)

    def test_buffer_with_styles(self, simple_geometry):
        """スタイル指定バッファテスト"""
        result = effects.buffer(simple_geometry, distance=0.3, join_style=0.8, resolution=0.9)
        assert isinstance(result, Geometry)


class TestArray:
    def test_array_basic(self, simple_geometry):
        """基本配列テスト"""
        result = effects.array(simple_geometry, n_duplicates=0.3, offset=(10, 0, 0))
        assert isinstance(result, Geometry)

    def test_array_no_duplicates(self, simple_geometry):
        """複製なし配列テスト"""
        result = effects.array(simple_geometry, n_duplicates=0.0)
        assert isinstance(result, Geometry)

    def test_array_with_rotation(self, simple_geometry):
        """回転付き配列テスト"""
        result = effects.array(
            simple_geometry,
            n_duplicates=0.5,
            offset=(5, 5, 0),
            rotate=(0.1, 0.0, 0.0)
        )
        assert isinstance(result, Geometry)


class TestPipeline:
    def test_pipeline_single_effect(self, simple_geometry):
        """単一エフェクトパイプラインテスト"""
        from effects import Rotation
        pipeline = effects.pipeline(Rotation())
        assert hasattr(pipeline, 'effects')
        assert len(pipeline.effects) == 1

    def test_pipeline_multiple_effects(self, simple_geometry):
        """複数エフェクトパイプラインテスト"""
        from effects import Rotation, Scaling, Translation
        pipeline = effects.pipeline(Rotation(), Scaling(), Translation())
        assert hasattr(pipeline, 'effects')
        assert len(pipeline.effects) == 3

    def test_pipeline_empty(self):
        """空パイプラインテスト"""
        pipeline = effects.pipeline()
        assert hasattr(pipeline, 'effects')
        assert len(pipeline.effects) == 0