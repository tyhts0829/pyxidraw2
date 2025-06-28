from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry
from shapes import ShapeFactory

# グローバル形状ファクトリインスタンス
_factory = ShapeFactory()


def polygon(
    n_sides: int | float = 3,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """正多角形を生成します。

    Args:
        n_sides: 辺の数。浮動小数点の場合0-100から指数的にマップ。
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        正多角形の頂点を含むGeometry
    """
    shape = _factory.create("polygon")
    return shape(n_sides=n_sides, center=center, scale=scale, rotate=rotate, **params)


def sphere(
    subdivisions: float = 0.5,
    sphere_type: float = 0.5,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """球体を生成します。

    Args:
        subdivisions: 細分化レベル (0.0-1.0、0-5にマップ)
        sphere_type: 描画スタイル (0.0-1.0):
                    0.0-0.2: 経緯線（デフォルト）
                    0.2-0.4: ワイヤーフレーム
                    0.4-0.6: ジグザグ
                    0.6-0.8: 正二十面体
                    0.8-1.0: リング
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        球体の頂点を含むGeometry
    """
    shape = _factory.create("sphere")
    return shape(subdivisions=subdivisions, sphere_type=sphere_type, center=center, scale=scale, rotate=rotate, **params)


def grid(
    n_divisions: tuple[float, float] = (0.1, 0.1),
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """グリッドを生成します。

    Args:
        n_divisions: (x_divisions, y_divisions) 0.0-1.0の浮動小数点
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        グリッド線の頂点を含むGeometry
    """
    shape = _factory.create("grid")
    return shape(n_divisions=n_divisions, center=center, scale=scale, rotate=rotate, **params)


def polyhedron(
    polygon_type: str | int = "tetrahedron",
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """正多面体を生成します。

    Args:
        polygon_type: 多面体のタイプ（名前または面の数）
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        多面体の端の頂点を含むGeometry
    """
    shape = _factory.create("polyhedron")
    return shape(polygon_type=polygon_type, center=center, scale=scale, rotate=rotate, **params)


def lissajous(
    freq_x: float = 3.0,
    freq_y: float = 2.0,
    phase: float = 0.0,
    points: int = 1000,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """リサージュ曲線を生成します。

    Args:
        freq_x: X軸の周波数
        freq_y: Y軸の周波数
        phase: 位相オフセット（ラジアン）
        points: 生成する点の数
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        頂点を含むGeometry
    """
    shape = _factory.create("lissajous")
    return shape(
        freq_x=freq_x, freq_y=freq_y, phase=phase, points=points, center=center, scale=scale, rotate=rotate, **params
    )


def torus(
    major_radius: float = 0.3,
    minor_radius: float = 0.1,
    major_segments: int = 32,
    minor_segments: int = 16,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """トーラスを生成します。

    Args:
        major_radius: 主半径（中心からチューブ中心まで）
        minor_radius: 副半径（チューブ半径）
        major_segments: 主円の周りのセグメント数
        minor_segments: 副円の周りのセグメント数
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        トーラス線の頂点を含むGeometry
    """
    shape = _factory.create("torus")
    return shape(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=major_segments,
        minor_segments=minor_segments,
        center=center,
        scale=scale,
        rotate=rotate,
        **params,
    )


def cylinder(
    radius: float = 0.3,
    height: float = 0.6,
    segments: int = 32,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """円柱を生成します。

    Args:
        radius: 円柱の半径
        height: 円柱の高さ
        segments: 周囲のセグメント数
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        円柱線の頂点を含むGeometry
    """
    shape = _factory.create("cylinder")
    return shape(radius=radius, height=height, segments=segments, center=center, scale=scale, rotate=rotate, **params)


def cone(
    radius: float = 0.3,
    height: float = 0.6,
    segments: int = 32,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """円錐を生成します。

    Args:
        radius: 底面半径
        height: 円錐の高さ
        segments: 周囲のセグメント数
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        円錐線の頂点を含むGeometry
    """
    shape = _factory.create("cone")
    return shape(radius=radius, height=height, segments=segments, center=center, scale=scale, rotate=rotate, **params)


def capsule(
    radius: float = 0.2,
    height: float = 0.4,
    segments: int = 32,
    latitude_segments: int = 16,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """カプセル形状を生成します。

    Args:
        radius: 半球の半径
        height: 円柱部分の高さ
        segments: 曲線のセグメント数
        latitude_segments: 半球の緯度方向のセグメント数
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        カプセル線の頂点を含むGeometry
    """
    shape = _factory.create("capsule")
    return shape(
        radius=radius,
        height=height,
        segments=segments,
        latitude_segments=latitude_segments,
        center=center,
        scale=scale,
        rotate=rotate,
        **params,
    )


def attractor(
    attractor_type: str = "lorenz",
    points: int = 10000,
    dt: float = 0.01,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """ストレンジアトラクターを生成します。

    Args:
        attractor_type: アトラクターのタイプ ("lorenz"、"rossler"、"chua")
        points: 生成する点の数
        dt: 積分の時間ステップ
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        頂点を含むGeometry
    """
    shape = _factory.create("attractor")
    return shape(
        attractor_type=attractor_type, points=points, dt=dt, center=center, scale=scale, rotate=rotate, **params
    )


def text(
    text: str = "HELLO",
    size: float = 0.1,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """テキストを線分として生成します。

    Args:
        text: レンダリングするテキスト文字列
        size: テキストサイズ
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        テキストアウトラインの頂点を含むGeometry
    """
    shape = _factory.create("text")
    return shape(text=text, size=size, center=center, scale=scale, rotate=rotate, **params)


def asemic_glyph(
    complexity: int = 5,
    seed: int | None = None,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> Geometry:
    """抽象的なグリフ状の形状を生成します。

    Args:
        complexity: ストローク数 (1-10)
        seed: 再現性のためのランダムシード
        center: 位置オフセット (x, y, z)
        scale: スケール係数 (x, y, z)
        rotate: 回転角度（ラジアン） (x, y, z)
        **params: 追加パラメータ

    Returns:
        グリフストロークの頂点を含むGeometry
    """
    shape = _factory.create("asemic_glyph")
    return shape(complexity=complexity, seed=seed, center=center, scale=scale, rotate=rotate, **params)
