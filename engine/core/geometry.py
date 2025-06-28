"""
このライブラリの中核となるデータ構造
coords   : float32 ndarray  (N, 3)   # 全頂点を 1 本の連続メモリで保持
offsets  : int32   ndarray  (M+1,)   # 各線の開始 index（最後に N を追加）
lines[i] = coords[offsets[i] : offsets[i+1]]
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class Geometry:
    coords: np.ndarray  # shape (N, 3)  float32
    offsets: np.ndarray  # shape (M+1,)  int32
    _cache_key: Optional[str] = None  # キャッシュキー

    # ── ファクトリ ───────────────────
    @classmethod
    def from_lines(cls, lines: list[np.ndarray]) -> "Geometry":
        """任意の線群を list[array] から構築。"""
        offsets = np.empty(len(lines) + 1, np.int32)
        offsets[0] = 0
        coords = []
        for i, ln in enumerate(lines, 1):
            offsets[i] = offsets[i - 1] + len(ln)
            coords.append(ln.astype(np.float32, copy=False))
        return cls(np.concatenate(coords, axis=0), offsets)

    def map(self, fn) -> "Geometry":
        """座標配列に任意の関数 fn: (N,3)->(N,3) を適用"""
        return Geometry(fn(self.coords), self.offsets)

    def as_arrays(self, *, copy: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """coords, offsets を返す。copy=True ならディープコピー。
        
        Args:
            copy: True なら独立したコピーを返す。False なら zero-copy view を返す。
            
        Returns:
            tuple[np.ndarray, np.ndarray]: (coords, offsets) のタプル
            
        Note:
            デフォルトの copy=False では、元の Geometry と同じメモリを共有する
            ビューを返すため、O(1) で取得でき、巨大データでも高速です。
            取得したビューを変更すると元の Geometry も変更されます。
        """
        if copy:
            return self.coords.copy(), self.offsets.copy()
        # zero-copy view を返す
        return self.coords, self.offsets

    def __add__(self, other: "Geometry"):
        off_other = other.offsets + len(self.coords)
        return Geometry(np.concatenate([self.coords, other.coords]), np.concatenate([self.offsets[:-1], off_other]))

    # ── キャッシュ関連メソッド ───────────────────
    def _get_hash(self) -> str:
        """Geometryデータのハッシュを計算"""
        if self._cache_key is None:
            # 座標とオフセットデータからハッシュを生成
            coords_bytes = self.coords.tobytes()
            offsets_bytes = self.offsets.tobytes()
            combined = coords_bytes + offsets_bytes
            self._cache_key = hashlib.md5(combined).hexdigest()
        return self._cache_key

    def _with_cache_key(self, operation: str, params: tuple) -> str:
        """操作履歴を含むキャッシュキーを生成"""
        base_hash = self._get_hash()
        operation_str = f"{operation}:{params}"
        combined = f"{base_hash}:{operation_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    @lru_cache(maxsize=256)
    def _cached_effect(cache_key: str, operation: str, geometry_hash: str, params: tuple) -> "Geometry":
        """キャッシュされたエフェクト適用（実際の処理は別途呼び出し）"""
        # このメソッドは直接使用せず、_apply_cached_effectから呼び出される
        return None  # プレースホルダー # type: ignore

    def _apply_cached_effect(self, operation: str, effect_func, *args, **kwargs) -> "Geometry":
        """キャッシュ付きエフェクト適用"""
        # パラメータをハッシュ可能な形に変換
        hashable_args = self._make_params_hashable(args, kwargs)
        cache_key = self._with_cache_key(operation, hashable_args)

        # キャッシュを確認（メモリ上に直接保持）
        if not hasattr(Geometry, "_effect_cache"):
            Geometry._effect_cache = {}

        if cache_key in Geometry._effect_cache:
            return Geometry._effect_cache[cache_key]

        # キャッシュミス：実際に計算
        result = effect_func(self, *args, **kwargs)

        # キャッシュサイズ制限（LRU風）
        if len(Geometry._effect_cache) >= 256:
            # 古いエントリを削除（簡易LRU）
            oldest_key = next(iter(Geometry._effect_cache))
            del Geometry._effect_cache[oldest_key]

        Geometry._effect_cache[cache_key] = result
        return result

    def _make_params_hashable(self, args: tuple, kwargs: dict) -> tuple:
        """パラメータをハッシュ可能な形に変換"""
        hashable_items = []

        # argsを処理
        for arg in args:
            if isinstance(arg, (tuple, list)):
                hashable_items.append(tuple(arg))
            elif isinstance(arg, np.ndarray):
                hashable_items.append(tuple(arg.flatten().tolist()))
            else:
                hashable_items.append(arg)

        # kwargsを処理（ソート済み）
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (tuple, list)):
                hashable_items.append((key, tuple(value)))
            elif isinstance(value, np.ndarray):
                hashable_items.append((key, tuple(value.flatten().tolist())))
            else:
                hashable_items.append((key, value))

        return tuple(hashable_items)

    # ── エフェクトメソッド（キャッシュ付きメソッドチェーン用） ───────────────────
    def scale(
        self, x: float = 1.0, y: float = 1.0, z: float = 1.0, center: tuple[float, float, float] = (0, 0, 0)
    ) -> "Geometry":
        """スケーリングを適用（キャッシュ付きメソッドチェーン対応）"""

        def _scale_effect(geom, *args, **kwargs):
            from api.effects import scaling

            return scaling(geom, center=kwargs["center"], scale=kwargs["scale"])

        return self._apply_cached_effect("scale", _scale_effect, center=center, scale=(x, y, z))

    def translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> "Geometry":
        """移動を適用（キャッシュ付きメソッドチェーン対応）"""

        def _translate_effect(geom, *args, **kwargs):
            from api.effects import translation

            return translation(
                geom, offset_x=kwargs["offset_x"], offset_y=kwargs["offset_y"], offset_z=kwargs["offset_z"]
            )

        return self._apply_cached_effect("translate", _translate_effect, offset_x=x, offset_y=y, offset_z=z)

    def rotate(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0, center: tuple[float, float, float] = (0, 0, 0)
    ) -> "Geometry":
        """回転を適用（キャッシュ付きメソッドチェーン対応）"""

        def _rotate_effect(geom, *args, **kwargs):
            from api.effects import rotation

            return rotation(geom, center=kwargs["center"], rotate=kwargs["rotate"])

        return self._apply_cached_effect("rotate", _rotate_effect, center=center, rotate=(x, y, z))

    def transform(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
    ) -> "Geometry":
        """複合変換を適用（キャッシュ付きメソッドチェーン対応）"""

        def _transform_effect(geom, *args, **kwargs):
            from api.effects import transform

            return transform(geom, center=kwargs["center"], scale=kwargs["scale"], rotate=kwargs["rotate"])

        return self._apply_cached_effect("transform", _transform_effect, center=center, scale=scale, rotate=rotate)

    # ── 便利なショートカットメソッド（キャッシュ対応） ───────────────────
    def scale_uniform(self, factor: float, center: tuple[float, float, float] = (0, 0, 0)) -> "Geometry":
        """一様スケーリング（ショートカット）"""
        return self.scale(factor, factor, factor, center)

    def rotate_z(self, angle: float, center: tuple[float, float, float] = (0, 0, 0)) -> "Geometry":
        """Z軸周りの回転（ショートカット）"""
        return self.rotate(0, 0, angle, center)

    def move_to(self, x: float, y: float, z: float = 0) -> "Geometry":
        """指定位置に移動（translate のエイリアス）"""
        return self.translate(x, y, z)

    def center_at(self, x: float, y: float, z: float = 0) -> "Geometry":
        """指定位置を中心にする（現在の重心から移動）"""

        def _center_at_effect(geom, *args, **kwargs):
            # 現在の重心を計算
            current_center = geom.coords.mean(axis=0)
            target = np.array([kwargs["x"], kwargs["y"], kwargs["z"]])
            offset = target - current_center
            from api.effects import translation

            return translation(geom, offset_x=offset[0], offset_y=offset[1], offset_z=offset[2])

        return self._apply_cached_effect("center_at", _center_at_effect, x=x, y=y, z=z)

    # ── 更に簡潔な記法のためのエイリアス ───────────────────
    def at(self, x: float, y: float, z: float = 0) -> "Geometry":
        """位置指定（center_at のエイリアス）"""
        return self.center_at(x, y, z)

    def size(self, factor: float) -> "Geometry":
        """サイズ指定（scale_uniform のエイリアス）"""
        return self.scale_uniform(factor)

    def spin(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> "Geometry":
        """動的に計算された中心点で3軸回転（自分の重心を中心に回転）"""
        def _spin_effect(geom, **kwargs):
            # 現在の重心を計算
            current_center = geom.coords.mean(axis=0)
            from api.effects import rotation
            return rotation(geom, center=tuple(current_center), rotate=(kwargs["x"], kwargs["y"], kwargs["z"]))
        
        return self._apply_cached_effect("spin", _spin_effect, x=x, y=y, z=z)

    # ── エフェクトメソッド（api/effectsの有効エフェクト） ───────────────────
    def subdivision(self, n_divisions: float = 0.5) -> "Geometry":
        """線を細分化（キャッシュ付きメソッドチェーン対応）"""
        def _subdivision_effect(geom, **kwargs):
            from api.effects import subdivision
            return subdivision(geom, n_divisions=kwargs["n_divisions"])
        
        return self._apply_cached_effect("subdivision", _subdivision_effect, n_divisions=n_divisions)

    def extrude(
        self,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        distance: float = 0.5,
        scale: float = 0.5,
        subdivisions: float = 0.5,
    ) -> "Geometry":
        """2D形状を3Dに押し出し（キャッシュ付きメソッドチェーン対応）"""
        def _extrude_effect(geom, **kwargs):
            from api.effects import extrude
            return extrude(
                geom,
                direction=kwargs["direction"],
                distance=kwargs["distance"],
                scale=kwargs["scale"],
                subdivisions=kwargs["subdivisions"]
            )
        
        return self._apply_cached_effect(
            "extrude", _extrude_effect,
            direction=direction, distance=distance, scale=scale, subdivisions=subdivisions
        )

    def filling(
        self,
        pattern: str = "lines",
        density: float = 0.5,
        angle: float = 0.0,
    ) -> "Geometry":
        """ハッチングパターンで塗りつぶし（キャッシュ付きメソッドチェーン対応）"""
        def _filling_effect(geom, **kwargs):
            from api.effects import filling
            return filling(
                geom,
                pattern=kwargs["pattern"],
                density=kwargs["density"],
                angle=kwargs["angle"]
            )
        
        return self._apply_cached_effect(
            "filling", _filling_effect,
            pattern=pattern, density=density, angle=angle
        )

    def noise(
        self,
        intensity: float = 0.5,
        frequency: tuple[float, float, float] | float = (0.5, 0.5, 0.5),
        time: float = 0.0,
    ) -> "Geometry":
        """Perlinノイズを適用（キャッシュ付きメソッドチェーン対応）"""
        def _noise_effect(geom, **kwargs):
            from api.effects import noise
            return noise(
                geom,
                intensity=kwargs["intensity"],
                frequency=kwargs["frequency"],
                time=kwargs["time"]
            )
        
        return self._apply_cached_effect(
            "noise", _noise_effect,
            intensity=intensity, frequency=frequency, time=time
        )

    def buffer(
        self,
        distance: float = 0.5,
        join_style: float = 0.5,
        resolution: float = 0.5,
    ) -> "Geometry":
        """パス周りにバッファ/オフセットを作成（キャッシュ付きメソッドチェーン対応）"""
        def _buffer_effect(geom, **kwargs):
            from api.effects import buffer
            return buffer(
                geom,
                distance=kwargs["distance"],
                join_style=kwargs["join_style"],
                resolution=kwargs["resolution"]
            )
        
        return self._apply_cached_effect(
            "buffer", _buffer_effect,
            distance=distance, join_style=join_style, resolution=resolution
        )

    def array(
        self,
        n_duplicates: float = 0.5,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float] = (0.5, 0.5, 0.5),
        scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> "Geometry":
        """入力のコピーを配列状に生成（キャッシュ付きメソッドチェーン対応）"""
        def _array_effect(geom, **kwargs):
            from api.effects import array
            return array(
                geom,
                n_duplicates=kwargs["n_duplicates"],
                offset=kwargs["offset"],
                rotate=kwargs["rotate"],
                scale=kwargs["scale"],
                center=kwargs["center"]
            )
        
        return self._apply_cached_effect(
            "array", _array_effect,
            n_duplicates=n_duplicates, offset=offset, rotate=rotate, scale=scale, center=center
        )

    # ── クラスメソッド（形状生成） ───────────────────
    @classmethod
    def polygon(
        cls,
        n_sides: int | float = 3,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """正多角形を生成"""
        from api.shapes import polygon
        return polygon(n_sides, center, scale, rotate, **params)

    @classmethod
    def sphere(
        cls,
        subdivisions: float = 0.5,
        sphere_type: float = 0.5,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """球体を生成"""
        from api.shapes import sphere
        return sphere(subdivisions, sphere_type, center, scale, rotate, **params)

    @classmethod
    def grid(
        cls,
        n_divisions: tuple[float, float] = (0.1, 0.1),
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """グリッドを生成"""
        from api.shapes import grid
        return grid(n_divisions, center, scale, rotate, **params)

    @classmethod
    def polyhedron(
        cls,
        polygon_type: str | int = "tetrahedron",
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """正多面体を生成"""
        from api.shapes import polyhedron
        return polyhedron(polygon_type, center, scale, rotate, **params)

    @classmethod
    def lissajous(
        cls,
        freq_x: float = 3.0,
        freq_y: float = 2.0,
        phase: float = 0.0,
        points: int = 1000,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """リサージュ曲線を生成"""
        from api.shapes import lissajous
        return lissajous(freq_x, freq_y, phase, points, center, scale, rotate, **params)

    @classmethod
    def torus(
        cls,
        major_radius: float = 0.3,
        minor_radius: float = 0.1,
        major_segments: int = 32,
        minor_segments: int = 16,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """トーラスを生成"""
        from api.shapes import torus
        return torus(major_radius, minor_radius, major_segments, minor_segments, center, scale, rotate, **params)

    @classmethod
    def cylinder(
        cls,
        radius: float = 0.3,
        height: float = 0.6,
        segments: int = 32,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """円柱を生成"""
        from api.shapes import cylinder
        return cylinder(radius, height, segments, center, scale, rotate, **params)

    @classmethod
    def cone(
        cls,
        radius: float = 0.3,
        height: float = 0.6,
        segments: int = 32,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """円錐を生成"""
        from api.shapes import cone
        return cone(radius, height, segments, center, scale, rotate, **params)

    @classmethod
    def capsule(
        cls,
        radius: float = 0.2,
        height: float = 0.4,
        segments: int = 32,
        latitude_segments: int = 16,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """カプセル形状を生成"""
        from api.shapes import capsule
        return capsule(radius, height, segments, latitude_segments, center, scale, rotate, **params)

    @classmethod
    def attractor(
        cls,
        attractor_type: str = "lorenz",
        points: int = 10000,
        dt: float = 0.01,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """ストレンジアトラクターを生成"""
        from api.shapes import attractor
        return attractor(attractor_type, points, dt, center, scale, rotate, **params)

    @classmethod
    def text(
        cls,
        text: str = "HELLO",
        size: float = 0.1,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """テキストを線分として生成"""
        from api.shapes import text as text_shape
        return text_shape(text, size, center, scale, rotate, **params)

    @classmethod
    def asemic_glyph(
        cls,
        complexity: int = 5,
        seed: int | None = None,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params,
    ) -> "Geometry":
        """抽象的なグリフ状の形状を生成"""
        from api.shapes import asemic_glyph
        return asemic_glyph(complexity, seed, center, scale, rotate, **params)

    # ── キャッシュ管理・デバッグ用メソッド ───────────────────
    @classmethod
    def clear_effect_cache(cls):
        """エフェクトキャッシュをクリア"""
        if hasattr(cls, "_effect_cache"):
            cls._effect_cache.clear()

    @classmethod
    def get_cache_stats(cls) -> dict:
        """キャッシュ統計を取得"""
        if not hasattr(cls, "_effect_cache"):
            return {"size": 0, "max_size": 256}

        return {
            "size": len(cls._effect_cache),
            "max_size": 256,
            "keys": list(cls._effect_cache.keys())[:5],  # 最初の5つのキーを表示
        }
