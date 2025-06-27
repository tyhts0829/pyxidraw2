"""
このライブラリの中核となるデータ構造
coords   : float32 ndarray  (N, 3)   # 全頂点を 1 本の連続メモリで保持
offsets  : int32   ndarray  (M+1,)   # 各線の開始 index（最後に N を追加）
lines[i] = coords[offsets[i] : offsets[i+1]]
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Geometry:
    coords: np.ndarray  # shape (N, 3)  float32
    offsets: np.ndarray  # shape (M+1,)  int32

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

    def as_arrays(self, copy=False):
        return (self.coords.copy() if copy else self.coords, self.offsets.copy() if copy else self.offsets)

    def __add__(self, other: "Geometry"):
        off_other = other.offsets + len(self.coords)
        return Geometry(np.concatenate([self.coords, other.coords]), np.concatenate([self.offsets[:-1], off_other]))
