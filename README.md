# PyxiDraw - MIDI制御リアルタイム3Dベクターグラフィックスシステム

PyxiDrawは、MIDIコントローラーと連携したリアルタイム3Dベクトルグラフィックス生成システムです。直感的なメソッドチェーンAPI、高性能キャッシュシステム、豊富な形状・エフェクトライブラリを特徴とし、AxiDrawなどのペンプロッターデバイスへの出力にも対応しています。

## 🚀 クイックスタート

### 基本セットアップ

```bash
# 基本デモの実行
python test_effects_geometry.py

# 全形状の展示
python all_shapes.py

# メソッドチェーンAPIデモ
python test_fluent_api.py
```

### 最初のスケッチ

```python
import arc
from api.runner import run_sketch
from api.shapes import polygon

def draw(t, cc):
    # 時間と共に回転する六角形
    return polygon(n_sides=6).size(50).at(100, 100).rotate_z(t * 0.1)

if __name__ == "__main__":
    arc.start(midi=True)  # MIDI制御を有効化
    run_sketch(draw, canvas_size="SQUARE_200", render_scale=4)
    arc.stop()
```

## 🏗️ システム構成

PyxiDrawは明確な4層アーキテクチャで構成されています：

```
pyxidraw2/
├── api/          # 🎯 高レベルユーザーAPI
├── engine/       # ⚙️ コアシステム（レンダリング、MIDI、並列処理）
├── shapes/       # 📐 形状生成ライブラリ
├── effects/      # ✨ エフェクト処理ライブラリ
└── util/         # 🔧 ユーティリティ・定数
```

## 🎨 形状ライブラリ

### 基本図形
```python
from api.shapes import polygon, sphere, polyhedron

# 正多角形（三角形〜多角形）
hex_shape = polygon(n_sides=6).size(50).at(100, 100)

# 球体（細分化レベル制御可能）
sphere_obj = sphere(subdivisions=0.5).scale(100, 100, 100)

# 正多面体（四面体、立方体、八面体、十二面体、二十面体）
tetrahedron = polyhedron("tetrahedron").scale(80, 80, 80)
```

### 高度な形状
```python
from api.shapes import lissajous, attractor, torus, text

# リサージュ曲線
lissa = lissajous(freq_x=3.0, freq_y=2.0).scale(50, 50, 50)

# カオスアトラクター（Lorenz、Rossler、Chua）
lorenz = attractor("lorenz").scale(5, 5, 5).at(150, 150)

# トーラス
donut = torus(major_radius=0.8, minor_radius=0.3).scale(100, 100, 100)

# テキスト
message = text("HELLO").size(30).at(100, 100)
```

## ✨ エフェクトシステム

### 基本変換
```python
from api.effects import rotation, scaling, translation, transform

# 個別変換
rotated = rotation(geometry, center=(0,0,0), rotate=(0,0,0.5))
scaled = scaling(geometry, center=(0,0,0), scale=(2,2,2))
moved = translation(geometry, offset_x=10, offset_y=20)

# 複合変換（推奨）
transformed = transform(geometry, 
                       center=(100, 100, 0), 
                       scale=(2, 2, 2), 
                       rotate=(0.5, 0.3, 0))
```

### 形状操作エフェクト
```python
from api.effects import subdivision, filling, noise, buffer

# 線の細分化（より滑らかな曲線）
smooth = subdivision(geometry, n_divisions=0.8)

# ハッチング塗りつぶし
filled = filling(geometry, pattern="lines", density=0.6)

# Perlinノイズによる有機的な歪み
organic = noise(geometry, intensity=0.3, time=t)

# パス周りのバッファ（太線効果）
thick = buffer(geometry, distance=2.0, join_style="round")
```

## 🎹 MIDI制御システム

### 基本的なMIDI制御
```python
def draw(t, cc):
    # t: 経過時間（秒）
    # cc: MIDI CC値辞書 {CC番号: 値(0.0-1.0)}
    
    size = cc.get(1, 0.5) * 100      # CC#1でサイズ制御
    rotation_speed = cc.get(2, 0.5)   # CC#2で回転速度制御
    noise_level = cc.get(3, 0.2)      # CC#3でノイズ強度制御
    
    base = polygon(n_sides=6).size(size).at(100, 100)
    rotating = base.rotate_z(t * rotation_speed)
    final = noise(rotating, intensity=noise_level, time=t)
    
    return final
```

### 複雑なMIDI制御例
```python
def advanced_midi_control(t, cc):
    # 形状の種類をMIDI制御
    n_sides = int(cc.get(1, 0.5) * 10 + 3)  # 3〜13角形
    
    # 複数パラメータの組み合わせ
    base = polygon(n_sides=n_sides)
    
    # エフェクトチェーン（MIDIで各段階を制御）
    transformed = transform(base,
                           center=(100, 100, 0),
                           scale=(cc.get(2, 0.5) * 100, cc.get(2, 0.5) * 100, 50),
                           rotate=(cc.get(3, 0) * 6.28, cc.get(4, 0) * 6.28, t * 0.1))
    
    subdivided = subdivision(transformed, n_divisions=cc.get(5, 0.5))
    final = noise(subdivided, intensity=cc.get(6, 0.2), time=t)
    
    return final
```

## 🚀 高性能メソッドチェーンAPI

PyxiDrawの最大の特徴は、直感的で高性能なメソッドチェーンAPIです：

### 新記法：エレガントなメソッドチェーン
```python
def draw(t, cc):
    # 読みやすく、書きやすい
    return (sphere(subdivisions=0.5)
            .scale(100, 100, 100)
            .translate(100, 100, 0)
            .rotate(t * 0.1, t * 0.2, 0))

def complex_scene(t, cc):
    # 複数オブジェクトの組み合わせ
    sphere_obj = (sphere(subdivisions=cc.get(1, 0.5))
                  .scale(80, 80, 80)
                  .at(100, 150)
                  .rotate(t * 0.2, t * 0.3, 0))
    
    polygon_obj = (polygon(n_sides=8)
                   .size(cc.get(2, 0.5) * 60)
                   .at(200, 150)
                   .spin(t * 0.5))
    
    return sphere_obj + polygon_obj
```

### ショートカットメソッド
```python
# 便利なエイリアス
geometry.size(50)        # scale_uniform(50) のエイリアス
geometry.at(100, 100)    # center_at(100, 100) のエイリアス
geometry.spin(0.5)       # rotate_z(0.5) のエイリアス
```

## ⚡ 高性能キャッシュシステム

PyxiDrawは**最大440倍の高速化**を実現するインテリジェントキャッシュを内蔵：

```python
# 同じ操作は自動的にキャッシュされ、即座に結果を返却
sphere1 = sphere(subdivisions=0.5).scale(100, 100, 100)  # 初回計算
sphere2 = sphere(subdivisions=0.5).scale(100, 100, 100)  # キャッシュヒット（440倍高速）

# パラメータが異なれば新たに計算
sphere3 = sphere(subdivisions=0.6).scale(100, 100, 100)  # 新規計算
```

## 🎛️ 対応MIDIデバイス

- **Teenage Engineering**: OP-Z, TX-6
- **monome**: Grid, Arc
- **一般的なMIDIコントローラー**: すべてのCC対応デバイス

### MIDI設定
```python
# MIDIデバイス確認
arc.list_midi_devices()

# MIDI有効化
arc.start(midi=True)

# MIDI無効モード
arc.start(midi=False)
```

## 📊 実行環境設定

### run_sketch関数の詳細設定
```python
run_sketch(
    user_draw_function,
    canvas_size="A4",                    # "A4", "A5", "SQUARE_200" or (width, height)
    render_scale=4,                      # 表示倍率（1-10）
    fps=60,                             # フレームレート
    background=(1, 1, 1, 1),            # 背景色 (R, G, B, A)
    workers=6                           # 並列処理ワーカー数
)
```

### キャンバスサイズプリセット
```python
from util.constants import CANVAS_SIZES

# 利用可能なサイズ
"A4"          # 210×297mm
"A5"          # 148×210mm  
"SQUARE_200"  # 200×200mm
"SQUARE_150"  # 150×150mm

# カスタムサイズ
run_sketch(draw, canvas_size=(120, 180))
```

## 🛠️ 開発者向け情報

### テストとベンチマーク
```bash
# パフォーマンステスト
python demo_cached_scene.py

# ベンチマーク実行
python -m pytest benchmarks/

# 全エフェクトテスト
python all_effects.py
```

### デバッグとモニタリング
```python
# パフォーマンス監視
from engine.monitor import performance_monitor

# デバッグ出力
import logging
logging.basicConfig(level=logging.DEBUG)
```

### プロジェクト拡張
```python
# カスタム形状の追加
from shapes.base import BaseShape
from engine.core.geometry import Geometry

class MyCustomShape(BaseShape):
    def create_geometry(self) -> Geometry:
        # カスタム形状ロジック
        pass

# カスタムエフェクトの追加
from effects.base import BaseEffect

class MyCustomEffect(BaseEffect):
    def apply(self, geometry: Geometry) -> Geometry:
        # カスタムエフェクトロジック
        pass
```

## 🔗 関連リソース

- **アーキテクチャ詳細**: `/CLAUDE.md`
- **サンプルコード**: `/examples/`
- **旧アーキテクチャ**: `/previous_design/`
- **ベンチマーク結果**: `/benchmark_results/`

## 🎯 学習ロードマップ

### 初心者
1. `test_effects_geometry.py` を実行して基本動作を確認
2. 簡単な形状生成から始める (`polygon`, `sphere`)
3. 基本的なMIDI制御を試す

### 中級者
1. メソッドチェーンAPIをマスター
2. 複数エフェクトの組み合わせを実験
3. 複雑なMIDI制御パターンを構築

### 上級者
1. カスタム形状・エフェクトの開発
2. 並列処理システムの活用
3. パフォーマンス最適化とベンチマーク

## ⚠️ 重要な注意事項

- **座標系**: 3D座標系のみ対応（Z=0で2D描画）
- **キャンバス**: A4サイズ（297mm×210mm）基準の設計
- **エフェクト引数**: 0.0-1.0の正規化レンジを使用
- **スレッドセーフティ**: MIDI入力は別スレッドで処理される

---

PyxiDrawで**美しいベクターアート**と**インタラクティブな表現**を作り上げましょう！ 🎨✨