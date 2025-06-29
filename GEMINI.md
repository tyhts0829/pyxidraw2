# GEMINI.md

## プロジェクト概要

`pyxidraw2` は、Pythonで手続き的な2D/3Dグラフィックスを生成し、PygletとModernGLを用いてリアルタイムにレンダリングするライブラリです。MIDIコントローラーからの入力を受け付け、パラメーターを動的に変更しながら描画できます。

## 中核となるコンセプト

### `Geometry` クラス (`engine/core/geometry.py`)

- **データ構造**:
    - `coords`: `(N, 3)` の `float32` NumPy配列。全頂点の座標を格納します。
    - `offsets`: `(M+1,)` の `int32` NumPy配列。各線（ストローク）の開始インデックスを格納します。
- **責務**:
    - 頂点データとオフセット情報を保持する、このライブラリの中心的なデータコンテナです。
    - 不変性(Immutability)を基本とし、操作（エフェクト適用）は新しい `Geometry` オブジェクトを返します。
    - メソッドチェーン (`.scale(...).rotate(...).noise(...)`) を可能にするインターフェースを提供します。
    - 形状生成（`Geometry.sphere()`など）とエフェクト適用（`geometry.noise()`など）の両方のためのメソッドを持ちます。
    - パフォーマンス向上のため、エフェクト適用結果をキャッシュする機能を持ちます。

### `run_sketch` 関数 (`api/runner.py`)

- **責務**:
    - アプリケーションのメインループを初期化し、実行します。
    - Pygletウィンドウ、ModernGLコンテキスト、MIDIサービス、ワーカースレッドなどをセットアップします。
    - ユーザーが定義した `draw` 関数を定期的に呼び出し、返された `Geometry` オブジェクトをレンダリングします。

### Shape (`shapes/` ディレクトリ)

- **基底クラス**: `shapes.base.BaseShape`
- **責務**:
    - `Geometry` オブジェクトを生成するクラスです。
    - `generate()` メソッドを実装し、具体的な形状の頂点データを計算します。
    - `__call__` メソッドにより、生成と同時に基本的な変形（移動、回転、スケール）を適用できます。
    - 例: `shapes.sphere.Sphere`, `shapes.polygon.Polygon`

### Effect (`effects/` ディレクトリ)

- **基底クラス**: `effects.base.BaseEffect`
- **責務**:
    - `Geometry` オブジェクトを入力として受け取り、変形させて新しい `Geometry` オブジェクトを返すクラスです。
    - `apply()` メソッドを実装し、具体的な変形処理を記述します。
    - 例: `effects.noise.Noise`, `effects.subdivision.Subdivision`

## 描画処理の流れ

`Geometry` オブジェクトが生成されてから画面に描画されるまでの処理は、主にワーカースレッドとメイン（レンダリング）スレッドの連携によって行われます。

1.  **`user_draw` 関数の実行 (Workerスレッド)**
    - `api.runner.run_sketch` に渡された `user_draw` 関数 (例: `main.py` の `draw` 関数) が `engine.pipeline.worker.WorkerPool` によってバックグラウンドのワーカースレッドで実行されます。
    - この関数内で `Geometry.sphere()` や `.noise()` といったメソッドチェーンが実行され、最終的な `Geometry` オブジェクトが構築されます。
    - 処理が完了すると、結果の `Geometry` オブジェクトは `worker_pool.result_q` (キュー) に入れられます。

2.  **結果の受信 (メインスレッド)**
    - `engine.pipeline.receiver.StreamReceiver` が、メインスレッドで `worker_pool.result_q` を監視しています。
    - 新しい `Geometry` オブジェクトがキューに追加されると、`StreamReceiver` はそれを取り出し、`engine.pipeline.buffer.SwapBuffer` に書き込みます。
    - `SwapBuffer` はダブルバッファリング（またはそれ以上）の仕組みを提供し、ワーカースレッドが次のフレームの計算を行っている間も、レンダラが安全に現在のフレームのデータを読み取れるようにします。

3.  **レンダリング (メインスレッド)**
    - `engine.render.renderer.LineRenderer` が、`SwapBuffer` から描画対象の `Geometry` データを読み取ります。
    - `LineRenderer.draw` メソッド内で、`Geometry` の `coords` と `offsets` データがVBO (Vertex Buffer Object) としてGPUに転送されます。
    - ModernGLを介してシェーダーが実行され、頂点データが画面上の線としてレンダリングされます。

この一連の流れは `engine.core.frame_clock.FrameClock` によって調整され、指定されたFPSで各コンポーネント (`WorkerPool`, `StreamReceiver`, `LineRenderer` など) の `tick` メソッドが呼び出されることで、フレームごとに繰り返されます。

## 実装のポイント

- **APIの統一**: `api` ディレクトリのモジュール (`api.shapes`, `api.effects`) が、`BaseShape` や `BaseEffect` のインスタンスをラップし、関数として呼び出せるようにしています。これにより、ユーザーはクラスを意識することなく `effects.noise(geom)` のように利用できます。
- **メソッドチェーンとキャッシュ**: `Geometry` クラスのメソッド（`.noise()`, `.scale()` など）は、内部で対応する `api.effects` の関数を呼び出します。この際、`_apply_cached_effect` メソッドを介して結果がキャッシュされ、同じパラメータでの再計算を防ぎます。
- **並列処理**: `run_sketch` は `WorkerPool` を使い、ユーザーの `draw` 関数を別スreadで実行します。これにより、メインのレンダリングスレッドがブロックされるのを防ぎます。

## 新しい機能を追加する方法

### 新しいShapeを追加する

1.  `shapes` ディレクトリに新しいファイルを作成します (例: `my_shape.py`)。
2.  `shapes.base.BaseShape` を継承したクラスを定義します。
3.  `generate(self, **params)` メソッドを実装し、`Geometry` オブジェクトを返します。
4.  `api/shapes.py` で、作成したクラスのインスタンスを作成し、関数としてエクスポートします。
5.  `Geometry` クラスに、新しい形状を生成するためのクラスメソッド (`@classmethod`) を追加します (例: `Geometry.my_shape(...)`)。

### 新しいEffectを追加する

1.  `effects` ディレクトリに新しいファイルを作成します (例: `my_effect.py`)。
2.  `effects.base.BaseEffect` を継承したクラスを定義します。
3.  `apply(self, geometry: Geometry, **params)` メソッドを実装し、変形後の `Geometry` オブジェクトを返します。
4.  `api/effects.py` で、作成したクラスのインスタンスを作成し、関数としてエクスポートします。
5.  `Geometry` クラスに、新しいエフェクトを適用するためのメソッドを追加します (例: `geometry.my_effect(...)`)。この際、`_apply_cached_effect` を使ってキャッシュを有効にします。