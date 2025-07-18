# pyxidraw2 コードレビュー

## 1. 総評

`pyxidraw2`は、手続き的なグラフィック生成のための強力でよく設計されたライブラリです。特に、`Geometry`オブジェクトを中心とした不変（immutable）なデータフロー、メソッドチェーンによる直感的なAPI、そしてパフォーマンスを考慮したキャッシュ機構と並列処理は高く評価できます。

コードは全体的にクリーンで、モジュール分割も適切です。ドキュメント（docstring）も整備されており、プロジェクトの理解を助けています。

以下に、さらなる改善のための具体的な指摘事項を挙げます。

---

## 2. 優れた点 (Good)

- **コアコンセプトの明確さ**: `Geometry`という不変のデータ構造を中心に、`Shape`で生成し`Effect`で変換するという流れが非常に明確です。これにより、APIの学習コストが下がり、予測可能なコードを書きやすくなっています。

- **流れるようなAPI (Fluent API)**: `Geometry.sphere().scale(2).noise()` のようにメソッドチェーンで処理を記述できるため、コードが直感的で読みやすいです。

- **パフォーマンスへの配慮**:
    - **キャッシュ機構**: `Geometry`と各`Effect`/`Shape`に`lru_cache`を利用したキャッシュが組み込まれており、同一パラメータでの再計算を効率的に防いでいます。これはインタラクティブなアプリケーションにおいて非常に重要です。
    - **並列処理**: `run_sketch`内で`WorkerPool`を使い、重い計算処理をバックグラウンドで実行する設計は、UIの応答性を保つ上で不可欠です。
    - **Numbaの活用**: `effects.noise`などで見られるように、計算量の多い部分で`@njit`を使い、Pythonコードをネイティブコード並みの速度にコンパイルしている点は素晴らしいです。

- **拡張性の高さ**: 新しい`Shape`や`Effect`を追加するための基底クラス (`BaseShape`, `BaseEffect`) と明確なディレクトリ構造が用意されており、ライブラリの機能を容易に拡張できます。

- **テストとベンチマーク**: `tests/`と`benchmarks/`が整備されており、品質保証と性能評価に対する意識の高さが伺えます。特に`benchmarks/effects_benchmark.py`は、各エフェクトの性能を多角的に測定・可視化する優れた作りになっています。

---

## 3. 改善提案 (Suggestions & Refactoring)

### 3.1. `Geometry`クラスのキャッシュ機構

- **現状**: `Geometry`クラス内に`_effect_cache`というクラス属性の辞書で自前のLRU風キャッシュを実装しています。 (`engine/core/geometry.py`)

- **問題点**: 
    1.  Python標準の`functools.lru_cache`デコレータが使える場面で、自前の実装はコードを複雑にし、潜在的なバグの原因となります。
    2.  現在の実装はスレッドセーフではありません。`WorkerPool`を使っているため、将来的に複数のワーカースレッドから同時にキャッシュアクセスが発生すると問題が起きる可能性があります。
    3.  LRUのロジックが「最も古いキーを削除する」という単純なもので、厳密なLRUではありません。

- **提案**: 
    - `_apply_cached_effect`メソッドを廃止し、`Geometry`の各エフェクトメソッド（`scale`, `rotate`など）を、`BaseEffect`のように`lru_cache`でデコレートされたキャッシュ用メソッドを呼び出す形に統一することを検討します。`BaseEffect`や`BaseShape`のキャッシュ実装が参考になります。
    - もしくは、`Geometry`オブジェクト自体を不変（hashable）にして、`lru_cache`の引数として直接渡せるように設計変更するのも一つの手です。

### 3.2. APIレイヤーと実装の結合度

- **現状**: `Geometry`クラスの各エフェクトメソッド（例: `noise`）内で、`from api.effects import noise` のように、具体的な実装モジュールを直接インポートしています。

- **問題点**: `engine`というコア層が`api`という上位層に依存しており、レイヤー間の依存関係が逆転しています。これは循環参照のリスクや、将来的なモジュール分割の妨げになる可能性があります。

- **提案**: 
    - **依存性逆転の原則 (DIP)** を適用します。`Geometry`クラスは具体的なエフェクト実装を知るべきではありません。
    - `Geometry`のコンストラクタやメソッドが、エフェクト処理を行う関数（Callable）を引数として受け取るようにします（依存性の注入）。
    - `api`層で、具体的な`Effect`オブジェクトと`Geometry`オブジェクトを結びつける処理を記述するように責務を分離します。

    ```python
    # 変更案 (Geometryクラス内)
    # from api.effects import noise # ← このような直接インポートをやめる

    def apply_effect(self, effect_func: Callable[..., 'Geometry'], *args, **kwargs) -> 'Geometry':
        # キャッシュロジックはここに集約
        return self._apply_cached_effect(effect_func.__name__, effect_func, *args, **kwargs)

    # api/effects.py 内
    def noise(geometry, **kwargs):
        noise_effect = Noise() # 具体的なEffectクラス
        return geometry.apply_effect(noise_effect.apply, **kwargs)
    ```

### 3.3. パラメータのマッピング処理

- **現状**: `shapes/polygon.py`の`generate`メソッド内で、`float`型の入力値を`_nonlinear_map_exp`を使って非線形に整数へマッピングしています。

- **問題点**: このようなパラメータの解釈・変換ロジックが各Shapeクラス内に散在すると、API全体としての一貫性が失われやすくなります。MIDIコントローラーの0-127の値を、意味のあるパラメータ範囲に変換する処理は多くの箇所で必要になるはずです。

- **提案**: 
    - MIDI入力やUIスライダーからの入力値（例: 0.0-1.0）を、各`Shape`や`Effect`が要求する具体的なパラメータ範囲（例: 3-100辺、0.1-10.0の強度）に変換するための専用ユーティリティモジュールやクラスを作成します。
    - 例えば、`util/mapping.py` を作り、線形、対数、指数など、さまざまなマッピング関数をそこにまとめることで、コードの再利用性が高まります。

### 3.4. Numba JITの定数

- **現状**: `effects/noise.py`では、`njit`でコンパイルされる関数内で、グローバルに定義された`perm`や`grad3`といったNumPy配列（定数）を参照しています。

- **問題点**: Numbaは`@njit`デコレータが適用された時点で関数をコンパイルしますが、グローバル変数を参照している場合、その変数がPythonオブジェクトであるためパフォーマンスのボトルネックになることがあります（リフレクションが発生する）。

- **提案**: 
    - `njit`関数には、使用する定数配列を引数として明示的に渡すようにします。これにより、Numbaは変数の型をコンパイル時に確定でき、最適化を最大限に活かすことができます。

    ```python
    # 変更案 (effects/noise.py)
    # @njit
    # def perlin_core(vertices, frequency): # グローバルのperm, grad3を参照
    #     ...

    @njit
    def perlin_core(vertices, frequency, perm_table, grad3_array): # 引数で受け取る
        ...
    ```

---

## 4. その他 (Minor Points)

- **型ヒント**: 全体的に型ヒントは付与されていますが、`Any`が散見されます。より厳密な型（例: `TypeVar`, `Callable`の引数と戻り値の型）を定義することで、静的解析の恩恵をさらに受けられます。
- **docstringの統一**: `api/runner.py`のdocstringは日本語ですが、他の多くは英語です。プロジェクト内で言語を統一すると、より一貫性が増します。
- **設定ファイル**: `benchmarks/config/default.yaml` がありますが、アプリケーション全体の設定（ウィンドウサイズ、FPSのデフォルト値など）も同様の仕組みで管理すると、コード内からハードコーディングされた値を減らせます。

以上が今回のコードレビューです。このプロジェクトは非常にポテンシャルが高く、今後の発展が楽しみです。
