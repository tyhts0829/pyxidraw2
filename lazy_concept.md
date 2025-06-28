1. ふつうの（即時実行）エフェクトの流れ
   geom = shape.circle(...)

# ノイズをかけるたびに coords が新しい配列へコピーされる

a = effects.noise(geom, 10) # ここで計算・メモリ確保
b = effects.scale(a, 2.0) # また計算
c = effects.rotate(b, 30) # またまた計算

draw(c) # 最終結果を描画
• 呼ぶたび NumPy が配列を作り直す → 大きなデータだと時間もメモリももったいない
• しかも途中の a, b は 一瞬しか使わないのにメモリに居座る

2. “遅延評価（lazy eval）” に切り替えると？ 1. まず 演算は「まだやらない」。 2. やることリスト（DAG: 有向非巡回グラフ） だけ積んでいく。 3. draw() や to_gcode() など 結果が本当に必要になった瞬間 に
   まとめて実行（materialize ＝具象化）→ 一回のバッファ確保で済む。

イメージ図
circle ─noise→ ○ ─scale→ ○ ─rotate→ ○───► ここで初めて計算
• 丸は「まだ計算していない中間ノード」
• 矢印は「どんなエフェクトをかけるか」という 関数＋パラメータの記録 3. GeometryLazy の API 例
from pyxidraw import shape, effects, lazy

geom = shape.circle(0, 0, 100).lazy() # ← ラップするだけ
geom = geom.noise(amp=10) # まだ計算しない
geom = geom.scale(2.0)
geom = geom.rotate(30)

draw(geom) # ← ここで初めて materialize して GPU へ
どう動く？
ステップ
内部でやること
.lazy()
元の Geometry を Source ノードとして DAG を作る
.noise()
DAG に「noise(amp=10)」ノードを追加
.scale()
「scale(2.0)」ノードを追加
.rotate()
「rotate(30)」ノードを追加
draw()

1.  末端ノードから辿り 依存順に実行 2. 結果は 1 回きりの NumPy 配列確保 3. GPU にアップロード後、すぐに中間バッファを破棄
2.  何がうれしいの？
    効   果
    説明
    コピー激減
    巨大配列を何度も複製しない。必要なのは最終配列 1 つだけ。
    メモリ節約
    中間ノードは「手順メモ」だけなので軽量（数十バイト）。
    リアルタイム編集が軽快
    パラメータを何度スライダでいじっても、確定まで計算しないから UI がサクサク。
    並列化しやすい
    materialize 時に DAG を一気に解析 ⇒ 依存の無い枝を GPU やマルチコアで並列実行できる。
    履歴が残る
    DAG をそのまま“レシピ”として保存すれば、再生成やパラメータ再調整が簡単。
3.  実装イメージ（超ざっくり）
    class LazyNode:
    def **init**(self, op, src, \*\*kw):
    self.op = op # 関数オブジェクト
    self.src = src # 依存ノード（1 個 or タプル）
    self.kw = kw # パラメータ辞書
    self.\_cache = None # materialize 済み Geometry

        def materialize(self) -> Geometry:
            if self._cache is None:
                # 依存ノードを先に解決
                src_geom = (
                    self.src.materialize() if isinstance(self.src, LazyNode) else self.src
                )
                self._cache = self.op(src_geom, **self.kw)
            return self._cache

class GeometryLazy:
def **init**(self, source: Geometry | LazyNode):
self.\_node = source if isinstance(source, LazyNode) else LazyNode(lambda x: x, source)

    # チェーン用ラッパー
    def noise(self, amp=10):
        return GeometryLazy(LazyNode(effects.noise, self._node, amp=amp))

    def scale(self, s):
        return GeometryLazy(LazyNode(effects.scale, self._node, s=s))

    def rotate(self, angle):
        return GeometryLazy(LazyNode(effects.rotate, self._node, angle=angle))

    # 実際に座標が必要になったとき
    def materialize(self) -> Geometry:
        return self._node.materialize()
    •	draw() 側では isinstance(obj, GeometryLazy) なら obj.materialize() を呼んでから通常パイプラインへ流すだけ。

6. 使うときの注意
   注意点
   具体例
   逐次プレビューが重い処理は変わらない
   例: 1 億点 SPH を作るエフェクト → materialize 時に時間はかかる。
   副作用禁止
   ノードは純粋関数（入力 → 出力が決まる）にしておかないと再実行で結果が変わる。
   一度 materialize したら同じ DAG で再利用
   そうしないと再計算になる。パラメータ変更したいときは新ノードで。
   まとめ
   • Lazy = 遅延、DAG = エフェクトの手順書
   • 書く側は今まで通り .noise().scale() と繋ぐだけ
   • 裏では「まだ計算しない」→ draw() で一括計算
   • メモリ・CPU・UX が全部軽くなる仕組みです 🎉
