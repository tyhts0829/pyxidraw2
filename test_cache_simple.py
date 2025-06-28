#!/usr/bin/env python3

import time
from api.shapes import polygon
from engine.core.geometry import Geometry

def test_cache_performance():
    print("=== キャッシュ性能テスト ===")
    
    # 初期キャッシュクリア
    Geometry.clear_effect_cache()
    
    # 基本形状を作成
    base_poly = polygon(n_sides=6)
    print(f"初期キャッシュ: {Geometry.get_cache_stats()}")
    
    # 同じ操作を複数回実行（キャッシュ効果テスト）
    operations = [
        lambda p: p.size(50).at(100, 100),
        lambda p: p.size(30).at(150, 50),  
        lambda p: p.size(50).at(100, 100),  # 重複（キャッシュヒットするはず）
        lambda p: p.size(30).at(150, 50),  # 重複（キャッシュヒットするはず）
    ]
    
    results = []
    for i, op in enumerate(operations):
        start_time = time.time()
        result = op(base_poly)
        end_time = time.time()
        
        duration = (end_time - start_time) * 1000  # ms
        cache_stats = Geometry.get_cache_stats()
        
        print(f"操作 {i+1}: {duration:.2f}ms - キャッシュサイズ: {cache_stats['size']}")
        results.append(result)
    
    print(f"最終キャッシュ統計: {Geometry.get_cache_stats()}")
    
    # メソッドチェーンテスト
    print("\n=== メソッドチェーンテスト ===")
    
    # 複雑なチェーン
    start_time = time.time()
    complex_chain1 = (polygon(n_sides=8)
                      .size(25)
                      .at(75, 75)
                      .spin(0.5))
    end_time = time.time()
    print(f"複雑チェーン1回目: {(end_time - start_time) * 1000:.2f}ms")
    
    # 同じチェーンを再実行（キャッシュされるはず）
    start_time = time.time()
    complex_chain2 = (polygon(n_sides=8)
                      .size(25)
                      .at(75, 75)
                      .spin(0.5))
    end_time = time.time()
    print(f"複雑チェーン2回目: {(end_time - start_time) * 1000:.2f}ms")
    
    print(f"最終キャッシュ統計: {Geometry.get_cache_stats()}")

if __name__ == "__main__":
    test_cache_performance()