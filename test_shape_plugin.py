#!/usr/bin/env python3
"""
形状プラグインの動作テスト
"""

from benchmarks.plugins.shapes import ShapeBenchmarkPlugin
from benchmarks.core.config import get_config

def main():
    print("Testing shape plugin...")
    
    # プラグインを作成
    config = get_config()
    plugin = ShapeBenchmarkPlugin("shape", config)
    
    # ターゲットを発見
    targets = plugin.discover_targets()
    print(f"Found {len(targets)} targets")
    
    # 最初のターゲットをテスト
    if targets:
        target = targets[0]
        print(f"Testing target: {target.name}")
        
        try:
            # 関数を実行してみる
            func = target.base_func
            result = func()
            print(f"Function executed successfully, result type: {type(result)}")
        except Exception as e:
            print(f"Error executing function: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()