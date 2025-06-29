#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークプラグインシステム基底クラス

プラグイン可能なベンチマークシステムのアーキテクチャを提供します。
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from benchmarks.core.types import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTarget,
    ModuleFeatures,
)
from benchmarks.core.exceptions import ModuleDiscoveryError, benchmark_operation


class BenchmarkPlugin(ABC):
    """ベンチマークプラグインの基底クラス"""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        self.name = name
        self.config = config
        self._targets: Optional[List[BenchmarkTarget]] = None
    
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """プラグインの種類を返す"""
        pass
    
    @abstractmethod
    def discover_targets(self) -> List[BenchmarkTarget]:
        """ベンチマーク対象を発見する"""
        pass
    
    @abstractmethod
    def create_benchmark_target(self, target_name: str, **kwargs) -> BenchmarkTarget:
        """ベンチマーク対象を作成する"""
        pass
    
    @abstractmethod
    def analyze_target_features(self, target: BenchmarkTarget) -> ModuleFeatures:
        """対象の特性を分析する"""
        pass
    
    def get_targets(self, refresh: bool = False) -> List[BenchmarkTarget]:
        """ベンチマーク対象リストを取得（キャッシュあり）"""
        if self._targets is None or refresh:
            self._targets = self.discover_targets()
        return self._targets
    
    def is_target_enabled(self, target_name: str) -> bool:
        """対象が有効かどうかをチェック"""
        # 設定ファイルから有効/無効を判定
        return True  # デフォルトは有効
    
    def get_target_config(self, target_name: str) -> Dict[str, Any]:
        """対象固有の設定を取得"""
        return {}
    
    def validate_target(self, target: BenchmarkTarget) -> bool:
        """ベンチマーク対象の妥当性を検証"""
        try:
            # 基本的な検証：nameとexecuteメソッドがあるか
            return hasattr(target, 'name') and hasattr(target, 'execute')
        except Exception:
            return False


class PluginManager:
    """プラグイン管理クラス"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.plugins: Dict[str, BenchmarkPlugin] = {}
        self._auto_discover_plugins()
    
    def register_plugin(self, plugin: BenchmarkPlugin) -> None:
        """プラグインを登録する"""
        self.plugins[plugin.name] = plugin
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """プラグインの登録を解除する"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[BenchmarkPlugin]:
        """プラグインを取得する"""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[BenchmarkPlugin]:
        """種類別にプラグインを取得する"""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.plugin_type == plugin_type
        ]
    
    def get_all_plugins(self) -> List[BenchmarkPlugin]:
        """すべてのプラグインを取得する"""
        return list(self.plugins.values())
    
    def get_all_targets(self) -> Dict[str, List[BenchmarkTarget]]:
        """すべてのプラグインからベンチマーク対象を取得"""
        all_targets = {}
        
        for plugin in self.plugins.values():
            try:
                targets = plugin.get_targets()
                all_targets[plugin.name] = targets
            except Exception as e:
                print(f"Warning: Failed to get targets from plugin {plugin.name}: {e}")
                all_targets[plugin.name] = []
        
        return all_targets
    
    def discover_plugin_classes(self, module_path: str) -> List[Type[BenchmarkPlugin]]:
        """モジュールからプラグインクラスを発見する"""
        plugin_classes = []
        
        try:
            module = importlib.import_module(module_path)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BenchmarkPlugin) and 
                    obj is not BenchmarkPlugin):
                    plugin_classes.append(obj)
        
        except Exception as e:
            print(f"Warning: Failed to discover plugins in {module_path}: {e}")
        
        return plugin_classes
    
    def _auto_discover_plugins(self) -> None:
        """プラグインを自動発見する"""
        # 既知のプラグインモジュールを検索
        plugin_modules = [
            "benchmarks.plugins.effects",
            "benchmarks.plugins.shapes",
        ]
        
        for module_path in plugin_modules:
            try:
                plugin_classes = self.discover_plugin_classes(module_path)
                
                for plugin_class in plugin_classes:
                    # プラグインクラスをインスタンス化
                    plugin_name = plugin_class.__name__.replace("BenchmarkPlugin", "").lower()
                    plugin_instance = plugin_class(plugin_name, self.config)
                    self.register_plugin(plugin_instance)
                    
            except Exception as e:
                print(f"Warning: Failed to auto-discover plugins from {module_path}: {e}")


class BaseBenchmarkTarget:
    """ベンチマーク対象の基底実装"""
    
    def __init__(self, name: str, execute_func: callable, **metadata):
        self.name = name
        self._execute_func = execute_func
        self.metadata = metadata
    
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """ベンチマーク対象を実行"""
        return self._execute_func(*args, **kwargs)
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """メタデータを取得"""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """メタデータを設定"""
        self.metadata[key] = value


class ModuleBenchmarkTarget(BaseBenchmarkTarget):
    """モジュールベースのベンチマーク対象"""
    
    def __init__(self, name: str, module_name: str, function_name: str, **metadata):
        self.module_name = module_name
        self.function_name = function_name
        
        # 動的に関数を取得
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)
            super().__init__(name, func, **metadata)
        except Exception as e:
            raise ModuleDiscoveryError(f"Failed to load {function_name} from {module_name}", module_name)
    
    def reload_function(self) -> None:
        """関数を再読み込み"""
        try:
            module = importlib.import_module(self.module_name)
            importlib.reload(module)
            self._execute_func = getattr(module, self.function_name)
        except Exception as e:
            raise ModuleDiscoveryError(f"Failed to reload {self.function_name} from {self.module_name}", self.module_name)


class ParametrizedBenchmarkTarget(BaseBenchmarkTarget):
    """パラメータ化されたベンチマーク対象"""
    
    def __init__(self, name: str, base_func: callable, parameters: Dict[str, Any], **metadata):
        self.base_func = base_func
        self.parameters = parameters
        
        # base_funcを直接使用（シリアライズ可能）
        super().__init__(name, base_func, **metadata)
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """パラメータを取得"""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """パラメータを設定"""
        self.parameters[key] = value
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """すべてのパラメータを取得"""
        return self.parameters.copy()


# 便利関数
def create_plugin_manager(config: BenchmarkConfig) -> PluginManager:
    """プラグインマネージャーを作成する便利関数"""
    return PluginManager(config)


def discover_all_targets(config: BenchmarkConfig) -> Dict[str, List[BenchmarkTarget]]:
    """すべてのプラグインからベンチマーク対象を発見する便利関数"""
    manager = create_plugin_manager(config)
    return manager.get_all_targets()