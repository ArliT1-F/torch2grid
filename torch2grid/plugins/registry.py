"""
Plugin registry for managing transformer plugins.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Optional
from torch2grid.plugins.base import TransformerPlugin


class PluginRegistry:
    """
    Registry for managing and discovering transformer plugins.
    """
    
    def __init__(self):
        self._plugins: Dict[str, TransformerPlugin] = {}
        self._load_builtin_plugins()
    
    def _load_builtin_plugins(self):
        """Load built-in plugins."""
        try:
            from torch2grid.plugins.builtin import (
                FlattenTransformer,
                LayerWeightedTransformer,
                SpiralTransformer,
                NormalizedTransformer,
                LayerSeparatedTransformer
            )
            
            self.register(FlattenTransformer())
            self.register(LayerWeightedTransformer())
            self.register(SpiralTransformer())
            self.register(NormalizedTransformer())
            self.register(LayerSeparatedTransformer())
        except ImportError:
            pass
    
    def register(self, plugin: TransformerPlugin):
        """
        Register a plugin.
        
        Args:
            plugin: TransformerPlugin instance
        """
        if not isinstance(plugin, TransformerPlugin):
            raise TypeError(f"Plugin must be instance of TransformerPlugin, got {type(plugin)}")
        
        self._plugins[plugin.name] = plugin
        print(f"Registered plugin: {plugin.name}")
    
    def unregister(self, name: str):
        """
        Unregister a plugin by name.
        
        Args:
            name: Plugin name
        """
        if name in self._plugins:
            del self._plugins[name]
    
    def get(self, name: str) -> Optional[TransformerPlugin]:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    def get_plugin_info(self, name: str) -> Optional[str]:
        """
        Get plugin description.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin description or None if not found
        """
        plugin = self.get(name)
        return plugin.description if plugin else None
    
    def load_from_file(self, filepath: str):
        """
        Load a plugin from a Python file.
        
        Args:
            filepath: Path to Python file containing plugin
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Plugin file not found: {filepath}")
        
        # Load module from file
        spec = importlib.util.spec_from_file_location("custom_plugin", filepath)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load plugin from {filepath}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_plugin"] = module
        spec.loader.exec_module(module)
        
        # Find and register all TransformerPlugin subclasses
        registered_count = 0
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, TransformerPlugin) and 
                attr is not TransformerPlugin):
                try:
                    plugin_instance = attr()
                    self.register(plugin_instance)
                    registered_count += 1
                except Exception as e:
                    print(f"Warning: Could not instantiate plugin {attr_name}: {e}")
        
        if registered_count == 0:
            print(f"Warning: No plugins found in {filepath}")
    
    def load_from_directory(self, directory: str):
        """
        Load all plugins from a directory.
        
        Args:
            directory: Path to directory containing plugin files
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        for filename in os.listdir(directory):
            if filename.endswith('.py') and not filename.startswith('_'):
                filepath = os.path.join(directory, filename)
                try:
                    self.load_from_file(filepath)
                except Exception as e:
                    print(f"Warning: Could not load plugin from {filepath}: {e}")
    
    def find_compatible_plugin(self, tensors: dict) -> Optional[TransformerPlugin]:
        """
        Find the first compatible plugin for given tensors.
        
        Args:
            tensors: Dictionary of layer names to arrays
            
        Returns:
            Compatible plugin or None
        """
        for plugin in self._plugins.values():
            if plugin.can_handle(tensors):
                return plugin
        return None
    
    def __repr__(self):
        return f"PluginRegistry({len(self._plugins)} plugins)"


# Global registry instance
_global_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return _global_registry
