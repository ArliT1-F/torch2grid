"""
Plugin system for custom transformers.

Plugins allow you to define custom transformation logic for converting
tensors into 2D grids for visualization.
"""

from torch2grid.plugins.base import TransformerPlugin
from torch2grid.plugins.registry import PluginRegistry

__all__ = ['TransformerPlugin', 'PluginRegistry']
