"""
Basic tests for torch2grid functionality.
"""

import unittest
import numpy as np
import tempfile
import os
import sys

# Add the parent directory to the path so we can import torch2grid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch2grid.transformer import to_neutral_grid
from torch2grid.plugins.builtin import FlattenTransformer, SpiralTransformer
from torch2grid.plugins.registry import PluginRegistry


class TestBasicFunctionality(unittest.TestCase):
    """Test basic torch2grid functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_tensors = {
            'layer1.weight': np.random.randn(10, 5),
            'layer1.bias': np.random.randn(10),
            'layer2.weight': np.random.randn(5, 10),
            'layer2.bias': np.random.randn(5),
        }
    
    def test_flatten_transformer(self):
        """Test the flatten transformer plugin."""
        transformer = FlattenTransformer()
        grid = transformer(self.sample_tensors)
        
        # Check that grid is 2D
        self.assertEqual(len(grid.shape), 2)
        
        # Check that grid is roughly square
        self.assertLessEqual(abs(grid.shape[0] - grid.shape[1]), 1)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(grid)))
    
    def test_spiral_transformer(self):
        """Test the spiral transformer plugin."""
        transformer = SpiralTransformer()
        grid = transformer(self.sample_tensors)
        
        # Check that grid is 2D
        self.assertEqual(len(grid.shape), 2)
        
        # Check that grid is roughly square
        self.assertLessEqual(abs(grid.shape[0] - grid.shape[1]), 1)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(grid)))
    
    def test_to_neutral_grid(self):
        """Test the to_neutral_grid function."""
        grid = to_neutral_grid(self.sample_tensors)
        
        # Check that grid is 2D
        self.assertEqual(len(grid.shape), 2)
        
        # Check that grid is roughly square
        self.assertLessEqual(abs(grid.shape[0] - grid.shape[1]), 1)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(grid)))
    
    def test_plugin_registry(self):
        """Test the plugin registry."""
        registry = PluginRegistry()
        
        # Check that built-in plugins are loaded
        self.assertIn('flatten', registry.list_plugins())
        self.assertIn('spiral', registry.list_plugins())
        
        # Test getting a plugin
        flatten_plugin = registry.get('flatten')
        self.assertIsNotNone(flatten_plugin)
        self.assertEqual(flatten_plugin.name, 'flatten')
    
    def test_empty_tensors(self):
        """Test handling of empty tensors."""
        empty_tensors = {}
        grid = to_neutral_grid(empty_tensors)
        
        # Should still produce a valid grid
        self.assertEqual(len(grid.shape), 2)
        self.assertTrue(np.all(grid == 0))
    
    def test_none_tensors(self):
        """Test handling of None values in tensors."""
        tensors_with_none = {
            'layer1.weight': np.random.randn(5, 3),
            'layer1.bias': None,
            'layer2.weight': np.random.randn(3, 5),
        }
        
        grid = to_neutral_grid(tensors_with_none)
        
        # Should still produce a valid grid
        self.assertEqual(len(grid.shape), 2)
        self.assertTrue(np.all(np.isfinite(grid)))


if __name__ == '__main__':
    unittest.main()