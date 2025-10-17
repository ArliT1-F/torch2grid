"""
Example custom transformer plugin.
"""

from torch2grid.plugins.base import TransformerPlugin
import numpy as np


class ReversedTransformer(TransformerPlugin):
    """Fills the grid in reverse order (bottom-right to top-left)."""
    
    @property
    def name(self) -> str:
        return "reversed"
    
    @property
    def description(self) -> str:
        return "Fills grid in reverse order for a different perspective"
    
    def transform(self, tensors: dict) -> np.ndarray:
        flat_values = []
        
        for name, arr in tensors.items():
            if arr is None:
                continue
            flat_values.extend(arr.flatten())
        
        if not flat_values:
            flat_values = [0.0]
        
        size = int(np.ceil(np.sqrt(len(flat_values))))
        grid = np.zeros((size, size))
        
        # Fill in reverse order
        for i, val in enumerate(flat_values[:size*size]):
            idx = (size * size - 1) - i
            grid[idx // size, idx % size] = val
        
        return grid
