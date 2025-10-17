"""
Built-in transformer plugins.
"""

import numpy as np
from torch2grid.plugins.base import TransformerPlugin


class FlattenTransformer(TransformerPlugin):
    """
    Default flatten transformer - flattens all tensors into a square grid.
    """
    
    @property
    def name(self) -> str:
        return "flatten"
    
    @property
    def description(self) -> str:
        return "Flattens all tensors into a square grid (default behavior)"
    
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
        for i, val in enumerate(flat_values[:size*size]):
            grid[i // size, i % size] = val
        
        return grid


class LayerWeightedTransformer(TransformerPlugin):
    """
    Arranges layers in blocks based on their size.
    Larger layers get more space in the grid.
    """
    
    @property
    def name(self) -> str:
        return "layer_weighted"
    
    @property
    def description(self) -> str:
        return "Arranges layers in blocks, larger layers get more space"
    
    def transform(self, tensors: dict) -> np.ndarray:
        if not tensors:
            return np.zeros((1, 1))
        
        # Calculate sizes
        layer_sizes = {}
        total_elements = 0
        for name, arr in tensors.items():
            if arr is not None:
                size = arr.size
                layer_sizes[name] = size
                total_elements += size
        
        if total_elements == 0:
            return np.zeros((1, 1))
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(total_elements)))
        grid = np.zeros((grid_size, grid_size))
        
        current_pos = 0
        for name, arr in tensors.items():
            if arr is None or name not in layer_sizes:
                continue
            
            flat = arr.flatten()
            for val in flat:
                if current_pos >= grid_size * grid_size:
                    break
                row = current_pos // grid_size
                col = current_pos % grid_size
                grid[row, col] = val
                current_pos += 1
        
        return grid


class SpiralTransformer(TransformerPlugin):
    """
    Arranges weights in a spiral pattern from center outward.
    """
    
    @property
    def name(self) -> str:
        return "spiral"
    
    @property
    def description(self) -> str:
        return "Arranges weights in a spiral pattern from center outward"
    
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
        
        # Generate spiral coordinates
        coords = self._generate_spiral(size)
        
        for i, val in enumerate(flat_values[:len(coords)]):
            row, col = coords[i]
            grid[row, col] = val
        
        return grid
    
    def _generate_spiral(self, size: int):
        """Generate spiral coordinates from center outward."""
        coords = []
        x, y = size // 2, size // 2
        coords.append((x, y))
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        dir_idx = 0
        steps = 1
        
        while len(coords) < size * size:
            for _ in range(2):
                dx, dy = directions[dir_idx]
                for _ in range(steps):
                    x, y = x + dx, y + dy
                    if 0 <= x < size and 0 <= y < size:
                        coords.append((x, y))
                    if len(coords) >= size * size:
                        return coords
                dir_idx = (dir_idx + 1) % 4
            steps += 1
        
        return coords


class NormalizedTransformer(TransformerPlugin):
    """
    Normalizes weights to [0, 1] range before creating grid.
    """
    
    @property
    def name(self) -> str:
        return "normalized"
    
    @property
    def description(self) -> str:
        return "Normalizes all weights to [0, 1] range before visualization"
    
    def transform(self, tensors: dict) -> np.ndarray:
        flat_values = []
        
        for name, arr in tensors.items():
            if arr is None:
                continue
            flat_values.extend(arr.flatten())
        
        if not flat_values:
            return np.zeros((1, 1))
        
        # Normalize values
        flat_array = np.array(flat_values)
        min_val, max_val = flat_array.min(), flat_array.max()
        if max_val - min_val > 1e-6:
            flat_array = (flat_array - min_val) / (max_val - min_val)
        
        size = int(np.ceil(np.sqrt(len(flat_array))))
        grid = np.zeros((size, size))
        for i, val in enumerate(flat_array[:size*size]):
            grid[i // size, i % size] = val
        
        return grid


class LayerSeparatedTransformer(TransformerPlugin):
    """
    Creates a grid where each layer is clearly separated by boundaries.
    """
    
    @property
    def name(self) -> str:
        return "layer_separated"
    
    @property
    def description(self) -> str:
        return "Separates layers with visible boundaries in the grid"
    
    def transform(self, tensors: dict) -> np.ndarray:
        if not tensors:
            return np.zeros((1, 1))
        
        # Calculate sizes and add padding
        layer_grids = []
        for name, arr in tensors.items():
            if arr is None:
                continue
            
            flat = arr.flatten()
            size = int(np.ceil(np.sqrt(len(flat))))
            layer_grid = np.zeros((size + 2, size + 2))  # +2 for border
            
            # Fill layer grid
            for i, val in enumerate(flat[:size*size]):
                row = (i // size) + 1
                col = (i % size) + 1
                layer_grid[row, col] = val
            
            layer_grids.append(layer_grid)
        
        if not layer_grids:
            return np.zeros((1, 1))
        
        # Arrange in grid
        n_layers = len(layer_grids)
        cols = int(np.ceil(np.sqrt(n_layers)))
        rows = int(np.ceil(n_layers / cols))
        
        max_h = max(g.shape[0] for g in layer_grids)
        max_w = max(g.shape[1] for g in layer_grids)
        
        final_grid = np.zeros((rows * max_h, cols * max_w))
        
        for idx, layer_grid in enumerate(layer_grids):
            row_idx = idx // cols
            col_idx = idx % cols
            
            start_row = row_idx * max_h
            start_col = col_idx * max_w
            
            h, w = layer_grid.shape
            final_grid[start_row:start_row+h, start_col:start_col+w] = layer_grid
        
        return final_grid
