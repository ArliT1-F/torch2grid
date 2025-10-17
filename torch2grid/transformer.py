import numpy as np


def to_neutral_grid(tensors, plugin_name=None):
    """
    Transform tensors into a 2D grid using a transformer plugin.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        plugin_name: Name of plugin to use (default: 'flatten')
        
    Returns:
        2D numpy array
    """
    # Try to use plugin system if available
    try:
        from torch2grid.plugins.registry import get_registry
        
        registry = get_registry()
        
        if plugin_name:
            plugin = registry.get(plugin_name)
            if plugin is None:
                print(f"Warning: Plugin '{plugin_name}' not found, using default flatten")
                plugin_name = "flatten"
                plugin = registry.get(plugin_name)
        else:
            plugin_name = "flatten"
            plugin = registry.get(plugin_name)
        
        if plugin:
            return plugin(tensors)
    except ImportError:
        pass
    
    # Fallback to original implementation
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