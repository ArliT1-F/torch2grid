import numpy as np


def to_neutral_grid(tensors):
    # Flattens all tensors into one big 2D grid (like a connectivity heatmap)
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