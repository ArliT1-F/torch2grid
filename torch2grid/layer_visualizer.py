import os
import re
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def visualize_layers(tensors, output_dir="grids/layers", show=False):
    """
    Visualize each layer's weights as a separate grid.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        output_dir: Directory to save layer visualizations
        show: Whether to display plots interactively
        
    Returns:
        List of paths to saved visualizations
    """
    try:
        import torch
        if isinstance(tensors, dict):
            for key, value in tensors.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for name, arr in tensors.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue
            
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_").lower()
        save_path = os.path.join(output_dir, f"{safe_name}.png")
        
        # Reshape to 2D if needed
        if arr.ndim == 1:
            size = int(np.ceil(np.sqrt(len(arr))))
            padded = np.zeros(size * size)
            padded[:len(arr)] = arr
            grid = padded.reshape(size, size)
        elif arr.ndim == 2:
            grid = arr
        elif arr.ndim == 3:
            # Conv kernels: flatten across channels
            grid = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 4:
            # Conv kernels: flatten to 2D
            grid = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2] * arr.shape[3])
        else:
            # Higher dimensions: just flatten to square
            flat = arr.flatten()
            size = int(np.ceil(np.sqrt(len(flat))))
            padded = np.zeros(size * size)
            padded[:len(flat)] = flat
            grid = padded.reshape(size, size)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(grid, cmap="viridis", interpolation="nearest", aspect="auto")
        plt.title(f"{name}\nShape: {arr.shape}")
        plt.colorbar(label="Weight magnitude")
        plt.tight_layout()
        
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        if show:
            plt.show()
        plt.close()
        
        saved_paths.append(save_path)
        print(f"Saved layer visualization: {os.path.abspath(save_path)}")
    
    return saved_paths


def create_layer_overview(tensors, output_path="grids/layer_overview.png", show=False):
    """
    Create a grid overview showing all layers in subplots.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        output_path: Path to save the overview image
        show: Whether to display plot interactively
        
    Returns:
        Path to saved overview
    """
    try:
        import torch
        if isinstance(tensors, dict):
            for key, value in tensors.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    valid_tensors = {name: arr for name, arr in tensors.items() 
                     if arr is not None and isinstance(arr, np.ndarray)}
    
    if not valid_tensors:
        print("No valid tensors to visualize")
        return None
    
    n_layers = len(valid_tensors)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, arr) in enumerate(valid_tensors.items()):
        ax = axes[idx]
        
        # Reshape to 2D
        if arr.ndim == 1:
            size = int(np.ceil(np.sqrt(len(arr))))
            padded = np.zeros(size * size)
            padded[:len(arr)] = arr
            grid = padded.reshape(size, size)
        elif arr.ndim == 2:
            grid = arr
        else:
            flat = arr.flatten()
            size = int(np.ceil(np.sqrt(len(flat))))
            padded = np.zeros(size * size)
            padded[:len(flat)] = flat
            grid = padded.reshape(size, size)
        
        im = ax.imshow(grid, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title(f"{name}\n{arr.shape}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Layer-by-Layer Overview", fontsize=14, y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved layer overview: {os.path.abspath(output_path)}")
    return output_path
