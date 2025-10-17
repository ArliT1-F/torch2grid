import os
import re
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def visualize_weight_histogram(tensors, layer_name, output_dir="grids/histograms", bins=50, show=False):
    """
    Create a histogram for a single layer's weight distribution.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        layer_name: Name of the layer to visualize
        output_dir: Directory to save histogram
        bins: Number of bins for histogram
        show: Whether to display plot interactively
        
    Returns:
        Path to saved histogram
    """
    try:
        import torch
        if isinstance(tensors, dict):
            for key, value in tensors.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    if layer_name not in tensors:
        print(f"Layer '{layer_name}' not found")
        return None
    
    arr = tensors[layer_name]
    if arr is None or not isinstance(arr, np.ndarray):
        print(f"Invalid tensor for layer '{layer_name}'")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", layer_name).strip("_").lower()
    save_path = os.path.join(output_dir, f"{safe_name}_hist.png")
    
    weights = arr.flatten()
    
    # Calculate statistics
    mean_val = np.mean(weights)
    std_val = np.std(weights)
    min_val = np.min(weights)
    max_val = np.max(weights)
    zero_pct = (np.abs(weights) < 1e-6).sum() / len(weights) * 100
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n, bins_edges, patches = plt.hist(weights, bins=bins, color='steelblue', 
                                       edgecolor='black', alpha=0.7)
    
    # Add vertical line for mean
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Weight Distribution: {layer_name}\nShape: {arr.shape}', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Mean: {mean_val:.4f}\n'
    stats_text += f'Std: {std_val:.4f}\n'
    stats_text += f'Min: {min_val:.4f}\n'
    stats_text += f'Max: {max_val:.4f}\n'
    stats_text += f'Near-zero: {zero_pct:.2f}%'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved histogram: {os.path.abspath(save_path)}")
    return save_path


def visualize_all_histograms(tensors, output_dir="grids/histograms", bins=50, show=False):
    """
    Create histograms for all layers.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        output_dir: Directory to save histograms
        bins: Number of bins for histogram
        show: Whether to display plots interactively
        
    Returns:
        List of paths to saved histograms
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
    
    for name in tensors.keys():
        path = visualize_weight_histogram(tensors, name, output_dir, bins, show)
        if path:
            saved_paths.append(path)
    
    return saved_paths


def create_histogram_overview(tensors, output_path="grids/histogram_overview.png", bins=30, show=False):
    """
    Create a multi-panel histogram overview showing all layers.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        output_path: Path to save the overview image
        bins: Number of bins for histograms
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
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (name, arr) in enumerate(valid_tensors.items()):
        ax = axes[idx]
        
        weights = arr.flatten()
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        
        # Create histogram
        ax.hist(weights, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                   label=f'μ={mean_val:.3f}')
        ax.axvline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        
        ax.set_title(f'{name}\n{arr.shape}', fontsize=9)
        ax.set_xlabel('Weight Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add std as text
        ax.text(0.98, 0.98, f'σ={std_val:.3f}', transform=ax.transAxes,
                fontsize=7, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle("Weight Distribution Overview", fontsize=14, y=0.995)
    plt.tight_layout()
    
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved histogram overview: {os.path.abspath(output_path)}")
    return output_path


def compare_layer_statistics(tensors):
    """
    Print statistical comparison of all layers.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
    """
    try:
        import torch
        if isinstance(tensors, dict):
            for key, value in tensors.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    print("\n" + "="*80)
    print("Layer Statistics Comparison")
    print("="*80)
    print(f"{'Layer Name':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'~Zero%':>8}")
    print("-"*80)
    
    for name, arr in tensors.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue
        
        weights = arr.flatten()
        mean_val = np.mean(weights)
        std_val = np.std(weights)
        min_val = np.min(weights)
        max_val = np.max(weights)
        zero_pct = (np.abs(weights) < 1e-6).sum() / len(weights) * 100
        
        print(f"{name:<40} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {zero_pct:>7.2f}%")
    
    print("="*80 + "\n")
