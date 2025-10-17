import os
import re
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def is_conv_layer(name, tensor):
    """
    Determine if a tensor represents a convolution kernel.
    
    Args:
        name: Layer name
        tensor: Numpy array
        
    Returns:
        Boolean indicating if this is a conv layer
    """
    if tensor is None or not isinstance(tensor, np.ndarray):
        return False
    
    # Conv kernels typically have 4 dimensions: (out_channels, in_channels, height, width)
    # or 3 dimensions for 1D conv: (out_channels, in_channels, width)
    if tensor.ndim in [3, 4]:
        # Also check naming patterns
        if any(pattern in name.lower() for pattern in ['conv', 'kernel']):
            return True
        # Check if it's a weight tensor with conv-like dimensions
        if 'weight' in name.lower() and tensor.ndim == 4 and tensor.shape[2] == tensor.shape[3]:
            return True
    return False


def visualize_conv_kernels(tensors, layer_name, output_dir="grids/conv_kernels", 
                           max_kernels=64, show=False):
    """
    Visualize convolution kernels for a specific layer.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        layer_name: Name of the conv layer to visualize
        output_dir: Directory to save visualization
        max_kernels: Maximum number of kernels to display
        show: Whether to display plot interactively
        
    Returns:
        Path to saved visualization
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
    
    if not is_conv_layer(layer_name, arr):
        print(f"Layer '{layer_name}' does not appear to be a convolution layer")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", layer_name).strip("_").lower()
    save_path = os.path.join(output_dir, f"{safe_name}_kernels.png")
    
    # Handle different dimensions
    if arr.ndim == 4:
        # 2D convolution: (out_channels, in_channels, height, width)
        out_channels, in_channels, kh, kw = arr.shape
        kernel_size = (kh, kw)
    elif arr.ndim == 3:
        # 1D convolution: (out_channels, in_channels, width)
        out_channels, in_channels, kw = arr.shape
        kernel_size = (1, kw)
        arr = arr[:, :, np.newaxis, :]
    else:
        print(f"Unsupported kernel dimensions: {arr.shape}")
        return None
    
    # Limit number of kernels to visualize
    num_kernels = min(out_channels, max_kernels)
    
    # Create visualization
    # For each output channel, we'll show a grid of input channel kernels
    cols = min(8, in_channels)
    rows_per_kernel = (in_channels + cols - 1) // cols
    total_rows = num_kernels * rows_per_kernel
    
    fig = plt.figure(figsize=(cols * 1.5, total_rows * 1.5))
    
    for out_idx in range(num_kernels):
        for in_idx in range(in_channels):
            subplot_idx = out_idx * in_channels + in_idx + 1
            
            if subplot_idx > num_kernels * in_channels:
                break
                
            ax = plt.subplot(num_kernels, in_channels, subplot_idx)
            
            # Get the kernel for this output-input channel pair
            kernel = arr[out_idx, in_idx]
            
            # Normalize for visualization
            vmin, vmax = kernel.min(), kernel.max()
            if vmax - vmin > 1e-6:
                kernel_norm = (kernel - vmin) / (vmax - vmin)
            else:
                kernel_norm = kernel
            
            # Display kernel
            ax.imshow(kernel_norm, cmap='viridis', interpolation='nearest')
            ax.axis('off')
            
            # Add title for first row
            if out_idx == 0:
                ax.set_title(f'In:{in_idx}', fontsize=6)
        
        # Add output channel label
        ax = plt.subplot(num_kernels, in_channels, out_idx * in_channels + 1)
        ax.text(-0.5, 0.5, f'Out {out_idx}', rotation=90, 
                verticalalignment='center', fontsize=8,
                transform=ax.transAxes)
    
    plt.suptitle(f'Convolution Kernels: {layer_name}\n'
                 f'Shape: {tensors[layer_name].shape} | '
                 f'Kernel Size: {kernel_size[0]}x{kernel_size[1]}',
                 fontsize=11)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved conv kernel visualization: {os.path.abspath(save_path)}")
    return save_path


def visualize_conv_kernels_grid(tensors, layer_name, output_dir="grids/conv_kernels",
                                max_kernels=64, channels_per_kernel=1, show=False):
    """
    Visualize convolution kernels in a simplified grid layout.
    Shows kernels as tiles, averaging across input channels if needed.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        layer_name: Name of the conv layer to visualize
        output_dir: Directory to save visualization
        max_kernels: Maximum number of kernels to display
        channels_per_kernel: How many input channels to show (1 = average all)
        show: Whether to display plot interactively
        
    Returns:
        Path to saved visualization
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
    
    if not is_conv_layer(layer_name, arr):
        print(f"Layer '{layer_name}' does not appear to be a convolution layer")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", layer_name).strip("_").lower()
    save_path = os.path.join(output_dir, f"{safe_name}_grid.png")
    
    # Handle different dimensions
    if arr.ndim == 4:
        out_channels, in_channels, kh, kw = arr.shape
    elif arr.ndim == 3:
        out_channels, in_channels, kw = arr.shape
        kh = 1
        arr = arr[:, :, np.newaxis, :]
    else:
        print(f"Unsupported kernel dimensions: {arr.shape}")
        return None
    
    # Limit number of kernels
    num_kernels = min(out_channels, max_kernels)
    
    # Average across input channels or select first few
    if channels_per_kernel == 1:
        # Average across all input channels
        kernels_to_show = np.mean(arr[:num_kernels], axis=1)
    else:
        # Show first channel
        kernels_to_show = arr[:num_kernels, 0]
    
    # Arrange in grid
    cols = int(np.ceil(np.sqrt(num_kernels)))
    rows = (num_kernels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if num_kernels == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx in range(num_kernels):
        ax = axes[idx]
        kernel = kernels_to_show[idx]
        
        # Normalize
        vmin, vmax = kernel.min(), kernel.max()
        if vmax - vmin > 1e-6:
            kernel_norm = (kernel - vmin) / (vmax - vmin)
        else:
            kernel_norm = kernel
        
        im = ax.imshow(kernel_norm, cmap='viridis', interpolation='nearest')
        ax.set_title(f'Filter {idx}', fontsize=8)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_kernels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Convolution Filters: {layer_name}\n'
                 f'{out_channels} filters, {in_channels} channels, '
                 f'{kh}x{kw} kernel',
                 fontsize=11)
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved conv kernel grid: {os.path.abspath(save_path)}")
    return save_path


def visualize_all_conv_layers(tensors, output_dir="grids/conv_kernels", 
                              max_kernels=64, show=False):
    """
    Visualize all convolution layers in the model.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        output_dir: Directory to save visualizations
        max_kernels: Maximum number of kernels per layer
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
    
    saved_paths = []
    conv_layers = []
    
    # Find all conv layers
    for name, arr in tensors.items():
        if is_conv_layer(name, arr):
            conv_layers.append(name)
    
    if not conv_layers:
        print("No convolution layers found in model")
        return saved_paths
    
    print(f"\nFound {len(conv_layers)} convolution layer(s):")
    for name in conv_layers:
        print(f"  - {name}: {tensors[name].shape}")
    
    # Visualize each conv layer
    for name in conv_layers:
        path = visualize_conv_kernels_grid(tensors, name, output_dir, max_kernels, show=show)
        if path:
            saved_paths.append(path)
    
    return saved_paths
