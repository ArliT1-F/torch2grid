"""
Gradient visualization support.
"""

import os
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def visualize_gradients(gradients, title="Gradient Magnitudes", output_dir="grids/gradients", show=False):
    """
    Visualize gradients as a grid similar to weights.
    
    Args:
        gradients: Dictionary of layer names to gradient arrays
        title: Title for visualization
        output_dir: Directory to save visualization
        show: Whether to display plot interactively
        
    Returns:
        Path to saved visualization
    """
    try:
        import torch
        if isinstance(gradients, dict):
            for key, value in gradients.items():
                if isinstance(value, torch.Tensor):
                    gradients[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten all gradients
    flat_values = []
    for name, arr in gradients.items():
        if arr is None:
            continue
        flat_values.extend(arr.flatten())
    
    if not flat_values:
        print("No gradients to visualize")
        return None
    
    # Create grid
    size = int(np.ceil(np.sqrt(len(flat_values))))
    grid = np.zeros((size, size))
    for i, val in enumerate(flat_values[:size*size]):
        grid[i // size, i % size] = val
    
    # Visualize
    save_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='RdBu_r', interpolation='nearest')
    plt.title(title)
    plt.colorbar(label='Gradient magnitude')
    plt.tight_layout()
    
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved gradient visualization: {os.path.abspath(save_path)}")
    return save_path


def visualize_gradient_flow(gradients, output_path="grids/gradient_flow.png", show=False):
    """
    Visualize gradient flow across layers (useful for detecting vanishing/exploding gradients).
    
    Args:
        gradients: Dictionary of layer names to gradient arrays
        output_path: Path to save visualization
        show: Whether to display plot interactively
        
    Returns:
        Path to saved visualization
    """
    try:
        import torch
        if isinstance(gradients, dict):
            for key, value in gradients.items():
                if isinstance(value, torch.Tensor):
                    gradients[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if not gradients:
        print("No gradients to visualize")
        return None
    
    # Compute statistics per layer
    layer_names = []
    means = []
    stds = []
    maxs = []
    mins = []
    
    for name, arr in gradients.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue
        
        layer_names.append(name)
        flat = arr.flatten()
        means.append(np.mean(np.abs(flat)))
        stds.append(np.std(flat))
        maxs.append(np.max(np.abs(flat)))
        mins.append(np.min(np.abs(flat)))
    
    if not layer_names:
        print("No valid gradient data")
        return None
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    x = np.arange(len(layer_names))
    
    # Plot 1: Mean absolute gradients with error bars
    ax1.bar(x, means, alpha=0.7, label='Mean |gradient|', color='steelblue')
    ax1.errorbar(x, means, yerr=stds, fmt='none', ecolor='red', 
                 capsize=3, label='Std deviation')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name[:20] for name in layer_names], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Mean Absolute Gradient', fontsize=10)
    ax1.set_title('Gradient Flow Across Layers', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')
    
    # Add warning lines for vanishing/exploding gradients
    if means:
        mean_grad = np.mean(means)
        ax1.axhline(y=mean_grad * 10, color='r', linestyle='--', 
                   linewidth=1, alpha=0.5, label='Exploding threshold')
        ax1.axhline(y=mean_grad * 0.1, color='orange', linestyle='--', 
                   linewidth=1, alpha=0.5, label='Vanishing threshold')
    
    # Plot 2: Min and Max gradients
    ax2.plot(x, maxs, 'o-', label='Max |gradient|', color='red', linewidth=2)
    ax2.plot(x, mins, 's-', label='Min |gradient|', color='blue', linewidth=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:20] for name in layer_names], rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Gradient Magnitude', fontsize=10)
    ax2.set_title('Gradient Range by Layer', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved gradient flow visualization: {os.path.abspath(output_path)}")
    return output_path


def analyze_gradient_health(gradients, vanishing_threshold=1e-7, exploding_threshold=1e2):
    """
    Analyze gradient health and detect vanishing/exploding gradients.
    
    Args:
        gradients: Dictionary of layer names to gradient arrays
        vanishing_threshold: Threshold below which gradients are considered vanishing
        exploding_threshold: Threshold above which gradients are considered exploding
        
    Returns:
        Dictionary with analysis results
    """
    try:
        import torch
        if isinstance(gradients, dict):
            for key, value in gradients.items():
                if isinstance(value, torch.Tensor):
                    gradients[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    results = {
        'layers': {},
        'vanishing': [],
        'exploding': [],
        'healthy': []
    }
    
    for name, arr in gradients.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue
        
        flat = arr.flatten()
        mean_abs_grad = np.mean(np.abs(flat))
        max_abs_grad = np.max(np.abs(flat))
        
        layer_info = {
            'mean': float(mean_abs_grad),
            'max': float(max_abs_grad),
            'std': float(np.std(flat)),
            'status': 'healthy'
        }
        
        if mean_abs_grad < vanishing_threshold:
            layer_info['status'] = 'vanishing'
            results['vanishing'].append(name)
        elif max_abs_grad > exploding_threshold:
            layer_info['status'] = 'exploding'
            results['exploding'].append(name)
        else:
            results['healthy'].append(name)
        
        results['layers'][name] = layer_info
    
    return results


def print_gradient_health_report(analysis):
    """
    Print gradient health analysis report.
    
    Args:
        analysis: Dictionary from analyze_gradient_health()
    """
    print("\n" + "="*80)
    print("Gradient Health Report")
    print("="*80)
    
    total = len(analysis['layers'])
    vanishing = len(analysis['vanishing'])
    exploding = len(analysis['exploding'])
    healthy = len(analysis['healthy'])
    
    print(f"Total layers: {total}")
    print(f"Healthy: {healthy} ({healthy/total*100:.1f}%)")
    print(f"Vanishing gradients: {vanishing} ({vanishing/total*100:.1f}%)")
    print(f"Exploding gradients: {exploding} ({exploding/total*100:.1f}%)")
    print("="*80)
    
    if analysis['layers']:
        print(f"\n{'Layer':<40} {'Mean':>12} {'Max':>12} {'Status':>12}")
        print("-"*80)
        
        for name, info in analysis['layers'].items():
            status_symbol = {
                'healthy': '✓',
                'vanishing': '⚠️  VANISH',
                'exploding': '⚠️  EXPLODE'
            }.get(info['status'], '')
            
            print(f"{name:<40} {info['mean']:>12.2e} {info['max']:>12.2e} {status_symbol:>12}")
        
        print("-"*80)
    
    # Warnings
    if vanishing > 0:
        print(f"\n⚠️  WARNING: {vanishing} layer(s) with vanishing gradients detected!")
        print("   Consider: gradient clipping, batch normalization, or residual connections")
    
    if exploding > 0:
        print(f"\n⚠️  WARNING: {exploding} layer(s) with exploding gradients detected!")
        print("   Consider: gradient clipping, lower learning rate, or weight initialization")
    
    if vanishing == 0 and exploding == 0:
        print("\n✓ All gradients are healthy.")
    
    print()


def compare_weights_and_gradients(weights, gradients, output_path="grids/weight_grad_comparison.png", show=False):
    """
    Create side-by-side comparison of weights and gradients.
    
    Args:
        weights: Dictionary of layer names to weight arrays
        gradients: Dictionary of layer names to gradient arrays
        output_path: Path to save visualization
        show: Whether to display plot interactively
        
    Returns:
        Path to saved visualization
    """
    try:
        import torch
        for tensors in [weights, gradients]:
            if isinstance(tensors, dict):
                for key, value in tensors.items():
                    if isinstance(value, torch.Tensor):
                        tensors[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Find common layers
    common_layers = set(weights.keys()) & set(gradients.keys())
    
    if not common_layers:
        print("No common layers between weights and gradients")
        return None
    
    n_layers = len(common_layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(10, 4*n_layers))
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for idx, layer_name in enumerate(sorted(common_layers)):
        weight = weights[layer_name]
        grad = gradients[layer_name]
        
        # Visualize weight
        ax_w = axes[idx, 0]
        if weight.ndim == 1:
            weight_vis = weight.reshape(-1, 1)
        elif weight.ndim > 2:
            weight_vis = weight.reshape(weight.shape[0], -1)
        else:
            weight_vis = weight
        
        im1 = ax_w.imshow(weight_vis, cmap='viridis', aspect='auto', interpolation='nearest')
        ax_w.set_title(f'{layer_name}\nWeights', fontsize=9)
        ax_w.axis('off')
        plt.colorbar(im1, ax=ax_w, fraction=0.046)
        
        # Visualize gradient
        ax_g = axes[idx, 1]
        if grad.ndim == 1:
            grad_vis = grad.reshape(-1, 1)
        elif grad.ndim > 2:
            grad_vis = grad.reshape(grad.shape[0], -1)
        else:
            grad_vis = grad
        
        im2 = ax_g.imshow(grad_vis, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        ax_g.set_title(f'{layer_name}\nGradients', fontsize=9)
        ax_g.axis('off')
        plt.colorbar(im2, ax=ax_g, fraction=0.046)
    
    plt.suptitle('Weights vs Gradients Comparison', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    if show:
        plt.show()
    plt.close()
    
    print(f"Saved weight-gradient comparison: {os.path.abspath(output_path)}")
    return output_path
