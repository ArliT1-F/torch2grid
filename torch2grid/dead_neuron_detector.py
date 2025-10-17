"""
Dead neuron detection and reporting.
"""

import numpy as np
import os
import json


class DeadNeuronReport:
    """Container for dead neuron detection results."""
    
    def __init__(self):
        self.layers = {}
        self.total_neurons = 0
        self.total_dead = 0
        self.threshold = 0.0
    
    def add_layer(self, name, total, dead, dead_indices, threshold):
        """Add layer results."""
        self.layers[name] = {
            'total': total,
            'dead': dead,
            'dead_indices': dead_indices,
            'percentage': (dead / total * 100) if total > 0 else 0,
            'threshold': threshold
        }
        self.total_neurons += total
        self.total_dead += dead
    
    def get_summary(self):
        """Get summary statistics."""
        return {
            'total_neurons': self.total_neurons,
            'total_dead': self.total_dead,
            'percentage': (self.total_dead / self.total_neurons * 100) if self.total_neurons > 0 else 0,
            'layers_affected': sum(1 for l in self.layers.values() if l['dead'] > 0),
            'total_layers': len(self.layers)
        }
    
    def to_dict(self):
        """Convert report to dictionary."""
        return {
            'summary': self.get_summary(),
            'layers': self.layers,
            'threshold': self.threshold
        }


def detect_dead_neurons(tensors, threshold=1e-6, output_layer_only=False):
    """
    Detect dead neurons (near-zero weights) in the model.
    
    A neuron is considered "dead" if all its weights are close to zero,
    meaning it doesn't contribute to the model's output.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        threshold: Absolute value threshold for considering weights as zero
        output_layer_only: If True, only check output layers (last dimension)
        
    Returns:
        DeadNeuronReport object
    """
    try:
        import torch
        if isinstance(tensors, dict):
            for key, value in tensors.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value.detach().cpu().numpy()
    except Exception:
        pass
    
    report = DeadNeuronReport()
    report.threshold = threshold
    
    for name, arr in tensors.items():
        if arr is None or not isinstance(arr, np.ndarray):
            continue
        
        # Only check weight tensors, skip biases
        if 'bias' in name.lower():
            continue
        
        # For weight matrices, check neurons (output dimension)
        if arr.ndim == 2:
            # Shape: (out_features, in_features)
            # Check each output neuron
            dead_neurons = []
            for i in range(arr.shape[0]):
                neuron_weights = arr[i, :]
                if np.all(np.abs(neuron_weights) < threshold):
                    dead_neurons.append(i)
            
            report.add_layer(name, arr.shape[0], len(dead_neurons), dead_neurons, threshold)
        
        elif arr.ndim == 4:
            # Convolution: (out_channels, in_channels, height, width)
            # Check each output channel
            dead_neurons = []
            for i in range(arr.shape[0]):
                channel_weights = arr[i, :, :, :]
                if np.all(np.abs(channel_weights) < threshold):
                    dead_neurons.append(i)
            
            report.add_layer(name, arr.shape[0], len(dead_neurons), dead_neurons, threshold)
        
        elif arr.ndim >= 1:
            # General case: check if all weights are near zero
            total = 1  # Treat as single unit
            dead = 1 if np.all(np.abs(arr) < threshold) else 0
            report.add_layer(name, total, dead, [0] if dead else [], threshold)
    
    return report


def print_dead_neuron_report(report, verbose=False):
    """
    Print dead neuron detection report.
    
    Args:
        report: DeadNeuronReport object
        verbose: If True, show detailed per-layer information
    """
    summary = report.get_summary()
    
    print("\n" + "="*80)
    print("Dead Neuron Detection Report")
    print("="*80)
    print(f"Threshold: {report.threshold}")
    print(f"Total neurons/channels: {summary['total_neurons']}")
    print(f"Dead neurons/channels: {summary['total_dead']} ({summary['percentage']:.2f}%)")
    print(f"Layers affected: {summary['layers_affected']}/{summary['total_layers']}")
    print("="*80)
    
    if verbose or summary['layers_affected'] > 0:
        print(f"\n{'Layer':<40} {'Total':>8} {'Dead':>8} {'%':>8}")
        print("-"*80)
        
        for name, layer_info in report.layers.items():
            if verbose or layer_info['dead'] > 0:
                status = " ⚠️ " if layer_info['dead'] > 0 else ""
                print(f"{name:<40} {layer_info['total']:>8} {layer_info['dead']:>8} "
                      f"{layer_info['percentage']:>7.2f}%{status}")
        
        print("-"*80)
    
    # Warnings
    if summary['percentage'] > 50:
        print("\n⚠️  WARNING: Over 50% of neurons are dead! Model may be undertrained.")
    elif summary['percentage'] > 20:
        print("\n⚠️  WARNING: Significant number of dead neurons detected.")
    elif summary['total_dead'] > 0:
        print(f"\n✓ {summary['total_dead']} dead neuron(s) detected across {summary['layers_affected']} layer(s).")
    else:
        print("\n✓ No dead neurons detected. All neurons are active.")
    
    print()


def save_dead_neuron_report(report, output_path="grids/dead_neurons_report.json"):
    """
    Save dead neuron report to JSON file.
    
    Args:
        report: DeadNeuronReport object
        output_path: Path to save JSON file
        
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Convert to serializable format
    report_dict = report.to_dict()
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    report_dict = convert(report_dict)
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"Saved dead neuron report: {os.path.abspath(output_path)}")
    return output_path


def visualize_dead_neurons(report, output_path="grids/dead_neurons_overview.png"):
    """
    Create visualization showing dead neuron distribution across layers.
    
    Args:
        report: DeadNeuronReport object
        output_path: Path to save visualization
        
    Returns:
        Path to saved visualization
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    if not report.layers:
        print("No layers to visualize")
        return None
    
    # Prepare data
    layer_names = list(report.layers.keys())
    percentages = [report.layers[name]['percentage'] for name in layer_names]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    colors = ['red' if p > 50 else 'orange' if p > 20 else 'yellow' if p > 0 else 'green' 
              for p in percentages]
    
    y_pos = np.arange(len(layer_names))
    ax1.barh(y_pos, percentages, color=colors, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([name[:30] for name in layer_names], fontsize=8)
    ax1.set_xlabel('Dead Neurons (%)', fontsize=10)
    ax1.set_title('Dead Neurons by Layer', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Summary pie chart
    summary = report.get_summary()
    alive = summary['total_neurons'] - summary['total_dead']
    
    if summary['total_dead'] > 0:
        sizes = [alive, summary['total_dead']]
        labels = [f'Active\n({alive})', f'Dead\n({summary["total_dead"]})']
        colors_pie = ['#90EE90', '#FF6B6B']
        explode = (0, 0.1)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax2.set_title(f'Overall Distribution\n({summary["percentage"]:.2f}% Dead)', fontsize=12)
    else:
        ax2.text(0.5, 0.5, '✓ No Dead Neurons', 
                ha='center', va='center', fontsize=16, color='green')
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()
    
    print(f"Saved dead neuron visualization: {os.path.abspath(output_path)}")
    return output_path
