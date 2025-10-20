#!/usr/bin/env python3
"""
Demo script showing torch2grid capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch2grid.loader import load_torch_model
from torch2grid.inspector import inspect_torch_object
from torch2grid.transformer import to_neutral_grid
from torch2grid.visualizer import visualize_grid
from torch2grid.layer_visualizer import visualize_layers, create_layer_overview
from torch2grid.histogram import visualize_all_histograms, create_histogram_overview
from torch2grid.conv_visualizer import visualize_all_conv_layers
from torch2grid.dead_neuron_detector import detect_dead_neurons, print_dead_neuron_report
from torch2grid.plugins.registry import get_registry


def create_sample_model():
    """Create a sample CNN model for demonstration."""
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model


def main():
    """Run the demo."""
    print("torch2grid Demo")
    print("=" * 50)
    
    # Create a sample model
    print("Creating sample CNN model...")
    model = create_sample_model()
    
    # Generate some dummy data and do a forward pass to initialize weights
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        _ = model(x)
    
    # Extract tensors
    print("Extracting model tensors...")
    tensors = inspect_torch_object(model)
    print(f"Found {len(tensors)} parameter tensors")
    
    # Show available plugins
    print("\nAvailable transformer plugins:")
    registry = get_registry()
    for name in registry.list_plugins():
        desc = registry.get_plugin_info(name)
        print(f"  - {name}: {desc}")
    
    # Create output directory
    output_dir = "demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic grid visualization
    print(f"\n1. Creating basic grid visualization...")
    grid = to_neutral_grid(tensors)
    visualize_grid(grid, title="Demo Model - Basic Grid", output_dir=output_dir)
    
    # 2. Layer-by-layer visualization
    print("2. Creating layer-by-layer visualizations...")
    visualize_layers(tensors, output_dir=f"{output_dir}/layers")
    create_layer_overview(tensors, output_path=f"{output_dir}/layer_overview.png")
    
    # 3. Histogram analysis
    print("3. Creating weight distribution histograms...")
    visualize_all_histograms(tensors, output_dir=f"{output_dir}/histograms")
    create_histogram_overview(tensors, output_path=f"{output_dir}/histogram_overview.png")
    
    # 4. Convolution kernel visualization
    print("4. Visualizing convolution kernels...")
    visualize_all_conv_layers(tensors, output_dir=f"{output_dir}/conv_kernels")
    
    # 5. Dead neuron detection
    print("5. Detecting dead neurons...")
    report = detect_dead_neurons(tensors)
    print_dead_neuron_report(report)
    
    # 6. Try different transformer plugins
    print("6. Testing different transformer plugins...")
    for plugin_name in ['spiral', 'normalized', 'layer_separated']:
        plugin = registry.get(plugin_name)
        if plugin:
            print(f"   Testing {plugin_name} transformer...")
            grid = plugin(tensors)
            visualize_grid(grid, title=f"Demo Model - {plugin_name.title()}", 
                          output_dir=f"{output_dir}/{plugin_name}")
    
    print(f"\nDemo complete! Check the '{output_dir}' directory for outputs.")
    print("Generated files:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.png', '.svg', '.pdf')):
                print(f"  - {os.path.join(root, file)}")


if __name__ == "__main__":
    main()