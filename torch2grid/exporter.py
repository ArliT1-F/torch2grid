"""
Export visualizations to multiple formats.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


def export_to_svg(fig, output_path):
    """
    Export matplotlib figure to SVG format.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save SVG file
        
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Exported to SVG: {os.path.abspath(output_path)}")
    return output_path


def export_to_pdf(fig, output_path):
    """
    Export matplotlib figure to PDF format.
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save PDF file
        
    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Exported to PDF: {os.path.abspath(output_path)}")
    return output_path


def export_grid_multi_format(grid, title="Neural Grid", output_dir="grids", 
                             formats=['png', 'svg', 'pdf']):
    """
    Export a grid visualization to multiple formats.
    
    Args:
        grid: 2D numpy array
        title: Title for the visualization
        output_dir: Directory to save files
        formats: List of formats to export ('png', 'svg', 'pdf')
        
    Returns:
        Dictionary mapping format to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="viridis", interpolation="nearest")
    plt.title(title)
    plt.colorbar(label="Weight magnitude")
    plt.tight_layout()
    
    saved_paths = {}
    base_name = title.lower().replace(' ', '_')
    
    for fmt in formats:
        output_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        
        if fmt == 'svg':
            export_to_svg(fig, output_path)
        elif fmt == 'pdf':
            export_to_pdf(fig, output_path)
        elif fmt == 'png':
            fig.savefig(output_path, format='png', bbox_inches='tight', dpi=120)
            print(f"Exported to PNG: {os.path.abspath(output_path)}")
        else:
            print(f"Warning: Unsupported format '{fmt}'")
            continue
        
        saved_paths[fmt] = output_path
    
    plt.close(fig)
    return saved_paths


def export_layers_to_pdf(tensors, output_path="grids/layers_report.pdf"):
    """
    Export all layers to a single multi-page PDF report.
    
    Args:
        tensors: Dictionary of layer names to numpy arrays
        output_path: Path to save PDF file
        
    Returns:
        Path to saved PDF
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
    
    with PdfPages(output_path) as pdf:
        for name, arr in tensors.items():
            if arr is None or not isinstance(arr, np.ndarray):
                continue
            
            # Create visualization
            fig = plt.figure(figsize=(8, 6))
            
            # Reshape to 2D if needed
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
            
            plt.imshow(grid, cmap='viridis', interpolation='nearest', aspect='auto')
            plt.title(f"{name}\nShape: {arr.shape}")
            plt.colorbar(label='Weight magnitude')
            
            # Add statistics
            stats_text = f'Mean: {np.mean(arr):.4f}\n'
            stats_text += f'Std: {np.std(arr):.4f}\n'
            stats_text += f'Min: {np.min(arr):.4f}\n'
            stats_text += f'Max: {np.max(arr):.4f}'
            
            plt.text(0.02, 0.02, stats_text, transform=fig.transFigure,
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = 'Neural Network Layers Report'
        d['Author'] = 'torch2grid'
        d['Subject'] = 'Layer-by-layer visualization'
        d['Keywords'] = 'PyTorch, Neural Network, Visualization'
    
    print(f"Exported {len(tensors)} layers to PDF: {os.path.abspath(output_path)}")
    return output_path


def get_supported_formats():
    """
    Get list of supported export formats.
    
    Returns:
        List of format strings
    """
    return ['png', 'svg', 'pdf']


def validate_format(format_str):
    """
    Check if a format is supported.
    
    Args:
        format_str: Format string to validate
        
    Returns:
        Boolean indicating if format is supported
    """
    return format_str.lower() in get_supported_formats()
