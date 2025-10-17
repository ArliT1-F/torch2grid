import sys
from torch2grid.loader import load_torch_model
from torch2grid.inspector import inspect_torch_object
from torch2grid.transformer import to_neutral_grid
from torch2grid.visualizer import visualize_grid
from torch2grid.layer_visualizer import visualize_layers, create_layer_overview
from torch2grid.interactive import interactive_mode
from torch2grid.histogram import (
    visualize_all_histograms,
    create_histogram_overview,
    compare_layer_statistics
)
from torch2grid.conv_visualizer import visualize_all_conv_layers


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m torch2grid <model.pt|model.pth|model.pkl> [options]")
        print("\nOptions:")
        print("  --layers        Visualize each layer separately")
        print("  --histogram     Generate weight distribution histograms")
        print("  --conv          Visualize convolution kernels")
        print("  --stats         Print layer statistics comparison")
        print("  --interactive   Interactive layer selection mode")
        print("  --help          Show this help message")
        return
    
    path = sys.argv[1]
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("torch2grid - PyTorch Model Visualization Tool")
        print("\nUsage: python -m torch2grid <model.pt|model.pth|model.pkl> [options]")
        print("\nOptions:")
        print("  --layers        Visualize each layer separately and create overview")
        print("  --histogram     Generate weight distribution histograms for all layers")
        print("  --conv          Visualize convolution kernels (filters)")
        print("  --stats         Print statistical comparison of all layers")
        print("  --interactive   Interactive mode for selecting specific layers")
        print("  --help, -h      Show this help message")
        print("\nExamples:")
        print("  python -m torch2grid model.pth")
        print("  python -m torch2grid model.pth --layers")
        print("  python -m torch2grid model.pth --histogram")
        print("  python -m torch2grid model.pth --conv")
        print("  python -m torch2grid model.pth --stats")
        print("  python -m torch2grid model.pth --interactive")
        print("  python -m torch2grid model.pth --layers --histogram --conv --stats")
        return

    obj = load_torch_model(path)
    tensors = inspect_torch_object(obj)
    
    # Handle stats flag (can be combined with other flags)
    if "--stats" in sys.argv:
        compare_layer_statistics(tensors)
    
    # Handle conv flag (can be combined with other flags)
    if "--conv" in sys.argv:
        visualize_all_conv_layers(tensors)
    
    # Handle primary visualization modes
    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode(tensors)
    elif "--histogram" in sys.argv:
        visualize_all_histograms(tensors)
        create_histogram_overview(tensors)
        if "--layers" in sys.argv:
            visualize_layers(tensors)
            create_layer_overview(tensors)
    elif "--layers" in sys.argv:
        visualize_layers(tensors)
        create_layer_overview(tensors)
    else:
        grid = to_neutral_grid(tensors)
        visualize_grid(grid)


if __name__ == "__main__":
    main()