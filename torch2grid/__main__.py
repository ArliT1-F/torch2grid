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
from torch2grid.plugins.registry import get_registry


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m torch2grid <model.pt|model.pth|model.pkl> [options]")
        print("\nOptions:")
        print("  --layers        Visualize each layer separately")
        print("  --histogram     Generate weight distribution histograms")
        print("  --conv          Visualize convolution kernels")
        print("  --stats         Print layer statistics comparison")
        print("  --interactive   Interactive layer selection mode")
        print("  --plugin NAME   Use specific transformer plugin")
        print("  --list-plugins  List available plugins")
        print("  --load-plugin FILE  Load custom plugin from file")
        print("  --help          Show this help message")
        return
    
    path = sys.argv[1]
    
    # Handle plugin listing
    if "--list-plugins" in sys.argv:
        registry = get_registry()
        plugins = registry.list_plugins()
        print("\nAvailable transformer plugins:")
        print("-" * 60)
        for name in plugins:
            desc = registry.get_plugin_info(name)
            print(f"  {name:<20} {desc}")
        print("-" * 60)
        return
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("torch2grid - PyTorch Model Visualization Tool")
        print("\nUsage: python -m torch2grid <model.pt|model.pth|model.pkl> [options]")
        print("\nOptions:")
        print("  --layers        Visualize each layer separately and create overview")
        print("  --histogram     Generate weight distribution histograms for all layers")
        print("  --conv          Visualize convolution kernels (filters)")
        print("  --stats         Print statistical comparison of all layers")
        print("  --interactive   Interactive mode for selecting specific layers")
        print("  --plugin NAME   Use specific transformer plugin (e.g., spiral, normalized)")
        print("  --list-plugins  List all available transformer plugins")
        print("  --load-plugin FILE  Load custom plugin from Python file")
        print("  --help, -h      Show this help message")
        print("\nExamples:")
        print("  python -m torch2grid model.pth")
        print("  python -m torch2grid model.pth --layers")
        print("  python -m torch2grid model.pth --histogram")
        print("  python -m torch2grid model.pth --conv")
        print("  python -m torch2grid model.pth --stats")
        print("  python -m torch2grid model.pth --plugin spiral")
        print("  python -m torch2grid model.pth --load-plugin my_plugin.py")
        print("  python -m torch2grid model.pth --list-plugins")
        print("  python -m torch2grid model.pth --interactive")
        print("  python -m torch2grid model.pth --layers --histogram --conv --stats")
        return
    
    # Load custom plugins if specified
    registry = get_registry()
    for i, arg in enumerate(sys.argv):
        if arg == "--load-plugin" and i + 1 < len(sys.argv):
            plugin_file = sys.argv[i + 1]
            try:
                registry.load_from_file(plugin_file)
            except Exception as e:
                print(f"Error loading plugin from {plugin_file}: {e}")

    obj = load_torch_model(path)
    tensors = inspect_torch_object(obj)
    
    # Get plugin name if specified
    plugin_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--plugin" and i + 1 < len(sys.argv):
            plugin_name = sys.argv[i + 1]
            break
    
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
        grid = to_neutral_grid(tensors, plugin_name=plugin_name)
        title = f"Neural Grid ({plugin_name})" if plugin_name else "Neural Grid"
        visualize_grid(grid, title=title)


if __name__ == "__main__":
    main()