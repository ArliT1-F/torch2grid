import sys
from torch2grid.loader import load_torch_model
from torch2grid.inspector import inspect_torch_object
from torch2grid.transformer import to_neutral_grid
from torch2grid.visualizer import visualize_grid
from torch2grid.layer_visualizer import visualize_layers, create_layer_overview
from torch2grid.interactive import interactive_mode


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m torch2grid <model.pt|model.pth|model.pkl> [options]")
        print("\nOptions:")
        print("  --layers        Visualize each layer separately")
        print("  --interactive   Interactive layer selection mode")
        print("  --help          Show this help message")
        return
    
    path = sys.argv[1]
    
    if "--help" in sys.argv or "-h" in sys.argv:
        print("torch2grid - PyTorch Model Visualization Tool")
        print("\nUsage: python -m torch2grid <model.pt|model.pth|model.pkl> [options]")
        print("\nOptions:")
        print("  --layers        Visualize each layer separately and create overview")
        print("  --interactive   Interactive mode for selecting specific layers")
        print("  --help, -h      Show this help message")
        print("\nExamples:")
        print("  python -m torch2grid model.pth")
        print("  python -m torch2grid model.pth --layers")
        print("  python -m torch2grid model.pth --interactive")
        return

    obj = load_torch_model(path)
    tensors = inspect_torch_object(obj)
    
    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode(tensors)
    elif "--layers" in sys.argv:
        visualize_layers(tensors)
        create_layer_overview(tensors)
    else:
        grid = to_neutral_grid(tensors)
        visualize_grid(grid)


if __name__ == "__main__":
    main()