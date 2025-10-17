import sys
from torch2grid.loader import load_torch_model
from torch2grid.inspector import inspect_torch_object
from torch2grid.transformer import to_neutral_grid
from torch2grid.visualizer import visualize_grid
from torch2grid.layer_visualizer import visualize_layers, create_layer_overview


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m torch2grid <model.pt|model.pth|model.pkl> [--layers]")
        print("Options:")
        print("  --layers    Visualize each layer separately")
        return
    
    path = sys.argv[1]
    show_layers = "--layers" in sys.argv

    obj = load_torch_model(path)
    tensors = inspect_torch_object(obj)
    
    if show_layers:
        visualize_layers(tensors)
        create_layer_overview(tensors)
    else:
        grid = to_neutral_grid(tensors)
        visualize_grid(grid)


if __name__ == "__main__":
    main()