import sys
from torch2grid.loader import load_torch_model
from torch2grid.inspector import inspect_torch_object
from torch2grid.transformer import to_neutral_grid
from torch2grid.visualizer import visualize_grid


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m torch2grid <model.pt|model.pth|model.pkl")
        return
    
    path = sys.argv[1]

    obj = load_torch_model(path)
    tensors = inspect_torch_object(obj)
    grid = to_neutral_grid(tensors)
    visualize_grid(grid)


if __name__ == "__main__":
    main()