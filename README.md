# torch2grid
Torch2Grid is a simple tool used to visualize the map brain of an ai model.
The current file types it can support are `.pt, .pth, .pkl`
A lightweight Python tool for visualizing PyTorch model weights and architectures. torch2grid transforms neural network parameters into intuitive visual grids, making it easy to inspect model structure, debug weight distributions, and understand layer compositions.

## Features

- **Unified Grid Visualization**: Flattens all model weights into a single heatmap
- **Layer-by-Layer Visualization**: Generate separate visualizations for each layer
- **Layer Overview**: Multi-panel view showing all layers at once
- **Headless Support**: Works in environments without GUI (Codespaces, SSH, CI/CD)
- **Multiple Input Formats**: Supports `.pt`, `.pth`, `.pkl` model files and state_dict objects

## Installation

```bash
pip install -r torch2grid/requirements.txt
```

## Usage

### Basic Visualization (Single Grid)

```bash
python -m torch2grid model.pth
```

This creates a single unified grid showing all weights combined.

### Layer-by-Layer Visualization

```bash
python -m torch2grid model.pth --layers
```

This generates:
- Individual PNG files for each layer in `grids/layers/`
- A comprehensive overview showing all layers in `grids/layer_overview.png`


### Interactive Mode

```bash
python -m torch2grid model.pth --interactive
```

Interactive mode provides a CLI menu for selective layer visualization:

**Selection Methods:**
- **By numbers**: `1,3,5` or ranges `1-10` or combined `1,3-5,7`
- **By pattern**: `fc*` (all fc layers), `*weight*` (all weight tensors), `*.bias` (all biases)
- **Commands**:
  - `all` - Visualize all layers
  - `overview` - Create multi-panel overview only
  - `unified` - Create single unified grid
  - `list` - Refresh layer list
  - `quit` - Exit interactive mode

**Example Session:**
```
Available layers:
------------------------------------------------------------
  1. fc1.weight                            torch.Size([20, 10])
  2. fc1.bias                              torch.Size([20])
  3. fc2.weight                            torch.Size([5, 20])
  4. fc2.bias                              torch.Size([5])
------------------------------------------------------------

Your choice: fc*weight*
Matched 2 layer(s):
  - fc1.weight
  - fc2.weight

Visualize these layers? (y/n): y
```

### Programmatic Usage

```python
from torch2grid.loader import load_torch_model
from torch2grid.inspector import inspect_torch_object
from torch2grid.layer_visualizer import visualize_layers, create_layer_overview
from torch2.grid.interactive import interactive_mode

# Load and inspect model
model = load_torch_model("model.pth")
tensors = inspect_torch_object(model)

# Visualize each layer separately
visualize_layers(tensors, output_dir="grids/layers")

# Create overview of all layers
create_layer_overview(tensors, output_path="grids/overview.png")

# Launch interactive mode programmatically
interactive_mode(tensors, output_dir="grids")
```

## Layer-by-Layer Visualization

The layer-by-layer visualization feature provides granular insight into your neural network by creating separate visualizations for each layer's weights.

**What it does:**
- Extracts weights from each layer (fully connected, convolutional, etc.)
- Reshapes multi-dimensional tensors into 2D grids for visualization
- Generates individual heatmaps showing weight distributions per layer
- Creates a combined overview displaying all layers in a single figure

**Use cases:**
- **Debug weight initialization**: Check if layers are properly initialized
- **Monitor training**: Compare layer visualizations across training checkpoints
- **Detect dead neurons**: Identify layers with near-zero weights
- **Analyze architecture**: Understand relative layer sizes and patterns
- **Compare models**: Visually diff weight distributions between model versions

Each visualization includes the layer name, original tensor shape, and a colorbar indicating weight magnitude.

## TODO: Upcoming Features

- [x] Interactive mode with layer selection
- [ ] Weight distribution histograms per layer
- [ ] Specialized convolution kernel visualization
- [ ] Export to multiple formats (SVG, PDF)
- [ ] Weight statistics dashboard (min/max/mean/std)
- [ ] Dead neuron detection and reporting
- [ ] Gradient visualization support
- [ ] Side-by-side model comparison
- [ ] Enhanced CLI with arguments (--output, --title, --colormap)
- [ ] Progress bar for large models
- [ ] Multiple colormap options
- [ ] Batch processing for multiple models
- [ ] Plugin system for custom transformers
- [ ] Config file support (YAML/JSON)
- [ ] Web viewer for interactive exploration

## Project Structure

```
torch2grid/
├── __main__.py          # CLI entry point
├── loader.py            # Model loading utilities
├── inspector.py         # Tensor extraction from models
├── transformer.py       # Grid transformation logic
├── visualizer.py        # Single unified grid visualization
└── layer_visualizer.py  # Layer-by-layer visualization
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT