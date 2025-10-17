# torch2grid

A lightweight Python tool for visualizing PyTorch model weights and architectures. torch2grid transforms neural network parameters into intuitive visual grids, making it easy to inspect model structure, debug weight distributions, and understand layer compositions.

## Features

- **Unified Grid Visualization**: Flattens all model weights into a single heatmap
- **Layer-by-Layer Visualization**: Generate separate visualizations for each layer
- **Layer Overview**: Multi-panel view showing all layers at once
- **Weight Distribution Histograms**: Analyze weight distributions with detailed statistics
- **Convolution Kernel Visualization**: Specialized visualization for CNN filters
- **Plugin System**: Extensible architecture for custom transformation algorithms
- **Interactive Mode**: CLI interface for selective layer visualization
- **Statistical Analysis**: Compare layer statistics (mean, std, min, max, sparsity)
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

### Weight Distribution Histograms

```bash
python -m torch2grid model.pth --histogram
```

Generates weight distribution histograms for each layer showing:
- Frequency distribution of weight values
- Mean and standard deviation
- Min/max values
- Percentage of near-zero weights (potential dead neurons)
- Multi-panel overview of all layers

**Combine with other visualizations:**
```bash
python -m torch2grid model.pth --layers --histogram --stats
```

### Convolution Kernel Visualization

```bash
python -m torch2grid model.pth --conv
```

Automatically detects and visualizes convolution layers:
- Displays filters as image grids
- Shows kernel patterns and learned features
- Supports 2D and 1D convolutions
- Averages across input channels for clarity

**Combine with other modes:**
```bash
python -m torch2grid model.pth --layers --conv --histogram
```

### Transformer Plugins

Use different algorithms to transform weights into grids:

```bash
# List available plugins
python -m torch2grid model.pth --list-plugins

# Use a specific plugin
python -m torch2grid model.pth --plugin spiral
python -m torch2grid model.pth --plugin normalized
python -m torch2grid model.pth --plugin layer_separated
```

**Built-in plugins:**
- `flatten` (default) - Simple flattening into square grid
- `spiral` - Arranges weights in spiral pattern from center
- `normalized` - Normalizes weights to [0, 1] before visualization
- `layer_weighted` - Larger layers get more space
- `layer_separated` - Separates layers with visible boundaries

**Load custom plugins:**
```bash
python -m torch2grid model.pth --load-plugin my_custom_transformer.py
```

### Statistical Analysis

```bash
python -m torch2grid model.pth --stats
```

Prints a table comparing statistics across all layers:
```
Layer Statistics Comparison
================================================================================
Layer Name                                 Mean        Std        Min        Max   ~Zero%
--------------------------------------------------------------------------------
fc1.weight                               0.0023     0.2891    -0.8654     0.8432    0.50%
fc1.bias                                 0.0000     0.0000     0.0000     0.0000  100.00%
fc2.weight                              -0.0156     0.3124    -0.9234     0.8765    0.75%
fc2.bias                                 0.0000     0.0000     0.0000     0.0000  100.00%
================================================================================
```

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
from torch2grid.interactive import interactive_mode
from torch2grid.histogram import (
    visualize_all_histograms,
    create_histogram_overview,
    compare_layer_statistics
)
from torch2grid.conv_visualizer import visualize_all_conv_layers

# Load and inspect model
model = load_torch_model("model.pth")
tensors = inspect_torch_object(model)

# Visualize each layer separately
visualize_layers(tensors, output_dir="grids/layers")

# Create overview of all layers
create_layer_overview(tensors, output_path="grids/overview.png")

# Generate histograms
visualize_all_histograms(tensors, output_dir="grids/histograms")
create_histogram_overview(tensors, output_path="grids/histogram_overview.png")

# Visualize convolution kernels
visualize_all_conv_layers(tensors, output_dir="grids/conv_kernels")

# Print statistics
compare_layer_statistics(tensors)

# Launch interactive mode
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

## Weight Distribution Histograms

Histogram visualization helps analyze the statistical properties of weights in each layer:

**Features:**
- Individual histograms per layer with statistics box
- Multi-panel overview for quick comparison
- Mean indicator line and zero reference
- Statistics: mean, std, min, max, near-zero percentage

**Use cases:**
- **Initialization verification**: Ensure proper weight initialization (e.g., Gaussian, Xavier)
- **Training monitoring**: Detect weight explosion or vanishing
- **Dead neuron detection**: High near-zero percentage indicates inactive neurons
- **Distribution analysis**: Compare weight distributions across layers
- **Model debugging**: Identify unusual distributions that may indicate issues

## Convolution Kernel Visualization

Specialized visualization for convolutional neural networks (CNNs) that displays learned filters:

**Features:**
- Automatic detection of convolution layers
- Grid layout showing individual filters
- Supports 2D convolutions (images) and 1D convolutions (sequences)
- Averages across input channels for simplified view
- Shows kernel patterns and edge detectors

**Use cases:**
- **Feature learning inspection**: See what patterns the model has learned
- **Architecture validation**: Verify conv layers are learning diverse features
- **Transfer learning**: Compare filters between pre-trained and fine-tuned models
- **Model interpretation**: Understand what low-level features are being extracted
- **Debug initialization**: Check if filters are properly initialized vs. all zeros/same

The visualization displays each output channel as a tile, making it easy to identify redundant or dead filters.

## Plugin System

The plugin system allows you to create custom transformers for converting model weights into 2D grids.

### Creating Custom Plugins

Create a Python file with a class that inherits from `TransformerPlugin`:

```python
# my_plugin.py
from torch2grid.plugins.base import TransformerPlugin
import numpy as np

class MyCustomTransformer(TransformerPlugin):
    @property
    def name(self) -> str:
        return "my_custom"
    
    @property
    def description(self) -> str:
        return "My custom transformation algorithm"
    
    def transform(self, tensors: dict) -> np.ndarray:
        # Your custom logic here
        flat_values = []
        for name, arr in tensors.items():
            if arr is not None:
                flat_values.extend(arr.flatten())
        
        # Create your custom grid layout
        size = int(np.ceil(np.sqrt(len(flat_values))))
        grid = np.zeros((size, size))
        
        # Fill grid with your algorithm
        for i, val in enumerate(flat_values[:size*size]):
            grid[i // size, i % size] = val
        
        return grid
```

**Load and use your plugin:**
```bash
python -m torch2grid model.pth --load-plugin my_plugin.py --plugin my_custom
```

### Plugin API

The `TransformerPlugin` base class provides these methods:

**Required methods:**
- `name` (property): Unique identifier for the plugin
- `transform(tensors)`: Main transformation logic

**Optional methods:**
- `description` (property): Human-readable description
- `can_handle(tensors)`: Check if plugin is compatible with tensors
- `preprocess(tensors)`: Filter/modify tensors before transformation
- `postprocess(grid)`: Modify grid after transformation

**Example with preprocessing:**
```python
class FilteredTransformer(TransformerPlugin):
    @property
    def name(self) -> str:
        return "weights_only"
    
    def preprocess(self, tensors: dict) -> dict:
        # Only keep weight tensors, skip biases
        return {k: v for k, v in tensors.items() if 'weight' in k}
    
    def transform(self, tensors: dict) -> np.ndarray:
        # Transform logic...
        pass
```

### Programmatic Plugin Usage

```python
from torch2grid.plugins.registry import get_registry
from torch2grid.plugins.base import TransformerPlugin

# Get registry
registry = get_registry()

# List plugins
plugins = registry.list_plugins()
print(f"Available: {plugins}")

# Load custom plugin
registry.load_from_file("my_plugin.py")

# Use plugin
plugin = registry.get("my_custom")
grid = plugin(tensors)

# Register plugin instance
class MyPlugin(TransformerPlugin):
    # ... implementation ...
    pass

registry.register(MyPlugin())
```

## TODO: Upcoming Features

- [x] Interactive mode with layer selection
- [x] Weight distribution histograms per layer
- [x] Weight statistics dashboard (min/max/mean/std)
- [x] Specialized convolution kernel visualization
- [x] Plugin system for custom transformers
- [ ] Export to multiple formats (SVG, PDF)
- [ ] Dead neuron detection and reporting
- [ ] Gradient visualization support
- [ ] Side-by-side model comparison
- [ ] Enhanced CLI with arguments (--output, --title, --colormap)
- [ ] Progress bar for large models
- [ ] Multiple colormap options
- [ ] Batch processing for multiple models
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
├── layer_visualizer.py  # Layer-by-layer visualization
├── histogram.py         # Weight distribution histograms
├── conv_visualizer.py   # Convolution kernel visualization
├── interactive.py       # Interactive CLI interface
└── plugins/             # Plugin system
    ├── __init__.py      # Plugin exports
    ├── base.py          # TransformerPlugin base class
    ├── registry.py      # Plugin registry and loader
    └── builtin.py       # Built-in transformer plugins
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT
