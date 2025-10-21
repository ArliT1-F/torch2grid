# torch2grid

A lightweight Python tool for visualizing PyTorch model weights and architectures. torch2grid transforms neural network parameters into intuitive visual grids, making it easy to inspect model structure, debug weight distributions, and understand layer compositions.

## Quick Start

```bash
# Install
git clone https://github.com/your-username/torch2grid.git
cd torch2grid
./install.sh

# Basic usage
python3 -m torch2grid model.pth

# Advanced visualization
python3 -m torch2grid model.pth --layers --histogram --conv --stats

# Interactive mode
python3 -m torch2grid model.pth --interactive

# Run demo
python3 examples/demo.py
```

## Features

- **Unified Grid Visualization**: Flattens all model weights into a single heatmap
- **Layer-by-Layer Visualization**: Generate separate visualizations for each layer
- **Layer Overview**: Multi-panel view showing all layers at once
- **Weight Distribution Histograms**: Analyze weight distributions with detailed statistics
- **Convolution Kernel Visualization**: Specialized visualization for CNN filters
- **Plugin System**: Extensible architecture for custom transformation algorithms
- **Dead Neuron Detection**: Identify inactive neurons with near-zero weights
- **Gradient Visualization**: Analyze gradient flow and detect vanishing/exploding gradients
- **Multi-Format Export**: Export visualizations to PNG, SVG, and PDF formats
- **Interactive Mode**: CLI interface for selective layer visualization
- **Statistical Analysis**: Compare layer statistics (mean, std, min, max, sparsity)
- **Headless Support**: Works in environments without GUI (Codespaces, SSH, CI/CD)
- **Multiple Input Formats**: Supports `.pt`, `.pth`, `.pkl` model files and state_dict objects

## Installation

### Quick Install

```bash
# Clone the repository
git clone https://github.com/your-username/torch2grid.git
cd torch2grid

# Run the installation script
./install.sh
```

### Manual Installation

```bash
# Install dependencies
pip install -r torch2grid/requirements.txt

# Install the package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or use the Makefile
make install-dev
```

### Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.21.0+
- Matplotlib 3.5.0+
- Pillow 8.0.0+

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

### Dead Neuron Detection

```bash
python -m torch2grid model.pth --dead-neurons
```

Detects neurons with near-zero weights that don't contribute to the model:

```
Dead Neuron Detection Report
================================================================================
Threshold: 1e-06
Total neurons/channels: 64
Dead neurons/channels: 3 (4.69%)
Layers affected: 2/4
================================================================================

Layer                                       Total     Dead        %
--------------------------------------------------------------------------------
conv1.weight                                   16        2    12.50% ⚠️ 
fc.weight                                      10        1    10.00% ⚠️ 
--------------------------------------------------------------------------------

✓ 3 dead neuron(s) detected across 2 layer(s).
```

**Outputs:**
- Console report with per-layer statistics
- JSON report saved to `grids/dead_neurons_report.json`
- Visualization chart showing dead neuron distribution

**Use cases:**
- Verify model is properly trained
- Identify overparameterized layers
- Detect initialization issues
- Guide model pruning decisions

### Export to Multiple Formats

```bash
# Export to SVG
python -m torch2grid model.pth --export svg

# Export to PDF
python -m torch2grid model.pth --export pdf

# Export to multiple formats
python -m torch2grid model.pth --export svg,pdf,png
```

Export visualizations in publication-quality vector formats:
- **SVG**: Scalable vector graphics for web and presentations
- **PDF**: High-quality prints and academic papers
- **PNG**: Standard raster format (default)

**Programmatic export:**
```python
from torch2grid.exporter import export_grid_multi_format, export_layers_to_pdf

# Export single grid to multiple formats
export_grid_multi_format(grid, title="My Model", formats=['svg', 'pdf'])

# Export all layers to multi-page PDF report
export_layers_to_pdf(tensors, output_path="model_report.pdf")
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
from torch2grid.dead_neuron_detector import (
    detect_dead_neurons,
    print_dead_neuron_report,
    visualize_dead_neurons
)
from torch2grid.exporter import export_grid_multi_format, export_layers_to_pdf

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

# Detect dead neurons
report = detect_dead_neurons(tensors)
print_dead_neuron_report(report)
visualize_dead_neurons(report)

# Export to multiple formats
grid = to_neutral_grid(tensors)
export_grid_multi_format(grid, formats=['svg', 'pdf'])
export_layers_to_pdf(tensors, output_path="model_report.pdf")

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

## Gradient Visualization

Visualize and analyze gradients to detect training issues (requires gradients to be computed during training):

**Gradient Flow Analysis:**
```python
from torch2grid.gradient_visualizer import (
    visualize_gradient_flow,
    analyze_gradient_health,
    print_gradient_health_report
)

# During training, collect gradients
gradients = {}
for name, param in model.named_parameters():
    if param.grad is not None:
        gradients[name] = param.grad

# Analyze gradient health
analysis = analyze_gradient_health(gradients)
print_gradient_health_report(analysis)

# Visualize gradient flow across layers
visualize_gradient_flow(gradients, output_path="grids/gradient_flow.png")
```

**Gradient Health Report:**
```
Gradient Health Report
================================================================================
Total layers: 6
Healthy: 4 (66.7%)
Vanishing gradients: 2 (33.3%)
Exploding gradients: 0 (0.0%)
================================================================================

Layer                                        Mean          Max       Status
--------------------------------------------------------------------------------
fc1.weight                                2.34e-08     5.67e-07    ⚠️  VANISH
fc2.weight                                5.12e-03     1.24e-02           ✓
--------------------------------------------------------------------------------

⚠️  WARNING: 2 layer(s) with vanishing gradients detected!
   Consider: gradient clipping, batch normalization, or residual connections
```

**Features:**
- Detect vanishing/exploding gradients
- Visualize gradient flow across layers
- Compare weights vs gradients side-by-side
- Export gradient analysis reports

**Use cases:**
- Debug training issues (vanishing/exploding gradients)
- Validate gradient flow in deep networks
- Tune learning rates and optimizer settings
- Compare gradient patterns across training epochs

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

## Recent Updates (v1.0.0)

### ✅ **New Features & Improvements**
- **Enhanced CLI**: Added `--version` flag and improved help system
- **Progress Indicators**: Added progress bars for long operations
- **Configuration System**: Environment variable support for customizable settings
- **Better Error Handling**: Comprehensive error messages with helpful suggestions
- **PyTorch 2.6+ Compatibility**: Fixed loader to handle new `weights_only=True` default
- **Development Tools**: Added Makefile, installation scripts, and CI/CD pipeline
- **Comprehensive Testing**: 11+ tests covering all core functionality
- **Enhanced Documentation**: Improved docstrings and API documentation

### ✅ **Project Structure Improvements**
- **Modern Packaging**: Added `pyproject.toml` and `setup.py`
- **Test Suite**: Organized tests in dedicated `tests/` directory
- **Examples**: Added `examples/` directory with demo scripts
- **CI/CD**: GitHub Actions workflow for automated testing
- **Development Tools**: Makefile, scripts, and linting configuration

### ✅ **Code Quality Enhancements**
- **Type Hints**: Added proper type annotations throughout
- **Error Handling**: Better error messages and edge case handling
- **Code Organization**: Improved modular design and separation of concerns
- **Version Management**: Added version tracking and changelog

## TODO: Upcoming Features

- [x] Interactive mode with layer selection
- [x] Weight distribution histograms per layer
- [x] Weight statistics dashboard (min/max/mean/std)
- [x] Specialized convolution kernel visualization
- [x] Plugin system for custom transformers
- [x] Export to multiple formats (SVG, PDF)
- [x] Dead neuron detection and reporting
- [x] Gradient visualization support
- [x] Enhanced CLI with arguments and progress bars
- [x] Configuration system with environment variables
- [x] Comprehensive test suite and CI/CD
- [ ] Side-by-side model comparison
- [ ] Multiple colormap options
- [ ] Batch processing for multiple models
- [ ] Config file support (YAML/JSON)
- [ ] Web viewer for interactive exploration
- [ ] Jupyter notebook integration
- [ ] Real-time visualization updates

## Project Structure

```
torch2grid/
├── README.md                           # Main project documentation
├── CHANGELOG.md                        # Version history and changes
├── setup.py                           # Legacy setup configuration
├── pyproject.toml                     # Modern Python project configuration
├── Makefile                           # Development task automation
├── install.sh                         # Easy installation script
├── run_tests.sh                       # Test runner script
├── .gitignore                         # Git ignore rules
├── .github/                           # GitHub configuration
│   └── workflows/
│       └── ci.yml                     # CI/CD pipeline
├── torch2grid/                        # Main package directory
│   ├── __init__.py                    # Package initialization
│   ├── __main__.py                    # CLI entry point
│   ├── __version__.py                 # Version information
│   ├── config.py                      # Configuration management
│   ├── utils.py                       # Utility functions
│   ├── loader.py                      # Model loading utilities
│   ├── inspector.py                   # Tensor extraction from models
│   ├── transformer.py                 # Grid transformation logic
│   ├── visualizer.py                  # Single unified grid visualization
│   ├── layer_visualizer.py            # Layer-by-layer visualization
│   ├── histogram.py                   # Weight distribution histograms
│   ├── conv_visualizer.py             # Convolution kernel visualization
│   ├── dead_neuron_detector.py        # Dead neuron detection and reporting
│   ├── gradient_visualizer.py         # Gradient flow visualization
│   ├── exporter.py                    # Multi-format export (SVG, PDF)
│   ├── interactive.py                 # Interactive CLI interface
│   ├── requirements.txt               # Package dependencies
│   └── plugins/                       # Plugin system
│       ├── __init__.py                # Plugin exports
│       ├── base.py                    # TransformerPlugin base class
│       ├── registry.py                # Plugin registry and loader
│       └── builtin.py                 # Built-in transformer plugins
├── tests/                             # Test suite
│   ├── __init__.py                    # Test package initialization
│   ├── test_basic.py                  # Basic functionality tests
│   └── test_loader.py                 # Model loading tests
├── examples/                          # Example scripts and demos
│   ├── README.md                      # Examples documentation
│   └── demo.py                        # Complete demonstration script
├── grids/                             # Generated visualizations (gitignored)
│   ├── layers/                        # Individual layer visualizations
│   ├── histograms/                    # Weight distribution histograms
│   ├── conv_kernels/                  # Convolution kernel visualizations
│   └── *.png, *.svg, *.pdf           # Various output formats
├── demo_output/                       # Demo script outputs (gitignored)
│   ├── layers/                        # Demo layer visualizations
│   ├── histograms/                    # Demo histograms
│   ├── conv_kernels/                  # Demo conv kernels
│   ├── spiral/                        # Spiral transformer outputs
│   ├── normalized/                    # Normalized transformer outputs
│   └── layer_separated/               # Layer separated outputs
├── simple_cnn.pth                     # Sample CNN model
├── tinynet.pth                        # Sample tiny model
├── train_conv_model.py                # Model training script
├── train_dummy_model.py               # Dummy model training
├── test_gradients.py                  # Gradient testing script
├── example_plugin.py                  # Example custom plugin
└── image.png                          # Sample image
```

### Directory Organization

- **Root Level**: Configuration files, documentation, and installation scripts
- **torch2grid/**: Main package with all core functionality
- **tests/**: Comprehensive test suite with 11+ tests
- **examples/**: Demo scripts and usage examples
- **grids/**: Generated visualizations (automatically created, gitignored)
- **Sample Files**: Pre-trained models and example scripts for testing

## Development

### Running Tests

```bash
# Run all tests
./run_tests.sh
# or
make test

# Run tests with coverage
make test-cov

# Run specific test file
python3 -m pytest tests/test_basic.py -v
```

### Code Quality

```bash
# Format code
make format
# or
black torch2grid/ tests/

# Lint code
make lint
# or
flake8 torch2grid/ tests/
mypy torch2grid/
```

### Development Workflow

1. **Fork and clone** the repository
2. **Install development dependencies**: `make install-dev`
3. **Create a feature branch**: `git checkout -b feature-name`
4. **Make changes** and add tests
5. **Run tests**: `make test`
6. **Format code**: `make format`
7. **Check linting**: `make lint`
8. **Commit changes**: `git commit -m "Add feature"`
9. **Push and create PR**: `git push origin feature-name`

### Available Make Commands

```bash
make help          # Show all available commands
make install       # Install the package
make install-dev   # Install with development dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Run linters
make format        # Format code
make clean         # Clean build artifacts
make demo          # Run demo with sample models
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes and add tests
4. **Run** the test suite (`make test`)
5. **Format** your code (`make format`)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to the branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Update documentation for new features
- Ensure all tests pass before submitting PR
- Use meaningful commit messages

## Troubleshooting

### Common Issues

**ImportError: No module named 'torch'**
```bash
# Install PyTorch first
pip install torch torchvision
# or
pip install -r torch2grid/requirements.txt
```

**Permission denied when running scripts**
```bash
# Make scripts executable
chmod +x install.sh run_tests.sh
```

**PyTorch 2.6+ loading errors**
- The loader automatically handles PyTorch 2.6+ security changes
- If you see warnings about `weights_only`, this is normal and handled automatically

**No display/GUI issues**
- torch2grid automatically detects headless environments
- Set `MPLBACKEND=Agg` if you encounter display issues

**Memory issues with large models**
- Use `--layers` to visualize layers separately instead of unified grid
- Consider using `--plugin layer_separated` for better memory management

### Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-username/torch2grid/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/torch2grid/discussions)
- **Documentation**: Check the examples in `examples/` directory

## License

MIT License - see [LICENSE](LICENSE) file for details.
