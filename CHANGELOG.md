# Changelog

All notable changes to torch2grid will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- Initial release of torch2grid
- Unified grid visualization of PyTorch model weights
- Layer-by-layer visualization with individual PNG files
- Weight distribution histograms with statistical analysis
- Convolution kernel visualization for CNN models
- Dead neuron detection and reporting
- Gradient visualization and health analysis
- Plugin system for custom transformation algorithms
- Multi-format export (PNG, SVG, PDF)
- Interactive CLI mode for selective layer visualization
- Statistical comparison of layer properties
- Headless support for environments without GUI
- Support for multiple input formats (.pt, .pth, .pkl)
- Comprehensive documentation and examples
- Test suite with basic functionality tests
- CI/CD pipeline with GitHub Actions
- Development tools (linting, formatting, type checking)

### Features
- **Unified Grid Visualization**: Flattens all model weights into a single heatmap
- **Layer-by-Layer Visualization**: Generate separate visualizations for each layer
- **Weight Distribution Histograms**: Analyze weight distributions with detailed statistics
- **Convolution Kernel Visualization**: Specialized visualization for CNN filters
- **Plugin System**: Extensible architecture for custom transformation algorithms
- **Dead Neuron Detection**: Identify inactive neurons with near-zero weights
- **Gradient Visualization**: Analyze gradient flow and detect vanishing/exploding gradients
- **Multi-Format Export**: Export visualizations to PNG, SVG, and PDF formats
- **Interactive Mode**: CLI interface for selective layer visualization
- **Statistical Analysis**: Compare layer statistics (mean, std, min, max, sparsity)
- **Headless Support**: Works in environments without GUI (Codespaces, SSH, CI/CD)
- **Multiple Input Formats**: Supports .pt, .pth, .pkl model files and state_dict objects

### Technical Details
- Python 3.8+ support
- Dependencies: torch, numpy, matplotlib, Pillow
- MIT License
- Cross-platform compatibility