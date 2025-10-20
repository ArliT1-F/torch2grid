# torch2grid Examples

This directory contains example scripts demonstrating torch2grid functionality.

## Files

- `demo.py` - Complete demonstration script showing all major features
- `README.md` - This file

## Running the Examples

Make sure you have torch2grid installed:

```bash
# From the project root
python3 -m pip install -e .
```

Then run the demo:

```bash
python3 examples/demo.py
```

This will create a `demo_output` directory with various visualizations of a sample CNN model.

## What the Demo Shows

1. **Basic Grid Visualization** - Single unified view of all model weights
2. **Layer-by-Layer Visualization** - Individual visualizations for each layer
3. **Weight Distribution Histograms** - Statistical analysis of weight distributions
4. **Convolution Kernel Visualization** - Specialized view of CNN filters
5. **Dead Neuron Detection** - Identification of inactive neurons
6. **Plugin System** - Different transformer algorithms (spiral, normalized, etc.)

## Custom Examples

You can create your own examples by:

1. Loading your own PyTorch model
2. Using the torch2grid API programmatically
3. Experimenting with different visualization options

See the main README.md for detailed API documentation.