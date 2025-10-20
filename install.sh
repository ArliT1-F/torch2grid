#!/bin/bash
# Installation script for torch2grid

set -e

echo "torch2grid Installation Script"
echo "=============================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3.8 or later and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or later is required, but found $PYTHON_VERSION"
    exit 1
fi

echo "Python version: $PYTHON_VERSION ✓"

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r torch2grid/requirements.txt

# Install the package
echo "Installing torch2grid..."
python3 -m pip install -e .

echo ""
echo "Installation complete! ✓"
echo ""
echo "You can now use torch2grid:"
echo "  python3 -m torch2grid --help"
echo "  python3 -m torch2grid model.pth --layers"
echo ""
echo "For development, install with:"
echo "  python3 -m pip install -e .[dev]"