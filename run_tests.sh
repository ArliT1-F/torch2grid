# Test runner script for torch2grid

set -e

echo "torch2grid Test Runner"
echo "======================"

# Check if Python3 is avaliable
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is required but not installed."
    exit 1
fi


# Install dependencies if needed
echo "Installing test dependencies..."
python3 -m pip install -e ".[dev]"



# Run tests
echo "Running tests..."
python3 -m pytest tests/ -v

echo ""
echo "Tests completed successfully."