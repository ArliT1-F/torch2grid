.PHONY: help install install-dev test clean lint format

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	python3 -m pip install -e .

install-dev:  ## Install the package with development dependencies
	python3 -m pip install -e ".[dev]"

test:  ## Run tests
	python3 -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python3 -m pytest tests/ -v --cov=torch2grid --cov-report=html

lint:  ## Run linters
	python3 -m flake8 torch2grid/ tests/
	python3 -m mypy torch2grid/

format:  ## Format code
	python3 -m black torch2grid/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo:  ## Run a demo with the provided models
	python3 -m torch2grid simple_cnn.pth --layers --histogram --stats
	python3 -m torch2grid tinynet.pth --conv --dead-neurons