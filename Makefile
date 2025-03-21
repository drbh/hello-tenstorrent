# Makefile for hello-tenstorrent project

# Include .env file if it exists
-include .env
export

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  container-full  - Build and run Docker container"
	@echo "  setup-patch     - Install llama-models from GitHub"
	@echo "  clean           - Remove build artifacts"
	@echo "  lint            - Run linting"
	@echo "  test            - Run tests"

# Docker commands
.PHONY: container-full container-build container-run
container-full: container-build container-run

container-build:
	docker build --platform linux/amd64 -t hello-tenstorrent .

container-run:
	docker run --rm hello-tenstorrent

# Development setup
.PHONY: setup-patch setup-dev
setup-patch:
	uv pip install git+https://github.com/tenstorrent/llama-models.git@tt_metal_tag

setup-dev:
	uv pip install -e ".[dev]"

# Cleanup
.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Testing and linting
.PHONY: lint test
lint:
	flake8 .
	isort --check .
	black --check .

test:
	pytest -xvs tests/