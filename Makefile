.PHONY: help install install-dev test lint format type-check security-check clean build docker-build docker-run docker-stop

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  test           - Run tests with coverage"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code with black"
	@echo "  type-check     - Run type checking with mypy"
	@echo "  security-check - Run security checks with bandit"
	@echo "  clean          - Clean up cache and build files"
	@echo "  build          - Build the project"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-stop    - Stop Docker container"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=utils --cov-report=html --cov-report=term

test-watch:
	pytest tests/ -v --cov=utils --cov-report=html --cov-report=term -f

# Code Quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

type-check:
	mypy utils.py --ignore-missing-imports

security-check:
	bandit -r . -f json -o bandit-report.json

# Quality checks (all)
quality: format lint type-check security-check test

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf bandit-report.json

# Build
build:
	python -m build

# Docker commands
docker-build:
	docker build -t kitchen-monitor .

docker-run:
	docker-compose up --build

docker-stop:
	docker-compose down

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete!"
	@echo "Run 'make quality' to check code quality"
	@echo "Run 'make test' to run tests"

# Pre-commit
pre-commit-all:
	pre-commit run --all-files

# Documentation
docs:
	@echo "Generating documentation..."
	# Add documentation generation commands here

# Release
release:
	@echo "Creating release..."
	# Add release commands here 