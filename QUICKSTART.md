# Quick Start Guide - Kitchen Dispatch Monitor

This guide will help you get up and running with the Kitchen Dispatch Monitor project quickly.

## Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized deployment)

## Option 1: Automated Setup (Recommended)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd dispatch-monitoring
```

### 2. Run the Setup Script
```bash
python scripts/setup_dev.py
```

This script will:
- Check Python version compatibility
- Create a virtual environment
- Install all dependencies
- Set up pre-commit hooks
- Create necessary directories
- Run initial tests

### 3. Activate the Virtual Environment
```bash
# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 4. Verify Installation
```bash
# Run tests
make test

# Run quality checks
make quality
```

## Option 2: Manual Setup

### 1. Create Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 3. Set Up Pre-commit Hooks
```bash
pre-commit install
```

### 4. Create Directories
```bash
mkdir -p videos feedback_data data/detection data/classification logs
```

## Running the Application

### Local Development
```bash
# Run the main application
python app-menu.py

# Run with custom video
python app-menu.py --video path/to/your/video.mp4

# Run with custom configuration
python app-menu.py --config path/to/config.yml
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

## Development Workflow

### 1. Code Quality Checks
```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Security checks
make security-check

# Run all quality checks
make quality
```

### 2. Testing
```bash
# Run all tests
make test

# Run tests with coverage
pytest tests/ -v --cov=utils --cov-report=html

# Run specific test
pytest tests/test_utils.py -v
```

### 3. Pre-commit Hooks
The project uses pre-commit hooks that run automatically on commit:
- Code formatting (Black)
- Import sorting (isort)
- Linting (Flake8)
- Type checking (MyPy)
- Security scanning (Bandit)
- Tests

### 4. Making Changes
1. Create a feature branch
2. Make your changes
3. Run quality checks: `make quality`
4. Commit your changes (pre-commit hooks will run)
5. Push and create a pull request

## Project Structure

```
dispatch-monitoring/
├── app-menu.py              # Main application
├── utils.py                 # Core utilities and tracking
├── config.yml               # Configuration
├── requirements.txt         # Dependencies
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
├── models/                 # ML models
├── videos/                 # Input videos
├── feedback_data/          # HITL feedback
└── data/                   # Training data
```

## Common Commands

### Development
```bash
make help                    # Show all available commands
make install-dev            # Install development dependencies
make test                   # Run tests
make quality                # Run all quality checks
make clean                  # Clean up cache files
```

### Docker
```bash
make docker-build           # Build Docker image
make docker-run             # Run with Docker Compose
make docker-stop            # Stop Docker containers
```

### Application
```bash
python app-menu.py          # Run application
python app-menu.py --help   # Show command line options
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **Pre-commit Hook Failures**
   - Run `make format` to fix formatting issues
   - Run `make lint` to fix linting issues

3. **Test Failures**
   - Check that all dependencies are installed
   - Ensure you're in the project root directory

4. **Docker Issues**
   - Ensure Docker is running
   - Check Docker Compose version compatibility

### Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review the [Development & Code Quality](README.md#development--code-quality) section
- Run `make help` for available commands
- Check the test files for usage examples

## Next Steps

1. **Explore the Code**: Start with `utils.py` to understand the tracking implementation
2. **Run Tests**: Execute `make test` to see the test coverage
3. **Try the Application**: Run `python app-menu.py` with a sample video
4. **Read Documentation**: Review the comprehensive README.md
5. **Contribute**: Follow the contributing guidelines in the README

## Support

- **Issues**: Use GitHub Issues for bug reports
- **Documentation**: Check the README.md file
- **Contact**: hunglg.341@gmail.com

---

Happy coding! 