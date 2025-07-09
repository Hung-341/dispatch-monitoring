#!/usr/bin/env python3
"""
Development Environment Setup Script
Automates the setup of the Kitchen Dispatch Monitor development environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a command and return success status."""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command, 
            cwd=cwd, 
            check=True, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False


def check_git() -> bool:
    """Check if git is installed."""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        print("Git is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git is not installed")
        print("Please install Git: https://git-scm.com/")
        return False


def create_virtual_environment(project_root: Path) -> bool:
    """Create a virtual environment."""
    venv_path = project_root / "venv"
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    if run_command([sys.executable, "-m", "venv", "venv"], cwd=project_root):
        print("Virtual environment created")
        return True
    else:
        print("Failed to create virtual environment")
        return False


def get_pip_command(project_root: Path) -> List[str]:
    """Get the appropriate pip command for the virtual environment."""
    if platform.system() == "Windows":
        pip_path = project_root / "venv" / "Scripts" / "pip.exe"
    else:
        pip_path = project_root / "venv" / "bin" / "pip"
    
    return [str(pip_path)]


def install_dependencies(project_root: Path) -> bool:
    """Install project dependencies."""
    pip_cmd = get_pip_command(project_root)
    
    # Upgrade pip
    if not run_command(pip_cmd + ["install", "--upgrade", "pip"], cwd=project_root):
        return False
    
    # Install requirements
    if not run_command(pip_cmd + ["install", "-r", "requirements.txt"], cwd=project_root):
        return False
    
    # Install development dependencies
    if not run_command(pip_cmd + ["install", "-e", ".[dev]"], cwd=project_root):
        return False
    
    print("Dependencies installed")
    return True


def setup_pre_commit(project_root: Path) -> bool:
    """Set up pre-commit hooks."""
    if platform.system() == "Windows":
        pre_commit_path = project_root / "venv" / "Scripts" / "pre-commit.exe"
    else:
        pre_commit_path = project_root / "venv" / "bin" / "pre-commit"
    
    if run_command([str(pre_commit_path), "install"], cwd=project_root):
        print("Pre-commit hooks installed")
        return True
    else:
        print("Failed to install pre-commit hooks")
        return False


def run_initial_tests(project_root: Path) -> bool:
    """Run initial tests to verify setup."""
    if platform.system() == "Windows":
        pytest_path = project_root / "venv" / "Scripts" / "pytest.exe"
    else:
        pytest_path = project_root / "venv" / "bin" / "pytest"
    
    if run_command([str(pytest_path), "tests/", "-v"], cwd=project_root):
        print("Initial tests passed")
        return True
    else:
        print("Initial tests failed")
        return False


def create_directories(project_root: Path) -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        "videos",
        "feedback_data",
        "data",
        "data/detection",
        "data/classification",
        "logs"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def main():
    """Main setup function."""
    print("Kitchen Dispatch Monitor - Development Setup")
    print("=" * 50)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    if not check_git():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment(project_root):
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(project_root):
        sys.exit(1)
    
    # Set up pre-commit hooks
    if not setup_pre_commit(project_root):
        print("Warning: Pre-commit setup failed, but continuing...")
    
    # Create necessary directories
    create_directories(project_root)
    
    # Run initial tests
    if not run_initial_tests(project_root):
        print("Warning: Initial tests failed, but setup is complete")
    
    print("\n" + "=" * 50)
    print("Development environment setup complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run quality checks: make quality")
    print("3. Start development: python app-menu.py")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main() 