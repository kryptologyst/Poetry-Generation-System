#!/usr/bin/env python3
"""Setup script for Poetry Generation System."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is 3.10 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def setup_environment():
    """Set up the development environment."""
    print("🚀 Setting up Poetry Generation System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    # Install the package in development mode
    if not run_command("pip install -e .", "Installing poetry generation system"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️  Development dependencies installation failed, continuing...")
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("⚠️  Pre-commit hooks installation failed, continuing...")
    
    # Create necessary directories
    directories = [
        "data",
        "data/cache", 
        "outputs",
        "checkpoints",
        "logs",
        "assets",
        "assets/samples",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Run tests to verify installation
    print("\n🧪 Running tests to verify installation...")
    if run_command("python -m pytest tests/ -v", "Running unit tests"):
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed, but installation is complete")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the demo: python 0382_modernized.py")
    print("2. Launch Streamlit demo: streamlit run demo/streamlit_demo.py")
    print("3. Train a model: python scripts/train.py")
    print("4. Generate samples: python scripts/sample.py")
    print("\nFor more information, see README.md")


def main():
    """Main setup function."""
    try:
        setup_environment()
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
