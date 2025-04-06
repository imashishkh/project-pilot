#!/usr/bin/env python3
"""
Setup script for MacAgent UI.

This script checks for the required UI dependencies and installs them if needed.
"""

import os
import sys
import subprocess
import platform
import importlib
import time

def check_dependency(package_name):
    """Check if a Python package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a Python package using pip."""
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Main entry point for the setup script."""
    print("MacAgent UI Setup")
    print("=================")
    
    # Check if running on macOS
    if platform.system() != "Darwin":
        print("Error: MacAgent is designed to run on macOS only.")
        sys.exit(1)
    
    # Check for PyQt5
    print("Checking for PyQt5...")
    if check_dependency("PyQt5"):
        print("✓ PyQt5 is installed")
    else:
        print("✗ PyQt5 is not installed")
        if install_package("PyQt5>=5.15.9"):
            print("✓ PyQt5 has been installed successfully")
        else:
            print("✗ Failed to install PyQt5. Please install it manually:")
            print("  pip install PyQt5>=5.15.9")
            return False
    
    # Check for SpeechRecognition (optional)
    print("\nChecking for SpeechRecognition (optional)...")
    if check_dependency("speech_recognition"):
        print("✓ SpeechRecognition is installed")
    else:
        print("✗ SpeechRecognition is not installed")
        print("  This is optional for voice commands. Install with:")
        print("  pip install speechrecognition>=3.10.0")
    
    print("\nSetup complete! You can now run MacAgent with:")
    print("python -m MacAgent.main")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    # Keep the terminal window open for a moment
    time.sleep(2) 