"""
Configuration file for pytest.

This file ensures that all tests can import from the MacAgent package
without having to modify sys.path in each test file.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to sys.path once, so imports will work across all tests
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# This helps tests find the src module directly
try:
    from MacAgent.src import vision, intelligence, interaction, core
except ModuleNotFoundError:
    # Alternative import path if we're already in the MacAgent directory
    try:
        from src import vision, intelligence, interaction, core
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src import vision, intelligence, interaction, core


@pytest.fixture
def test_resources_dir():
    """Fixture providing the path to test resources."""
    return os.path.join(os.path.dirname(__file__), "resources")


@pytest.fixture
def test_images_dir():
    """Fixture providing the path to test images."""
    return os.path.join(os.path.dirname(__file__), "resources", "images")


@pytest.fixture
def test_output_dir():
    """Fixture providing the path to test output directory."""
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@pytest.fixture
def create_output_subdir():
    """Fixture providing a function to create subdirectories in the output directory."""
    def _create_subdir(subdir_name):
        output_dir = os.path.join(os.path.dirname(__file__), "output", subdir_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return _create_subdir


# Helper function that can be imported by tests to get proper paths
def get_test_path(relative_path):
    """
    Get absolute path to a test resource.
    
    Args:
        relative_path: Path relative to the tests directory
        
    Returns:
        Absolute path
    """
    return os.path.join(os.path.dirname(__file__), relative_path) 