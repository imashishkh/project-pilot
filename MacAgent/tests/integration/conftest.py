"""
pytest configuration file for integration tests.

This file provides fixtures specifically for integration tests.
"""

import pytest
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the main fixtures from parent conftest
from MacAgent.tests.conftest import (
    test_resources_dir,
    test_images_dir,
    test_output_dir,
    create_output_subdir
)


@pytest.fixture
def integration_test_output_dir(test_output_dir):
    """Fixture providing a directory for integration test outputs."""
    output_dir = os.path.join(test_output_dir, "integration")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


@pytest.fixture
def integration_context():
    """Fixture providing a context object for sharing data between test steps."""
    return {} 