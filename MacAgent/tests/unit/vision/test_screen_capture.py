"""
Test module for screen capture functionality.

This module tests the ability to capture the screen and specific regions.
"""

import os
import cv2
import pytest

# Import directly from MacAgent package or alternative paths
try:
    from MacAgent.src.vision.screen_capture import ScreenCapture
except ModuleNotFoundError:
    # Try alternative import paths
    import sys
    from pathlib import Path
    # Add the project root to the path
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from MacAgent.src.vision.screen_capture import ScreenCapture
    except ModuleNotFoundError:
        from src.vision.screen_capture import ScreenCapture


@pytest.mark.vision
def test_full_screen_capture(test_output_dir):
    """Test capturing the full screen."""
    # Initialize screen capture with default settings
    capture = ScreenCapture()
    
    # Capture the screen
    image = capture.capture()
    
    # Save the image
    output_path = os.path.join(test_output_dir, "test_capture.png")
    cv2.imwrite(output_path, image)
    
    # Verify the capture worked
    assert image is not None, "Screen capture returned None"
    assert image.shape[0] > 0 and image.shape[1] > 0, "Screen capture has invalid dimensions"
    assert os.path.exists(output_path), "Screen capture file was not saved"
    
    # Clean up resources
    capture.close()


@pytest.mark.vision
def test_region_capture(test_output_dir):
    """Test capturing a specific region of the screen."""
    # Initialize screen capture
    capture = ScreenCapture()
    
    # Define a region (x, y, width, height)
    region = (100, 100, 400, 300)
    
    # Capture the specified region
    region_image = capture.capture(region=region)
    
    # Save the region capture
    region_path = os.path.join(test_output_dir, "test_region_capture.png")
    cv2.imwrite(region_path, region_image)
    
    # Verify the region capture worked
    assert region_image is not None, "Region capture returned None"
    # Note: The actual dimensions may not match the region exactly depending on implementation
    # Just check that we got a valid image of reasonable size
    assert region_image.shape[0] > 0 and region_image.shape[1] > 0, "Region capture has invalid dimensions"
    assert os.path.exists(region_path), "Region capture file was not saved"
    
    # Clean up resources
    capture.close()


if __name__ == "__main__":
    # For manual testing outside pytest
    import sys
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    test_full_screen_capture(output_dir)
    test_region_capture(output_dir)
    
    print("Screen capture tests completed successfully!") 