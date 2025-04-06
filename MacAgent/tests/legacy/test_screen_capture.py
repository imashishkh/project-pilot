"""
Simple test script for testing screen capture functionality.

This script captures the screen and saves it as a PNG file for verification.
"""

import os
import sys
import cv2

# Add the parent directory to the Python path to find the MacAgent package
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from MacAgent.src.vision.screen_capture import ScreenCapture

def test_capture():
    # Create output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Initialize screen capture with default settings
    capture = ScreenCapture()
    
    # Capture the screen
    image = capture.capture()
    
    # Save the image
    output_path = "test_output/test_capture.png"
    cv2.imwrite(output_path, image)
    
    print(f"Screen capture successful! Image dimensions: {image.shape[1]}x{image.shape[0]}")
    print(f"Screen capture saved to: {output_path}")
    
    # Optional: capture a specific region
    region = (100, 100, 400, 300)  # x, y, width, height
    region_image = capture.capture(region=region)
    
    # Save the region capture
    region_path = "test_output/test_region_capture.png"
    cv2.imwrite(region_path, region_image)
    print(f"Region capture saved to: {region_path}")
    
    # Clean up resources
    capture.close()

if __name__ == "__main__":
    test_capture() 