"""
Test module for image processing functionality.

This module tests the ability to process and analyze images.
"""

import os
import sys
import logging
import pytest
import numpy as np
import cv2
from pathlib import Path

# Import directly from MacAgent package or alternative paths
try:
    from MacAgent.src.vision.image_processor import ImageProcessor
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
        from MacAgent.src.vision.image_processor import ImageProcessor
        from MacAgent.src.vision.screen_capture import ScreenCapture
    except ModuleNotFoundError:
        from src.vision.image_processor import ImageProcessor
        from src.vision.screen_capture import ScreenCapture

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_processor():
    # Create output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Initialize screen capture with default settings
    capture = ScreenCapture()
    
    # Capture the screen
    image = capture.capture()
    
    # Create an image processor
    processor = ImageProcessor()
    
    # 1. Basic grayscale conversion
    gray = processor.to_grayscale(image)
    cv2.imwrite("test_output/gray.png", gray)
    print("✓ Grayscale conversion")
    
    # 2. Different contrast enhancement methods
    clahe_enhanced = processor.enhance_contrast(image, method='clahe')
    histeq_enhanced = processor.enhance_contrast(image, method='histogram_eq')
    stretch_enhanced = processor.enhance_contrast(image, method='linear_stretch')
    
    cv2.imwrite("test_output/clahe_enhanced.png", clahe_enhanced)
    cv2.imwrite("test_output/histeq_enhanced.png", histeq_enhanced)
    cv2.imwrite("test_output/stretch_enhanced.png", stretch_enhanced)
    print("✓ Contrast enhancement (CLAHE, histogram eq, linear stretch)")
    
    # 3. Noise reduction methods
    gaussian_blur = processor.reduce_noise(image, method='gaussian', kernel_size=5)
    median_blur = processor.reduce_noise(image, method='median', kernel_size=5)
    bilateral_blur = processor.reduce_noise(image, method='bilateral', kernel_size=9)
    
    cv2.imwrite("test_output/gaussian_blur.png", gaussian_blur)
    cv2.imwrite("test_output/median_blur.png", median_blur)
    cv2.imwrite("test_output/bilateral_blur.png", bilateral_blur)
    print("✓ Noise reduction (Gaussian, median, bilateral)")
    
    # 4. Edge detection methods
    canny_edges = processor.detect_edges(image, method='canny')
    sobel_edges = processor.detect_edges(image, method='sobel')
    laplacian_edges = processor.detect_edges(image, method='laplacian')
    
    cv2.imwrite("test_output/canny_edges.png", canny_edges)
    cv2.imwrite("test_output/sobel_edges.png", sobel_edges)
    cv2.imwrite("test_output/laplacian_edges.png", laplacian_edges)
    print("✓ Edge detection (Canny, Sobel, Laplacian)")
    
    # 5. Thresholding methods
    binary_thresh = processor.apply_threshold(gray, method='binary')
    adaptive_thresh = processor.apply_threshold(gray, method='adaptive')
    otsu_thresh = processor.apply_threshold(gray, method='otsu')
    
    cv2.imwrite("test_output/binary_thresh.png", binary_thresh)
    cv2.imwrite("test_output/adaptive_thresh.png", adaptive_thresh)
    cv2.imwrite("test_output/otsu_thresh.png", otsu_thresh)
    print("✓ Thresholding (binary, adaptive, Otsu)")
    
    # 6. Morphological operations
    dilated = processor.apply_morphology(binary_thresh, operation='dilate')
    eroded = processor.apply_morphology(binary_thresh, operation='erode')
    opened = processor.apply_morphology(binary_thresh, operation='open')
    closed = processor.apply_morphology(binary_thresh, operation='close')
    
    cv2.imwrite("test_output/dilated.png", dilated)
    cv2.imwrite("test_output/eroded.png", eroded)
    cv2.imwrite("test_output/opened.png", opened)
    cv2.imwrite("test_output/closed.png", closed)
    print("✓ Morphological operations (dilate, erode, open, close)")
    
    # 7. UI element enhancement
    buttons_enhanced = processor.enhance_ui_elements(image, element_type='buttons')
    text_fields_enhanced = processor.enhance_ui_elements(image, element_type='text_fields')
    checkboxes_enhanced = processor.enhance_ui_elements(image, element_type='checkboxes')
    
    cv2.imwrite("test_output/buttons_enhanced.png", buttons_enhanced)
    cv2.imwrite("test_output/text_fields_enhanced.png", text_fields_enhanced)
    cv2.imwrite("test_output/checkboxes_enhanced.png", checkboxes_enhanced)
    print("✓ UI element enhancement (buttons, text fields, checkboxes)")
    
    # 8. Text enhancement
    text_enhanced = processor.enhance_text(image)
    cv2.imwrite("test_output/text_enhanced.png", text_enhanced)
    print("✓ Text enhancement")
    
    # 9. Sharpening
    sharpened = processor.sharpen(image, amount=1.5)
    cv2.imwrite("test_output/sharpened.png", sharpened)
    print("✓ Image sharpening")
    
    # 10. Processing pipeline
    pipeline_result = processor.preprocess_pipeline(image, [
        {'method': 'to_grayscale'},
        {'method': 'enhance_contrast', 'method_param': 'clahe'},
        {'method': 'reduce_noise', 'method_param': 'gaussian', 'kernel_size': 3},
        {'method': 'sharpen', 'amount': 0.5},
        {'method': 'apply_threshold', 'method_param': 'adaptive'}
    ])
    
    cv2.imwrite("test_output/pipeline_result.png", pipeline_result)
    print("✓ Processing pipeline")
    
    # Print summary
    print("\nImage processing tests completed.")
    print(f"Original image dimensions: {image.shape[1]}x{image.shape[0]}")
    print(f"Total processed images saved: 20")
    print("All outputs saved to test_output/ directory")

if __name__ == "__main__":
    test_image_processor() 