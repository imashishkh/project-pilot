#!/usr/bin/env python3
"""
Test module for text recognition functionality.

This module tests the ability to recognize and extract text from images.
"""

import os
import sys
import logging
import pytest
import numpy as np
import cv2
import json
from pathlib import Path

# Import directly from MacAgent package or alternative paths
try:
    from MacAgent.src.vision.text_recognition import TextRecognizer
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
        from MacAgent.src.vision.text_recognition import TextRecognizer
        from MacAgent.src.vision.screen_capture import ScreenCapture
    except ModuleNotFoundError:
        from src.vision.text_recognition import TextRecognizer
        from src.vision.screen_capture import ScreenCapture

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_text_recognition():
    # Create output directory if it doesn't exist
    os.makedirs("test_output", exist_ok=True)
    
    # Initialize screen capture with default settings
    capture = ScreenCapture()
    
    # Capture the screen
    image = capture.capture()
    
    # Save the original image for reference
    cv2.imwrite("test_output/original_for_ocr.png", image)
    print("✓ Captured screen for text recognition")
    
    # Create a text recognizer
    recognizer = TextRecognizer(preprocess=True)
    
    # Test 1: Basic text extraction
    print("\nTest 1: Basic text extraction")
    text = recognizer.read_text(image)
    print(f"Extracted {len(text.split())} words")
    with open("test_output/extracted_text.txt", "w") as f:
        f.write(text)
    print("✓ Saved extracted text to test_output/extracted_text.txt")
    
    # Test 2: Text block detection
    print("\nTest 2: Text block detection")
    # Using our TextRecognizer's method
    try:
        text_blocks = recognizer.detect_text_blocks(image)
        print(f"Detected {len(text_blocks)} text blocks")
        
        # Visualize text blocks on image
        annotated_image = image.copy()
        for block in text_blocks:
            x, y, w, h = block.box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated_image, block.text[:10], (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite("test_output/text_blocks.png", annotated_image)
        print(f"Saved text blocks visualization to test_output/text_blocks.png")
    except Exception as e:
        print(f"Error in text block detection: {e}")
    
    # Direct approach using pytesseract.image_to_boxes
    try:
        import pytesseract
        # Get character boxes
        h, w = image.shape[:2]
        boxes = pytesseract.image_to_boxes(image)
        
        # Visualize character boxes
        char_image = image.copy()
        if boxes:
            for b in boxes.splitlines():
                parts = b.split(' ')
                if len(parts) >= 6:
                    char, x1, y1, x2, y2 = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    # Convert from bottom-left origin to top-left origin
                    y1 = h - y1
                    y2 = h - y2
                    # Draw rectangle
                    cv2.rectangle(char_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            
            cv2.imwrite("test_output/text_boxes.png", char_image)
            print(f"Saved character boxes visualization to test_output/text_boxes.png")
        else:
            print("No character boxes detected")
    except Exception as e:
        print(f"Error in direct character box detection: {e}")
    
    # Test 3: Finding specific text
    print("\nTest 3: Text search")
    search_terms = ["menu", "file", "button", "click"]
    
    for term in search_terms:
        matches = recognizer.find_text(image, term, case_sensitive=False)
        print(f"Found {len(matches)} matches for '{term}'")
        
        if matches:
            # Visualize the first match
            match = matches[0]
            match_image = image.copy()
            x, y, w, h = match.box
            cv2.rectangle(match_image, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(match_image, f"Found: {term}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imwrite(f"test_output/found_{term}.png", match_image)
            print(f"✓ Saved match for '{term}' to test_output/found_{term}.png")
    
    # Test 4: Paragraph detection
    print("\nTest 4: Paragraph detection")
    paragraphs = recognizer.detect_paragraphs(image)
    print(f"Detected {len(paragraphs)} paragraphs")
    
    # Visualize paragraphs
    paragraph_image = image.copy()
    for i, para in enumerate(paragraphs):
        x, y, w, h = para.box
        # Use a different color for each paragraph (cycling through 5 colors)
        color = [
            (255, 0, 0),   # Blue
            (0, 255, 0),   # Green
            (0, 0, 255),   # Red
            (255, 255, 0), # Cyan
            (0, 255, 255)  # Yellow
        ][i % 5]
        
        cv2.rectangle(paragraph_image, (x, y), (x+w, y+h), color, 2)
        # Add paragraph number
        cv2.putText(paragraph_image, f"Para {i+1}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv2.imwrite("test_output/paragraphs.png", paragraph_image)
    print("✓ Saved image with detected paragraphs to test_output/paragraphs.png")
    
    # Test 5: Text orientation detection
    print("\nTest 5: Text orientation detection")
    orientation = recognizer.get_text_orientation(image)
    print(f"Detected text orientation: {orientation}")
    
    # Test 6: Text cleaning
    print("\nTest 6: Text cleaning")
    sample_text = "This   is a\n\n  text with\textra  spaces and |ines and 0ther issues."
    cleaned_text = recognizer.clean_text(sample_text)
    print(f"Original: '{sample_text}'")
    print(f"Cleaned:  '{cleaned_text}'")
    
    # Test 7: Preprocessing for OCR
    print("\nTest 7: Image preprocessing for OCR")
    preprocessed = recognizer._preprocess_for_ocr(image)
    cv2.imwrite("test_output/preprocessed_for_ocr.png", preprocessed)
    print("✓ Saved preprocessed image to test_output/preprocessed_for_ocr.png")
    
    # Test 8: Direct test of pytesseract.image_to_data to debug the issue
    print("\nTest 8: Direct test of pytesseract.image_to_data")
    try:
        import pytesseract
        # Test directly with a small, controlled image
        test_img = np.zeros((100, 300, 3), dtype=np.uint8)
        test_img.fill(255)  # White background
        cv2.putText(test_img, "Test Text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(os.path.join("test_output", "test_text.png"), test_img)
        
        print("Testing image_to_data with a simple test image...")
        # Use a different config with explicit PSM mode
        data = pytesseract.image_to_data(
            test_img,
            lang='eng',
            config='--psm 6',  # Assume a single uniform block of text
            output_type=pytesseract.Output.DICT
        )
        
        if data and 'text' in data:
            print(f"Success! Found {len(data['text'])} entries")
            print(f"Text entries: {data['text']}")
        else:
            print("No text data returned")
    except Exception as e:
        print(f"Error testing image_to_data directly: {e}")
    
    print("\nText recognition tests completed.")
    print(f"All outputs saved to test_output/ directory")

if __name__ == "__main__":
    test_text_recognition() 