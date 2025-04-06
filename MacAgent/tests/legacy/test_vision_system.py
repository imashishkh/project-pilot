#!/usr/bin/env python3
"""
Tests for the MacAgent vision system.

This script tests the screen capture, element detection, and context analysis
functionality of the MacAgent vision system.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MacAgent.src.vision.screen_capture import ScreenCapture
from MacAgent.src.vision.element_detector import UIElementDetector, ElementType
from MacAgent.src.vision.context_analyzer import ContextAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_element_detection():
    """Test the UI element detection system."""
    logger.info("Starting UI element detection test...")
    
    # Initialize screen capture and element detector
    capture = ScreenCapture()
    detector = UIElementDetector()
    
    # Capture screen and detect elements
    image = capture.capture()
    if image is None:
        logger.error("Failed to capture screen")
        return
    
    logger.info(f"Captured screen with dimensions: {image.shape}")
    
    # Detect UI elements
    elements = detector.detect_elements(image)
    
    # Display results
    logger.info(f"Detected {len(elements)} UI elements")
    for i, element in enumerate(elements):
        logger.info(f"Element {i}: {element.element_type.name} at {element.bounding_box}, confidence: {element.confidence:.2f}")
        if element.text:
            logger.info(f"  Text: {element.text}")
    
    # Save a visualization of detected elements
    output_dir = Path(__file__).parent.parent / "test_output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        import cv2
        import numpy as np
        
        # Create a copy of the image for visualization
        viz_image = image.copy()
        
        # Draw bounding boxes around detected elements
        for element in elements:
            x, y, w, h = element.bounding_box
            # Choose color based on element type
            color = (0, 255, 0)  # Default: green
            
            if element.element_type == ElementType.BUTTON:
                color = (0, 0, 255)  # Red for buttons
            elif element.element_type == ElementType.TEXT_FIELD:
                color = (255, 0, 0)  # Blue for text fields
            elif element.element_type == ElementType.CHECKBOX:
                color = (255, 255, 0)  # Cyan for checkboxes
            
            # Draw rectangle and element type label
            cv2.rectangle(viz_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(viz_image, str(element.element_type.name), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save the visualization
        output_path = output_dir / f"element_detection_{int(time.time())}.png"
        cv2.imwrite(str(output_path), viz_image)
        logger.info(f"Saved visualization to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")


def test_context_analysis():
    """Test the context analysis system."""
    logger.info("Starting context analysis test...")
    
    # Initialize screen capture and context analyzer
    capture = ScreenCapture()
    context_analyzer = ContextAnalyzer()
    
    # Capture screen and analyze context
    image = capture.capture()
    if image is None:
        logger.error("Failed to capture screen")
        return
    
    # Analyze context
    context = context_analyzer.analyze_context(image)
    
    # Display results
    logger.info("Context Analysis Results:")
    
    # Display active window information
    active_window = context.get("active_window")
    if active_window:
        logger.info(f"Active Window: {active_window.app_name} - {active_window.window_name}")
        logger.info(f"Window Bounds: {active_window.bounds}")
        logger.info(f"Application Context: {context.get('app_context')}")
    else:
        logger.info("No active window detected")
    
    # Display all windows
    windows = context.get("windows", [])
    logger.info(f"Detected {len(windows)} windows:")
    for i, window in enumerate(windows):
        logger.info(f"  Window {i}: {window.app_name} - {window.window_name}")
    
    # Display task context
    task_context = context.get("task_context", {})
    logger.info(f"Task Type: {task_context.get('task_type', 'unknown')}")
    logger.info(f"Form Filling: {task_context.get('form_filling', False)}")
    logger.info(f"Reading Content: {task_context.get('reading_content', False)}")
    logger.info(f"Media Controls: {task_context.get('media_controls', False)}")
    logger.info(f"Navigation: {task_context.get('navigation', False)}")
    
    # Display available actions
    actions = task_context.get("actions_available", [])
    if actions:
        logger.info(f"Available Actions: {len(actions)}")
        for action in actions:
            logger.info(f"  {action.get('action')} {action.get('target')}")
    
    # Display UI elements count
    elements = context.get("elements", [])
    logger.info(f"Total UI Elements: {len(elements)}")
    
    # Display theme
    logger.info(f"Interface Theme: {context.get('overall_theme')}")


def main():
    """Run all vision system tests."""
    logger.info("Starting vision system tests...")
    
    try:
        test_element_detection()
        logger.info("-" * 50)
        test_context_analysis()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
    
    logger.info("Vision system tests completed.")


if __name__ == "__main__":
    main() 