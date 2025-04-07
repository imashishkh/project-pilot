"""
Perception Module for MacAgent.

This module handles all screen capture and analysis functionality for the agent.
It provides capabilities to observe the screen, detect UI elements, and extract text.
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import pyautogui
from PIL import Image

# Configure logger
logger = logging.getLogger(__name__)

class PerceptionModule:
    """
    Handles screen observation and analysis functionality.
    
    This module is responsible for capturing the screen, analyzing UI elements,
    detecting text, and providing visual information to the agent.
    """
    
    def __init__(self, capture_region: Optional[Tuple[int, int, int, int]] = None,
                 screenshot_interval: float = 0.1):
        """
        Initialize the perception module.
        
        Args:
            capture_region: Optional tuple of (x, y, width, height) defining screen region to capture.
                           If None, captures the entire screen.
            screenshot_interval: Time in seconds between screenshot captures.
        """
        self.capture_region = capture_region
        self.screenshot_interval = screenshot_interval
        self.current_screenshot: Optional[np.ndarray] = None
        self.last_capture_time: float = 0.0
        logger.info("PerceptionModule initialized")
    
    async def capture_screen(self) -> np.ndarray:
        """
        Capture a screenshot asynchronously.
        
        Returns:
            Screenshot as numpy array in BGR format.
        """
        # Rate limit captures to avoid excessive CPU usage
        current_time = time.time()
        if current_time - self.last_capture_time < self.screenshot_interval:
            await asyncio.sleep(self.screenshot_interval - (current_time - self.last_capture_time))
        
        # Capture screen
        try:
            if self.capture_region:
                logger.debug(f"Capturing screen region: {self.capture_region}")
                screenshot = pyautogui.screenshot(region=self.capture_region)
            else:
                logger.debug("Capturing full screen")
                screenshot = pyautogui.screenshot()
            
            # Convert to numpy array in BGR format (cv2 format)
            self.current_screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            self.last_capture_time = time.time()
            return self.current_screenshot
        except Exception as e:
            logger.error(f"Error capturing screen: {str(e)}")
            if self.current_screenshot is not None:
                return self.current_screenshot
            # Return a black image if no previous screenshot available
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    async def initialize(self) -> None:
        """
        Initialize the perception module.
        Called by the AgentLoop during startup.
        """
        logger.debug("Initializing perception module...")
        # Reset the screenshot timing
        self.last_capture_time = 0.0
        # Take an initial screenshot to verify capture works
        try:
            await self.capture_screen()
            logger.info("Successfully captured initial screenshot")
        except Exception as e:
            logger.error(f"Error capturing initial screenshot: {str(e)}")
        logger.debug("Perception module initialization complete")
        
    async def cleanup(self) -> None:
        """
        Clean up perception module resources.
        Called by the AgentLoop during shutdown.
        """
        logger.debug("Cleaning up perception module resources...")
        # Release any resources if needed
        self.current_screenshot = None
        logger.debug("Perception module cleanup complete")
    
    async def get_ui_elements(self) -> List[Dict[str, Any]]:
        """
        Identify UI elements in the current screenshot.
        
        Returns:
            List of dictionaries containing UI element information.
        """
        if self.current_screenshot is None:
            await self.capture_screen()
            
        if self.current_screenshot is None:
            logger.error("Failed to get screenshot for UI element detection")
            return []
            
        # TODO: Implement UI element detection using computer vision
        # This is a placeholder that would be replaced with actual UI detection
        # Could use template matching, object detection models, or other CV techniques
        
        logger.debug("UI element detection requested - implementation pending")
        return []
    
    async def extract_text(self, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Extract text from the specified region of the current screenshot.
        
        Args:
            region: Optional tuple of (x, y, width, height) to extract text from.
                   If None, extracts from the entire screenshot.
        
        Returns:
            Extracted text as string.
        """
        if self.current_screenshot is None:
            await self.capture_screen()
            
        if self.current_screenshot is None:
            logger.error("Failed to get screenshot for text extraction")
            return ""
        
        try:
            # If region specified, crop the screenshot
            target_image = self.current_screenshot
            if region:
                x, y, w, h = region
                target_image = target_image[y:y+h, x:x+w]
            
            # TODO: Implement text extraction using OCR (e.g., pytesseract)
            # This placeholder would be replaced with actual OCR implementation
            
            logger.debug("Text extraction requested - implementation pending")
            return ""
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def find_template(self, template_image: np.ndarray, 
                      threshold: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a template image within the current screenshot.
        
        Args:
            template_image: The template image to find (numpy array).
            threshold: Matching threshold (0.0 to 1.0).
            
        Returns:
            Tuple of (x, y, width, height) if found, None otherwise.
        """
        if self.current_screenshot is None:
            logger.error("No screenshot available for template matching")
            return None
        
        try:
            # Convert template to grayscale if it's color
            if len(template_image.shape) == 3:
                template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_image
                
            # Convert screenshot to grayscale
            screenshot_gray = cv2.cvtColor(self.current_screenshot, cv2.COLOR_BGR2GRAY)
            
            # Template matching
            h, w = template_gray.shape
            result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            
            # Find position with highest match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val >= threshold:
                x, y = max_loc
                logger.debug(f"Template found at ({x}, {y}) with confidence {max_val:.2f}")
                return (x, y, w, h)
            else:
                logger.debug(f"Template not found (best match: {max_val:.2f})")
                return None
                
        except Exception as e:
            logger.error(f"Error during template matching: {str(e)}")
            return None
