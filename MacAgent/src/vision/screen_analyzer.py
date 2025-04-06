"""
Screen Analyzer Module for MacAgent

This module provides comprehensive screen analysis capabilities, including
text extraction (OCR), UI element detection, and image preprocessing
optimized for macOS interfaces.
"""

import time
import logging
import numpy as np
import cv2
import os
import hashlib
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# OCR Dependencies
import pytesseract
from PIL import Image

# Set pytesseract command if needed (macOS often needs this)
if os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
elif os.path.exists('/usr/local/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
else:
    logging.warning("Tesseract not found in common locations. OCR may not work correctly.")

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Class representing a UI element detected on screen."""
    
    element_type: str  # Type of element (button, text, etc.)
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float  # Detection confidence (0-1)
    text: Optional[str] = None  # Text content if available
    properties: Optional[Dict[str, Any]] = None  # Additional properties
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the element."""
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        """Get the area of the element in pixels."""
        _, _, w, h = self.bounds
        return w * h
    
    def distance_to(self, other: 'UIElement') -> float:
        """Calculate distance to another UI element (center to center)."""
        x1, y1 = self.center
        x2, y2 = other.center
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def overlaps_with(self, other: 'UIElement', threshold: float = 0.5) -> bool:
        """Check if this element overlaps with another element."""
        x1, y1, w1, h1 = self.bounds
        x2, y2, w2, h2 = other.bounds
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        intersection = x_overlap * y_overlap
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # Calculate IoU (Intersection over Union)
        iou = intersection / union if union > 0 else 0
        
        return iou >= threshold
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """Check if the element contains the given point."""
        x, y, w, h = self.bounds
        px, py = point
        return (x <= px <= x + w) and (y <= py <= y + h)


class ScreenAnalyzer:
    """Class for analyzing screen content and detecting UI elements."""
    
    def __init__(self, 
                ocr_enabled: bool = True, 
                preprocess_image: bool = True,
                ui_detection_enabled: bool = True,
                template_dir: Optional[str] = None):
        """Initialize the screen analyzer.
        
        Args:
            ocr_enabled: Whether to enable OCR text extraction
            preprocess_image: Whether to preprocess images before analysis
            ui_detection_enabled: Whether to enable UI element detection
            template_dir: Directory containing UI element templates
        """
        self.ocr_enabled = ocr_enabled
        self.preprocess_image = preprocess_image
        self.ui_detection_enabled = ui_detection_enabled
        
        # Set up template matching
        self.template_dir = Path(template_dir) if template_dir else None
        self._templates = {}
        
        # Set up cache
        self._cache = {}
        self._load_templates()
        
        logger.info(f"Initialized ScreenAnalyzer: OCR={ocr_enabled}, "
                   f"Preprocessing={preprocess_image}, UI Detection={ui_detection_enabled}")
    
    def _load_templates(self) -> None:
        """Load UI element templates from the template directory."""
        if not self.template_dir or not self.template_dir.exists():
            logger.warning("Template directory not found. Template matching will be limited.")
            return
        
        template_types = {
            "button": "buttons",
            "checkbox": "checkboxes",
            "radio": "radio_buttons",
            "textfield": "textfields",
            "menu": "menus",
            "tab": "tabs"
        }
        
        # Load templates for each UI element type
        for element_type, subdir in template_types.items():
            template_path = self.template_dir / subdir
            if not template_path.exists():
                continue
            
            templates = []
            for img_path in template_path.glob("*.png"):
                try:
                    template = cv2.imread(str(img_path))
                    if template is not None:
                        # Store template with its name
                        template_name = img_path.stem
                        templates.append({
                            "name": template_name,
                            "image": template,
                            "path": img_path
                        })
                        logger.debug(f"Loaded template: {img_path}")
                except Exception as e:
                    logger.error(f"Error loading template {img_path}: {e}")
            
            if templates:
                self._templates[element_type] = templates
                logger.info(f"Loaded {len(templates)} templates for {element_type}")
    
    def _image_hash(self, image: np.ndarray) -> str:
        """Generate a hash for an image to use as cache key."""
        # Resize to small dimensions for faster hashing
        small = cv2.resize(image, (32, 32))
        # Create a hash
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def _check_cache(self, image: np.ndarray, analysis_type: str) -> Optional[Any]:
        """Check if we have cached results for this image and analysis type."""
        image_hash = self._image_hash(image)
        cache_key = f"{image_hash}_{analysis_type}"
        
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {analysis_type}")
            return self._cache[cache_key]
        
        return None
    
    def _add_to_cache(self, image: np.ndarray, analysis_type: str, result: Any) -> None:
        """Add results to the cache."""
        image_hash = self._image_hash(image)
        cache_key = f"{image_hash}_{analysis_type}"
        self._cache[cache_key] = result
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache = {}
        logger.debug("Analysis cache cleared")
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR to improve text recognition."""
        if not self.preprocess_image:
            return image
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get black and white image
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            return opening
        except Exception as e:
            logger.error(f"Error in OCR preprocessing: {e}")
            return image
    
    def _preprocess_for_element_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for UI element detection."""
        if not self.preprocess_image:
            return image
        
        try:
            # Denoise
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Enhance contrast
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception as e:
            logger.error(f"Error in element detection preprocessing: {e}")
            return image
    
    def extract_text(self, image: np.ndarray, lang: str = 'eng') -> str:
        """Extract text from an image using OCR.
        
        Args:
            image: Image to extract text from
            lang: Language for OCR
            
        Returns:
            Extracted text as string
        """
        if not self.ocr_enabled:
            logger.warning("OCR is disabled")
            return ""
        
        # Check cache
        cached = self._check_cache(image, f"ocr_{lang}")
        if cached is not None:
            return cached
        
        try:
            # Preprocess the image
            processed = self._preprocess_for_ocr(image)
            
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(processed)
            
            # Perform OCR
            text = pytesseract.image_to_string(pil_image, lang=lang)
            
            # Cache the result
            self._add_to_cache(image, f"ocr_{lang}", text)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text with OCR: {e}")
            return ""
    
    def extract_text_blocks(self, image: np.ndarray, lang: str = 'eng') -> List[Dict[str, Any]]:
        """Extract text blocks with position information.
        
        Args:
            image: Image to extract text from
            lang: Language for OCR
            
        Returns:
            List of text blocks with position information
        """
        if not self.ocr_enabled:
            logger.warning("OCR is disabled")
            return []
        
        # Check cache
        cached = self._check_cache(image, f"ocr_blocks_{lang}")
        if cached is not None:
            return cached
        
        try:
            # Preprocess the image
            processed = self._preprocess_for_ocr(image)
            
            # Get text blocks with bounding box information
            data = pytesseract.image_to_data(processed, lang=lang, output_type=pytesseract.Output.DICT)
            
            # Organize into blocks
            blocks = []
            for i in range(len(data['text'])):
                # Skip empty text
                if not data['text'][i].strip():
                    continue
                
                # Check for confidence (Tesseract 4+)
                confidence = data['conf'][i] if 'conf' in data else 100.0
                
                block = {
                    'text': data['text'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'confidence': confidence
                }
                blocks.append(block)
            
            # Cache the result
            self._add_to_cache(image, f"ocr_blocks_{lang}", blocks)
            
            return blocks
        except Exception as e:
            logger.error(f"Error extracting text blocks with OCR: {e}")
            return []
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges in the image for UI element detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            return edges
        except Exception as e:
            logger.error(f"Error detecting edges: {e}")
            return np.zeros_like(image[:,:,0])
    
    def _find_contours(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Find contours that might represent UI elements."""
        edges = self._detect_edges(image)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        processed_contours = []
        min_area = 100  # Minimum area to consider
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Skip tiny contours
            if area < min_area:
                continue
            
            # Skip excessively large contours (likely false positives)
            if area > 0.5 * image.shape[0] * image.shape[1]:
                continue
            
            # Skip contours with irregular aspect ratios
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:
                continue
            
            processed_contours.append({
                "contour": contour,
                "bounds": (x, y, w, h),
                "area": area
            })
        
        return processed_contours
    
    def _classify_element_type(self, image: np.ndarray, bounds: Tuple[int, int, int, int]) -> str:
        """Classify the type of UI element based on appearance."""
        x, y, w, h = bounds
        
        # Extract the region
        if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
            return "unknown"
        
        element_img = image[y:y+h, x:x+w]
        
        # Simple classification based on aspect ratio and appearance
        aspect_ratio = w / h if h > 0 else 0
        
        # Basic heuristics for element classification
        if 2.5 <= aspect_ratio <= 6.0 and w >= 50 and h >= 20:
            # Likely a button
            return "button"
        elif 0.8 <= aspect_ratio <= 1.2 and w <= 40:
            # Might be a checkbox or radio button
            return "checkbox"
        elif aspect_ratio >= 4.0 and h <= 40:
            # Likely a text field
            return "textfield"
        elif h <= 30:
            # Might be a menu item
            return "menu"
        else:
            # Generic container or unknown
            return "container"
    
    def detect_ui_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements in the image.
        
        Args:
            image: Image to analyze
            
        Returns:
            List of detected UIElement objects
        """
        if not self.ui_detection_enabled:
            logger.warning("UI detection is disabled")
            return []
        
        # Check cache
        cached = self._check_cache(image, "ui_elements")
        if cached is not None:
            return cached
        
        try:
            # Preprocess the image
            processed = self._preprocess_for_element_detection(image)
            
            # First, get text blocks (they are likely UI elements or contained within them)
            text_blocks = self.extract_text_blocks(image)
            
            # Then find contours for potential UI elements
            contours = self._find_contours(processed)
            
            # Track already detected regions to avoid duplicates
            detected_regions = set()
            ui_elements = []
            
            # Convert text blocks to UI elements
            for block in text_blocks:
                x, y, w, h = block['x'], block['y'], block['width'], block['height']
                
                # Skip tiny or empty text blocks
                if w * h < 100 or not block['text'].strip():
                    continue
                
                # Expand the region slightly to get the whole UI element
                x_expanded = max(0, x - 5)
                y_expanded = max(0, y - 5)
                w_expanded = min(image.shape[1] - x_expanded, w + 10)
                h_expanded = min(image.shape[0] - y_expanded, h + 10)
                
                # Create a region key to check for duplicates
                region_key = f"{x_expanded},{y_expanded},{w_expanded},{h_expanded}"
                if region_key in detected_regions:
                    continue
                
                detected_regions.add(region_key)
                
                # Create UI element (most text blocks are labels or interactive elements)
                element = UIElement(
                    element_type="text",
                    bounds=(x_expanded, y_expanded, w_expanded, h_expanded),
                    confidence=float(block['confidence']) / 100.0,
                    text=block['text'],
                    properties={"source": "text_block"}
                )
                
                ui_elements.append(element)
            
            # Process contours to find additional UI elements
            for contour_info in contours:
                bounds = contour_info["bounds"]
                x, y, w, h = bounds
                
                # Check if this region overlaps significantly with already detected elements
                region_key = f"{x},{y},{w},{h}"
                if region_key in detected_regions:
                    continue
                
                # Check for overlap with existing elements
                overlapping = False
                for element in ui_elements:
                    x2, y2, w2, h2 = element.bounds
                    
                    # Calculate intersection
                    intersection_w = min(x + w, x2 + w2) - max(x, x2)
                    intersection_h = min(y + h, y2 + h2) - max(y, y2)
                    
                    if intersection_w > 0 and intersection_h > 0:
                        intersection_area = intersection_w * intersection_h
                        min_area = min(w * h, w2 * h2)
                        
                        # If the overlap is significant, skip this contour
                        if intersection_area > 0.7 * min_area:
                            overlapping = True
                            break
                
                if overlapping:
                    continue
                
                # Not overlapping with existing elements, so classify and add
                element_type = self._classify_element_type(image, bounds)
                detected_regions.add(region_key)
                
                # Extract text within this region if it's likely to contain text
                element_text = None
                if element_type in ["button", "textfield", "menu"]:
                    # Extract a bit larger region to ensure we get all text
                    text_region = image[
                        max(0, y-5):min(image.shape[0], y+h+5), 
                        max(0, x-5):min(image.shape[1], x+w+5)
                    ]
                    element_text = self.extract_text(text_region).strip()
                
                element = UIElement(
                    element_type=element_type,
                    bounds=bounds,
                    confidence=0.7,  # Default confidence for contour-based detection
                    text=element_text,
                    properties={"source": "contour", "area": contour_info["area"]}
                )
                
                ui_elements.append(element)
            
            # Template matching for common UI elements
            template_elements = self.detect_ui_elements_with_templates(image)
            for element in template_elements:
                # Check for overlap with existing elements
                overlapping = False
                x, y, w, h = element.bounds
                region_key = f"{x},{y},{w},{h}"
                
                if region_key in detected_regions:
                    continue
                
                for existing in ui_elements:
                    if element.overlaps_with(existing, threshold=0.3):
                        overlapping = True
                        break
                
                if not overlapping:
                    detected_regions.add(region_key)
                    ui_elements.append(element)
            
            # Sort elements by y-coordinate for a more logical order
            ui_elements.sort(key=lambda e: (e.bounds[1], e.bounds[0]))
            
            # Cache the result
            self._add_to_cache(image, "ui_elements", ui_elements)
            
            return ui_elements
        
        except Exception as e:
            logger.error(f"Error detecting UI elements: {e}")
            return []
    
    def detect_ui_elements_with_templates(self, image: np.ndarray) -> List[UIElement]:
        """Detect UI elements using template matching.
        
        Args:
            image: Image to analyze
            
        Returns:
            List of detected UIElement objects
        """
        if not self._templates:
            return []
        
        detected_elements = []
        
        try:
            # Convert to grayscale for template matching
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Process each template type
            for element_type, templates in self._templates.items():
                for template_info in templates:
                    template = template_info["image"]
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    
                    # Match template
                    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.7  # Confidence threshold
                    
                    # Get matches above threshold
                    locations = np.where(result >= threshold)
                    for pt in zip(*locations[::-1]):
                        x, y = pt
                        w, h = template_gray.shape[1], template_gray.shape[0]
                        
                        # Get the confidence score
                        confidence = result[y, x]
                        
                        # Extract the element region
                        element_region = image[y:y+h, x:x+w]
                        
                        # Try to extract text if the element might contain it
                        text = None
                        if element_type in ["button", "textfield", "menu"]:
                            text = self.extract_text(element_region).strip()
                        
                        # Create UI element
                        element = UIElement(
                            element_type=element_type,
                            bounds=(x, y, w, h),
                            confidence=float(confidence),
                            text=text,
                            properties={
                                "source": "template",
                                "template_name": template_info["name"]
                            }
                        )
                        
                        detected_elements.append(element)
            
            return detected_elements
        
        except Exception as e:
            logger.error(f"Error in template matching: {e}")
            return []
    
    def find_element_by_text(self, image: np.ndarray, text: str, 
                           element_type: Optional[str] = None) -> Optional[UIElement]:
        """Find a UI element containing the specified text.
        
        Args:
            image: Image to search in
            text: Text to search for (case insensitive)
            element_type: Optional type of element to limit search to
            
        Returns:
            UIElement if found, None otherwise
        """
        # Get all UI elements
        elements = self.detect_ui_elements(image)
        
        # Normalize search text
        text_lower = text.lower()
        
        # Find elements with matching text
        matching_elements = []
        for element in elements:
            if element.text and text_lower in element.text.lower():
                if element_type is None or element.element_type == element_type:
                    matching_elements.append((element, element.text.lower().find(text_lower)))
        
        if not matching_elements:
            return None
        
        # Sort by relevance (exact match, then position of match, then confidence)
        matching_elements.sort(key=lambda x: (
            0 if x[0].text.lower() == text_lower else 1,  # Exact match first
            x[1],  # Position of match in text
            -x[0].confidence  # Higher confidence first
        ))
        
        return matching_elements[0][0]
    
    def analyze_screen(self, image: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive analysis of the screen.
        
        Args:
            image: Image to analyze
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        # Check cache
        cached = self._check_cache(image, "full_analysis")
        if cached is not None:
            return cached
        
        results = {}
        
        # 1. UI Element Detection
        if self.ui_detection_enabled:
            results["ui_elements"] = self.detect_ui_elements(image)
        else:
            results["ui_elements"] = []
        
        # 2. Text Extraction
        if self.ocr_enabled:
            results["text"] = self.extract_text(image)
            results["text_blocks"] = self.extract_text_blocks(image)
        else:
            results["text"] = ""
            results["text_blocks"] = []
        
        # 3. Calculate analysis metrics
        results["analysis_time"] = time.time() - start_time
        results["image_dimensions"] = (image.shape[1], image.shape[0])
        
        # Cache the results
        self._add_to_cache(image, "full_analysis", results)
        
        return results


def test_screen_analyzer():
    """Test function to demonstrate the screen analyzer functionality."""
    # Import screen capture to get a test image
    from screen_capture import ScreenCapture
    
    # Initialize components
    screen_capture = ScreenCapture()
    screen_analyzer = ScreenAnalyzer(
        ocr_enabled=True,
        preprocess_image=True,
        ui_detection_enabled=True
    )
    
    # Capture screen
    screenshot = screen_capture.capture()
    
    # Analyze screen
    print("Analyzing screen...")
    results = screen_analyzer.analyze_screen(screenshot)
    
    # Print results summary
    print(f"Analysis completed in {results['analysis_time']:.2f} seconds")
    print(f"Detected {len(results['ui_elements'])} UI elements")
    print(f"Extracted {len(results['text'])} characters of text")
    
    # Save annotated image
    annotated = screenshot.copy()
    
    # Draw UI elements
    for element in results['ui_elements']:
        x, y, w, h = element.bounds
        
        # Color based on element type
        if element.element_type == "button":
            color = (0, 255, 0)  # Green for buttons
        elif element.element_type == "text":
            color = (255, 0, 0)  # Blue for text
        elif element.element_type == "textfield":
            color = (0, 0, 255)  # Red for text fields
        else:
            color = (255, 255, 0)  # Yellow for other elements
        
        # Draw rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        # Add element type label
        cv2.putText(annotated, element.element_type, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Save the annotated image
    cv2.imwrite("annotated_screen.png", annotated)
    print("Saved annotated image to 'annotated_screen.png'")
    
    # Clean up
    screen_capture.close()


if __name__ == "__main__":
    test_screen_analyzer() 