"""
Text Recognition Module for MacAgent

This module provides text recognition capabilities for detecting and extracting
text from screen captures and UI elements, supporting OCR operations.
"""

import os
import re
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import pytesseract for OCR functionality
try:
    import pytesseract
    HAS_TESSERACT = True
    
    # Try to find tesseract executable
    if not pytesseract.pytesseract.tesseract_cmd:
        # Common locations for tesseract on macOS
        common_locations = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract'
        ]
        
        for location in common_locations:
            if os.path.exists(location):
                pytesseract.pytesseract.tesseract_cmd = location
                break
        
        if not pytesseract.pytesseract.tesseract_cmd:
            logger.warning("Tesseract not found in common locations. OCR may not work correctly.")
            HAS_TESSERACT = False
    
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed. OCR functionality will be limited.")

# Import our image processor for preprocessing
from .image_processor import ImageProcessor


class TextBlock:
    """Represents a block of text detected in an image."""
    
    def __init__(self, text: str, box: Tuple[int, int, int, int], confidence: float = 0.0):
        """
        Initialize a text block.
        
        Args:
            text: The extracted text content
            box: Bounding box coordinates (x, y, width, height)
            confidence: Confidence score of the OCR result (0-100)
        """
        self.text = text
        self.box = box
        self.confidence = confidence
        
        # Calculate additional properties
        self.x, self.y, self.width, self.height = box
        self.area = self.width * self.height
        self.center = (self.x + self.width // 2, self.y + self.height // 2)
    
    def __str__(self) -> str:
        """String representation of the text block."""
        return f"TextBlock('{self.text}', box={self.box}, confidence={self.confidence:.1f}%)"
    
    def get_image_region(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the image region corresponding to this text block.
        
        Args:
            image: The source image
            
        Returns:
            The image region containing this text block
        """
        x, y, w, h = self.box
        # Ensure coordinates are within image boundaries
        h_max, w_max = image.shape[:2]
        x = max(0, min(x, w_max - 1))
        y = max(0, min(y, h_max - 1))
        w = min(w, w_max - x)
        h = min(h, h_max - y)
        
        return image[y:y+h, x:x+w]


class TextMatch(TextBlock):
    """Represents a match for a text pattern search in an image."""
    
    def __init__(self, text: str, box: Tuple[int, int, int, int], confidence: float = 0.0):
        """
        Initialize a text match.
        
        Args:
            text: The matched text
            box: Bounding box coordinates of the match (x, y, width, height)
            confidence: Confidence score (0-100)
        """
        super().__init__(text, box, confidence)
    
    def __str__(self) -> str:
        """String representation of the text match."""
        return f"TextMatch('{self.text}', box={self.box}, confidence={self.confidence:.1f}%)"


class TextRecognizer:
    """
    Recognizes and extracts text from images using OCR.
    
    This class provides methods for detecting text blocks and reading text content
    from screen captures and UI elements.
    """
    
    def __init__(self, lang: str = 'eng', 
                 config: Optional[str] = None,
                 min_confidence: float = 60.0,
                 preprocess: bool = True):
        """
        Initialize the text recognizer.
        
        Args:
            lang: Language for OCR (default: 'eng')
            config: Custom configuration for Tesseract
            min_confidence: Minimum confidence threshold for text detection (0-100)
            preprocess: Whether to preprocess images before OCR
        """
        self.lang = lang
        self.config = config
        self.min_confidence = min_confidence
        self.preprocess = preprocess
        self.image_processor = ImageProcessor()
        
        # Check if OCR is available
        if not HAS_TESSERACT:
            logger.warning("Tesseract OCR is not available. Install pytesseract and tesseract for full functionality.")
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image to optimize for OCR.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Create a processing pipeline optimized for text recognition
        preprocessing_steps = [
            {'method': 'to_grayscale'},
            {'method': 'enhance_contrast', 'method_param': 'clahe'},
            {'method': 'reduce_noise', 'method_param': 'gaussian', 'kernel_size': 3},
            {'method': 'sharpen', 'amount': 1.0}
        ]
        
        # Apply the preprocessing pipeline
        preprocessed = self.image_processor.preprocess_pipeline(image, preprocessing_steps)
        
        return preprocessed
    
    def read_text(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        Extract text from an image or specified region.
        
        Args:
            image: Input image
            region: Optional tuple of (x, y, width, height) to specify region of interest
            
        Returns:
            Extracted text
        """
        if not HAS_TESSERACT:
            logger.error("Cannot read text: Tesseract is not available.")
            return ""
        
        # Extract the region if specified
        if region:
            x, y, w, h = region
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
                roi = image[y:y+h, x:x+w]
            else:
                logger.warning(f"Invalid region: {region}, using full image")
                roi = image
        else:
            roi = image
        
        # Preprocess the image if enabled
        if self.preprocess:
            processed_img = self._preprocess_for_ocr(roi)
        else:
            processed_img = roi
        
        # Use PSM mode 6 (assume a single uniform block of text) for consistency
        custom_config = f"{self.config} --psm 6" if self.config else "--psm 6"
        
        try:
            text = pytesseract.image_to_string(
                processed_img,
                lang=self.lang,
                config=custom_config
            )
            return text.strip()
        except Exception as e:
            logger.error(f"Error reading text: {e}")
            return ""
    
    def detect_text_blocks(self, image: np.ndarray) -> List[TextBlock]:
        """
        Detect and extract text blocks from an image.
        
        Args:
            image: Input image
            
        Returns:
            List of TextBlock objects containing detected text
        """
        if not HAS_TESSERACT:
            logger.error("Cannot detect text blocks: Tesseract is not available.")
            return []
        
        # Preprocess the image if enabled
        if self.preprocess:
            processed_image = self._preprocess_for_ocr(image)
        else:
            processed_image = image
        
        text_blocks = []
        
        # Method 1: Try using image_to_data with a specific PSM mode
        try:
            # Use PSM mode 6 (assume a single uniform block of text) to avoid NoneType error
            custom_config = f"{self.config} --psm 6" if self.config else "--psm 6"
            logger.debug("Calling pytesseract.image_to_data with custom config")
            
            data = pytesseract.image_to_data(
                processed_image, 
                lang=self.lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Debug the data structure
            if data is None:
                logger.warning("pytesseract.image_to_data returned None")
                data = {}  # Initialize empty dict to avoid NoneType errors
            
            # Validate the data structure before proceeding
            if 'text' in data and isinstance(data['text'], list) and len(data['text']) > 0:
                logger.debug(f"Found {len(data['text'])} text entries in data")
                n_boxes = len(data['text'])
                
                for i in range(n_boxes):
                    try:
                        # Debug the current text entry
                        logger.debug(f"Processing text entry {i}: {repr(data['text'][i])}")
                        
                        # Skip empty text or None values with extra safe handling
                        text_entry = data['text'][i]
                        if text_entry is None:
                            logger.debug(f"Skipping None text entry at index {i}")
                            continue
                        
                        # Convert to string if needed and then check if it's empty
                        try:
                            # Convert to string safely if it's not already a string
                            if not isinstance(text_entry, str):
                                text_entry = str(text_entry)
                            
                            # Now check if it's empty after stripping
                            if not text_entry.strip():
                                logger.debug(f"Skipping empty text entry at index {i}")
                                continue
                        except Exception as str_e:
                            logger.debug(f"Error processing text entry: {str_e}")
                            continue
                        
                        # Now we have a valid, non-empty text entry
                        # Check if confidence is available
                        conf = 0  # Default confidence
                        if ('conf' in data and isinstance(data['conf'], list) and 
                            i < len(data['conf']) and data['conf'][i] is not None):
                            try:
                                conf_val = data['conf'][i]
                                if conf_val != -1:  # -1 indicates no confidence
                                    conf = float(conf_val)
                            except (ValueError, TypeError):
                                # If conversion fails, keep default confidence
                                pass
                            
                            if conf < self.min_confidence:
                                continue
                        
                        # Get bounding box coordinates with robust checking
                        if all(key in data for key in ['left', 'top', 'width', 'height']):
                            # Check if all required lists exist and have sufficient length
                            lists_valid = all(
                                isinstance(data[key], list) and i < len(data[key])
                                for key in ['left', 'top', 'width', 'height']
                            )
                            
                            if lists_valid:
                                # Check if all values are not None
                                values_valid = all(
                                    data[key][i] is not None
                                    for key in ['left', 'top', 'width', 'height']
                                )
                                
                                if values_valid:
                                    try:
                                        x = int(data['left'][i])
                                        y = int(data['top'][i])
                                        w = int(data['width'][i])
                                        h = int(data['height'][i])
                                        
                                        # Add to text blocks list
                                        text_blocks.append(TextBlock(
                                            text=data['text'][i].strip(),
                                            box=(x, y, w, h),
                                            confidence=conf
                                        ))
                                    except (ValueError, TypeError) as e:
                                        logger.debug(f"Could not convert box values to int: {e}")
                    except (ValueError, TypeError, IndexError) as e:
                        logger.debug(f"Skipping text block due to data error: {e}")
                        continue
        except Exception as e:
            logger.warning(f"Image_to_data method failed: {e}")
        
        # Method 2: If method 1 failed or found no blocks, try a simpler approach with image_to_string
        if not text_blocks:
            try:
                # Extract text using image_to_string
                text = pytesseract.image_to_string(
                    processed_image,
                    lang=self.lang,
                    config=self.config
                )
                
                # Make sure we have valid text
                if isinstance(text, str) and text.strip():
                    # If we have text but no blocks, create a single block with the whole text
                    # and approximate the bounding box to the whole image or a portion of it
                    h, w = processed_image.shape[:2]
                    # Use a heuristic - assume text is in the center area
                    x = int(w * 0.1)  # 10% from left
                    y = int(h * 0.1)  # 10% from top
                    block_w = int(w * 0.8)  # 80% of width
                    block_h = int(h * 0.8)  # 80% of height
                    
                    text_blocks.append(TextBlock(
                        text=text.strip(),
                        box=(x, y, block_w, block_h),
                        confidence=90.0  # Arbitrary high confidence
                    ))
            except Exception as e:
                logger.warning(f"Image_to_string method failed: {e}")
        
        # Method 3: If both methods failed, try using image_to_boxes for character-level detection
        if not text_blocks:
            try:
                boxes = pytesseract.image_to_boxes(
                    processed_image,
                    lang=self.lang,
                    config=self.config
                )
                
                if boxes:
                    # Process character boxes to group into word blocks
                    # (this is a simplified approach - in a real implementation you would
                    # want a more sophisticated grouping algorithm)
                    char_boxes = []
                    h, w = processed_image.shape[:2]
                    
                    for b in boxes.splitlines():
                        parts = b.split(' ')
                        if len(parts) >= 6:
                            char = parts[0]
                            x1, y1, x2, y2 = int(parts[1]), h - int(parts[2]), int(parts[3]), h - int(parts[4])
                            char_boxes.append((char, x1, y1, x2, y2))
                    
                    # Group characters into words using a simple distance threshold
                    if char_boxes:
                        # Sort by x-coordinate
                        char_boxes.sort(key=lambda box: box[1])
                        
                        # Group characters that are close horizontally
                        word_groups = []
                        current_group = [char_boxes[0]]
                        
                        for i in range(1, len(char_boxes)):
                            prev_box = char_boxes[i-1]
                            curr_box = char_boxes[i]
                            
                            # If this character is close to the previous one, add to current group
                            if curr_box[1] - prev_box[3] < 20:  # Threshold for horizontal distance
                                current_group.append(curr_box)
                            else:
                                # Start a new group
                                word_groups.append(current_group)
                                current_group = [curr_box]
                        
                        # Add the last group
                        if current_group:
                            word_groups.append(current_group)
                        
                        # Create text blocks from word groups
                        for group in word_groups:
                            if group:
                                # Get text from characters
                                word = ''.join(box[0] for box in group)
                                
                                # Get bounding box for the group
                                x1 = min(box[1] for box in group)
                                y1 = min(box[2] for box in group)
                                x2 = max(box[3] for box in group)
                                y2 = max(box[4] for box in group)
                                
                                text_blocks.append(TextBlock(
                                    text=word,
                                    box=(x1, y1, x2 - x1, y2 - y1),
                                    confidence=80.0  # Arbitrary confidence
                                ))
            except Exception as e:
                logger.warning(f"Image_to_boxes method failed: {e}")
        
        return text_blocks
    
    def find_text(self, image: np.ndarray, text_pattern: str, 
                  case_sensitive: bool = False, 
                  region: Optional[Tuple[int, int, int, int]] = None) -> List[TextMatch]:
        """
        Search for specific text patterns within the image.
        
        Args:
            image: Input image
            text_pattern: Text pattern to search for
            case_sensitive: Whether to match case
            region: Optional tuple of (x, y, width, height) to specify region of interest
            
        Returns:
            List of TextMatch objects with text and bounding box
        """
        if not text_pattern:
            return []
        
        matches = []
        
        # First, detect all text blocks
        text_blocks = self.detect_text_blocks(image)
        
        # Filter text blocks by region if needed
        if region:
            x, y, w, h = region
            region_blocks = []
            for block in text_blocks:
                bx, by, bw, bh = block.box
                # Check if block is inside the region
                if (bx >= x and by >= y and
                    bx + bw <= x + w and by + bh <= y + h):
                    region_blocks.append(block)
            text_blocks = region_blocks
        
        # Search for the pattern in each block
        for block in text_blocks:
            text = block.text
            
            # Apply case insensitivity if needed
            search_text = text if case_sensitive else text.lower()
            search_pattern = text_pattern if case_sensitive else text_pattern.lower()
            
            if search_pattern in search_text:
                # Calculate the position of the match within the text
                start_pos = search_text.find(search_pattern)
                text_before = text[:start_pos]
                
                # Approximate position by character ratio
                if len(text) > 0:
                    char_width = block.box[2] / len(text)  # Average character width
                    offset_x = int(len(text_before) * char_width)
                    
                    # Create a match with adjusted bounding box
                    x, y, w, h = block.box
                    match_width = int(len(text_pattern) * char_width)
                    match_box = (x + offset_x, y, match_width, h)
                    
                    matches.append(TextMatch(
                        text=text_pattern,
                        box=match_box,
                        confidence=block.confidence
                    ))
                else:
                    # If we can't calculate character width, use the whole block box
                    matches.append(TextMatch(
                        text=text_pattern,
                        box=block.box,
                        confidence=block.confidence
                    ))
        
        return matches
    
    def get_text_from_element(self, image: np.ndarray, 
                             element_box: Tuple[int, int, int, int],
                             padding: int = 5) -> str:
        """
        Extract text from a UI element.
        
        Args:
            image: Input image
            element_box: Bounding box of the UI element (x, y, width, height)
            padding: Additional padding around the element box
            
        Returns:
            Extracted text from the UI element
        """
        # Apply padding to the element box
        x, y, w, h = element_box
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = w + 2 * padding
        h = h + 2 * padding
        
        # Ensure coordinates are within image boundaries
        h_max, w_max = image.shape[:2]
        x = min(x, w_max - 1)
        y = min(y, h_max - 1)
        w = min(w, w_max - x)
        h = min(h, h_max - y)
        
        # Extract text from the padded region
        return self.read_text(image, region=(x, y, w, h))
    
    def detect_paragraphs(self, image: np.ndarray, 
                         line_gap_threshold: int = 10) -> List[TextBlock]:
        """
        Detect paragraphs in an image by grouping text blocks.
        
        Args:
            image: Input image
            line_gap_threshold: Maximum vertical distance between lines in the same paragraph
            
        Returns:
            List of TextBlock objects representing paragraphs
        """
        # Get all text blocks
        blocks = self.detect_text_blocks(image)
        
        # Sort blocks by vertical position
        blocks.sort(key=lambda block: block.y)
        
        # Group blocks into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for block in blocks:
            if not current_paragraph:
                # Start a new paragraph
                current_paragraph.append(block)
            else:
                prev_block = current_paragraph[-1]
                # Check if this block is part of the current paragraph
                prev_bottom = prev_block.y + prev_block.height
                if block.y - prev_bottom <= line_gap_threshold:
                    # This block is part of the current paragraph
                    current_paragraph.append(block)
                else:
                    # This block starts a new paragraph
                    # Finalize the current paragraph
                    if current_paragraph:
                        text = " ".join([b.text for b in current_paragraph])
                        
                        # Calculate bounding box encompassing all blocks in the paragraph
                        min_x = min(b.x for b in current_paragraph)
                        min_y = min(b.y for b in current_paragraph)
                        max_x = max(b.x + b.width for b in current_paragraph)
                        max_y = max(b.y + b.height for b in current_paragraph)
                        
                        width = max_x - min_x
                        height = max_y - min_y
                        
                        # Create a text block for the paragraph
                        paragraph_block = TextBlock(
                            text=text,
                            box=(min_x, min_y, width, height),
                            confidence=sum(b.confidence for b in current_paragraph) / len(current_paragraph)
                        )
                        paragraphs.append(paragraph_block)
                    
                    # Start a new paragraph with this block
                    current_paragraph = [block]
        
        # Add the last paragraph if it exists
        if current_paragraph:
            text = " ".join([b.text for b in current_paragraph])
            
            # Calculate bounding box for the paragraph
            min_x = min(b.x for b in current_paragraph)
            min_y = min(b.y for b in current_paragraph)
            max_x = max(b.x + b.width for b in current_paragraph)
            max_y = max(b.y + b.height for b in current_paragraph)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Create a text block for the paragraph
            paragraph_block = TextBlock(
                text=text,
                box=(min_x, min_y, width, height),
                confidence=sum(b.confidence for b in current_paragraph) / len(current_paragraph)
            )
            paragraphs.append(paragraph_block)
        
        return paragraphs
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize recognized text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR errors
        text = text.replace('|', 'I')  # Vertical bar often mistaken for 'I'
        text = text.replace('0', 'O')  # 0 often mistaken for 'O'
        
        # Remove non-printable characters
        text = ''.join(c for c in text if c.isprintable())
        
        return text
    
    def get_text_orientation(self, image: np.ndarray) -> str:
        """
        Detect the orientation of text in an image.
        
        Args:
            image: Input image
            
        Returns:
            Orientation as 'horizontal', 'vertical', or 'unknown'
        """
        if not HAS_TESSERACT:
            return "unknown"
        
        try:
            # Use Tesseract's OSD (orientation and script detection)
            osd = pytesseract.image_to_osd(image)
            
            # Extract rotation angle
            rotation = int(re.search(r'Rotate: (\d+)', osd).group(1))
            
            # Determine orientation based on rotation
            if rotation in [0, 180]:
                return "horizontal"
            elif rotation in [90, 270]:
                return "vertical"
            else:
                return "unknown"
        except Exception as e:
            logger.error(f"Error detecting text orientation: {e}")
            return "unknown"


# Example usage for testing
def test_text_recognition():
    """Test function to demonstrate text recognition functionality."""
    from .screen_capture import ScreenCapture
    
    # Capture a screen region
    capture = ScreenCapture()
    image = capture.capture()
    
    # Create a text recognizer
    recognizer = TextRecognizer()
    
    # Extract all text from the image
    text = recognizer.read_text(image)
    print("Full text content:")
    print(text)
    print("-" * 50)
    
    # Detect text blocks
    text_blocks = recognizer.detect_text_blocks(image)
    print(f"Detected {len(text_blocks)} text blocks:")
    for i, block in enumerate(text_blocks[:5]):  # Show first 5 blocks
        print(f"{i+1}. {block}")
    print("-" * 50)
    
    # Find specific text
    search_term = "menu"
    matches = recognizer.find_text(image, search_term, case_sensitive=False)
    print(f"Found {len(matches)} matches for '{search_term}':")
    for match in matches[:3]:  # Show first 3 matches
        print(f"- {match}")
    print("-" * 50)
    
    # Detect paragraphs
    paragraphs = recognizer.detect_paragraphs(image)
    print(f"Detected {len(paragraphs)} paragraphs:")
    for i, para in enumerate(paragraphs[:3]):  # Show first 3 paragraphs
        print(f"Paragraph {i+1}: {para.text[:100]}...")
    
    print("\nText recognition test completed.")


if __name__ == "__main__":
    test_text_recognition()
