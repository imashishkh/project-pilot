"""
Image Processor Module for MacAgent

This module provides image processing capabilities for enhancing screen captures
to improve UI element detection, OCR accuracy, and overall visual analysis.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any, List, TYPE_CHECKING

# For type checking only, not executed at runtime
if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class ImageProcessor:
    """
    Processes images from screen captures to enhance element detection.
    
    This class provides methods for common image preprocessing tasks such as
    grayscale conversion, contrast enhancement, noise reduction, and more
    specialized operations for UI element detection.
    """
    
    def __init__(self, default_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the image processor with optional default parameters.
        
        Args:
            default_params: Dictionary of default parameters for processing operations
        """
        self.default_params = default_params or {}
        
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            # Already grayscale
            return image
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe', 
                         clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Enhance image contrast using various methods.
        
        Args:
            image: Input image
            method: Contrast enhancement method ('clahe', 'histogram_eq', 'linear_stretch')
            clip_limit: Clipping limit for CLAHE method
            tile_grid_size: Tile grid size for CLAHE method
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to grayscale if needed for certain operations
        gray = self.to_grayscale(image) if len(image.shape) > 2 else image
        
        if method == 'clahe':
            # Create a CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(gray)
        
        elif method == 'histogram_eq':
            # Apply histogram equalization
            enhanced = cv2.equalizeHist(gray)
        
        elif method == 'linear_stretch':
            # Apply linear contrast stretch
            min_val = np.min(gray)
            max_val = np.max(gray)
            enhanced = np.uint8(255 * (gray - min_val) / (max_val - min_val + 1e-8))
        
        else:
            raise ValueError(f"Unknown contrast enhancement method: {method}")
        
        # If the input was color, convert back to color
        if len(image.shape) > 2:
            # Create a 3-channel image where each channel is the enhanced grayscale
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return enhanced_color
        
        return enhanced
    
    def reduce_noise(self, image: np.ndarray, method: str = 'gaussian', 
                    kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
        """
        Apply noise reduction to an image.
        
        Args:
            image: Input image
            method: Noise reduction method ('gaussian', 'median', 'bilateral')
            kernel_size: Size of the kernel (must be odd)
            sigma: Standard deviation for Gaussian or bilateral filter
            
        Returns:
            Noise-reduced image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if method == 'gaussian':
            # Apply Gaussian blur
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif method == 'median':
            # Apply median blur
            return cv2.medianBlur(image, kernel_size)
        
        elif method == 'bilateral':
            # Apply bilateral filter
            if sigma <= 0:
                sigma = 75  # Default value for bilateral filter
            return cv2.bilateralFilter(image, kernel_size, sigma, sigma)
        
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
    
    def detect_edges(self, image: np.ndarray, method: str = 'canny', 
                    threshold1: float = 100, threshold2: float = 200,
                    aperture_size: int = 3) -> np.ndarray:
        """
        Detect edges in an image.
        
        Args:
            image: Input image
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            threshold1: First threshold for Canny edge detector
            threshold2: Second threshold for Canny edge detector
            aperture_size: Aperture size for the Sobel/Laplacian operator
            
        Returns:
            Edge-detected image
        """
        # Convert to grayscale if needed
        gray = self.to_grayscale(image) if len(image.shape) > 2 else image
        
        if method == 'canny':
            # Apply Canny edge detection
            return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)
        
        elif method == 'sobel':
            # Apply Sobel edge detection
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=aperture_size)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=aperture_size)
            
            # Compute the magnitude
            magnitude = cv2.magnitude(grad_x, grad_y)
            
            # Normalize and convert to uint8
            return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        elif method == 'laplacian':
            # Apply Laplacian edge detection
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=aperture_size)
            
            # Convert to absolute values and normalize
            return cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
    
    def apply_threshold(self, image: np.ndarray, method: str = 'adaptive', 
                       threshold: int = 127, max_value: int = 255,
                       adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                       block_size: int = 11, c: int = 2) -> np.ndarray:
        """
        Apply thresholding to an image.
        
        Args:
            image: Input image
            method: Thresholding method ('binary', 'adaptive', 'otsu')
            threshold: Threshold value for binary thresholding
            max_value: Maximum value for pixels above threshold
            adaptive_method: Adaptive thresholding method (cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            block_size: Block size for adaptive thresholding (must be odd)
            c: Constant subtracted from the mean or weighted mean in adaptive thresholding
            
        Returns:
            Thresholded image
        """
        # Convert to grayscale if needed
        gray = self.to_grayscale(image) if len(image.shape) > 2 else image
        
        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        if method == 'binary':
            # Apply binary thresholding
            _, thresholded = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
        
        elif method == 'adaptive':
            # Apply adaptive thresholding
            thresholded = cv2.adaptiveThreshold(
                gray, max_value, adaptive_method, cv2.THRESH_BINARY, block_size, c
            )
        
        elif method == 'otsu':
            # Apply Otsu's thresholding
            _, thresholded = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
        
        return thresholded
    
    def apply_morphology(self, image: np.ndarray, operation: str = 'open', 
                        kernel_size: Tuple[int, int] = (5, 5), 
                        iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to an image.
        
        Args:
            image: Input image
            operation: Morphological operation ('erode', 'dilate', 'open', 'close')
            kernel_size: Size of the structuring element
            iterations: Number of times to apply the operation
            
        Returns:
            Morphologically processed image
        """
        kernel = np.ones(kernel_size, np.uint8)
        
        if operation == 'erode':
            # Apply erosion
            return cv2.erode(image, kernel, iterations=iterations)
        
        elif operation == 'dilate':
            # Apply dilation
            return cv2.dilate(image, kernel, iterations=iterations)
        
        elif operation == 'open':
            # Apply opening (erosion followed by dilation)
            return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        
        elif operation == 'close':
            # Apply closing (dilation followed by erosion)
            return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
    
    def enhance_text(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance text in an image for better OCR.
        
        Args:
            image: Input image
            
        Returns:
            Text-enhanced image
        """
        # Convert to grayscale
        gray = self.to_grayscale(image)
        
        # Apply adaptive thresholding
        thresh = self.apply_threshold(gray, method='adaptive', max_value=255, 
                                    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    block_size=11, c=2)
        
        # Apply morphological operations to remove noise
        processed = self.apply_morphology(thresh, operation='close', kernel_size=(3, 3))
        
        return processed
    
    def enhance_ui_elements(self, image: np.ndarray, element_type: str = 'buttons') -> np.ndarray:
        """
        Enhance specific UI elements for better detection.
        
        Args:
            image: Input image
            element_type: Type of UI element to enhance ('buttons', 'text_fields', 'checkboxes')
            
        Returns:
            Enhanced image optimized for the specified element type
        """
        if element_type == 'buttons':
            # Enhance contrast for button detection
            enhanced = self.enhance_contrast(image, method='clahe')
            
            # Apply edge detection to highlight button boundaries
            edges = self.detect_edges(enhanced, method='canny')
            
            # Apply dilation to connect edges
            dilated_edges = self.apply_morphology(edges, operation='dilate', kernel_size=(3, 3))
            
            return dilated_edges
        
        elif element_type == 'text_fields':
            # Enhance text fields by focusing on rectangular shapes
            # First enhance the image
            enhanced = self.enhance_contrast(image)
            
            # Convert to grayscale and apply threshold
            gray = self.to_grayscale(enhanced)
            thresh = self.apply_threshold(gray, method='adaptive')
            
            # Apply morphological operations to highlight rectangular regions
            processed = self.apply_morphology(thresh, operation='close', kernel_size=(5, 5))
            
            return processed
        
        elif element_type == 'checkboxes':
            # Enhance checkbox detection by focusing on small square shapes
            # Convert to grayscale
            gray = self.to_grayscale(image)
            
            # Apply adaptive threshold
            thresh = self.apply_threshold(gray, method='adaptive')
            
            # Apply morphological operations
            processed = self.apply_morphology(thresh, operation='open', kernel_size=(3, 3))
            
            return processed
        
        else:
            raise ValueError(f"Unknown UI element type: {element_type}")
    
    def sharpen(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """
        Sharpen an image using unsharp masking.
        
        Args:
            image: Input image
            amount: Sharpening intensity
            
        Returns:
            Sharpened image
        """
        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Apply unsharp masking
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def preprocess_pipeline(self, image: np.ndarray, steps: List[Dict[str, Any]]) -> np.ndarray:
        """
        Apply a sequence of preprocessing steps to an image.
        
        Args:
            image: Input image
            steps: List of preprocessing steps, where each step is a dictionary
                  with 'method' and additional parameters
                  
        Returns:
            Processed image after applying all steps
        """
        processed = image.copy()
        
        for step in steps:
            # Make a copy of the step dictionary to avoid modifying the original
            step_params = step.copy()
            
            # Extract the method name
            method_name = step_params.pop('method')
            
            # Handle name conflicts - if there's a parameter called 'method'
            # which conflicts with the key, it should be passed as 'method_param'
            if 'method_param' in step_params:
                step_params['method'] = step_params.pop('method_param')
            
            # Get the corresponding method from the class
            if hasattr(self, method_name) and callable(getattr(self, method_name)):
                method = getattr(self, method_name)
                
                # Apply the method with the provided parameters
                processed = method(processed, **step_params)
            else:
                raise ValueError(f"Unknown preprocessing method: {method_name}")
        
        return processed


# Example usage
def test_image_processor():
    """Test function to demonstrate image processing functionality."""
    try:
        from screen_capture import ScreenCapture
    except ImportError:
        from .screen_capture import ScreenCapture
    
    # Capture a screen region
    screen_capture = ScreenCapture()
    image = screen_capture.capture(region=(100, 100, 800, 600))
    
    # Create an image processor
    processor = ImageProcessor()
    
    # Process the image with a pipeline
    processed = processor.preprocess_pipeline(image, [
        {'method': 'to_grayscale'},
        {'method': 'enhance_contrast', 'method_param': 'clahe'},
        {'method': 'reduce_noise', 'method_param': 'gaussian', 'kernel_size': 3},
        {'method': 'sharpen', 'amount': 0.5}
    ])
    
    # Try to visualize results if matplotlib is available
    _try_visualize_results(image, processed)
    
    # Save the processed image
    cv2.imwrite('processed_image.png', processed)
    
    print("Image processing complete. Processed image saved as 'processed_image.png'.")


def _try_visualize_results(original_image: np.ndarray, processed_image: np.ndarray) -> None:
    """
    Attempt to visualize original and processed images using matplotlib if available.
    
    Args:
        original_image: The original input image
        processed_image: The processed output image
    """
    # Isolate matplotlib import in this function to handle it cleanly
    try:
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt  # type: ignore
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(processed_image, cmap='gray')
        plt.title('Processed Image')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Visualization skipped.")


if __name__ == "__main__":
    test_image_processor()
