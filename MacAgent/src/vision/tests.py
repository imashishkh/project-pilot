"""
Test suite for screen capture and analyzer modules.

This module provides comprehensive tests for the vision components,
including performance benchmarks and accuracy tests.
"""

import unittest
import time
import os
import numpy as np
import cv2
from pathlib import Path
import sys
import logging
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import the modules properly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the modules to test
from MacAgent.src.vision.screen_capture import ScreenCapture, ContinuousCaptureSession
from MacAgent.src.vision.screen_analyzer import ScreenAnalyzer, UIElement


class PerformanceMetrics:
    """Class to track and report performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """Start a timer for a specific metric."""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and record the elapsed time."""
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was never started")
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(elapsed)
        return elapsed
    
    def get_avg(self, name: str) -> float:
        """Get the average time for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_min(self, name: str) -> float:
        """Get the minimum time for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        return min(self.metrics[name])
    
    def get_max(self, name: str) -> float:
        """Get the maximum time for a metric."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        return max(self.metrics[name])
    
    def report(self) -> Dict[str, Dict[str, float]]:
        """Generate a report of all metrics."""
        report = {}
        
        for name in self.metrics:
            if not self.metrics[name]:
                continue
                
            report[name] = {
                "avg": self.get_avg(name),
                "min": self.get_min(name),
                "max": self.get_max(name),
                "samples": len(self.metrics[name])
            }
        
        return report
    
    def print_report(self) -> None:
        """Print a formatted report of all metrics."""
        report = self.report()
        
        print("\n=== Performance Report ===")
        for name, stats in report.items():
            print(f"{name}:")
            print(f"  Avg: {stats['avg']*1000:.2f} ms")
            print(f"  Min: {stats['min']*1000:.2f} ms")
            print(f"  Max: {stats['max']*1000:.2f} ms")
            print(f"  Samples: {stats['samples']}")
        print("==========================\n")


class TestScreenCapture(unittest.TestCase):
    """Test cases for the ScreenCapture class."""
    
    def setUp(self):
        """Set up test environment."""
        self.metrics = PerformanceMetrics()
        self.screen_capture = ScreenCapture(
            capture_mode='auto',
            cache_ttl=0.1
        )
        
        # Create output directory for test artifacts
        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'screen_capture'):
            self.screen_capture.close()
        
        # Report performance metrics
        self.metrics.print_report()
    
    def test_capture_full_screen(self):
        """Test capturing the full screen."""
        # Capture full screen
        self.metrics.start_timer("capture_full_screen")
        screenshot = self.screen_capture.capture(force_refresh=True)
        elapsed = self.metrics.stop_timer("capture_full_screen")
        
        # Verify the screenshot
        self.assertIsNotNone(screenshot)
        self.assertIsInstance(screenshot, np.ndarray)
        self.assertEqual(len(screenshot.shape), 3)  # Should be 3D (height, width, channels)
        self.assertEqual(screenshot.shape[2], 3)  # BGR has 3 channels
        
        # Save the screenshot for inspection
        cv2.imwrite(str(self.output_dir / "full_screen.png"), screenshot)
        
        logger.info(f"Full screen capture took {elapsed*1000:.2f} ms, "
                    f"resolution: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    def test_capture_region(self):
        """Test capturing a specific region of the screen."""
        # Get primary monitor info to define a region
        monitor_info = self.screen_capture.get_monitor_info(0)
        
        # Define a small region in the middle of the screen
        width, height = monitor_info["width"], monitor_info["height"]
        region_size = min(400, width // 4, height // 4)
        region = (
            width // 2 - region_size // 2,
            height // 2 - region_size // 2,
            region_size,
            region_size
        )
        
        # Capture the region
        self.metrics.start_timer("capture_region")
        screenshot = self.screen_capture.capture(region=region, force_refresh=True)
        elapsed = self.metrics.stop_timer("capture_region")
        
        # Verify the screenshot
        self.assertIsNotNone(screenshot)
        self.assertIsInstance(screenshot, np.ndarray)
        self.assertEqual(len(screenshot.shape), 3)
        
        # Check dimensions
        # Note: There can sometimes be a slight mismatch in dimensions due to
        # scaling or OS constraints, so we check approximately
        self.assertAlmostEqual(screenshot.shape[1], region[2], delta=5)
        self.assertAlmostEqual(screenshot.shape[0], region[3], delta=5)
        
        # Save the screenshot for inspection
        cv2.imwrite(str(self.output_dir / "region.png"), screenshot)
        
        logger.info(f"Region capture took {elapsed*1000:.2f} ms, "
                    f"region: {region}, "
                    f"resolution: {screenshot.shape[1]}x{screenshot.shape[0]}")
    
    def test_capture_modes(self):
        """Test different capture modes (quartz vs mss)."""
        # Test Quartz capture
        self.screen_capture.capture_mode = 'quartz'
        self.metrics.start_timer("capture_quartz")
        quartz_img = self.screen_capture.capture(force_refresh=True)
        quartz_time = self.metrics.stop_timer("capture_quartz")
        
        # Test MSS capture
        self.screen_capture.capture_mode = 'mss'
        self.metrics.start_timer("capture_mss")
        mss_img = self.screen_capture.capture(force_refresh=True)
        mss_time = self.metrics.stop_timer("capture_mss")
        
        # Verify captures
        self.assertIsNotNone(quartz_img)
        self.assertIsNotNone(mss_img)
        
        # Check they have similar dimensions (should be the same screen)
        self.assertEqual(quartz_img.shape, mss_img.shape)
        
        # Save for comparison
        cv2.imwrite(str(self.output_dir / "quartz_capture.png"), quartz_img)
        cv2.imwrite(str(self.output_dir / "mss_capture.png"), mss_img)
        
        logger.info(f"Quartz capture: {quartz_time*1000:.2f} ms, "
                    f"MSS capture: {mss_time*1000:.2f} ms")
    
    def test_caching(self):
        """Test the caching mechanism."""
        # First capture (should hit the backend)
        self.metrics.start_timer("first_capture")
        first = self.screen_capture.capture(force_refresh=True)
        first_time = self.metrics.stop_timer("first_capture")
        
        # Second capture immediately after (should use cache)
        self.metrics.start_timer("cached_capture")
        cached = self.screen_capture.capture(force_refresh=False)
        cached_time = self.metrics.stop_timer("cached_capture")
        
        # Verify first is the same as cached
        self.assertTrue(np.array_equal(first, cached))
        
        # Cached should be much faster
        self.assertLess(cached_time, first_time)
        
        logger.info(f"First capture: {first_time*1000:.2f} ms, "
                    f"Cached capture: {cached_time*1000:.2f} ms")
    
    def test_continuous_capture(self):
        """Test continuous capture functionality."""
        # Start continuous capture session
        session = self.screen_capture.start_continuous_capture(
            interval=0.05,
            buffer_size=10
        )
        
        # Start the session
        session.start()
        
        # Wait a bit to get some frames
        time.sleep(0.3)
        
        # Get the latest frame
        frame = session.get_latest_frame()
        self.assertIsNotNone(frame)
        
        # Get buffer frames
        buffer_frames = session.get_buffer_frames()
        self.assertGreater(len(buffer_frames), 0)
        
        # Check FPS
        fps = session.get_fps()
        self.assertGreater(fps, 0)
        
        # Stop the session
        session.stop()
        
        logger.info(f"Continuous capture: {len(buffer_frames)} frames, "
                    f"FPS: {fps:.2f}")
    
    def test_multi_monitor(self):
        """Test capturing from multiple monitors (if available)."""
        # Get number of monitors
        self._refresh_monitors_info = getattr(self.screen_capture, '_refresh_monitors_info', None)
        if self._refresh_monitors_info:
            self._refresh_monitors_info()
        
        monitors = getattr(self.screen_capture, '_monitors_info', [])
        monitor_count = len(monitors)
        
        logger.info(f"Detected {monitor_count} monitors")
        
        # Skip test if there's only one monitor
        if monitor_count <= 1:
            logger.warning("Skipping multi-monitor test as only one monitor detected")
            return
        
        # Capture from each monitor
        for i in range(monitor_count):
            self.metrics.start_timer(f"capture_monitor_{i}")
            screenshot = self.screen_capture.capture(monitor_index=i, force_refresh=True)
            elapsed = self.metrics.stop_timer(f"capture_monitor_{i}")
            
            # Verify the screenshot
            self.assertIsNotNone(screenshot)
            self.assertIsInstance(screenshot, np.ndarray)
            
            # Save for inspection
            cv2.imwrite(str(self.output_dir / f"monitor_{i}.png"), screenshot)
            
            logger.info(f"Monitor {i} capture: {elapsed*1000:.2f} ms, "
                        f"resolution: {screenshot.shape[1]}x{screenshot.shape[0]}")


class TestScreenAnalyzer(unittest.TestCase):
    """Test cases for the ScreenAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        # Create output directory for test artifacts
        cls.output_dir = Path("test_output")
        cls.output_dir.mkdir(exist_ok=True)
        
        # Create directory for test images
        cls.test_images_dir = cls.output_dir / "test_images"
        cls.test_images_dir.mkdir(exist_ok=True)
        
        # Create or copy test images if needed
        cls._setup_test_images()
    
    @classmethod
    def _setup_test_images(cls):
        """Set up test images for consistent testing."""
        # Capture a real screenshot for testing
        screen_capture = ScreenCapture()
        screenshot = screen_capture.capture()
        cv2.imwrite(str(cls.test_images_dir / "screenshot.png"), screenshot)
        screen_capture.close()
        
        # Create a synthetic image with known UI elements
        cls._create_synthetic_ui_image()
    
    @classmethod
    def _create_synthetic_ui_image(cls):
        """Create a synthetic image with known UI elements for testing."""
        # Create a blank image
        width, height = 800, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Add a button
        button_x, button_y = 100, 100
        button_w, button_h = 200, 50
        cv2.rectangle(image, 
                     (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     (200, 200, 200), 
                     -1)  # Filled rectangle
        cv2.rectangle(image, 
                     (button_x, button_y), 
                     (button_x + button_w, button_y + button_h), 
                     (150, 150, 150), 
                     2)  # Border
        
        # Add text to the button
        text = "Click Me"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = button_x + (button_w - text_size[0]) // 2
        text_y = button_y + (button_h + text_size[1]) // 2
        cv2.putText(image, text, (text_x, text_y), font, 0.7, (0, 0, 0), 2)
        
        # Add a text field
        field_x, field_y = 400, 100
        field_w, field_h = 300, 50
        cv2.rectangle(image, 
                     (field_x, field_y), 
                     (field_x + field_w, field_y + field_h), 
                     (255, 255, 255), 
                     -1)  # White background
        cv2.rectangle(image, 
                     (field_x, field_y), 
                     (field_x + field_w, field_y + field_h), 
                     (100, 100, 100), 
                     1)  # Thin border
        
        # Add placeholder text to the field
        placeholder = "Enter text here"
        text_size = cv2.getTextSize(placeholder, font, 0.6, 1)[0]
        text_x = field_x + 10
        text_y = field_y + (field_h + text_size[1]) // 2
        cv2.putText(image, placeholder, (text_x, text_y), font, 0.6, (180, 180, 180), 1)
        
        # Add a checkbox
        checkbox_x, checkbox_y = 100, 200
        checkbox_size = 30
        cv2.rectangle(image, 
                     (checkbox_x, checkbox_y), 
                     (checkbox_x + checkbox_size, checkbox_y + checkbox_size), 
                     (255, 255, 255), 
                     -1)  # White background
        cv2.rectangle(image, 
                     (checkbox_x, checkbox_y), 
                     (checkbox_x + checkbox_size, checkbox_y + checkbox_size), 
                     (100, 100, 100), 
                     2)  # Border
        
        # Add label for checkbox
        label = "Remember me"
        text_x = checkbox_x + checkbox_size + 10
        text_y = checkbox_y + checkbox_size // 2 + 5
        cv2.putText(image, label, (text_x, text_y), font, 0.6, (0, 0, 0), 1)
        
        # Add some paragraph text
        paragraph_x, paragraph_y = 100, 300
        lines = [
            "This is a test image for the ScreenAnalyzer class.",
            "It contains various UI elements like buttons,",
            "text fields, and checkboxes for testing the",
            "element detection and OCR capabilities."
        ]
        
        for i, line in enumerate(lines):
            y = paragraph_y + i * 30
            cv2.putText(image, line, (paragraph_x, y), font, 0.6, (0, 0, 0), 1)
        
        # Save the image
        cv2.imwrite(str(cls.test_images_dir / "synthetic_ui.png"), image)
    
    def setUp(self):
        """Set up each test."""
        self.metrics = PerformanceMetrics()
        self.screen_analyzer = ScreenAnalyzer(
            ocr_enabled=True,
            preprocess_image=True,
            ui_detection_enabled=True
        )
        
        # Load test images
        self.real_screenshot = cv2.imread(str(self.test_images_dir / "screenshot.png"))
        self.synthetic_ui = cv2.imread(str(self.test_images_dir / "synthetic_ui.png"))
    
    def tearDown(self):
        """Clean up after each test."""
        # Report performance metrics
        self.metrics.print_report()
    
    def test_text_extraction(self):
        """Test text extraction capabilities."""
        # Test on synthetic image
        self.metrics.start_timer("ocr_synthetic")
        text = self.screen_analyzer.extract_text(self.synthetic_ui)
        ocr_time = self.metrics.stop_timer("ocr_synthetic")
        
        # Verify some expected text is found
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)
        
        # The synthetic image contains "Click Me" and "Remember me"
        # However, OCR might not be exact, so we check for parts
        self.assertTrue(any(keyword in text for keyword in ["Click", "Remember"]), 
                        f"Expected keywords not found in extracted text: {text}")
        
        logger.info(f"OCR on synthetic image took {ocr_time*1000:.2f} ms")
        logger.info(f"Extracted text: {text[:100]}...")
        
        # Test on real screenshot (may be less predictable)
        self.metrics.start_timer("ocr_real")
        real_text = self.screen_analyzer.extract_text(self.real_screenshot)
        real_ocr_time = self.metrics.stop_timer("ocr_real")
        
        logger.info(f"OCR on real screenshot took {real_ocr_time*1000:.2f} ms")
        logger.info(f"Extracted text length: {len(real_text)} characters")
    
    def test_ui_element_detection(self):
        """Test UI element detection."""
        # Test on synthetic image
        self.metrics.start_timer("detect_ui_synthetic")
        elements = self.screen_analyzer.detect_ui_elements(self.synthetic_ui)
        detection_time = self.metrics.stop_timer("detect_ui_synthetic")
        
        # Verify elements were detected
        self.assertIsNotNone(elements)
        self.assertIsInstance(elements, list)
        self.assertGreater(len(elements), 0)
        
        # Save annotated image with detected elements
        annotated = self.synthetic_ui.copy()
        for element in elements:
            x, y, w, h = element.bounds
            color = (0, 255, 0) if element.element_type == "button" else (0, 0, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        
        cv2.imwrite(str(self.output_dir / "detected_elements_synthetic.png"), annotated)
        
        logger.info(f"UI detection on synthetic image took {detection_time*1000:.2f} ms")
        logger.info(f"Detected {len(elements)} elements: " + 
                    ", ".join([f"{e.element_type}" for e in elements[:5]]) + 
                    (", ..." if len(elements) > 5 else ""))
        
        # Test on real screenshot
        self.metrics.start_timer("detect_ui_real")
        real_elements = self.screen_analyzer.detect_ui_elements(self.real_screenshot)
        real_detection_time = self.metrics.stop_timer("detect_ui_real")
        
        # Annotate real screenshot too
        real_annotated = self.real_screenshot.copy()
        for element in real_elements:
            x, y, w, h = element.bounds
            if element.element_type == "button":
                color = (0, 255, 0)  # Green for buttons
            elif element.element_type == "text":
                color = (255, 0, 0)  # Blue for text
            else:
                color = (0, 0, 255)  # Red for other
            
            cv2.rectangle(real_annotated, (x, y), (x + w, y + h), color, 2)
        
        cv2.imwrite(str(self.output_dir / "detected_elements_real.png"), real_annotated)
        
        logger.info(f"UI detection on real screenshot took {real_detection_time*1000:.2f} ms")
        logger.info(f"Detected {len(real_elements)} elements on real screenshot")
    
    def test_screen_analysis(self):
        """Test the complete screen analysis pipeline."""
        # Test the complete analysis
        self.metrics.start_timer("analyze_screen")
        results = self.screen_analyzer.analyze_screen(self.synthetic_ui)
        analysis_time = self.metrics.stop_timer("analyze_screen")
        
        # Verify results contain expected keys
        self.assertIn("ui_elements", results)
        self.assertIn("text", results)
        self.assertIn("text_blocks", results)
        self.assertIn("analysis_time", results)
        
        # Check if results are reasonable
        self.assertGreater(len(results["ui_elements"]), 0)
        self.assertGreater(len(results["text"]), 0)
        
        logger.info(f"Complete analysis took {analysis_time*1000:.2f} ms")
        logger.info(f"Analysis results: {len(results['ui_elements'])} elements, " + 
                    f"{len(results['text_blocks'])} text blocks, " + 
                    f"{len(results['text'])} chars of text")
    
    def test_find_element_by_text(self):
        """Test finding elements by contained text."""
        # Search for "Click Me" button in synthetic image
        self.metrics.start_timer("find_element")
        element = self.screen_analyzer.find_element_by_text(self.synthetic_ui, "Click")
        find_time = self.metrics.stop_timer("find_element")
        
        # Verify element was found
        self.assertIsNotNone(element)
        self.assertIsInstance(element, UIElement)
        
        # Draw the found element
        if element:
            result_img = self.synthetic_ui.copy()
            x, y, w, h = element.bounds
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result_img, f"Found: {element.text}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imwrite(str(self.output_dir / "found_element.png"), result_img)
        
        logger.info(f"Finding element took {find_time*1000:.2f} ms")
        if element:
            logger.info(f"Found element: {element.element_type} with text '{element.text}'")
    
    def test_caching(self):
        """Test the caching functionality."""
        # First analysis (should hit backend)
        self.metrics.start_timer("first_analysis")
        first_results = self.screen_analyzer.analyze_screen(self.synthetic_ui)
        first_time = self.metrics.stop_timer("first_analysis")
        
        # Second analysis of the same image (should use cache)
        self.metrics.start_timer("cached_analysis")
        cached_results = self.screen_analyzer.analyze_screen(self.synthetic_ui)
        cached_time = self.metrics.stop_timer("cached_analysis")
        
        # Cached should be much faster
        self.assertLess(cached_time, first_time)
        
        # Results should be identical
        self.assertEqual(len(first_results["ui_elements"]), len(cached_results["ui_elements"]))
        self.assertEqual(first_results["text"], cached_results["text"])
        
        logger.info(f"First analysis: {first_time*1000:.2f} ms, "
                    f"Cached analysis: {cached_time*1000:.2f} ms")
        
        # Clear cache and verify it causes a full reanalysis
        self.screen_analyzer.clear_cache()
        
        self.metrics.start_timer("after_clear_cache")
        new_results = self.screen_analyzer.analyze_screen(self.synthetic_ui)
        after_clear_time = self.metrics.stop_timer("after_clear_cache")
        
        # Should be closer to first analysis time
        self.assertGreater(after_clear_time, cached_time)
        
        logger.info(f"Analysis after cache clear: {after_clear_time*1000:.2f} ms")


def run_benchmark():
    """Run performance benchmarks."""
    logger.info("Running performance benchmarks...")
    
    # Create output directory
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    screen_capture = ScreenCapture()
    screen_analyzer = ScreenAnalyzer()
    
    # Benchmark settings
    iterations = 10
    metrics = PerformanceMetrics()
    
    # Benchmark screen capture
    logger.info(f"Benchmarking screen capture ({iterations} iterations)...")
    for i in range(iterations):
        metrics.start_timer(f"capture_{i}")
        screenshot = screen_capture.capture(force_refresh=True)
        metrics.stop_timer(f"capture_{i}")
    
    capture_avg = metrics.get_avg("capture_0")
    for i in range(1, iterations):
        capture_avg += metrics.get_avg(f"capture_{i}")
    capture_avg /= iterations
    
    # Benchmark OCR
    logger.info(f"Benchmarking OCR ({iterations} iterations)...")
    screenshot = screen_capture.capture()
    for i in range(iterations):
        metrics.start_timer(f"ocr_{i}")
        text = screen_analyzer.extract_text(screenshot)
        metrics.stop_timer(f"ocr_{i}")
    
    ocr_avg = metrics.get_avg("ocr_0")
    for i in range(1, iterations):
        ocr_avg += metrics.get_avg(f"ocr_{i}")
    ocr_avg /= iterations
    
    # Benchmark UI detection
    logger.info(f"Benchmarking UI detection ({iterations} iterations)...")
    for i in range(iterations):
        metrics.start_timer(f"ui_detection_{i}")
        elements = screen_analyzer.detect_ui_elements(screenshot)
        metrics.stop_timer(f"ui_detection_{i}")
    
    ui_avg = metrics.get_avg("ui_detection_0")
    for i in range(1, iterations):
        ui_avg += metrics.get_avg(f"ui_detection_{i}")
    ui_avg /= iterations
    
    # Benchmark full pipeline
    logger.info(f"Benchmarking full pipeline ({iterations} iterations)...")
    for i in range(iterations):
        metrics.start_timer(f"pipeline_{i}")
        
        # Capture
        screenshot = screen_capture.capture(force_refresh=True)
        
        # Analyze
        results = screen_analyzer.analyze_screen(screenshot)
        
        metrics.stop_timer(f"pipeline_{i}")
    
    pipeline_avg = metrics.get_avg("pipeline_0")
    for i in range(1, iterations):
        pipeline_avg += metrics.get_avg(f"pipeline_{i}")
    pipeline_avg /= iterations
    
    # Write benchmark results
    with open(output_dir / "benchmark_results.txt", "w") as f:
        f.write("=== Vision System Benchmark Results ===\n")
        f.write(f"Screen Capture: {capture_avg*1000:.2f} ms avg\n")
        f.write(f"OCR: {ocr_avg*1000:.2f} ms avg\n")
        f.write(f"UI Detection: {ui_avg*1000:.2f} ms avg\n")
        f.write(f"Full Pipeline: {pipeline_avg*1000:.2f} ms avg\n")
        f.write("\n")
        f.write(f"Test system: {os.uname().sysname} {os.uname().release}\n")
        f.write(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Print benchmark results
    logger.info("=== Vision System Benchmark Results ===")
    logger.info(f"Screen Capture: {capture_avg*1000:.2f} ms avg")
    logger.info(f"OCR: {ocr_avg*1000:.2f} ms avg")
    logger.info(f"UI Detection: {ui_avg*1000:.2f} ms avg")
    logger.info(f"Full Pipeline: {pipeline_avg*1000:.2f} ms avg")
    
    # Clean up
    screen_capture.close()


if __name__ == "__main__":
    # Create test directory
    os.makedirs("test_output", exist_ok=True)
    
    # Run unittest tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    # Run benchmarks
    run_benchmark() 