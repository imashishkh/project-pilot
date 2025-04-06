"""
Screen Capture Module for MacAgent

This module provides high-performance screen capture functionality
optimized for macOS, supporting multiple monitors, region capture,
and continuous capture modes with various performance settings.
"""

import time
import threading
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import deque

# Import macOS-specific libraries
try:
    import Quartz
    from Quartz import CGWindowListCreateImage, CGRectMake, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    from AppKit import NSBitmapImageRep, NSDeviceRGBColorSpace
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False
    logging.warning("Quartz libraries not available. Some Mac-specific functionality will be limited.")

# Try to import mss for faster capture in some cases
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False
    logging.warning("MSS library not available. Falling back to PyAutoGUI for screen capture.")

# Import pyautogui as a fallback
import pyautogui

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContinuousCaptureSession:
    """Session for continuous screen capturing at defined intervals."""
    
    def __init__(self, capture_func: Callable, interval: float = 0.1, buffer_size: int = 5):
        """Initialize a continuous capture session.
        
        Args:
            capture_func: Function to call for capturing screens
            interval: Time between captures in seconds
            buffer_size: Number of frames to keep in buffer
        """
        self.capture_func = capture_func
        self.interval = interval
        self.buffer_size = buffer_size
        self.is_running = False
        self.thread = None
        self.buffer = deque(maxlen=buffer_size)
        self.latest_frame = None
        self.frame_times = deque(maxlen=100)  # For FPS calculation
        self.lock = threading.RLock()
    
    def _capture_loop(self):
        """Main capture loop running in a separate thread."""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Capture frame
                frame = self.capture_func(force_refresh=True)
                
                # Store in buffer and as latest frame
                with self.lock:
                    self.buffer.append(frame)
                    self.latest_frame = frame
                    self.frame_times.append(start_time)
                
                # Calculate sleep time to maintain the interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.5)  # Sleep on error to avoid tight loop
    
    def start(self):
        """Start the continuous capture session."""
        if self.is_running:
            logger.warning("Continuous capture session already running")
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started continuous capture with interval={self.interval}s")
    
    def stop(self):
        """Stop the continuous capture session."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Stopped continuous capture")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent captured frame."""
        with self.lock:
            return self.latest_frame
    
    def get_buffer_frames(self) -> List[np.ndarray]:
        """Get all frames in the buffer."""
        with self.lock:
            return list(self.buffer)
    
    def get_fps(self) -> float:
        """Calculate the current capture FPS."""
        with self.lock:
            if len(self.frame_times) < 2:
                return 0.0
            
            # Calculate FPS from the frame timestamps
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff <= 0:
                return 0.0
            
            return (len(self.frame_times) - 1) / time_diff
    
    def is_active(self) -> bool:
        """Check if the capture session is active."""
        return self.is_running and (self.thread is not None) and self.thread.is_alive()


class ScreenCapture:
    """High-performance screen capture for macOS.
    
    This class provides multiple capture methods optimized for macOS,
    with fallbacks to cross-platform solutions when needed.
    """
    
    def __init__(self, capture_mode: str = 'auto', cache_ttl: float = 0.1):
        """Initialize screen capture with the specified mode.
        
        Args:
            capture_mode: Capture method to use ('auto', 'quartz', 'mss', 'pyautogui')
            cache_ttl: Time in seconds to cache screen captures
        """
        self.capture_mode = self._validate_capture_mode(capture_mode)
        self.cache_ttl = cache_ttl
        self._cached_image = None
        self._cached_region = None
        self._cached_monitor = None
        self._last_capture_time = 0
        self._monitors_info = []
        self._mss_instance = None
        
        # Initialize the capture system
        self._initialize_capture_system()
        logger.info(f"Initialized ScreenCapture with mode: {self.capture_mode}")
    
    def _validate_capture_mode(self, mode: str) -> str:
        """Validate and normalize the capture mode."""
        mode = mode.lower()
        
        if mode == 'auto':
            # Choose the best available method
            if HAS_QUARTZ:
                return 'quartz'
            elif HAS_MSS:
                return 'mss'
            else:
                return 'pyautogui'
        
        # Check specific mode availability
        if mode == 'quartz' and not HAS_QUARTZ:
            logger.warning("Quartz not available, falling back to next best method")
            return 'mss' if HAS_MSS else 'pyautogui'
        
        if mode == 'mss' and not HAS_MSS:
            logger.warning("MSS not available, falling back to PyAutoGUI")
            return 'pyautogui'
        
        # Ensure it's a valid mode
        if mode not in ['quartz', 'mss', 'pyautogui']:
            logger.warning(f"Invalid capture mode '{mode}', using auto selection")
            return self._validate_capture_mode('auto')
        
        return mode
    
    def _initialize_capture_system(self):
        """Initialize the screen capture system based on the selected mode."""
        # Set up MSS if needed
        if self.capture_mode == 'mss' or (self.capture_mode == 'auto' and HAS_MSS):
            self._mss_instance = mss.mss()
        
        # Get monitor information
        self._refresh_monitors_info()
    
    def _refresh_monitors_info(self):
        """Refresh information about connected monitors."""
        self._monitors_info = []
        
        if self.capture_mode == 'quartz':
            # Get monitor info using Quartz
            try:
                # Simpler approach: just use the main display as a fallback
                # This works reliably even if we can't get the full display list
                main_display = Quartz.CGMainDisplayID()
                
                # Get the bounds of the main display
                bounds = Quartz.CGDisplayBounds(main_display)
                info = {
                    "id": main_display,
                    "index": 0,
                    "x": bounds.origin.x,
                    "y": bounds.origin.y,
                    "width": bounds.size.width,
                    "height": bounds.size.height,
                    "is_primary": True
                }
                self._monitors_info.append(info)
                
                logger.debug(f"Successfully detected main display using Quartz")
            except Exception as e:
                logger.error(f"Error getting Quartz display info: {e}")
                
        elif self.capture_mode == 'mss':
            # Get monitor info using MSS
            try:
                for i, monitor in enumerate(self._mss_instance.monitors[1:], 0):  # Skip the first "all monitors" entry
                    info = {
                        "id": i,
                        "index": i,
                        "x": monitor["left"],
                        "y": monitor["top"],
                        "width": monitor["width"],
                        "height": monitor["height"],
                        "is_primary": i == 0  # Assuming first monitor is primary
                    }
                    self._monitors_info.append(info)
            except Exception as e:
                logger.error(f"Error getting MSS monitor info: {e}")
        
        else:
            # Fallback to PyAutoGUI
            try:
                size = pyautogui.size()
                info = {
                    "id": 0,
                    "index": 0,
                    "x": 0,
                    "y": 0,
                    "width": size[0],
                    "height": size[1],
                    "is_primary": True
                }
                self._monitors_info.append(info)
            except Exception as e:
                logger.error(f"Error getting PyAutoGUI screen info: {e}")
        
        logger.debug(f"Detected {len(self._monitors_info)} monitors")
    
    def get_monitor_info(self, monitor_index: int = 0) -> Dict[str, Any]:
        """Get information about a specific monitor.
        
        Args:
            monitor_index: Index of the monitor (0 for primary)
            
        Returns:
            Dictionary with monitor information
        """
        if not self._monitors_info:
            self._refresh_monitors_info()
        
        if not self._monitors_info:
            # Fallback if we couldn't get monitor info
            size = pyautogui.size()
            return {
                "id": 0,
                "index": 0,
                "x": 0,
                "y": 0,
                "width": size[0],
                "height": size[1],
                "is_primary": True
            }
        
        if monitor_index >= len(self._monitors_info):
            logger.warning(f"Monitor index {monitor_index} out of range, using primary monitor")
            # Find primary monitor
            for monitor in self._monitors_info:
                if monitor["is_primary"]:
                    return monitor
            # Fallback to first monitor
            return self._monitors_info[0]
        
        return self._monitors_info[monitor_index]
    
    def _capture_with_quartz(self, region: Optional[Tuple[int, int, int, int]] = None, 
                            monitor_index: Optional[int] = None) -> Optional[np.ndarray]:
        """Capture screen using macOS Quartz/CoreGraphics API.
        
        Args:
            region: Region to capture (x, y, width, height)
            monitor_index: Index of monitor to capture
            
        Returns:
            Captured image as numpy array
        """
        try:
            # Determine capture region
            if region is not None:
                x, y, width, height = region
                rect = CGRectMake(x, y, width, height)
            elif monitor_index is not None:
                monitor = self.get_monitor_info(monitor_index)
                rect = CGRectMake(
                    monitor["x"], monitor["y"], 
                    monitor["width"], monitor["height"]
                )
            else:
                # Capture main display
                main_display = self.get_monitor_info(0)
                rect = CGRectMake(
                    main_display["x"], main_display["y"], 
                    main_display["width"], main_display["height"]
                )
            
            # Capture the image
            image_ref = CGWindowListCreateImage(
                rect,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
                0
            )
            
            if not image_ref:
                logger.error("Failed to capture screen with Quartz")
                return None
            
            # Convert to bitmap
            width = Quartz.CGImageGetWidth(image_ref)
            height = Quartz.CGImageGetHeight(image_ref)
            bitmap = NSBitmapImageRep.alloc().initWithCGImage_(image_ref)
            
            if not bitmap:
                logger.error("Failed to create bitmap from Quartz capture")
                return None
            
            # Get raw data
            data = bitmap.bitmapData()
            bytesPerRow = bitmap.bytesPerRow()
            
            # Convert to numpy array
            buffer = data[:bytesPerRow * height]
            image_array = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytesPerRow // 4, 4))
            
            # Convert RGBA to BGR (OpenCV format)
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            
            return bgr_image
        
        except Exception as e:
            logger.error(f"Error in Quartz screen capture: {e}")
            return None
    
    def _capture_with_mss(self, region: Optional[Tuple[int, int, int, int]] = None, 
                         monitor_index: Optional[int] = None) -> Optional[np.ndarray]:
        """Capture screen using MSS library.
        
        Args:
            region: Region to capture (x, y, width, height)
            monitor_index: Index of monitor to capture
            
        Returns:
            Captured image as numpy array
        """
        try:
            if self._mss_instance is None:
                self._mss_instance = mss.mss()
            
            # Determine capture region
            if region is not None:
                x, y, width, height = region
                monitor = {"left": x, "top": y, "width": width, "height": height}
            elif monitor_index is not None:
                # MSS uses 1-based indexing, with 0 being "all monitors"
                mss_monitor_index = monitor_index + 1
                # Ensure we don't go out of bounds
                if mss_monitor_index >= len(self._mss_instance.monitors):
                    logger.warning(f"Monitor index {monitor_index} out of range for MSS, using primary")
                    mss_monitor_index = 1
                
                monitor = self._mss_instance.monitors[mss_monitor_index]
            else:
                # Capture primary monitor
                monitor = self._mss_instance.monitors[1]  # Primary monitor
            
            # Capture the screen
            screenshot = self._mss_instance.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to BGR
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            return img_bgr
        
        except Exception as e:
            logger.error(f"Error in MSS screen capture: {e}")
            return None
    
    def _capture_with_pyautogui(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """Capture screen using PyAutoGUI.
        
        Args:
            region: Region to capture (x, y, width, height)
            
        Returns:
            Captured image as numpy array
        """
        try:
            if region is not None:
                x, y, width, height = region
                screenshot = pyautogui.screenshot(region=(x, y, width, height))
            else:
                screenshot = pyautogui.screenshot()
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert RGB to BGR (OpenCV format)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            return img_bgr
        
        except Exception as e:
            logger.error(f"Error in PyAutoGUI screen capture: {e}")
            return None
    
    def _check_cache(self, region: Optional[Tuple[int, int, int, int]] = None, 
                    monitor_index: Optional[int] = None) -> Optional[np.ndarray]:
        """Check if we have a cached image for the requested region/monitor.
        
        Args:
            region: Region to capture (x, y, width, height)
            monitor_index: Index of monitor to capture
            
        Returns:
            Cached image if available and valid, None otherwise
        """
        current_time = time.time()
        
        # Check if cache is still valid
        if (current_time - self._last_capture_time <= self.cache_ttl and 
            self._cached_image is not None):
            
            # Check if the region/monitor matches the cached one
            if ((region is None and self._cached_region is None and 
                 monitor_index is None and self._cached_monitor is None) or
                (region is not None and self._cached_region == region) or
                (monitor_index is not None and self._cached_monitor == monitor_index)):
                
                logger.debug("Using cached screenshot")
                return self._cached_image
        
        return None
    
    def _update_cache(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None, 
                     monitor_index: Optional[int] = None):
        """Update the screenshot cache.
        
        Args:
            image: The captured image
            region: The region that was captured
            monitor_index: The monitor index that was captured
        """
        self._cached_image = image
        self._cached_region = region
        self._cached_monitor = monitor_index
        self._last_capture_time = time.time()
    
    def capture(self, region: Optional[Tuple[int, int, int, int]] = None, 
               monitor_index: Optional[int] = None, 
               force_refresh: bool = False) -> np.ndarray:
        """Capture the screen or a portion of it.
        
        Args:
            region: Optional region to capture (x, y, width, height)
            monitor_index: Optional monitor index to capture
            force_refresh: Force a new capture even if cached
            
        Returns:
            Captured image as BGR numpy array
        """
        # Check cache first if not forcing a refresh
        if not force_refresh:
            cached = self._check_cache(region, monitor_index)
            if cached is not None:
                return cached
        
        # Perform the capture based on the selected mode
        image = None
        
        if self.capture_mode == 'quartz':
            image = self._capture_with_quartz(region, monitor_index)
            
            # Fall back to MSS if Quartz fails
            if image is None and HAS_MSS:
                logger.warning("Quartz capture failed, falling back to MSS")
                image = self._capture_with_mss(region, monitor_index)
        
        elif self.capture_mode == 'mss':
            image = self._capture_with_mss(region, monitor_index)
        
        # Fall back to PyAutoGUI as last resort
        if image is None:
            logger.warning(f"Primary capture method failed, falling back to PyAutoGUI")
            image = self._capture_with_pyautogui(region)
        
        # Update cache if successful
        if image is not None:
            self._update_cache(image, region, monitor_index)
        else:
            logger.error("All screen capture methods failed")
            # Return a black image as last resort
            if region:
                x, y, width, height = region
                return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                # Use the primary monitor dimensions
                monitor = self.get_monitor_info(0)
                return np.zeros((monitor["height"], monitor["width"], 3), dtype=np.uint8)
        
        return image
    
    def start_continuous_capture(self, interval: float = 0.1, buffer_size: int = 5, 
                                region: Optional[Tuple[int, int, int, int]] = None, 
                                monitor_index: Optional[int] = None) -> ContinuousCaptureSession:
        """Start a continuous capture session.
        
        Args:
            interval: Time between captures in seconds
            buffer_size: Number of frames to keep in buffer
            region: Optional region to capture
            monitor_index: Optional monitor index to capture
            
        Returns:
            ContinuousCaptureSession object
        """
        # Create a capture function that includes the region/monitor
        def capture_func(force_refresh: bool = True):
            return self.capture(region=region, monitor_index=monitor_index, force_refresh=force_refresh)
        
        # Create and return the session
        session = ContinuousCaptureSession(
            capture_func=capture_func,
            interval=interval,
            buffer_size=buffer_size
        )
        
        return session
    
    def close(self):
        """Release any resources used by the screen capture."""
        if self._mss_instance:
            self._mss_instance.close()
            self._mss_instance = None


def test_screen_capture():
    """Test function to demonstrate screen capture functionality."""
    # Initialize the screen capture
    screen_capture = ScreenCapture()
    
    # Print available monitors
    print("Available monitors:")
    for i, monitor in enumerate(screen_capture._monitors_info):
        print(f"Monitor {i}: {monitor['width']}x{monitor['height']} at ({monitor['x']}, {monitor['y']})")
    
    # Capture the entire screen
    print("\nCapturing entire screen...")
    full_screen = screen_capture.capture()
    print(f"Full screen captured: {full_screen.shape[1]}x{full_screen.shape[0]} pixels")
    cv2.imwrite("full_screen.png", full_screen)
    
    # Capture a region
    print("\nCapturing region of screen...")
    region = (100, 100, 400, 300)  # x, y, width, height
    region_capture = screen_capture.capture(region=region)
    print(f"Region captured: {region_capture.shape[1]}x{region_capture.shape[0]} pixels")
    cv2.imwrite("region_capture.png", region_capture)
    
    # Test caching
    print("\nTesting caching...")
    start_time = time.time()
    screen_capture.capture(force_refresh=True)
    first_capture_time = time.time() - start_time
    
    start_time = time.time()
    screen_capture.capture(force_refresh=False)
    cached_capture_time = time.time() - start_time
    
    print(f"First capture: {first_capture_time*1000:.2f}ms")
    print(f"Cached capture: {cached_capture_time*1000:.2f}ms")
    
    # Test continuous capture
    print("\nTesting continuous capture...")
    session = screen_capture.start_continuous_capture(interval=0.1, buffer_size=10)
    session.start()
    
    # Wait for a few frames
    time.sleep(1)
    
    # Get the latest frame
    latest = session.get_latest_frame()
    print(f"Latest frame dimensions: {latest.shape[1]}x{latest.shape[0]}")
    cv2.imwrite("continuous_capture.png", latest)
    
    # Check FPS
    fps = session.get_fps()
    print(f"Continuous capture FPS: {fps:.2f}")
    
    # Stop the session
    session.stop()
    
    # Clean up
    screen_capture.close()
    print("\nScreen capture test completed.")


if __name__ == "__main__":
    test_screen_capture()
