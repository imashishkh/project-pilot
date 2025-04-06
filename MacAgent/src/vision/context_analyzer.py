"""
Context Analyzer Module for MacAgent

This module provides functionality for analyzing the overall context of screens,
identifying application windows, and determining the current state of the interface.
It works with other vision modules to provide a higher-level understanding of what's
happening on screen.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass
from enum import Enum, auto
import os
import time

# Try to import macOS-specific libraries
try:
    import Quartz
    from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
    from Quartz import kCGWindowLayer, kCGWindowName, kCGWindowOwnerName, kCGWindowBounds
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False
    logging.warning("Quartz libraries not available. Some Mac-specific functionality will be limited.")

# Local imports
from .element_detector import UIElement, ElementType, ElementState, ThemeMode
from .screen_analyzer import ScreenAnalyzer

# Configure logging
logger = logging.getLogger(__name__)


class AppContext(Enum):
    """Enum for application context types."""
    UNKNOWN = auto()
    FINDER = auto()
    BROWSER = auto()
    TERMINAL = auto()
    TEXT_EDITOR = auto()
    SYSTEM_DIALOG = auto()
    PREFERENCES = auto()
    MEDIA_PLAYER = auto()
    PRODUCTIVITY = auto()
    COMMUNICATION = auto()
    DEVELOPMENT = auto()


class WindowState(Enum):
    """Enum for window states."""
    UNKNOWN = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    MINIMIZED = auto()
    MAXIMIZED = auto()
    DIALOG = auto()
    MODAL = auto()


@dataclass
class AppWindow:
    """Class representing an application window."""
    window_id: int
    app_name: str
    window_name: str
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    is_active: bool = False
    state: WindowState = WindowState.UNKNOWN
    process_id: Optional[int] = None
    layer: Optional[int] = None
    alpha: Optional[float] = None
    context: AppContext = AppContext.UNKNOWN
    elements: List[UIElement] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.elements is None:
            self.elements = []
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the window."""
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        """Get the area of the window in pixels."""
        _, _, w, h = self.bounds
        return w * h


class ContextAnalyzer:
    """
    Analyzes screen context, identifies application windows, and
    determines the current state of the interface.
    
    This class integrates with other vision components to provide
    a higher-level understanding of what's happening on screen.
    """
    
    def __init__(self, 
                 screen_analyzer: Optional[ScreenAnalyzer] = None,
                 enable_quartz: bool = HAS_QUARTZ,
                 cache_ttl: float = 1.0):
        """
        Initialize the context analyzer.
        
        Args:
            screen_analyzer: ScreenAnalyzer instance (created if not provided)
            enable_quartz: Whether to use Quartz for window detection (if available)
            cache_ttl: How long to cache window info (seconds)
        """
        self.screen_analyzer = screen_analyzer or ScreenAnalyzer()
        self.enable_quartz = enable_quartz and HAS_QUARTZ
        self.cache_ttl = cache_ttl
        
        # Caching for performance
        self._cached_windows = []
        self._last_window_scan = 0
        
        logger.info(f"Initialized ContextAnalyzer: Quartz={self.enable_quartz}")
    
    def analyze_context(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the full context of a screen image.
        
        Args:
            image: The screen image to analyze
            
        Returns:
            Dictionary containing context analysis results
        """
        logger.info("Starting context analysis")
        
        # Get window information
        windows = self.get_windows()
        
        # Analyze UI elements in the image
        ui_analysis = self.screen_analyzer.analyze_screen(image)
        
        # Detect the active application context
        active_window = self._find_active_window(windows)
        app_context = self._determine_app_context(active_window)
        
        # Match detected UI elements to windows
        self._match_elements_to_windows(windows, ui_analysis.get("elements", []))
        
        # Determine the current task context
        task_context = self._determine_task_context(active_window, ui_analysis)
        
        # Build the complete context
        context = {
            "timestamp": time.time(),
            "active_window": active_window,
            "windows": windows,
            "app_context": app_context,
            "task_context": task_context,
            "elements": ui_analysis.get("elements", []),
            "text_blocks": ui_analysis.get("text_blocks", []),
            "overall_theme": self._determine_theme(image)
        }
        
        logger.info(f"Context analysis complete. Found {len(windows)} windows, {len(context['elements'])} UI elements")
        return context
    
    def get_windows(self) -> List[AppWindow]:
        """
        Get information about all visible windows.
        
        Returns:
            List of AppWindow objects representing visible windows
        """
        current_time = time.time()
        
        # Return cached windows if they're fresh enough
        if self._cached_windows and (current_time - self._last_window_scan) < self.cache_ttl:
            return self._cached_windows
        
        windows = []
        
        # Use Quartz CGWindowListCopyWindowInfo if available (macOS-specific)
        if self.enable_quartz:
            # Get window info from Quartz
            window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
            
            for window_info in window_list:
                try:
                    window_id = window_info.get("kCGWindowNumber", 0)
                    app_name = window_info.get(kCGWindowOwnerName, "")
                    window_name = window_info.get(kCGWindowName, "")
                    
                    # Skip windows without names or from background processes
                    if not app_name or window_id == 0:
                        continue
                    
                    # Get window bounds
                    bounds_dict = window_info.get(kCGWindowBounds, {})
                    if bounds_dict:
                        x = bounds_dict.get("X", 0)
                        y = bounds_dict.get("Y", 0)
                        w = bounds_dict.get("Width", 0)
                        h = bounds_dict.get("Height", 0)
                        bounds = (int(x), int(y), int(w), int(h))
                    else:
                        continue  # Skip windows without bounds
                    
                    # Get other window properties
                    layer = window_info.get(kCGWindowLayer, 0)
                    alpha = window_info.get("kCGWindowAlpha", 1.0)
                    pid = window_info.get("kCGWindowOwnerPID", None)
                    
                    # Determine if this is the active window (typically layer 0)
                    is_active = (layer == 0)
                    
                    # Determine window state
                    state = WindowState.ACTIVE if is_active else WindowState.INACTIVE
                    if alpha < 0.1:
                        state = WindowState.MINIMIZED
                    
                    # Create an AppWindow object
                    app_window = AppWindow(
                        window_id=window_id,
                        app_name=app_name,
                        window_name=window_name,
                        bounds=bounds,
                        is_active=is_active,
                        state=state,
                        process_id=pid,
                        layer=layer,
                        alpha=alpha
                    )
                    
                    # Determine app context for this window
                    app_window.context = self._determine_app_context(app_window)
                    
                    windows.append(app_window)
                except Exception as e:
                    logger.error(f"Error processing window info: {e}")
        else:
            # Fallback for non-macOS or when Quartz is unavailable
            # We'll have to rely on visual analysis in this case
            logger.warning("Using fallback window detection (less accurate)")
            
            # Detect windows based on visual features
            windows = self._detect_windows_visually(None)  # Will use cached image if available
        
        # Sort windows by layer (active window first) and size
        windows.sort(key=lambda w: (0 if w.is_active else (w.layer or 999), -w.area))
        
        # Update cache
        self._cached_windows = windows
        self._last_window_scan = current_time
        
        return windows
    
    def _detect_windows_visually(self, image: Optional[np.ndarray] = None) -> List[AppWindow]:
        """
        Detect windows visually in an image using computer vision.
        
        Args:
            image: The screen image to analyze (or None to use cached)
            
        Returns:
            List of detected AppWindow objects
        """
        # This is a fallback when Quartz is unavailable
        # In a real implementation, this would use computer vision to detect window boundaries
        # For now, return a minimal implementation
        
        if image is None:
            # No image provided and no cached image
            return []
        
        windows = []
        
        # Simple placeholder window detection 
        # (In a real implementation, this would use contour detection, edge detection, etc.)
        h, w = image.shape[:2]
        
        # Assume one main window for now
        main_window = AppWindow(
            window_id=1,
            app_name="Unknown",
            window_name="Main Window",
            bounds=(0, 0, w, h),
            is_active=True,
            state=WindowState.ACTIVE
        )
        windows.append(main_window)
        
        return windows
    
    def _find_active_window(self, windows: List[AppWindow]) -> Optional[AppWindow]:
        """
        Find the currently active window.
        
        Args:
            windows: List of window objects
            
        Returns:
            The active window, or None if not found
        """
        for window in windows:
            if window.is_active:
                return window
        
        # If no window is marked as active, return the topmost window
        return windows[0] if windows else None
    
    def _determine_app_context(self, window: Optional[AppWindow]) -> AppContext:
        """
        Determine the application context based on window information.
        
        Args:
            window: Window information
            
        Returns:
            AppContext enum value
        """
        if not window:
            return AppContext.UNKNOWN
        
        app_name = window.app_name.lower()
        window_name = window.window_name.lower()
        
        # Determine context based on app name
        if any(name in app_name for name in ["finder", "explorer"]):
            return AppContext.FINDER
        elif any(name in app_name for name in ["chrome", "safari", "firefox", "edge"]):
            return AppContext.BROWSER
        elif any(name in app_name for name in ["terminal", "iterm", "command"]):
            return AppContext.TERMINAL
        elif any(name in app_name for name in ["word", "pages", "text", "notes"]):
            return AppContext.TEXT_EDITOR
        elif any(name in app_name for name in ["preferences", "settings"]):
            return AppContext.PREFERENCES
        elif any(name in app_name for name in ["vlc", "quicktime", "itunes", "music", "spotify"]):
            return AppContext.MEDIA_PLAYER
        elif any(name in app_name for name in ["slack", "teams", "zoom", "messages"]):
            return AppContext.COMMUNICATION
        elif any(name in app_name for name in ["code", "xcode", "pycharm", "intellij", "android studio"]):
            return AppContext.DEVELOPMENT
        elif any(name in app_name for name in ["keynote", "powerpoint", "excel", "numbers"]):
            return AppContext.PRODUCTIVITY
        
        # Check window name for dialog indicators
        if any(term in window_name for term in ["dialog", "alert", "warning", "error"]):
            return AppContext.SYSTEM_DIALOG
        
        return AppContext.UNKNOWN
    
    def _determine_task_context(self, 
                               active_window: Optional[AppWindow],
                               ui_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the current task context based on window and UI analysis.
        
        Args:
            active_window: The active window
            ui_analysis: UI analysis data
            
        Returns:
            Dictionary containing task context information
        """
        task_context = {
            "task_type": "unknown",
            "actions_available": [],
            "form_filling": False,
            "reading_content": False,
            "media_controls": False,
            "navigation": False
        }
        
        if not active_window:
            return task_context
        
        # Extract UI elements
        elements = ui_analysis.get("elements", [])
        
        # Look for forms
        text_fields = [e for e in elements if e.element_type in 
                      [ElementType.TEXT_FIELD, ElementType.DROPDOWN]]
        buttons = [e for e in elements if e.element_type == ElementType.BUTTON]
        
        # Determine task type based on elements and app context
        if len(text_fields) > 1 and any(b.text and "submit" in b.text.lower() for b in buttons):
            task_context["task_type"] = "form_filling"
            task_context["form_filling"] = True
        
        # Check for media playback controls
        media_controls = [e for e in elements if e.text and 
                         any(c in e.text.lower() for c in ["play", "pause", "stop", "volume"])]
        if media_controls or active_window.context == AppContext.MEDIA_PLAYER:
            task_context["media_controls"] = True
            task_context["task_type"] = "media_playback"
        
        # Check for navigation elements
        nav_elements = [e for e in elements if e.text and 
                       any(n in e.text.lower() for n in ["back", "forward", "home", "menu"])]
        if nav_elements or active_window.context == AppContext.BROWSER:
            task_context["navigation"] = True
            if task_context["task_type"] == "unknown":
                task_context["task_type"] = "browsing"
        
        # Check for text content
        text_blocks = ui_analysis.get("text_blocks", [])
        if len(text_blocks) > 3:
            task_context["reading_content"] = True
            if task_context["task_type"] == "unknown":
                task_context["task_type"] = "reading"
        
        # Determine available actions based on UI elements
        for element in elements:
            if element.element_type == ElementType.BUTTON and element.text:
                task_context["actions_available"].append({
                    "action": "click",
                    "target": element.text,
                    "element_id": element.element_id
                })
        
        return task_context
    
    def _determine_theme(self, image: np.ndarray) -> ThemeMode:
        """
        Determine if the interface is using light or dark theme.
        
        Args:
            image: Screen image
            
        Returns:
            ThemeMode enum value
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # If average brightness is less than 128, it's likely a dark theme
        if avg_brightness < 128:
            return ThemeMode.DARK
        else:
            return ThemeMode.LIGHT
    
    def _match_elements_to_windows(self, 
                                  windows: List[AppWindow], 
                                  elements: List[UIElement]) -> None:
        """
        Associate UI elements with their containing windows.
        
        Args:
            windows: List of window objects
            elements: List of UI elements
        """
        for element in elements:
            elem_x, elem_y, elem_w, elem_h = element.bounding_box
            elem_center = (elem_x + elem_w // 2, elem_y + elem_h // 2)
            
            for window in windows:
                win_x, win_y, win_w, win_h = window.bounds
                
                # Check if element center is within window bounds
                if (win_x <= elem_center[0] <= win_x + win_w and 
                    win_y <= elem_center[1] <= win_y + win_h):
                    window.elements.append(element)
                    break
    
    def get_focused_ui_element(self, 
                              image: np.ndarray, 
                              windows: Optional[List[AppWindow]] = None) -> Optional[UIElement]:
        """
        Identify the currently focused UI element (e.g., text field with cursor).
        
        Args:
            image: Screen image
            windows: List of window objects (or None to detect)
            
        Returns:
            The focused UI element, or None if not found
        """
        windows = windows or self.get_windows()
        active_window = self._find_active_window(windows)
        
        if not active_window:
            return None
        
        # If we already have elements associated with the window, check those first
        if active_window.elements:
            for element in active_window.elements:
                if element.state == ElementState.FOCUSED:
                    return element
        
        # Analyze the image to find elements
        ui_analysis = self.screen_analyzer.analyze_screen(image)
        elements = ui_analysis.get("elements", [])
        
        # Look for elements that appear to be focused
        for element in elements:
            if element.state == ElementState.FOCUSED:
                return element
        
        # As a fallback, look for text fields
        text_fields = [e for e in elements if e.element_type == ElementType.TEXT_FIELD]
        if text_fields:
            # Sort by likelihood of being focused (this could be improved)
            text_fields.sort(key=lambda e: e.confidence, reverse=True)
            return text_fields[0]
        
        return None
