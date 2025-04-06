"""
Mouse Controller Module for MacAgent

This module provides precise and reliable mouse control for an AI agent on macOS,
featuring natural movement patterns, configurable parameters, and error recovery.
"""

import time
import random
import logging
import numpy as np
from typing import Tuple, List, Optional, Union, Callable
import math

# macOS-specific libraries
try:
    import Quartz
    from Quartz import CGEventCreateMouseEvent, CGEventPost, kCGEventMouseMoved
    from Quartz import kCGEventLeftMouseDown, kCGEventLeftMouseUp, kCGEventRightMouseDown
    from Quartz import kCGEventRightMouseUp, kCGHIDEventTap, kCGMouseButtonLeft, kCGMouseButtonRight
    from Quartz import CGEventCreateScrollWheelEvent, kCGScrollEventUnitPixel
    from Quartz import kCGEventLeftMouseDragged
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False
    logging.warning("Quartz libraries not available. Some Mac-specific functionality will be limited.")

# Cross-platform fallback
import pyautogui
# Configure pyautogui to be safer
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MouseController:
    """
    Advanced mouse controller for Mac with precise control and human-like movement.
    
    This class provides methods for moving the cursor, clicking, dragging, scrolling,
    and performing other mouse operations with configurable parameters and error recovery.
    """
    
    # Bezier curve parameters for human-like movement
    BEZIER_POINTS = 50
    
    # Movement profiles - speed multipliers for different movement styles
    MOVEMENT_PROFILES = {
        'normal': {
            'start_slowdown': 0.3,
            'end_slowdown': 0.3,
            'precision_mode': False,
            'jitter': 0.5
        },
        'fast': {
            'start_slowdown': 0.1,
            'end_slowdown': 0.2,
            'precision_mode': False,
            'jitter': 0.3
        },
        'precise': {
            'start_slowdown': 0.4,
            'end_slowdown': 0.5,
            'precision_mode': True,
            'jitter': 0.1
        },
        'natural': {
            'start_slowdown': 0.3,
            'end_slowdown': 0.4,
            'precision_mode': False,
            'jitter': 0.8
        }
    }
    
    def __init__(self, 
                 use_quartz: bool = HAS_QUARTZ,
                 default_speed: float = 1.0,
                 default_profile: str = 'normal',
                 recovery_attempts: int = 3,
                 recovery_precision: int = 3):
        """
        Initialize the mouse controller.
        
        Args:
            use_quartz: Whether to use macOS-specific Quartz framework
            default_speed: Default movement speed multiplier
            default_profile: Default movement profile
            recovery_attempts: Number of attempts for error recovery
            recovery_precision: Precision level for error recovery (1-5)
        """
        self.use_quartz = use_quartz and HAS_QUARTZ
        self.default_speed = default_speed
        self.default_profile = default_profile
        self.recovery_attempts = recovery_attempts
        self.recovery_precision = max(1, min(recovery_precision, 5))
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Current position cache
        self._current_x = None
        self._current_y = None
        
        # Last action success flag and timestamp
        self.last_action_success = True
        self.last_action_time = time.time()
        
        logger.info(f"MouseController initialized with: Quartz={self.use_quartz}, "
                   f"Profile={default_profile}, Speed={default_speed}")
    
    def get_position(self) -> Tuple[int, int]:
        """
        Get the current mouse position.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        try:
            if self.use_quartz:
                # Get current mouse position using Quartz
                mouse_loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
                return (int(mouse_loc.x), int(mouse_loc.y))
            else:
                # Use pyautogui as fallback
                x, y = pyautogui.position()
                return (x, y)
        except Exception as e:
            logger.error(f"Error getting mouse position: {e}")
            return (0, 0)
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get the screen size.
        
        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            if self.use_quartz:
                # Get screen size using Quartz
                main_monitor = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
                width = int(main_monitor.size.width)
                height = int(main_monitor.size.height)
                return (width, height)
            else:
                # Use pyautogui as fallback
                width, height = pyautogui.size()
                return (width, height)
        except Exception as e:
            logger.error(f"Error getting screen size: {e}")
            return (1440, 900)  # Return a default fallback size
    
    def _move_quartz(self, x: int, y: int):
        """
        Move mouse using Quartz (macOS-specific).
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
        """
        # Create and post a mouse moved event
        move_event = CGEventCreateMouseEvent(
            None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft
        )
        CGEventPost(kCGHIDEventTap, move_event)
    
    def _click_quartz(self, x: int, y: int, button: str = 'left'):
        """
        Click using Quartz (macOS-specific).
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
            button: Mouse button ('left' or 'right')
        """
        # Determine button constants
        if button == 'left':
            mouse_button = kCGMouseButtonLeft
            down_event_type = kCGEventLeftMouseDown
            up_event_type = kCGEventLeftMouseUp
        else:  # 'right'
            mouse_button = kCGMouseButtonRight
            down_event_type = kCGEventRightMouseDown
            up_event_type = kCGEventRightMouseUp
        
        # Create and post mouse down event
        down_event = CGEventCreateMouseEvent(
            None, down_event_type, (x, y), mouse_button
        )
        CGEventPost(kCGHIDEventTap, down_event)
        
        # Small delay
        time.sleep(0.01)
        
        # Create and post mouse up event
        up_event = CGEventCreateMouseEvent(
            None, up_event_type, (x, y), mouse_button
        )
        CGEventPost(kCGHIDEventTap, up_event)
    
    def _scroll_quartz(self, scroll_amount: int):
        """
        Scroll using Quartz (macOS-specific).
        
        Args:
            scroll_amount: Scroll amount (positive for up, negative for down)
        """
        # Create and post a scroll wheel event
        scroll_event = CGEventCreateScrollWheelEvent(
            None, kCGScrollEventUnitPixel, 1, scroll_amount
        )
        CGEventPost(kCGHIDEventTap, scroll_event)
    
    def _bezier_curve(self, 
                     start: Tuple[int, int], 
                     end: Tuple[int, int], 
                     control_points: int = 2) -> List[Tuple[int, int]]:
        """
        Generate a bezier curve for natural mouse movement.
        
        Args:
            start: Starting coordinates (x, y)
            end: Ending coordinates (x, y)
            control_points: Number of control points (higher = more complex curve)
            
        Returns:
            List of points along the bezier curve
        """
        # Start and end points
        path_points = [start, end]
        
        # Generate random control points
        for _ in range(control_points):
            # Random offset up to 25% of the movement distance
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            # Calculate distance
            distance = math.sqrt(dx ** 2 + dy ** 2)
            
            # Generate random control point
            control_x = start[0] + dx * random.uniform(0.3, 0.7)
            control_y = start[1] + dy * random.uniform(0.3, 0.7)
            
            # Add perpendicular offset
            perpendicular_x = -dy / distance * random.uniform(-0.25, 0.25) * distance
            perpendicular_y = dx / distance * random.uniform(-0.25, 0.25) * distance
            
            control_x += perpendicular_x
            control_y += perpendicular_y
            
            # Add control point
            path_points.insert(-1, (int(control_x), int(control_y)))
        
        # Calculate bezier curve points
        curve = []
        for t in np.linspace(0, 1, self.BEZIER_POINTS):
            point = self._bezier_point(t, path_points)
            curve.append(point)
        
        return curve
    
    def _bezier_point(self, t: float, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Calculate a point on a bezier curve.
        
        Args:
            t: Parameter value (0-1)
            points: Control points
            
        Returns:
            Coordinates (x, y) of point on the curve
        """
        n = len(points) - 1
        x = sum(math.comb(n, i) * (1 - t) ** (n - i) * t ** i * points[i][0] for i in range(n + 1))
        y = sum(math.comb(n, i) * (1 - t) ** (n - i) * t ** i * points[i][1] for i in range(n + 1))
        return int(x), int(y)
    
    def _apply_movement_profile(self, 
                               curve: List[Tuple[int, int]], 
                               profile: str) -> List[Tuple[int, int]]:
        """
        Apply a movement profile to a curve to adjust timing and precision.
        
        Args:
            curve: List of points along the curve
            profile: Movement profile name
            
        Returns:
            Modified curve with adjusted points
        """
        # Get profile parameters
        profile_params = self.MOVEMENT_PROFILES.get(profile, self.MOVEMENT_PROFILES['normal'])
        
        # Adjust points based on profile
        adjusted_curve = []
        
        for i, point in enumerate(curve):
            # Calculate progress along the curve (0 to 1)
            progress = i / (len(curve) - 1) if len(curve) > 1 else 0
            
            # Apply slowdown at start and end of movement
            if progress < profile_params['start_slowdown']:
                # Slow start
                adjusted_point = point
            elif progress > (1 - profile_params['end_slowdown']):
                # Slow end
                if profile_params['precision_mode'] and i > 0:
                    # For precision, make a smoother approach to target
                    prev = curve[i-1]
                    target = curve[-1]
                    
                    # Move closer to target at decreasing rate
                    factor = (1 - progress) / profile_params['end_slowdown']
                    x = prev[0] + (target[0] - prev[0]) * (1 - factor * 0.8)
                    y = prev[1] + (target[1] - prev[1]) * (1 - factor * 0.8)
                    adjusted_point = (int(x), int(y))
                else:
                    adjusted_point = point
            else:
                # Normal movement with optional jitter
                if profile_params['jitter'] > 0 and random.random() < 0.3:
                    jitter_amount = profile_params['jitter']
                    x = point[0] + random.randint(-int(jitter_amount), int(jitter_amount))
                    y = point[1] + random.randint(-int(jitter_amount), int(jitter_amount))
                    adjusted_point = (x, y)
                else:
                    adjusted_point = point
            
            adjusted_curve.append(adjusted_point)
        
        return adjusted_curve
    
    def move_to(self, 
               x: int, 
               y: int, 
               speed: float = None, 
               profile: str = None, 
               timeout: float = 5.0) -> bool:
        """
        Move the mouse cursor to the specified coordinates.
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
            speed: Movement speed multiplier (1.0 = normal)
            profile: Movement profile ('normal', 'fast', 'precise', 'natural')
            timeout: Maximum time to spend on movement in seconds
            
        Returns:
            True if movement was successful, False otherwise
        """
        # Use default values if not specified
        speed = speed or self.default_speed
        profile = profile or self.default_profile
        
        # Get current position
        start_x, start_y = self.get_position()
        
        # Ensure coordinates are within screen bounds
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        
        logger.debug(f"Moving from ({start_x}, {start_y}) to ({x}, {y}) with profile={profile}, speed={speed}")
        
        # If already at destination, do nothing
        if start_x == x and start_y == y:
            return True
        
        try:
            # Generate movement curve
            curve = self._bezier_curve((start_x, start_y), (x, y))
            
            # Apply movement profile
            curve = self._apply_movement_profile(curve, profile)
            
            # Calculate movement duration based on distance and speed
            distance = math.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
            duration = min(timeout, max(0.1, distance / (2000 * speed)))
            
            # Calculate time per step
            time_per_step = duration / len(curve)
            
            # Execute movement
            start_time = time.time()
            for point_x, point_y in curve:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Movement timeout: took more than {timeout}s")
                    break
                
                # Move to next point
                if self.use_quartz:
                    self._move_quartz(point_x, point_y)
                else:
                    pyautogui.moveTo(point_x, point_y, _pause=False)
                
                # Update current position
                self._current_x, self._current_y = point_x, point_y
                
                # Sleep for calculated time
                time.sleep(time_per_step * (0.8 + random.random() * 0.4))  # Randomize sleep time slightly
            
            # Ensure we end up exactly at the target
            if self.use_quartz:
                self._move_quartz(x, y)
            else:
                pyautogui.moveTo(x, y, _pause=False)
            
            # Update current position
            self._current_x, self._current_y = x, y
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in move_to: {e}")
            self.last_action_success = False
            return False
    
    def click(self, 
             x: Optional[int] = None, 
             y: Optional[int] = None, 
             button: str = 'left', 
             clicks: int = 1, 
             interval: float = 0.1,
             move_speed: float = None,
             move_profile: str = None) -> bool:
        """
        Click at the specified coordinates or current position.
        
        Args:
            x: Target x coordinate (None for current position)
            y: Target y coordinate (None for current position)
            button: Mouse button ('left' or 'right')
            clicks: Number of clicks
            interval: Time between clicks in seconds
            move_speed: Movement speed if coordinates are provided
            move_profile: Movement profile if coordinates are provided
            
        Returns:
            True if click was successful, False otherwise
        """
        try:
            # Get current position if x,y not specified
            if x is None or y is None:
                x, y = self.get_position()
            else:
                # Move to specified position first
                move_success = self.move_to(
                    x, y, 
                    speed=move_speed or self.default_speed,
                    profile=move_profile or 'precise'
                )
                if not move_success:
                    return False
            
            # Execute clicks
            for i in range(clicks):
                if self.use_quartz:
                    self._click_quartz(x, y, button)
                else:
                    if button == 'left':
                        pyautogui.click(x, y, _pause=False)
                    else:
                        pyautogui.rightClick(x, y, _pause=False)
                
                # Wait between clicks if multiple
                if i < clicks - 1:
                    time.sleep(interval)
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in click: {e}")
            self.last_action_success = False
            return False
    
    def double_click(self, 
                    x: Optional[int] = None, 
                    y: Optional[int] = None,
                    move_speed: float = None,
                    move_profile: str = None) -> bool:
        """
        Perform a double-click at the specified coordinates or current position.
        
        Args:
            x: Target x coordinate (None for current position)
            y: Target y coordinate (None for current position)
            move_speed: Movement speed if coordinates are provided
            move_profile: Movement profile if coordinates are provided
            
        Returns:
            True if double-click was successful, False otherwise
        """
        return self.click(x, y, 'left', 2, 0.1, move_speed, move_profile)
    
    def right_click(self, 
                   x: Optional[int] = None, 
                   y: Optional[int] = None,
                   move_speed: float = None,
                   move_profile: str = None) -> bool:
        """
        Perform a right-click at the specified coordinates or current position.
        
        Args:
            x: Target x coordinate (None for current position)
            y: Target y coordinate (None for current position)
            move_speed: Movement speed if coordinates are provided
            move_profile: Movement profile if coordinates are provided
            
        Returns:
            True if right-click was successful, False otherwise
        """
        return self.click(x, y, 'right', 1, 0, move_speed, move_profile)
    
    def drag_to(self, 
               x: int, 
               y: int, 
               from_x: Optional[int] = None, 
               from_y: Optional[int] = None,
               speed: float = None,
               profile: str = None) -> bool:
        """
        Drag from one position to another.
        
        Args:
            x: Target x coordinate
            y: Target y coordinate
            from_x: Starting x coordinate (None for current position)
            from_y: Starting y coordinate (None for current position)
            speed: Movement speed multiplier
            profile: Movement profile
            
        Returns:
            True if drag was successful, False otherwise
        """
        try:
            # Use current position if starting position not specified
            if from_x is None or from_y is None:
                from_x, from_y = self.get_position()
            else:
                # Move to starting position first
                move_success = self.move_to(from_x, from_y, speed, profile)
                if not move_success:
                    return False
            
            # Ensure coordinates are within screen bounds
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            
            if self.use_quartz:
                # Quartz implementation
                # Mouse down at starting position
                down_event = CGEventCreateMouseEvent(
                    None, kCGEventLeftMouseDown, (from_x, from_y), kCGMouseButtonLeft
                )
                CGEventPost(kCGHIDEventTap, down_event)
                
                # Generate drag movement curve
                curve = self._bezier_curve((from_x, from_y), (x, y))
                curve = self._apply_movement_profile(curve, profile or self.default_profile)
                
                # Calculate drag duration based on distance and speed
                speed_factor = speed or self.default_speed
                distance = math.sqrt((x - from_x) ** 2 + (y - from_y) ** 2)
                duration = max(0.2, distance / (1000 * speed_factor))
                time_per_step = duration / len(curve)
                
                # Execute drag movement
                for point_x, point_y in curve:
                    # Create drag event
                    drag_event = CGEventCreateMouseEvent(
                        None, kCGEventLeftMouseDragged, (point_x, point_y), kCGMouseButtonLeft
                    )
                    CGEventPost(kCGHIDEventTap, drag_event)
                    
                    # Update current position
                    self._current_x, self._current_y = point_x, point_y
                    
                    # Sleep for calculated time
                    time.sleep(time_per_step)
                
                # Final position adjustment
                final_drag = CGEventCreateMouseEvent(
                    None, kCGEventLeftMouseDragged, (x, y), kCGMouseButtonLeft
                )
                CGEventPost(kCGHIDEventTap, final_drag)
                
                # Mouse up at ending position
                up_event = CGEventCreateMouseEvent(
                    None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft
                )
                CGEventPost(kCGHIDEventTap, up_event)
                
            else:
                # PyAutoGUI implementation
                # Move to start position
                pyautogui.moveTo(from_x, from_y, _pause=False)
                
                # Drag to end position
                pyautogui.dragTo(
                    x, y, 
                    duration=max(0.2, distance / (1000 * (speed or self.default_speed))),
                    _pause=False
                )
            
            # Update current position
            self._current_x, self._current_y = x, y
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in drag_to: {e}")
            self.last_action_success = False
            return False
    
    def scroll(self, 
              amount: int, 
              x: Optional[int] = None, 
              y: Optional[int] = None,
              incremental: bool = True,
              duration: float = 0.5) -> bool:
        """
        Scroll the mouse wheel.
        
        Args:
            amount: Scroll amount (positive for up, negative for down)
            x: X coordinate to position mouse before scrolling (None for current position)
            y: Y coordinate to position mouse before scrolling (None for current position)
            incremental: Whether to scroll incrementally (smoother)
            duration: Duration of scrolling in seconds (only used if incremental=True)
            
        Returns:
            True if scroll was successful, False otherwise
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self.move_to(x, y, profile='fast')
            
            if incremental and abs(amount) > 5:
                # Incremental scrolling (smoother)
                increments = min(abs(amount), 20)  # Max 20 increments
                increment_size = amount / increments
                increment_delay = duration / increments
                
                for _ in range(increments):
                    if self.use_quartz:
                        self._scroll_quartz(int(increment_size))
                    else:
                        pyautogui.scroll(int(increment_size), _pause=False)
                    time.sleep(increment_delay)
            else:
                # Single scroll
                if self.use_quartz:
                    self._scroll_quartz(amount)
                else:
                    pyautogui.scroll(amount, _pause=False)
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in scroll: {e}")
            self.last_action_success = False
            return False
    
    def move_relative(self, 
                     dx: int, 
                     dy: int, 
                     speed: float = None,
                     profile: str = None) -> bool:
        """
        Move the cursor relative to its current position.
        
        Args:
            dx: X offset
            dy: Y offset
            speed: Movement speed multiplier
            profile: Movement profile
            
        Returns:
            True if movement was successful, False otherwise
        """
        current_x, current_y = self.get_position()
        return self.move_to(
            current_x + dx, 
            current_y + dy, 
            speed=speed, 
            profile=profile
        )
    
    def drag(self,
            start_x: int,
            start_y: int,
            end_x: int,
            end_y: int,
            speed: float = None,
            move_profile: str = None,
            hold_duration: float = 0.1) -> bool:
        """
        Drag from start position to end position.
        
        Args:
            start_x: Starting x coordinate
            start_y: Starting y coordinate
            end_x: Target x coordinate
            end_y: Target y coordinate
            speed: Movement speed multiplier
            move_profile: Movement profile
            hold_duration: How long to hold before dragging
            
        Returns:
            True if drag was successful, False otherwise
        """
        # First move to start position
        move_success = self.move_to(start_x, start_y, speed, move_profile)
        if not move_success:
            return False
            
        # Small delay to simulate holding before drag
        if hold_duration > 0:
            time.sleep(hold_duration)
            
        # Perform the drag operation
        return self.drag_to(end_x, end_y, start_x, start_y, speed, move_profile)
    
    def click_on_element(self, 
                        element: Union[Tuple[int, int, int, int], dict],
                        button: str = 'left',
                        clicks: int = 1,
                        offset_x: float = 0.5,
                        offset_y: float = 0.5,
                        retry_on_failure: bool = True) -> bool:
        """
        Click on a UI element at a relative offset.
        
        Args:
            element: Either (x, y, width, height) tuple or dict with 'bounds' key
            button: Mouse button ('left' or 'right')
            clicks: Number of clicks
            offset_x: Relative x offset within element (0-1)
            offset_y: Relative y offset within element (0-1)
            retry_on_failure: Whether to retry with error recovery
            
        Returns:
            True if click was successful, False otherwise
        """
        try:
            # Extract bounds from element
            if isinstance(element, tuple) and len(element) == 4:
                x, y, width, height = element
            elif isinstance(element, dict) and 'bounds' in element:
                x, y, width, height = element['bounds']
            else:
                logger.error("Invalid element format in click_on_element")
                return False
            
            # Calculate target position with offset
            target_x = int(x + width * offset_x)
            target_y = int(y + height * offset_y)
            
            # Click on target
            success = self.click(
                target_x, target_y,
                button=button,
                clicks=clicks,
                move_profile='precise'
            )
            
            # Error recovery if needed
            if not success and retry_on_failure:
                return self._recover_click(x, y, width, height, button, clicks)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in click_on_element: {e}")
            self.last_action_success = False
            return False
    
    def _recover_click(self, 
                      x: int, 
                      y: int, 
                      width: int, 
                      height: int,
                      button: str,
                      clicks: int) -> bool:
        """
        Attempt to recover from a failed click by trying different positions.
        
        Args:
            x, y, width, height: Element bounds
            button: Mouse button to click
            clicks: Number of clicks
            
        Returns:
            True if recovery was successful, False otherwise
        """
        logger.info("Attempting click recovery")
        
        # Try different positions within the element
        positions = [
            (0.5, 0.5),  # Center
            (0.25, 0.25),  # Top-left quadrant
            (0.75, 0.25),  # Top-right quadrant
            (0.25, 0.75),  # Bottom-left quadrant
            (0.75, 0.75),  # Bottom-right quadrant
            (0.5, 0.25),   # Top center
            (0.5, 0.75),   # Bottom center
            (0.25, 0.5),   # Left center
            (0.75, 0.5)    # Right center
        ]
        
        # Higher precision during recovery
        original_profile = self.default_profile
        self.default_profile = 'precise'
        
        for attempt in range(self.recovery_attempts):
            for offset_x, offset_y in positions[:min(len(positions), attempt+3)]:
                # Calculate target position
                target_x = int(x + width * offset_x)
                target_y = int(y + height * offset_y)
                
                # Click on target with increased precision
                success = self.click(
                    target_x, target_y,
                    button=button,
                    clicks=clicks,
                    move_profile='precise'
                )
                
                if success:
                    logger.info(f"Click recovery successful at position ({offset_x}, {offset_y})")
                    self.default_profile = original_profile
                    return True
                
                # Small delay between attempts
                time.sleep(0.1)
        
        # Restore original profile
        self.default_profile = original_profile
        logger.warning("Click recovery failed after all attempts")
        return False
    
    def hover_over_element(self, 
                          element: Union[Tuple[int, int, int, int], dict],
                          offset_x: float = 0.5,
                          offset_y: float = 0.5,
                          duration: float = 0.5) -> bool:
        """
        Hover over a UI element at a relative offset.
        
        Args:
            element: Either (x, y, width, height) tuple or dict with 'bounds' key
            offset_x: Relative x offset within element (0-1)
            offset_y: Relative y offset within element (0-1)
            duration: Time to hover in seconds
            
        Returns:
            True if hover was successful, False otherwise
        """
        try:
            # Extract bounds from element
            if isinstance(element, tuple) and len(element) == 4:
                x, y, width, height = element
            elif isinstance(element, dict) and 'bounds' in element:
                x, y, width, height = element['bounds']
            else:
                logger.error("Invalid element format in hover_over_element")
                return False
            
            # Calculate target position with offset
            target_x = int(x + width * offset_x)
            target_y = int(y + height * offset_y)
            
            # Move to target
            success = self.move_to(
                target_x, target_y,
                profile='precise'
            )
            
            if success:
                # Add subtle movement during hover to mimic human behavior
                start_time = time.time()
                while time.time() - start_time < duration:
                    # Small random movement
                    jitter_x = random.randint(-2, 2)
                    jitter_y = random.randint(-2, 2)
                    
                    # Ensure we stay within the element
                    new_x = min(max(x, target_x + jitter_x), x + width)
                    new_y = min(max(y, target_y + jitter_y), y + height)
                    
                    if self.use_quartz:
                        self._move_quartz(new_x, new_y)
                    else:
                        pyautogui.moveTo(new_x, new_y, _pause=False)
                    
                    # Update current position
                    self._current_x, self._current_y = new_x, new_y
                    
                    # Sleep for a bit
                    time.sleep(0.05)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in hover_over_element: {e}")
            self.last_action_success = False
            return False
