"""
Action Module for MacAgent.

This module handles all mouse and keyboard interactions for the agent.
It provides capabilities to move the mouse, click, type, and perform other UI interactions.
"""

import logging
import asyncio
import time
import inspect
from typing import Tuple, Optional, List, Dict, Any, Union, Callable
import pyautogui
import pynput
from pynput.keyboard import Key
from pynput.mouse import Button

# Configure logger
logger = logging.getLogger(__name__)

# Set up safety measures
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1  # Small delay between PyAutoGUI actions


async def _filter_params(func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter parameters dictionary to only include parameters that exist in the function signature.
    
    Args:
        func: The function whose signature will be checked
        params: Dictionary of parameters to filter
        
    Returns:
        Filtered parameters dictionary containing only valid parameters for the function
    """
    try:
        # Get the function signature
        sig = inspect.signature(func)
        
        # Create a new dictionary with only the parameters that exist in the function signature
        filtered_params = {
            param_name: param_value 
            for param_name, param_value in params.items() 
            if param_name in sig.parameters
        }
        
        # Log parameter mismatches
        await _log_parameter_mismatch(func.__name__, params, sig.parameters)
        
        logger.debug(f"Filtered parameters for {func.__name__}: {filtered_params}")
        return filtered_params
    except Exception as e:
        logger.error(f"Failed to filter parameters: {str(e)}")
        return {}


async def _log_parameter_mismatch(func_name: str, provided_params: Dict[str, Any], 
                                 accepted_params: Dict[str, inspect.Parameter]) -> None:
    """
    Log warnings for parameter mismatches between provided and accepted parameters.
    
    Args:
        func_name: Name of the function being called
        provided_params: Dictionary of parameters provided to the function
        accepted_params: Dictionary of parameters accepted by the function
    """
    try:
        # Identify unexpected parameters (provided but not accepted)
        unexpected_params = set(provided_params.keys()) - set(accepted_params.keys())
        if unexpected_params:
            logger.warning(f"Function '{func_name}' received unexpected parameters: {unexpected_params}")
        
        # Identify missing required parameters (required but not provided)
        missing_required = [
            param_name for param_name, param in accepted_params.items()
            if param.default == inspect.Parameter.empty and param_name not in provided_params
            and param.kind != inspect.Parameter.VAR_POSITIONAL
            and param.kind != inspect.Parameter.VAR_KEYWORD
        ]
        
        if missing_required:
            logger.warning(f"Function '{func_name}' missing required parameters: {missing_required}")
            
    except Exception as e:
        logger.error(f"Error analyzing parameter mismatch for {func_name}: {str(e)}")


class ActionModule:
    """
    Handles mouse and keyboard interaction functionality.
    
    This module is responsible for controlling the mouse, sending keystrokes,
    and performing UI interactions based on agent decisions.
    """
    
    def __init__(self, move_speed: float = 0.3, click_delay: float = 0.1, debug_mode: bool = False):
        """
        Initialize the action module.
        
        Args:
            move_speed: Duration of mouse movement in seconds (slower is more realistic)
            click_delay: Delay between clicks in seconds
            debug_mode: Whether to enable detailed debug logging
        """
        self.move_speed = move_speed
        self.click_delay = click_delay
        self.debug_mode = debug_mode
        self.mouse_controller = pynput.mouse.Controller()
        self.keyboard_controller = pynput.keyboard.Controller()
        logger.info("ActionModule initialized")
    
    def set_debug_mode(self, enabled: bool) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            enabled: Whether to enable debug mode
        """
        self.debug_mode = enabled
        logger.info(f"Debug mode {'enabled' if enabled else 'disabled'}")
    
    async def _debug_trace(self, action: str, params: Dict[str, Any], result: Any = None) -> None:
        """
        Log detailed debug information about an action.
        
        Args:
            action: Name of the action being performed
            params: Parameters for the action
            result: Result of the action (if available)
        """
        if not self.debug_mode:
            return
        
        # Create a detailed log message
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        message = f"DEBUG TRACE: {action}({param_str})"
        
        if result is not None:
            message += f" => {result}"
        
        logger.debug(message)
    
    async def move_to(self, x: int, y: int, duration: Optional[float] = None) -> bool:
        """
        Move the mouse to the specified coordinates.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            duration: Optional override for movement duration
            
        Returns:
            True if successful, False otherwise
        """
        params = {"x": x, "y": y, "duration": duration if duration is not None else self.move_speed}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("move_to", params, None)
                
            move_duration = duration if duration is not None else self.move_speed
            logger.debug(f"Moving mouse to ({x}, {y}) over {move_duration}s")
            
            # Use PyAutoGUI for smoother movement with duration
            pyautogui.moveTo(x, y, duration=move_duration)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("move_to", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("move_to", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to move mouse: {str(e)}")
            return False
    
    async def click(self, 
                   button: str = "left", 
                   clicks: int = 1, 
                   interval: Optional[float] = None) -> bool:
        """
        Perform a mouse click at the current position.
        
        Args:
            button: Mouse button to click ("left", "right", "middle")
            clicks: Number of clicks to perform
            interval: Time between clicks in seconds
            
        Returns:
            True if successful, False otherwise
        """
        params = {"button": button, "clicks": clicks, "interval": interval if interval is not None else self.click_delay}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("click", params, None)
                
            click_interval = interval if interval is not None else self.click_delay
            
            # Map string button name to pynput Button
            button_map = {
                "left": Button.left,
                "right": Button.right,
                "middle": Button.middle
            }
            pynput_button = button_map.get(button.lower(), Button.left)
            
            logger.debug(f"Clicking {button} button {clicks} times")
            
            for i in range(clicks):
                self.mouse_controller.click(pynput_button)
                if i < clicks - 1:
                    await asyncio.sleep(click_interval)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("click", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("click", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to click: {str(e)}")
            return False
    
    async def click_at(self, 
                      x: int, 
                      y: int, 
                      button: str = "left", 
                      clicks: int = 1) -> bool:
        """
        Move to coordinates and perform a mouse click.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            button: Mouse button to click ("left", "right", "middle")
            clicks: Number of clicks to perform
            
        Returns:
            True if successful, False otherwise
        """
        params = {"x": x, "y": y, "button": button, "clicks": clicks}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("click_at", params, None)
                
            move_result = await self.move_to(x, y)
            if not move_result:
                # Debug trace for early return
                if self.debug_mode:
                    await self._debug_trace("click_at", params, False)
                return False
            
            # Small pause after moving before clicking
            await asyncio.sleep(0.1)
            
            result = await self.click(button, clicks)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("click_at", params, result)
                
            return result
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("click_at", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to click at coordinates: {str(e)}")
            return False
    
    async def drag_to(self, 
                     start_x: int, 
                     start_y: int, 
                     end_x: int, 
                     end_y: int, 
                     duration: Optional[float] = None) -> bool:
        """
        Perform a drag operation from start to end coordinates.
        
        Args:
            start_x: Starting X-coordinate
            start_y: Starting Y-coordinate
            end_x: Ending X-coordinate
            end_y: Ending Y-coordinate
            duration: Optional override for drag duration
            
        Returns:
            True if successful, False otherwise
        """
        params = {
            "start_x": start_x, 
            "start_y": start_y, 
            "end_x": end_x, 
            "end_y": end_y, 
            "duration": duration if duration is not None else self.move_speed
        }
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("drag_to", params, None)
                
            drag_duration = duration if duration is not None else self.move_speed
            logger.debug(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            
            # Move to start position
            await self.move_to(start_x, start_y)
            await asyncio.sleep(0.1)
            
            # Use PyAutoGUI for the drag operation
            pyautogui.dragTo(end_x, end_y, duration=drag_duration)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("drag_to", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("drag_to", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to drag: {str(e)}")
            return False
    
    async def type_text(self, text: str, interval: Optional[float] = None) -> bool:
        """
        Type text at the current cursor position.
        
        Args:
            text: Text to type
            interval: Time between keystrokes in seconds
            
        Returns:
            True if successful, False otherwise
        """
        params = {"text": text, "interval": interval if interval is not None else 0.01}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("type_text", params, None)
                
            # Default interval is 0.01s between keystrokes
            type_interval = interval if interval is not None else 0.01
            
            logger.debug(f"Typing text: {text[:10]}{'...' if len(text) > 10 else ''}")
            
            # Type the text with PyAutoGUI to maintain interval timing
            pyautogui.write(text, interval=type_interval)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("type_text", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("type_text", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to type text: {str(e)}")
            return False
    
    async def press_key(self, key: Union[str, Key] = None, modifiers: List[str] = None, key_combination: List[str] = None) -> bool:
        """
        Press a keyboard key with optional modifiers.
        
        Args:
            key: Key to press (string or pynput.keyboard.Key)
            modifiers: List of modifier keys ("ctrl", "shift", "alt", "cmd")
            key_combination: Alternative way to specify a key combination as a list of keys
            
        Returns:
            True if successful, False otherwise
        """
        params = {"key": key, "modifiers": modifiers, "key_combination": key_combination}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("press_key", params, None)
                
            # If key_combination is provided, extract the last key as the main key
            # and the rest as modifiers
            if key_combination and not key and not modifiers:
                if len(key_combination) > 1:
                    key = key_combination[-1]
                    modifiers = key_combination[:-1]
                elif len(key_combination) == 1:
                    key = key_combination[0]
                    modifiers = []
            
            # Ensure we have a key to press
            if not key:
                logger.error("No key specified for press_key action")
                
                # Debug trace for error
                if self.debug_mode:
                    await self._debug_trace("press_key", params, "Error: No key specified")
                    
                return False
            
            # Initialize modifiers list if not provided
            if modifiers is None:
                modifiers = []
            
            modifier_keys = []
            for mod in modifiers:
                if mod.lower() == "ctrl":
                    modifier_keys.append(Key.ctrl)
                elif mod.lower() == "shift":
                    modifier_keys.append(Key.shift)
                elif mod.lower() == "alt":
                    modifier_keys.append(Key.alt)
                elif mod.lower() in ("cmd", "command", "win"):
                    modifier_keys.append(Key.cmd)
            
            # Convert string key to pynput Key if it's a special key
            if isinstance(key, str) and key.lower() in dir(Key):
                key = getattr(Key, key.lower())
            
            logger.debug(f"Pressing key: {key} with modifiers: {modifiers or []}")
            
            # Press modifier keys
            for mod_key in modifier_keys:
                self.keyboard_controller.press(mod_key)
            
            # Press main key
            self.keyboard_controller.press(key)
            self.keyboard_controller.release(key)
            
            # Release modifier keys in reverse order
            for mod_key in reversed(modifier_keys):
                self.keyboard_controller.release(mod_key)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("press_key", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("press_key", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to press key: {str(e)}")
            return False
    
    async def perform_hotkey(self, *keys: str) -> bool:
        """
        Perform a hotkey combination.
        
        Args:
            *keys: Sequence of keys in the hotkey
            
        Returns:
            True if successful, False otherwise
        """
        params = {"keys": keys}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("perform_hotkey", params, None)
                
            logger.debug(f"Performing hotkey: {keys}")
            pyautogui.hotkey(*keys)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("perform_hotkey", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("perform_hotkey", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to perform hotkey: {str(e)}")
            return False
    
    async def scroll(self, clicks: int) -> bool:
        """
        Scroll the mouse wheel.
        
        Args:
            clicks: Number of scroll clicks (positive for up, negative for down)
            
        Returns:
            True if successful, False otherwise
        """
        params = {"clicks": clicks}
        
        try:
            # Debug trace at start
            if self.debug_mode:
                await self._debug_trace("scroll", params, None)
                
            logger.debug(f"Scrolling {clicks} clicks")
            pyautogui.scroll(clicks)
            
            # Debug trace at end
            if self.debug_mode:
                await self._debug_trace("scroll", params, True)
                
            return True
        except Exception as e:
            # Debug trace for error
            if self.debug_mode:
                await self._debug_trace("scroll", params, f"Error: {str(e)}")
                
            logger.error(f"Failed to scroll: {str(e)}")
            return False
