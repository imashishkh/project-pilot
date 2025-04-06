"""
Interaction Coordinator Module for MacAgent

This module provides high-level coordination of mouse and keyboard interactions,
managing complex sequences of interaction events with timing and error recovery.
"""

import time
import logging
import random
from typing import Dict, List, Tuple, Callable, Any, Optional, Union

from .mouse_controller import MouseController
from .keyboard_controller import KeyboardController

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractionCoordinator:
    """
    Coordinator for mouse and keyboard interactions.
    
    This class provides high-level methods for complex interaction patterns
    like form filling, drag-and-drop operations, and multi-step sequences.
    It ensures natural timing between actions and handles error recovery.
    """
    
    # Interaction timing patterns (in seconds)
    TIMING_PATTERNS = {
        'fast': {
            'click_to_type': (0.2, 0.5),      # Delay between click and typing
            'type_to_click': (0.3, 0.6),      # Delay between typing and next click
            'click_to_click': (0.2, 0.4),     # Delay between successive clicks
            'sequence_delay': (0.5, 1.0),     # Delay between interaction sequences
            'scroll_pause': (0.3, 0.8),       # Pause after scrolling
            'drag_preparation': (0.2, 0.4),   # Preparation time before drag
            'post_action_delay': (0.2, 0.5)   # Delay after completing an action
        },
        'normal': {
            'click_to_type': (0.5, 1.0),
            'type_to_click': (0.7, 1.2),
            'click_to_click': (0.4, 0.8),
            'sequence_delay': (1.0, 2.0),
            'scroll_pause': (0.7, 1.5),
            'drag_preparation': (0.3, 0.6),
            'post_action_delay': (0.5, 1.0)
        },
        'slow': {
            'click_to_type': (1.0, 2.0),
            'type_to_click': (1.2, 2.5),
            'click_to_click': (0.8, 1.5),
            'sequence_delay': (2.0, 3.5),
            'scroll_pause': (1.0, 2.0),
            'drag_preparation': (0.5, 1.0),
            'post_action_delay': (1.0, 2.0)
        }
    }
    
    # Common interaction patterns
    INTERACTION_PATTERNS = {
        'form_filling': {
            'description': 'Click field, type content, tab or click to next field',
            'timing_profile': 'normal',
            'recovery_attempts': 2
        },
        'browsing': {
            'description': 'Navigate links, scroll, read content',
            'timing_profile': 'normal',
            'recovery_attempts': 2
        },
        'content_creation': {
            'description': 'Type content with occasional formatting',
            'timing_profile': 'slow',
            'recovery_attempts': 3
        },
        'gaming': {
            'description': 'Quick precise movements and key presses',
            'timing_profile': 'fast',
            'recovery_attempts': 1
        },
        'file_management': {
            'description': 'Select, drag-drop, rename files',
            'timing_profile': 'normal',
            'recovery_attempts': 3
        }
    }
    
    def __init__(self, 
                 mouse_controller: MouseController = None,
                 keyboard_controller: KeyboardController = None,
                 timing_profile: str = 'normal',
                 max_recovery_attempts: int = 3,
                 enable_human_timing: bool = True):
        """
        Initialize the interaction coordinator.
        
        Args:
            mouse_controller: MouseController instance or None to create a new one
            keyboard_controller: KeyboardController instance or None to create a new one
            timing_profile: Default timing profile ('fast', 'normal', 'slow')
            max_recovery_attempts: Maximum number of attempts for action recovery
            enable_human_timing: Whether to use human-like timing between actions
        """
        # Initialize controllers if not provided
        self.mouse = mouse_controller if mouse_controller else MouseController()
        self.keyboard = keyboard_controller if keyboard_controller else KeyboardController()
        
        self.timing_profile = timing_profile
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_human_timing = enable_human_timing
        
        # State tracking
        self.last_action_type = None  # 'mouse', 'keyboard', or None
        self.last_action_time = time.time()
        self.action_history = []
        self.recovery_mode = False
        
        logger.info(f"InteractionCoordinator initialized with timing profile: {timing_profile}")
    
    def _add_human_delay(self, action_transition: str) -> None:
        """
        Add a human-like delay between actions based on the transition type.
        
        Args:
            action_transition: Type of transition (e.g., 'click_to_type')
        """
        if not self.enable_human_timing:
            return
        
        # Get delay range for this transition
        timing_profile = self.TIMING_PATTERNS.get(
            self.timing_profile, self.TIMING_PATTERNS['normal'])
        
        delay_range = timing_profile.get(
            action_transition, timing_profile['post_action_delay'])
        
        # Add random delay within range
        delay = random.uniform(delay_range[0], delay_range[1])
        time.sleep(delay)
    
    def _update_action_history(self, action_type: str, action_name: str, 
                              target: Any, success: bool) -> None:
        """
        Update the action history with details of the most recent action.
        
        Args:
            action_type: Type of action ('mouse', 'keyboard')
            action_name: Name of the action (e.g., 'click', 'type')
            target: Target of the action (coordinates, text, etc.)
            success: Whether the action was successful
        """
        timestamp = time.time()
        
        # Add to history
        self.action_history.append({
            'timestamp': timestamp,
            'action_type': action_type,
            'action_name': action_name,
            'target': target,
            'success': success,
            'recovery_mode': self.recovery_mode
        })
        
        # Limit history length to avoid memory issues
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        
        # Update last action tracking
        self.last_action_type = action_type
        self.last_action_time = timestamp
    
    def _handle_transition_timing(self, from_action: str, to_action: str) -> None:
        """
        Handle timing between different types of actions.
        
        Args:
            from_action: The action just completed ('mouse', 'keyboard', None)
            to_action: The action about to start ('mouse', 'keyboard')
        """
        # Skip if no previous action or timing disabled
        if from_action is None or not self.enable_human_timing:
            return
        
        # Determine transition type
        if from_action == 'mouse' and to_action == 'keyboard':
            # Clicked something, now going to type
            self._add_human_delay('click_to_type')
        elif from_action == 'keyboard' and to_action == 'mouse':
            # Typed something, now going to click
            self._add_human_delay('type_to_click')
        elif from_action == 'mouse' and to_action == 'mouse':
            # Multiple mouse actions in sequence
            self._add_human_delay('click_to_click')
        elif from_action == 'keyboard' and to_action == 'keyboard':
            # Multiple keyboard actions in sequence
            # No additional delay needed as type_text handles its own timing
            pass
    
    def click_and_type(self, 
                      position: Tuple[int, int], 
                      text: str, 
                      click_type: str = 'single',
                      wpm: int = None,
                      move_profile: str = None,
                      typing_profile: str = None) -> bool:
        """
        Click at a position and then type text with natural timing.
        
        Args:
            position: (x, y) coordinates to click
            text: Text to type after clicking
            click_type: Type of click ('single', 'double', 'right')
            wpm: Words per minute for typing
            move_profile: Mouse movement profile
            typing_profile: Typing rhythm profile
            
        Returns:
            True if all actions successful, False otherwise
        """
        try:
            # Handle transition timing from previous action
            self._handle_transition_timing(self.last_action_type, 'mouse')
            
            # Move and click
            if click_type == 'single':
                success = self.mouse.click(position[0], position[1], 
                                          move_profile=move_profile)
            elif click_type == 'double':
                success = self.mouse.double_click(position[0], position[1], 
                                                 move_profile=move_profile)
            elif click_type == 'right':
                success = self.mouse.right_click(position[0], position[1], 
                                               move_profile=move_profile)
            else:
                logger.error(f"Unknown click type: {click_type}")
                return False
            
            # Update action history
            self._update_action_history('mouse', f'{click_type}_click', position, success)
            
            # If click failed, return False
            if not success:
                return False
            
            # Add appropriate delay between click and typing
            self._add_human_delay('click_to_type')
            
            # Type text
            typing_success = self.keyboard.type_text(text, wpm=wpm, profile=typing_profile)
            
            # Update action history
            self._update_action_history('keyboard', 'type', text, typing_success)
            
            return typing_success
            
        except Exception as e:
            logger.error(f"Error in click_and_type: {e}")
            return False
    
    def fill_form(self, fields: List[Dict[str, Any]], 
                 use_tab_navigation: bool = True,
                 timing_profile: str = None) -> bool:
        """
        Fill a form with multiple fields.
        
        Args:
            fields: List of dictionaries with field information
                Each dict should have 'position' and 'text' keys
                Optional keys: 'click_type', 'clear_field'
            use_tab_navigation: Whether to use Tab key to navigate between fields
            timing_profile: Timing profile to use
            
        Returns:
            True if all fields were filled successfully, False otherwise
        """
        try:
            # Set timing profile for this interaction
            old_profile = self.timing_profile
            if timing_profile:
                self.timing_profile = timing_profile
            
            # Track success
            all_successful = True
            
            # Process each field
            for i, field in enumerate(fields):
                # Extract field information
                position = field['position']
                text = field['text']
                click_type = field.get('click_type', 'single')
                clear = field.get('clear_field', False)
                
                # Click on the field
                click_success = False
                if click_type == 'single':
                    click_success = self.mouse.click(position[0], position[1])
                elif click_type == 'double':
                    click_success = self.mouse.double_click(position[0], position[1])
                else:
                    click_success = self.mouse.click(position[0], position[1])
                
                # Update action history
                self._update_action_history('mouse', f'{click_type}_click', position, click_success)
                
                # If click failed, continue to next field but mark as unsuccessful
                if not click_success:
                    all_successful = False
                    continue
                
                # Add delay between click and typing
                self._add_human_delay('click_to_type')
                
                # Clear field if requested
                if clear:
                    self.keyboard.clear_field()
                    # Small delay after clearing
                    time.sleep(0.3)
                
                # Type the text
                type_success = self.keyboard.type_text(text)
                
                # Update action history
                self._update_action_history('keyboard', 'type', text, type_success)
                
                if not type_success:
                    all_successful = False
                
                # If not the last field, navigate to the next one
                if i < len(fields) - 1:
                    if use_tab_navigation:
                        # Add delay before tabbing
                        time.sleep(0.3)
                        
                        # Press Tab to move to next field
                        tab_success = self.keyboard.press_key('tab')
                        
                        # Update action history
                        self._update_action_history('keyboard', 'tab', None, tab_success)
                        
                        if not tab_success:
                            all_successful = False
                    else:
                        # Add delay between fields
                        self._add_human_delay('type_to_click')
            
            # Restore original timing profile
            self.timing_profile = old_profile
            
            return all_successful
            
        except Exception as e:
            logger.error(f"Error in fill_form: {e}")
            self.timing_profile = old_profile  # Ensure profile is restored
            return False
    
    def drag_and_drop(self, 
                     start_pos: Tuple[int, int], 
                     end_pos: Tuple[int, int],
                     move_profile: str = None,
                     hold_duration: float = 0.1) -> bool:
        """
        Perform a drag and drop operation.
        
        Args:
            start_pos: Starting position (x, y)
            end_pos: Ending position (x, y)
            move_profile: Mouse movement profile
            hold_duration: How long to hold before dragging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle transition timing from previous action
            self._handle_transition_timing(self.last_action_type, 'mouse')
            
            # Add preparation delay
            self._add_human_delay('drag_preparation')
            
            # Perform drag and drop
            success = self.mouse.drag(
                start_pos[0], start_pos[1], 
                end_pos[0], end_pos[1],
                move_profile=move_profile,
                hold_duration=hold_duration
            )
            
            # Update action history
            self._update_action_history('mouse', 'drag_drop', 
                                      {'start': start_pos, 'end': end_pos}, 
                                      success)
            
            return success
            
        except Exception as e:
            logger.error(f"Error in drag_and_drop: {e}")
            return False
    
    def scroll_and_click(self, 
                        scroll_amount: int,
                        scroll_direction: str = 'down',
                        click_position: Tuple[int, int] = None,
                        click_type: str = 'single',
                        move_profile: str = None) -> bool:
        """
        Scroll the page and then click at a position.
        
        Args:
            scroll_amount: Amount to scroll
            scroll_direction: Direction to scroll ('up', 'down')
            click_position: Position to click after scrolling, or None to skip click
            click_type: Type of click ('single', 'double', 'right')
            move_profile: Mouse movement profile
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Handle transition timing from previous action
            self._handle_transition_timing(self.last_action_type, 'mouse')
            
            # Scroll
            scroll_success = self.mouse.scroll(
                amount=scroll_amount, 
                direction=scroll_direction
            )
            
            # Update action history
            self._update_action_history('mouse', 'scroll', 
                                      {'amount': scroll_amount, 'direction': scroll_direction}, 
                                      scroll_success)
            
            # If scroll failed or no click position, return scroll result
            if not scroll_success or click_position is None:
                return scroll_success
            
            # Add pause after scrolling
            self._add_human_delay('scroll_pause')
            
            # Click
            if click_type == 'single':
                click_success = self.mouse.click(
                    click_position[0], click_position[1], 
                    move_profile=move_profile
                )
            elif click_type == 'double':
                click_success = self.mouse.double_click(
                    click_position[0], click_position[1], 
                    move_profile=move_profile
                )
            elif click_type == 'right':
                click_success = self.mouse.right_click(
                    click_position[0], click_position[1],
                    move_profile=move_profile
                )
            else:
                logger.error(f"Unknown click type: {click_type}")
                return False
            
            # Update action history
            self._update_action_history('mouse', f'{click_type}_click', 
                                      click_position, click_success)
            
            return click_success
            
        except Exception as e:
            logger.error(f"Error in scroll_and_click: {e}")
            return False
    
    def execute_sequence(self, 
                        actions: List[Dict[str, Any]],
                        stop_on_failure: bool = False,
                        pattern: str = None) -> Dict[str, Any]:
        """
        Execute a sequence of interaction actions.
        
        Args:
            actions: List of action dictionaries with keys:
                - 'type': 'mouse' or 'keyboard'
                - 'action': Specific action (e.g., 'click', 'type', 'drag')
                - 'params': Dictionary of parameters for the action
            stop_on_failure: Whether to stop if an action fails
            pattern: Name of predefined interaction pattern
            
        Returns:
            Dictionary with results including success rate and failed actions
        """
        try:
            # Apply pattern settings if specified
            old_profile = self.timing_profile
            old_recovery = self.max_recovery_attempts
            
            if pattern and pattern in self.INTERACTION_PATTERNS:
                pattern_settings = self.INTERACTION_PATTERNS[pattern]
                self.timing_profile = pattern_settings['timing_profile']
                self.max_recovery_attempts = pattern_settings['recovery_attempts']
            
            # Results tracking
            results = {
                'total_actions': len(actions),
                'successful_actions': 0,
                'failed_actions': [],
                'complete': False
            }
            
            # Execute each action
            for i, action_info in enumerate(actions):
                action_type = action_info['type']
                action_name = action_info['action']
                params = action_info.get('params', {})
                
                # Handle transition timing between actions
                self._handle_transition_timing(self.last_action_type, action_type)
                
                # Execute the appropriate action
                success = False
                
                # Mouse actions
                if action_type == 'mouse':
                    if action_name == 'click':
                        x, y = params['position']
                        click_type = params.get('click_type', 'single')
                        move_profile = params.get('move_profile', None)
                        
                        if click_type == 'single':
                            success = self.mouse.click(x, y, move_profile=move_profile)
                        elif click_type == 'double':
                            success = self.mouse.double_click(x, y, move_profile=move_profile)
                        elif click_type == 'right':
                            success = self.mouse.right_click(x, y, move_profile=move_profile)
                        
                    elif action_name == 'move':
                        x, y = params['position']
                        move_profile = params.get('move_profile', None)
                        success = self.mouse.move_to(x, y, profile=move_profile)
                        
                    elif action_name == 'drag':
                        start_x, start_y = params['start_position']
                        end_x, end_y = params['end_position']
                        move_profile = params.get('move_profile', None)
                        hold_duration = params.get('hold_duration', 0.1)
                        success = self.mouse.drag(
                            start_x, start_y, end_x, end_y,
                            move_profile=move_profile,
                            hold_duration=hold_duration
                        )
                        
                    elif action_name == 'scroll':
                        amount = params['amount']
                        direction = params.get('direction', 'down')
                        success = self.mouse.scroll(amount=amount, direction=direction)
                
                # Keyboard actions
                elif action_type == 'keyboard':
                    if action_name == 'type':
                        text = params['text']
                        wpm = params.get('wpm', None)
                        profile = params.get('profile', None)
                        success = self.keyboard.type_text(text, wpm=wpm, profile=profile)
                        
                    elif action_name == 'key':
                        key = params['key']
                        modifiers = params.get('modifiers', [])
                        success = self.keyboard.press_key(key, modifiers)
                        
                    elif action_name == 'shortcut':
                        shortcut_name = params['name']
                        success = self.keyboard.keyboard_shortcut(shortcut_name)
                        
                    elif action_name == 'hold':
                        key = params['key']
                        duration = params.get('duration', 0.5)
                        modifiers = params.get('modifiers', [])
                        success = self.keyboard.hold_key(key, duration, modifiers)
                
                # Combined actions
                elif action_type == 'combined':
                    if action_name == 'click_and_type':
                        position = params['position']
                        text = params['text']
                        click_type = params.get('click_type', 'single')
                        wpm = params.get('wpm', None)
                        move_profile = params.get('move_profile', None)
                        typing_profile = params.get('typing_profile', None)
                        success = self.click_and_type(
                            position, text, click_type, wpm, move_profile, typing_profile
                        )
                
                # Update action history
                self._update_action_history(action_type, action_name, params, success)
                
                # Update results
                if success:
                    results['successful_actions'] += 1
                else:
                    results['failed_actions'].append({
                        'index': i,
                        'action_type': action_type,
                        'action_name': action_name,
                        'params': params
                    })
                    
                    if stop_on_failure:
                        break
                
                # Add delay between sequential actions if not the last one
                if i < len(actions) - 1:
                    self._add_human_delay('sequence_delay')
            
            # Restore original settings
            self.timing_profile = old_profile
            self.max_recovery_attempts = old_recovery
            
            # Mark sequence as complete if all actions executed
            results['complete'] = len(results['failed_actions']) == 0 or not stop_on_failure
            
            return results
            
        except Exception as e:
            logger.error(f"Error in execute_sequence: {e}")
            
            # Restore original settings in case of exception
            self.timing_profile = old_profile
            self.max_recovery_attempts = old_recovery
            
            return {
                'total_actions': len(actions),
                'successful_actions': results['successful_actions'] if 'results' in locals() else 0,
                'failed_actions': results['failed_actions'] if 'results' in locals() else [],
                'complete': False,
                'error': str(e)
            }
    
    def perform_recovery(self, failed_action: Dict[str, Any], 
                        retry_count: int = 0,
                        alternative_actions: List[Dict[str, Any]] = None) -> bool:
        """
        Attempt to recover from a failed action.
        
        Args:
            failed_action: Dictionary describing the failed action
            retry_count: Number of previous retry attempts
            alternative_actions: List of alternative actions to try
            
        Returns:
            True if recovery was successful, False otherwise
        """
        try:
            # Set recovery mode flag
            self.recovery_mode = True
            
            # If exceeded max retry attempts, return failure
            if retry_count >= self.max_recovery_attempts:
                logger.warning(f"Max recovery attempts ({self.max_recovery_attempts}) exceeded")
                self.recovery_mode = False
                return False
            
            # Log recovery attempt
            logger.info(f"Attempting recovery for {failed_action['action_type']}:"
                      f"{failed_action['action_name']} (attempt {retry_count + 1})")
            
            # Retry original action first
            result = False
            
            # Execute the appropriate action based on type
            action_type = failed_action['action_type']
            action_name = failed_action['action_name']
            params = failed_action.get('params', {})
            
            # Simple retry of the exact same action
            if alternative_actions is None:
                if action_type == 'combined' and action_name == 'click_and_type':
                    # For click_and_type, retry with more precise movement
                    position = params['position']
                    text = params['text']
                    click_type = params.get('click_type', 'single')
                    
                    # Use more precise movement and typing
                    result = self.click_and_type(
                        position, text, click_type,
                        move_profile='precise',
                        typing_profile='precise'
                    )
                else:
                    # Package the action into a sequence and execute
                    action_sequence = [{
                        'type': action_type,
                        'action': action_name,
                        'params': params
                    }]
                    
                    sequence_result = self.execute_sequence(action_sequence)
                    result = sequence_result['complete'] and sequence_result['successful_actions'] == 1
            
            # Try alternative actions if provided and first retry failed
            if not result and alternative_actions:
                for alt_action in alternative_actions:
                    logger.info(f"Trying alternative action: {alt_action['action_name']}")
                    
                    # Execute the alternative action
                    alt_sequence = [alt_action]
                    alt_result = self.execute_sequence(alt_sequence)
                    
                    if alt_result['complete'] and alt_result['successful_actions'] == 1:
                        result = True
                        break
            
            # If still failed and retry attempts remain, recurse with higher retry count
            if not result and retry_count + 1 < self.max_recovery_attempts:
                # Wait a bit longer before next attempt
                time.sleep(1.0)
                result = self.perform_recovery(failed_action, retry_count + 1, alternative_actions)
            
            # Clear recovery mode flag
            self.recovery_mode = False
            
            return result
            
        except Exception as e:
            logger.error(f"Error in perform_recovery: {e}")
            self.recovery_mode = False
            return False
    
    def wait_for_condition(self, 
                          condition_checker: Callable[[], bool],
                          timeout: float = 10.0,
                          check_interval: float = 0.5,
                          action_on_timeout: Dict[str, Any] = None) -> bool:
        """
        Wait for a condition to be true with timeout.
        
        Args:
            condition_checker: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            check_interval: Time between condition checks
            action_on_timeout: Action to execute if timeout occurs
            
        Returns:
            True if condition was met, False if timed out
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Check condition
                if condition_checker():
                    return True
                
                # Wait before checking again
                time.sleep(check_interval)
            
            # If timeout and action specified, execute it
            if action_on_timeout:
                action_type = action_on_timeout['type']
                action_name = action_on_timeout['action']
                params = action_on_timeout.get('params', {})
                
                if action_type == 'mouse':
                    if action_name == 'click':
                        x, y = params['position']
                        click_type = params.get('click_type', 'single')
                        
                        if click_type == 'single':
                            self.mouse.click(x, y)
                        elif click_type == 'double':
                            self.mouse.double_click(x, y)
                        elif click_type == 'right':
                            self.mouse.right_click(x, y)
                
                elif action_type == 'keyboard':
                    if action_name == 'key':
                        key = params['key']
                        modifiers = params.get('modifiers', [])
                        self.keyboard.press_key(key, modifiers)
            
            return False
            
        except Exception as e:
            logger.error(f"Error in wait_for_condition: {e}")
            return False 