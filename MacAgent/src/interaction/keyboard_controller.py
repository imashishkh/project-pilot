"""
Keyboard Controller Module for MacAgent

This module provides advanced keyboard control for an AI agent on macOS,
featuring natural typing patterns, special key support, and text manipulation.
"""

import time
import random
import logging
import re
from typing import Dict, List, Optional, Union, Tuple, Any

# macOS-specific libraries
try:
    import Quartz
    from Quartz import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap
    from Quartz import CGEventSetFlags, kCGEventFlagMaskShift, kCGEventFlagMaskCommand
    from Quartz import kCGEventFlagMaskControl, kCGEventFlagMaskAlternate
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False
    logging.warning("Quartz libraries not available. Some Mac-specific functionality will be limited.")

# Cross-platform fallback
import pyautogui
pyautogui.PAUSE = 0.01  # Lower default pause for typing

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KeyboardController:
    """
    Advanced keyboard controller for Mac with natural typing and special key support.
    
    This class provides methods for typing text with configurable speed and rhythm,
    handling special keys and keyboard shortcuts, and performing text selection and editing.
    """
    
    # Mac-specific key codes
    MAC_KEY_CODES = {
        'a': 0x00, 'b': 0x0B, 'c': 0x08, 'd': 0x02, 'e': 0x0E, 'f': 0x03, 'g': 0x05,
        'h': 0x04, 'i': 0x22, 'j': 0x26, 'k': 0x28, 'l': 0x25, 'm': 0x2E, 'n': 0x2D,
        'o': 0x1F, 'p': 0x23, 'q': 0x0C, 'r': 0x0F, 's': 0x01, 't': 0x11, 'u': 0x20,
        'v': 0x09, 'w': 0x0D, 'x': 0x07, 'y': 0x10, 'z': 0x06,
        '1': 0x12, '2': 0x13, '3': 0x14, '4': 0x15, '5': 0x17, '6': 0x16, '7': 0x1A,
        '8': 0x1C, '9': 0x19, '0': 0x1D,
        '-': 0x1B, '=': 0x18, '[': 0x21, ']': 0x1E, '\\': 0x2A, ';': 0x29, "'": 0x27,
        ',': 0x2B, '.': 0x2F, '/': 0x2C, '`': 0x32, ' ': 0x31,
        'return': 0x24, 'tab': 0x30, 'delete': 0x33, 'escape': 0x35, 'command': 0x37,
        'shift': 0x38, 'capslock': 0x39, 'option': 0x3A, 'control': 0x3B,
        'right_shift': 0x3C, 'right_option': 0x3D, 'right_control': 0x3E,
        'function': 0x3F, 'f1': 0x7A, 'f2': 0x78, 'f3': 0x63, 'f4': 0x76,
        'f5': 0x60, 'f6': 0x61, 'f7': 0x62, 'f8': 0x64, 'f9': 0x65, 'f10': 0x6D,
        'f11': 0x67, 'f12': 0x6F, 'f13': 0x69, 'f14': 0x6B, 'f15': 0x71,
        'f16': 0x6A, 'f17': 0x40, 'f18': 0x4F, 'f19': 0x50, 'f20': 0x5A,
        'left_arrow': 0x7B, 'right_arrow': 0x7C, 'down_arrow': 0x7D, 'up_arrow': 0x7E,
        'home': 0x73, 'end': 0x77, 'page_up': 0x74, 'page_down': 0x79,
        'forward_delete': 0x75
    }
    
    # Mapping for shift-modified keys
    SHIFT_MAP = {
        '!': '1', '@': '2', '#': '3', '$': '4', '%': '5', '^': '6', '&': '7',
        '*': '8', '(': '9', ')': '0', '_': '-', '+': '=', '{': '[', '}': ']',
        '|': '\\', ':': ';', '"': "'", '<': ',', '>': '.', '?': '/', '~': '`',
        'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd', 'E': 'e', 'F': 'f', 'G': 'g',
        'H': 'h', 'I': 'i', 'J': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
        'O': 'o', 'P': 'p', 'Q': 'q', 'R': 'r', 'S': 's', 'T': 't', 'U': 'u',
        'V': 'v', 'W': 'w', 'X': 'x', 'Y': 'y', 'Z': 'z'
    }
    
    # Common Mac keyboard shortcuts
    MAC_SHORTCUTS = {
        'copy': ('c', ['command']),
        'paste': ('v', ['command']),
        'cut': ('x', ['command']),
        'select_all': ('a', ['command']),
        'undo': ('z', ['command']),
        'redo': ('z', ['command', 'shift']),
        'save': ('s', ['command']),
        'new': ('n', ['command']),
        'open': ('o', ['command']),
        'find': ('f', ['command']),
        'print': ('p', ['command']),
        'close': ('w', ['command']),
        'quit': ('q', ['command']),
        'switch_app': ('tab', ['command']),
        'hide_app': ('h', ['command']),
        'minimize': ('m', ['command']),
        'spotlight': ('space', ['command']),
        'screenshots': ('5', ['command', 'shift']),
        'screen_recording': ('5', ['command', 'shift', 'control']),
        'mission_control': ('up_arrow', ['control']),
        'app_windows': ('down_arrow', ['control']),
        'show_desktop': ('d', ['command', 'function']),
        'delete_word': ('delete', ['option']),
        'delete_line': ('delete', ['command']),
        'home': ('left_arrow', ['command']),
        'end': ('right_arrow', ['command']),
        'select_to_start': ('left_arrow', ['command', 'shift']),
        'select_to_end': ('right_arrow', ['command', 'shift']),
        'word_left': ('left_arrow', ['option']),
        'word_right': ('right_arrow', ['option']),
        'select_word_left': ('left_arrow', ['option', 'shift']),
        'select_word_right': ('right_arrow', ['option', 'shift']),
        'emoji_picker': ('e', ['command', 'control', 'space']),
    }
    
    # Typing rhythm profiles
    TYPING_PROFILES = {
        'normal': {
            'base_delay': 0.08,         # Base delay between keystrokes
            'variance': 0.05,           # Random variance in timing
            'mistake_rate': 0.005,      # Probability of typing mistake
            'correction_delay': 0.2,    # Delay before correcting a mistake
            'pause_punctuation': 0.15,  # Extra delay after punctuation
            'pause_newline': 0.3,       # Extra delay after newline
            'burst_probability': 0.2,   # Probability of typing a burst
            'burst_length': (2, 5)      # Range of characters in a burst
        },
        'fast': {
            'base_delay': 0.04,
            'variance': 0.03,
            'mistake_rate': 0.002,
            'correction_delay': 0.1,
            'pause_punctuation': 0.08,
            'pause_newline': 0.15,
            'burst_probability': 0.4,
            'burst_length': (3, 8)
        },
        'slow': {
            'base_delay': 0.15,
            'variance': 0.08,
            'mistake_rate': 0.01,
            'correction_delay': 0.25,
            'pause_punctuation': 0.3,
            'pause_newline': 0.5,
            'burst_probability': 0.1,
            'burst_length': (2, 3)
        },
        'precise': {
            'base_delay': 0.1,
            'variance': 0.02,
            'mistake_rate': 0.0,        # No mistakes
            'correction_delay': 0.0,
            'pause_punctuation': 0.1,
            'pause_newline': 0.2,
            'burst_probability': 0.0,   # No bursts
            'burst_length': (0, 0)
        }
    }
    
    def __init__(self, 
                 use_quartz: bool = HAS_QUARTZ,
                 typing_profile: str = 'normal',
                 default_wpm: int = 60):
        """
        Initialize the keyboard controller.
        
        Args:
            use_quartz: Whether to use macOS-specific Quartz framework
            typing_profile: Default typing profile ('normal', 'fast', 'slow', 'precise')
            default_wpm: Default typing speed in words per minute
        """
        self.use_quartz = use_quartz and HAS_QUARTZ
        self.typing_profile = typing_profile
        self.default_wpm = default_wpm
        
        # Calculate base delay from WPM
        # Assuming 5 chars per word on average, convert WPM to CPM then to CPS
        self.base_delay = 60.0 / (default_wpm * 5)
        
        # Last action state tracking
        self.last_action_success = True
        self.last_action_time = time.time()
        
        logger.info(f"KeyboardController initialized with: Quartz={self.use_quartz}, "
                   f"Profile={typing_profile}, WPM={default_wpm}")
    
    def _key_down_quartz(self, key_code: int, flags: int = 0):
        """
        Press a key down using Quartz (macOS-specific).
        
        Args:
            key_code: Mac-specific key code
            flags: Modifier flags
        """
        event = CGEventCreateKeyboardEvent(None, key_code, True)
        if flags:
            CGEventSetFlags(event, flags)
        CGEventPost(kCGHIDEventTap, event)
    
    def _key_up_quartz(self, key_code: int, flags: int = 0):
        """
        Release a key using Quartz (macOS-specific).
        
        Args:
            key_code: Mac-specific key code
            flags: Modifier flags
        """
        event = CGEventCreateKeyboardEvent(None, key_code, False)
        if flags:
            CGEventSetFlags(event, flags)
        CGEventPost(kCGHIDEventTap, event)
    
    def _get_key_code(self, key: str) -> int:
        """
        Get the Mac-specific key code for a key.
        
        Args:
            key: Key character or name
            
        Returns:
            Key code for the specified key
        """
        if key.lower() in self.MAC_KEY_CODES:
            return self.MAC_KEY_CODES[key.lower()]
        else:
            raise ValueError(f"Unknown key: {key}")
    
    def _get_modifier_flags(self, modifiers: List[str]) -> int:
        """
        Get the combined flags for the specified modifiers.
        
        Args:
            modifiers: List of modifier key names
            
        Returns:
            Combined flags value
        """
        flags = 0
        if 'shift' in modifiers:
            flags |= kCGEventFlagMaskShift
        if 'command' in modifiers:
            flags |= kCGEventFlagMaskCommand
        if 'control' in modifiers:
            flags |= kCGEventFlagMaskControl
        if 'option' in modifiers or 'alt' in modifiers:
            flags |= kCGEventFlagMaskAlternate
        return flags
    
    def press_key(self, key: str, modifiers: List[str] = None) -> bool:
        """
        Press and release a single key, optionally with modifiers.
        
        Args:
            key: Key to press
            modifiers: List of modifier keys to hold
            
        Returns:
            True if successful, False otherwise
        """
        try:
            modifiers = modifiers or []
            
            if self.use_quartz:
                # Get key code
                key_code = self._get_key_code(key)
                
                # Get modifier flags
                flags = self._get_modifier_flags(modifiers)
                
                # Press and release key
                self._key_down_quartz(key_code, flags)
                time.sleep(0.01)  # Brief delay
                self._key_up_quartz(key_code, flags)
            else:
                # Build key combination for pyautogui
                key_combo = []
                for mod in modifiers:
                    key_combo.append(mod)
                key_combo.append(key)
                
                # Press keys
                pyautogui.hotkey(*key_combo, _pause=False)
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in press_key: {e}")
            self.last_action_success = False
            return False
    
    def hold_key(self, key: str, duration: float = 0.5, modifiers: List[str] = None) -> bool:
        """
        Hold down a key for a specified duration.
        
        Args:
            key: Key to hold
            duration: Duration to hold the key in seconds
            modifiers: List of modifier keys to hold
            
        Returns:
            True if successful, False otherwise
        """
        try:
            modifiers = modifiers or []
            
            if self.use_quartz:
                # Get key code
                key_code = self._get_key_code(key)
                
                # Get modifier flags
                flags = self._get_modifier_flags(modifiers)
                
                # Press key
                self._key_down_quartz(key_code, flags)
                
                # Hold for duration
                time.sleep(duration)
                
                # Release key
                self._key_up_quartz(key_code, flags)
            else:
                # Hold with pyautogui
                modifier_keys = []
                for mod in modifiers:
                    modifier_keys.append(mod)
                
                # Press modifiers
                for mod in modifier_keys:
                    pyautogui.keyDown(mod, _pause=False)
                
                # Press and hold main key
                pyautogui.keyDown(key, _pause=False)
                time.sleep(duration)
                
                # Release keys in reverse order
                pyautogui.keyUp(key, _pause=False)
                
                for mod in reversed(modifier_keys):
                    pyautogui.keyUp(mod, _pause=False)
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in hold_key: {e}")
            self.last_action_success = False
            return False
    
    def type_text(self, 
                 text: str, 
                 wpm: int = None, 
                 profile: str = None, 
                 natural_mistakes: bool = True) -> bool:
        """
        Type text with configurable speed and rhythm.
        
        Args:
            text: Text to type
            wpm: Words per minute typing speed
            profile: Typing profile to use
            natural_mistakes: Whether to simulate natural typing mistakes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use defaults if not specified
            profile = profile or self.typing_profile
            profile_params = self.TYPING_PROFILES.get(profile, self.TYPING_PROFILES['normal'])
            
            # Calculate base delay from WPM if specified
            if wpm:
                base_delay = 60.0 / (wpm * 5)  # 5 chars per word average
            else:
                base_delay = profile_params['base_delay']
            
            # Process text
            i = 0
            while i < len(text):
                # Check for burst typing
                burst_length = 0
                if random.random() < profile_params['burst_probability']:
                    burst_length = random.randint(*profile_params['burst_length'])
                    burst_length = min(burst_length, len(text) - i)
                
                # Type character by character (with bursts)
                if burst_length > 0:
                    burst_text = text[i:i+burst_length]
                    
                    # Type burst of characters faster
                    if self.use_quartz:
                        for char in burst_text:
                            self._type_char_quartz(char)
                    else:
                        pyautogui.write(burst_text, interval=base_delay * 0.4, _pause=False)
                    
                    i += burst_length
                    
                    # Longer pause after burst
                    time.sleep(base_delay * 1.2)
                else:
                    # Type single character
                    char = text[i]
                    
                    # Simulate mistake and correction
                    if natural_mistakes and random.random() < profile_params['mistake_rate']:
                        # Determine a "wrong" key near the intended one
                        wrong_char = self._get_adjacent_key(char)
                        
                        # Type the wrong character
                        if self.use_quartz:
                            self._type_char_quartz(wrong_char)
                        else:
                            pyautogui.press(wrong_char, _pause=False)
                        
                        # Wait before correction
                        time.sleep(profile_params['correction_delay'])
                        
                        # Delete the wrong character
                        if self.use_quartz:
                            self._key_down_quartz(self.MAC_KEY_CODES['delete'])
                            time.sleep(0.01)
                            self._key_up_quartz(self.MAC_KEY_CODES['delete'])
                        else:
                            pyautogui.press('backspace', _pause=False)
                        
                        # Wait before typing correct character
                        time.sleep(profile_params['correction_delay'] * 0.5)
                    
                    # Type the correct character
                    if self.use_quartz:
                        self._type_char_quartz(char)
                    else:
                        pyautogui.press(char, _pause=False)
                    
                    # Move to next character
                    i += 1
                
                # Calculate delay for next character
                delay = base_delay
                
                # Add variance
                delay += random.uniform(-profile_params['variance'], profile_params['variance'])
                
                # Add extra delay for punctuation and newlines
                if i < len(text):
                    next_char = text[i]
                    if next_char in '.,:;!?':
                        delay += profile_params['pause_punctuation']
                    elif next_char == '\n':
                        delay += profile_params['pause_newline']
                
                # Wait before next character
                time.sleep(max(0.01, delay))
            
            self.last_action_success = True
            self.last_action_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error in type_text: {e}")
            self.last_action_success = False
            return False
    
    def _type_char_quartz(self, char: str):
        """
        Type a single character using Quartz.
        
        Args:
            char: Character to type
        """
        # Handle shifted characters
        if char in self.SHIFT_MAP:
            base_char = self.SHIFT_MAP[char]
            key_code = self._get_key_code(base_char)
            
            # Press shift down
            self._key_down_quartz(self.MAC_KEY_CODES['shift'])
            
            # Type the character
            self._key_down_quartz(key_code)
            time.sleep(0.01)
            self._key_up_quartz(key_code)
            
            # Release shift
            self._key_up_quartz(self.MAC_KEY_CODES['shift'])
        else:
            # Handle special keys
            if char == '\n':
                key_code = self._get_key_code('return')
            elif char == '\t':
                key_code = self._get_key_code('tab')
            else:
                # Regular character
                try:
                    key_code = self._get_key_code(char)
                except ValueError:
                    # Skip characters we don't have codes for
                    logger.warning(f"No key code for character: {char}")
                    return
            
            # Type the character
            self._key_down_quartz(key_code)
            time.sleep(0.01)
            self._key_up_quartz(key_code)
    
    def _get_adjacent_key(self, char: str) -> str:
        """
        Get a random adjacent key on a QWERTY keyboard.
        
        Args:
            char: Character to find adjacent key for
            
        Returns:
            Adjacent character
        """
        keyboard_layout = {
            'q': ['w', '1', 'a'],
            'w': ['q', 'e', '2', 's', 'a'],
            'e': ['w', 'r', '3', 'd', 's'],
            'r': ['e', 't', '4', 'f', 'd'],
            't': ['r', 'y', '5', 'g', 'f'],
            'y': ['t', 'u', '6', 'h', 'g'],
            'u': ['y', 'i', '7', 'j', 'h'],
            'i': ['u', 'o', '8', 'k', 'j'],
            'o': ['i', 'p', '9', 'l', 'k'],
            'p': ['o', '[', '0', 'l'],
            'a': ['q', 'w', 's', 'z'],
            's': ['a', 'w', 'e', 'd', 'x', 'z'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'],
            'f': ['d', 'r', 't', 'g', 'v', 'c'],
            'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'],
            'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p', ';'],
            'z': ['a', 's', 'x'],
            'x': ['z', 's', 'd', 'c'],
            'c': ['x', 'd', 'f', 'v'],
            'v': ['c', 'f', 'g', 'b'],
            'b': ['v', 'g', 'h', 'n'],
            'n': ['b', 'h', 'j', 'm'],
            'm': ['n', 'j', 'k', ','],
            ',': ['m', 'k', 'l', '.'],
            '.': [',', 'l', ';', '/'],
            '/': ['.', ';', "'"],
            '1': ['q', '2'],
            '2': ['1', 'q', 'w', '3'],
            '3': ['2', 'w', 'e', '4'],
            '4': ['3', 'e', 'r', '5'],
            '5': ['4', 'r', 't', '6'],
            '6': ['5', 't', 'y', '7'],
            '7': ['6', 'y', 'u', '8'],
            '8': ['7', 'u', 'i', '9'],
            '9': ['8', 'i', 'o', '0'],
            '0': ['9', 'o', 'p', '-'],
            '-': ['0', 'p', '[', '='],
            '=': ['-', '[', ']'],
            ' ': ['c', 'v', 'b', 'n', 'm']  # Space bar
        }
        
        # Default to a random character if not in map
        if char.lower() not in keyboard_layout:
            return random.choice('abcdefghijklmnopqrstuvwxyz')
        
        # Get adjacent keys and pick one randomly
        adjacent_keys = keyboard_layout[char.lower()]
        wrong_char = random.choice(adjacent_keys)
        
        # Preserve case
        if char.isupper():
            wrong_char = wrong_char.upper()
            
        return wrong_char
    
    def keyboard_shortcut(self, shortcut_name: str) -> bool:
        """
        Execute a common keyboard shortcut by name.
        
        Args:
            shortcut_name: Name of the shortcut (e.g., 'copy', 'paste')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if shortcut_name not in self.MAC_SHORTCUTS:
                logger.error(f"Unknown shortcut: {shortcut_name}")
                return False
            
            key, modifiers = self.MAC_SHORTCUTS[shortcut_name]
            return self.press_key(key, modifiers)
            
        except Exception as e:
            logger.error(f"Error in keyboard_shortcut: {e}")
            self.last_action_success = False
            return False
    
    def custom_shortcut(self, key: str, modifiers: List[str]) -> bool:
        """
        Execute a custom keyboard shortcut.
        
        Args:
            key: Main key to press
            modifiers: List of modifier keys to hold
            
        Returns:
            True if successful, False otherwise
        """
        return self.press_key(key, modifiers)
    
    def select_all(self) -> bool:
        """
        Select all text (Command+A).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('select_all')
    
    def copy(self) -> bool:
        """
        Copy selected text (Command+C).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('copy')
    
    def paste(self) -> bool:
        """
        Paste clipboard contents (Command+V).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('paste')
    
    def cut(self) -> bool:
        """
        Cut selected text (Command+X).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('cut')
    
    def undo(self) -> bool:
        """
        Undo last action (Command+Z).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('undo')
    
    def redo(self) -> bool:
        """
        Redo last undone action (Command+Shift+Z).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('redo')
    
    def delete_line(self) -> bool:
        """
        Delete current line (Command+Delete).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('delete_line')
    
    def move_to_start(self) -> bool:
        """
        Move cursor to start of line (Command+Left Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('home')
    
    def move_to_end(self) -> bool:
        """
        Move cursor to end of line (Command+Right Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('end')
    
    def select_to_start(self) -> bool:
        """
        Select from cursor to start of line (Command+Shift+Left Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('select_to_start')
    
    def select_to_end(self) -> bool:
        """
        Select from cursor to end of line (Command+Shift+Right Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('select_to_end')
    
    def move_word_left(self) -> bool:
        """
        Move cursor one word left (Option+Left Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('word_left')
    
    def move_word_right(self) -> bool:
        """
        Move cursor one word right (Option+Right Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('word_right')
    
    def select_word_left(self) -> bool:
        """
        Select one word to the left (Option+Shift+Left Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('select_word_left')
    
    def select_word_right(self) -> bool:
        """
        Select one word to the right (Option+Shift+Right Arrow).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('select_word_right')
    
    def delete_word(self) -> bool:
        """
        Delete word to the left of cursor (Option+Delete).
        
        Returns:
            True if successful, False otherwise
        """
        return self.keyboard_shortcut('delete_word')
    
    def arrow_key(self, direction: str, count: int = 1, modifiers: List[str] = None) -> bool:
        """
        Press arrow key multiple times.
        
        Args:
            direction: Direction ('up', 'down', 'left', 'right')
            count: Number of times to press
            modifiers: Optional modifier keys to hold
            
        Returns:
            True if successful, False otherwise
        """
        try:
            modifiers = modifiers or []
            
            # Map direction to key name
            key_map = {
                'up': 'up_arrow',
                'down': 'down_arrow',
                'left': 'left_arrow',
                'right': 'right_arrow'
            }
            
            if direction.lower() not in key_map:
                logger.error(f"Invalid arrow direction: {direction}")
                return False
            
            key = key_map[direction.lower()]
            
            for _ in range(count):
                success = self.press_key(key, modifiers)
                if not success:
                    return False
                time.sleep(0.05)  # Small delay between presses
            
            return True
            
        except Exception as e:
            logger.error(f"Error in arrow_key: {e}")
            self.last_action_success = False
            return False
    
    def type_with_backspaces(self, text: str, error_rate: float = 0.1) -> bool:
        """
        Type text with occasional backspaces to simulate corrections.
        
        Args:
            text: Text to type
            error_rate: Rate of simulated errors (0-1)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            i = 0
            while i < len(text):
                # Type current character
                char = text[i]
                self.press_key(char)
                i += 1
                
                # Random pause
                time.sleep(random.uniform(0.05, 0.2))
                
                # Randomly make a "mistake"
                if i < len(text) - 1 and random.random() < error_rate:
                    # Type an extra character
                    extra_char = self._get_adjacent_key(text[i])
                    self.press_key(extra_char)
                    
                    # Pause before correction
                    time.sleep(random.uniform(0.2, 0.5))
                    
                    # Delete the mistake
                    self.press_key('delete')
                    
                    # Pause again
                    time.sleep(random.uniform(0.1, 0.3))
            
            return True
            
        except Exception as e:
            logger.error(f"Error in type_with_backspaces: {e}")
            self.last_action_success = False
            return False
    
    def clear_field(self) -> bool:
        """
        Clear current text field (select all + delete).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Select all text
            self.select_all()
            time.sleep(0.1)  # Small delay
            
            # Delete selection
            self.press_key('delete')
            
            return True
            
        except Exception as e:
            logger.error(f"Error in clear_field: {e}")
            self.last_action_success = False
            return False
    
    def tab_key(self, count: int = 1) -> bool:
        """
        Press the Tab key multiple times.
        
        Args:
            count: Number of times to press Tab
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for _ in range(count):
                self.press_key('tab')
                time.sleep(0.1)  # Small delay between presses
            
            return True
            
        except Exception as e:
            logger.error(f"Error in tab_key: {e}")
            self.last_action_success = False
            return False
    
    def press_return(self, count: int = 1) -> bool:
        """
        Press the Return/Enter key multiple times.
        
        Args:
            count: Number of times to press Return
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for _ in range(count):
                self.press_key('return')
                time.sleep(0.1)  # Small delay between presses
            
            return True
            
        except Exception as e:
            logger.error(f"Error in press_return: {e}")
            self.last_action_success = False
            return False 