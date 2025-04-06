"""
Interaction Module for MacAgent

This module provides a comprehensive set of tools for simulating human-like 
interaction with a Mac system, including mouse and keyboard control with
natural movement patterns, timing, and error recovery capabilities.
It also includes AppleScript integration for high-level application control.
"""

from .mouse_controller import MouseController
from .keyboard_controller import KeyboardController
from .interaction_coordinator import InteractionCoordinator
from .applescript_bridge import AppleScriptBridge
from .script_library import ScriptLibrary
from .application_controller import ApplicationController

__all__ = [
    'MouseController',
    'KeyboardController',
    'InteractionCoordinator',
    'AppleScriptBridge',
    'ScriptLibrary',
    'ApplicationController'
]
