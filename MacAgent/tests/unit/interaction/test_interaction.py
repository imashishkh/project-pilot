#!/usr/bin/env python3
"""
Test module for the interaction system.

This module tests the ability of the system to interact with macOS.
"""

import os
import sys
import time
import logging
import pytest
from pathlib import Path

# Import directly from MacAgent package or alternative paths
try:
    from MacAgent.src.interaction.mouse_controller import MouseController
    from MacAgent.src.interaction.keyboard_controller import KeyboardController
    from MacAgent.src.interaction.interaction_coordinator import InteractionCoordinator
except ModuleNotFoundError:
    # Try alternative import paths
    import sys
    from pathlib import Path
    # Add the project root to the path
    project_root = Path(__file__).parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from MacAgent.src.interaction.mouse_controller import MouseController
        from MacAgent.src.interaction.keyboard_controller import KeyboardController
        from MacAgent.src.interaction.interaction_coordinator import InteractionCoordinator
    except ModuleNotFoundError:
        from src.interaction.mouse_controller import MouseController
        from src.interaction.keyboard_controller import KeyboardController
        from src.interaction.interaction_coordinator import InteractionCoordinator

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_mouse_movement():
    """Test basic mouse movement with different profiles."""
    logger.info("Testing mouse movement...")
    
    mouse = MouseController()
    
    # Get screen size
    screen_width, screen_height = mouse.get_screen_size()
    logger.info(f"Screen size: {screen_width}x{screen_height}")
    
    # Test different movement profiles
    profiles = ["normal", "fast", "precise", "natural"]
    
    for profile in profiles:
        logger.info(f"Testing '{profile}' movement profile")
        
        # Define random target point
        target_x = random.randint(100, screen_width - 100)
        target_y = random.randint(100, screen_height - 100)
        
        # Move to target
        success = mouse.move_to(target_x, target_y, profile=profile)
        logger.info(f"Move to ({target_x}, {target_y}) with {profile} profile: {'Success' if success else 'Failed'}")
        
        # Pause between movements
        time.sleep(1)
    
    logger.info("Mouse movement tests completed")


def test_mouse_clicking():
    """Test mouse clicking operations."""
    logger.info("Testing mouse clicking...")
    
    mouse = MouseController()
    
    # Get current position for clicks
    current_x, current_y = mouse.get_position()
    logger.info(f"Current position: ({current_x}, {current_y})")
    
    # Test single click
    logger.info("Testing single click")
    success = mouse.click(current_x, current_y)
    logger.info(f"Single click: {'Success' if success else 'Failed'}")
    time.sleep(1)
    
    # Test double click
    logger.info("Testing double click")
    success = mouse.double_click(current_x, current_y)
    logger.info(f"Double click: {'Success' if success else 'Failed'}")
    time.sleep(1)
    
    # Test right click
    logger.info("Testing right click")
    success = mouse.right_click(current_x, current_y)
    logger.info(f"Right click: {'Success' if success else 'Failed'}")
    
    logger.info("Mouse clicking tests completed")


def test_mouse_dragging():
    """Test mouse drag operations."""
    logger.info("Testing mouse dragging...")
    
    mouse = MouseController()
    
    # Get screen size
    screen_width, screen_height = mouse.get_screen_size()
    
    # Define start and end points
    start_x = random.randint(100, screen_width // 2)
    start_y = random.randint(100, screen_height // 2)
    
    end_x = random.randint(screen_width // 2, screen_width - 100)
    end_y = random.randint(screen_height // 2, screen_height - 100)
    
    # Move to start position first
    mouse.move_to(start_x, start_y)
    time.sleep(1)
    
    # Perform drag
    logger.info(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y})")
    success = mouse.drag(start_x, start_y, end_x, end_y)
    logger.info(f"Drag operation: {'Success' if success else 'Failed'}")
    
    logger.info("Mouse dragging tests completed")


def test_keyboard_typing():
    """Test keyboard typing with different profiles."""
    logger.info("Testing keyboard typing...")
    
    keyboard = KeyboardController()
    
    # Test typing with different profiles
    test_text = "This is a test of the MacAgent keyboard controller."
    profiles = ["normal", "fast", "slow", "precise"]
    
    for profile in profiles:
        logger.info(f"Testing typing with '{profile}' profile")
        success = keyboard.type_text(test_text, profile=profile)
        logger.info(f"Typing with {profile} profile: {'Success' if success else 'Failed'}")
        
        # Press enter to create a new line
        keyboard.press_return()
        time.sleep(1)
    
    logger.info("Keyboard typing tests completed")


def test_keyboard_shortcuts():
    """Test keyboard shortcuts."""
    logger.info("Testing keyboard shortcuts...")
    
    keyboard = KeyboardController()
    
    # Test common shortcuts
    shortcuts = ["select_all", "copy", "paste"]
    
    for shortcut in shortcuts:
        logger.info(f"Testing '{shortcut}' shortcut")
        success = keyboard.keyboard_shortcut(shortcut)
        logger.info(f"{shortcut}: {'Success' if success else 'Failed'}")
        time.sleep(1)
    
    # Test custom shortcut (Cmd+Shift+4 for screenshot)
    logger.info("Testing custom shortcut (Cmd+Shift+4 for screenshot)")
    success = keyboard.custom_shortcut("4", ["command", "shift"])
    logger.info(f"Custom shortcut: {'Success' if success else 'Failed'}")
    
    # Immediately press Escape to cancel the screenshot
    time.sleep(0.5)
    keyboard.press_key("escape")
    
    logger.info("Keyboard shortcut tests completed")


def test_coordinator_click_and_type():
    """Test the interaction coordinator's click and type functionality."""
    logger.info("Testing coordinator click and type...")
    
    coordinator = InteractionCoordinator()
    
    # Get mouse position for click
    x, y = coordinator.mouse.get_position()
    
    # Test click and type
    test_text = "This is a test of the click and type functionality."
    logger.info(f"Clicking at ({x}, {y}) and typing text")
    
    success = coordinator.click_and_type((x, y), test_text)
    logger.info(f"Click and type: {'Success' if success else 'Failed'}")
    
    logger.info("Coordinator click and type test completed")


def test_form_filling():
    """Test form filling with the interaction coordinator."""
    logger.info("Testing form filling...")
    
    coordinator = InteractionCoordinator()
    
    # Get current position as a starting point
    base_x, base_y = coordinator.mouse.get_position()
    
    # Define simulated form fields (normally these would come from UI analysis)
    form_fields = [
        {"position": (base_x, base_y), "text": "John Doe", "clear_field": True},
        {"position": (base_x, base_y + 50), "text": "johndoe@example.com", "clear_field": True},
        {"position": (base_x, base_y + 100), "text": "123 Main St", "clear_field": True}
    ]
    
    # Fill the form
    logger.info("Filling form with tab navigation")
    success = coordinator.fill_form(form_fields, use_tab_navigation=True)
    logger.info(f"Form filling: {'Success' if success else 'Failed'}")
    
    logger.info("Form filling test completed")


def test_action_sequence():
    """Test executing a sequence of actions."""
    logger.info("Testing action sequence...")
    
    coordinator = InteractionCoordinator()
    
    # Get screen dimensions
    screen_width, screen_height = coordinator.mouse.get_screen_size()
    
    # Define a sequence of actions
    actions = [
        {
            "type": "mouse",
            "action": "move",
            "params": {"position": (screen_width // 4, screen_height // 4)}
        },
        {
            "type": "mouse",
            "action": "click",
            "params": {"position": (screen_width // 4, screen_height // 4)}
        },
        {
            "type": "keyboard",
            "action": "type",
            "params": {"text": "Action sequence test"}
        },
        {
            "type": "keyboard",
            "action": "key",
            "params": {"key": "return"}
        },
        {
            "type": "mouse",
            "action": "move",
            "params": {"position": (screen_width // 2, screen_height // 2)}
        }
    ]
    
    # Execute the sequence
    logger.info("Executing action sequence")
    results = coordinator.execute_sequence(actions, pattern="content_creation")
    
    # Log results
    logger.info(f"Sequence complete: {results['complete']}")
    logger.info(f"Successful actions: {results['successful_actions']}/{results['total_actions']}")
    
    if results['failed_actions']:
        logger.info(f"Failed actions: {len(results['failed_actions'])}")
        for failed in results['failed_actions']:
            logger.info(f"  - {failed['action_type']}:{failed['action_name']}")
    
    logger.info("Action sequence test completed")


def main():
    """Run interaction system tests."""
    logger.info("Starting interaction system tests")
    
    # Ask user which tests to run
    print("\nMacAgent Interaction System Tests")
    print("=================================")
    print("1. Mouse Movement")
    print("2. Mouse Clicking")
    print("3. Mouse Dragging")
    print("4. Keyboard Typing")
    print("5. Keyboard Shortcuts")
    print("6. Coordinator Click and Type")
    print("7. Form Filling")
    print("8. Action Sequence")
    print("9. Run All Tests")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-9): ")
    
    # Give the user time to switch to a suitable window
    print("\nTests will begin in 3 seconds. Please switch to your desired application window.")
    time.sleep(3)
    
    # Execute chosen tests
    if choice == "1" or choice == "9":
        test_mouse_movement()
    
    if choice == "2" or choice == "9":
        test_mouse_clicking()
    
    if choice == "3" or choice == "9":
        test_mouse_dragging()
    
    if choice == "4" or choice == "9":
        test_keyboard_typing()
    
    if choice == "5" or choice == "9":
        test_keyboard_shortcuts()
    
    if choice == "6" or choice == "9":
        test_coordinator_click_and_type()
    
    if choice == "7" or choice == "9":
        test_form_filling()
    
    if choice == "8" or choice == "9":
        test_action_sequence()
    
    logger.info("All selected interaction system tests completed")


if __name__ == "__main__":
    main() 