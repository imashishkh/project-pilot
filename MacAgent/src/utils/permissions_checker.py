"""
MacOS Permissions Checker Module

This module provides utilities to check and request necessary macOS permissions
for the MacAgent application to function properly.
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Set, Optional, Tuple
import platform

logger = logging.getLogger(__name__)

class MacOSPermissionChecker:
    """
    Utility class to check and request macOS permissions needed for MacAgent.
    
    The class checks for the following permissions:
    - Accessibility: For controlling mouse and keyboard
    - Screen Recording: For capturing screen contents
    - Automation: For controlling other applications
    """
    
    def __init__(self):
        """Initialize the permissions checker."""
        self.permissions_status = {
            "accessibility": None,  # None = unknown, False = denied, True = granted
            "screen_recording": None,
            "automation": None,
        }
        
        # Verify we're running on macOS
        if platform.system() != "Darwin":
            logger.error("MacOSPermissionChecker can only be used on macOS")
            raise RuntimeError("MacOSPermissionChecker requires macOS")
    
    def check_all_permissions(self) -> Dict[str, bool]:
        """
        Check all required permissions.
        
        Returns:
            Dictionary of permission names and their status (True if granted, False if not)
        """
        self.permissions_status["accessibility"] = self.check_accessibility_permission()
        self.permissions_status["screen_recording"] = self.check_screen_recording_permission()
        self.permissions_status["automation"] = self.check_automation_permission()
        
        return self.permissions_status
    
    def check_accessibility_permission(self) -> bool:
        """
        Check if Accessibility permission is granted.
        
        Returns:
            True if granted, False if not
        """
        try:
            # Test if we can get mouse position, which requires accessibility permission
            import pyautogui
            pyautogui.position()
            return True
        except Exception as e:
            logger.warning(f"Accessibility permission check failed: {str(e)}")
            return False
    
    def check_screen_recording_permission(self) -> bool:
        """
        Check if Screen Recording permission is granted.
        
        Returns:
            True if granted, False if not
        """
        try:
            # Attempt to take a screenshot, which requires screen recording permission
            import pyautogui
            screenshot = pyautogui.screenshot()
            
            # If screenshot is blank or very small, permission likely not granted
            if screenshot.size[0] < 10 or screenshot.size[1] < 10:
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Screen Recording permission check failed: {str(e)}")
            return False
    
    def check_automation_permission(self) -> bool:
        """
        Check if Automation permission is granted.
        
        Returns:
            True if granted, False if not
        """
        try:
            # This is a simple test using AppleScript to check Finder, which
            # requires automation permission
            result = subprocess.run([
                "osascript", 
                "-e", 
                'tell application "Finder" to get name of front window'
            ], capture_output=True, text=True)
            
            # If command fails or returns error, permission isn't granted
            if result.returncode != 0 or "not allowed" in result.stderr:
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Automation permission check failed: {str(e)}")
            return False
    
    def open_accessibility_preferences(self) -> None:
        """Open the Accessibility section of Privacy & Security settings."""
        try:
            if int(platform.mac_ver()[0].split('.')[0]) >= 13:  # macOS 13 (Ventura) or newer
                subprocess.run([
                    "open", 
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
                ])
            else:  # Older macOS versions
                subprocess.run([
                    "open", 
                    "/System/Library/PreferencePanes/Security.prefPane"
                ])
        except Exception as e:
            logger.error(f"Failed to open Accessibility preferences: {str(e)}")
    
    def open_screen_recording_preferences(self) -> None:
        """Open the Screen Recording section of Privacy & Security settings."""
        try:
            if int(platform.mac_ver()[0].split('.')[0]) >= 13:  # macOS 13 (Ventura) or newer
                subprocess.run([
                    "open", 
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
                ])
            else:  # Older macOS versions
                subprocess.run([
                    "open", 
                    "/System/Library/PreferencePanes/Security.prefPane"
                ])
        except Exception as e:
            logger.error(f"Failed to open Screen Recording preferences: {str(e)}")
    
    def open_automation_preferences(self) -> None:
        """Open the Automation section of Privacy & Security settings."""
        try:
            if int(platform.mac_ver()[0].split('.')[0]) >= 13:  # macOS 13 (Ventura) or newer
                subprocess.run([
                    "open", 
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_Automation"
                ])
            else:  # Older macOS versions
                subprocess.run([
                    "open", 
                    "/System/Library/PreferencePanes/Security.prefPane"
                ])
        except Exception as e:
            logger.error(f"Failed to open Automation preferences: {str(e)}")
    
    def display_permissions_guide(self) -> str:
        """
        Generate a guide for setting up permissions based on current status.
        
        Returns:
            A formatted string with instructions
        """
        # First check the current status
        self.check_all_permissions()
        
        guide = "MacAgent Permissions Guide\n"
        guide += "==========================\n\n"
        
        # Accessibility section
        guide += "1. Accessibility Permission:\n"
        guide += "   Status: " + ("✅ Granted" if self.permissions_status["accessibility"] else "❌ Not Granted") + "\n"
        if not self.permissions_status["accessibility"]:
            guide += "   - Open System Preferences > Security & Privacy > Privacy > Accessibility\n"
            guide += "   - Click the lock icon to make changes\n"
            guide += "   - Ensure that Python or your application is in the list and checked\n"
            guide += "   - If not, click '+' and add your application\n\n"
        else:
            guide += "   - No action needed\n\n"
            
        # Screen Recording section
        guide += "2. Screen Recording Permission:\n"
        guide += "   Status: " + ("✅ Granted" if self.permissions_status["screen_recording"] else "❌ Not Granted") + "\n"
        if not self.permissions_status["screen_recording"]:
            guide += "   - Open System Preferences > Security & Privacy > Privacy > Screen Recording\n"
            guide += "   - Click the lock icon to make changes\n"
            guide += "   - Ensure that Python or your application is in the list and checked\n"
            guide += "   - If not, click '+' and add your application\n"
            guide += "   - You may need to restart your application after granting permission\n\n"
        else:
            guide += "   - No action needed\n\n"
            
        # Automation section
        guide += "3. Automation Permission:\n"
        guide += "   Status: " + ("✅ Granted" if self.permissions_status["automation"] else "❌ Not Granted") + "\n"
        if not self.permissions_status["automation"]:
            guide += "   - When prompted, click 'OK' to allow your application to control other applications\n"
            guide += "   - Or open System Preferences > Security & Privacy > Privacy > Automation\n"
            guide += "   - Click the lock icon to make changes\n"
            guide += "   - Ensure your application has necessary apps checked (e.g., Finder, Safari)\n\n"
        else:
            guide += "   - No action needed\n\n"
            
        guide += "After granting permissions, you may need to restart MacAgent.\n"
        
        return guide

# Singleton instance for easier imports
permission_checker = MacOSPermissionChecker() 