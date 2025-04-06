"""
Profile Manager Module

This module provides functionality for creating, managing, and switching between
different usage profiles with specific settings and behaviors.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable, Set

from MacAgent.src.utils.settings_manager import SettingsManager

logger = logging.getLogger(__name__)


class Profile:
    """
    Represents a user profile with specific settings and behaviors.
    """
    
    def __init__(self, name: str, settings: Dict[str, Any], behaviors: Optional[Set[str]] = None):
        """Initialize a profile.
        
        Args:
            name: Profile name
            settings: Profile settings dictionary
            behaviors: Set of enabled behaviors for this profile
        """
        self.name = name
        self.settings = settings
        self.behaviors = behaviors or set()
    
    def has_behavior(self, behavior: str) -> bool:
        """Check if the profile has a specific behavior enabled.
        
        Args:
            behavior: Behavior to check
            
        Returns:
            True if the behavior is enabled, False otherwise
        """
        return behavior in self.behaviors
    
    def enable_behavior(self, behavior: str) -> None:
        """Enable a specific behavior for this profile.
        
        Args:
            behavior: Behavior to enable
        """
        self.behaviors.add(behavior)
    
    def disable_behavior(self, behavior: str) -> None:
        """Disable a specific behavior for this profile.
        
        Args:
            behavior: Behavior to disable
        """
        if behavior in self.behaviors:
            self.behaviors.remove(behavior)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the profile
        """
        return {
            "name": self.name,
            "settings": self.settings,
            "behaviors": list(self.behaviors)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Profile':
        """Create a profile from a dictionary.
        
        Args:
            data: Dictionary containing profile data
            
        Returns:
            New Profile instance
        """
        return cls(
            name=data.get("name", "Unnamed"),
            settings=data.get("settings", {}),
            behaviors=set(data.get("behaviors", []))
        )


class ProfileManager:
    """
    Manages user profiles with specific settings and behaviors.
    
    Features:
    - Create, delete, and modify profiles
    - Switch between profiles
    - Enable/disable behaviors for profiles
    - Load and save profiles using SettingsManager
    """
    
    # Common behavior flags
    BEHAVIOR_SCREEN_RECORDING = "screen_recording"
    BEHAVIOR_FILE_ACCESS = "file_access"
    BEHAVIOR_UI_AUTOMATION = "ui_automation"
    BEHAVIOR_INTERNET_ACCESS = "internet_access"
    BEHAVIOR_USE_CLIPBOARD = "use_clipboard"
    BEHAVIOR_NOTIFICATIONS = "notifications"
    BEHAVIOR_BACKGROUND_PROCESSING = "background_processing"
    BEHAVIOR_PROACTIVE_SUGGESTIONS = "proactive_suggestions"
    
    # Predefined profile types
    PROFILE_TYPE_STANDARD = "standard"
    PROFILE_TYPE_RESTRICTED = "restricted"
    PROFILE_TYPE_POWER_USER = "power_user"
    PROFILE_TYPE_DEVELOPER = "developer"
    
    def __init__(self, settings_manager: Optional[SettingsManager] = None):
        """Initialize the profile manager.
        
        Args:
            settings_manager: SettingsManager instance to use, or None to create a new one
        """
        self.settings_manager = settings_manager or SettingsManager()
        self.profiles: Dict[str, Profile] = {}
        self.active_profile_name: str = "Default"
        
        # Callbacks for profile changes
        self.profile_change_callbacks: List[Callable[[str], None]] = []
        
        # Load profiles
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load profiles from settings."""
        # Get profile data from settings
        all_profiles = self.settings_manager.get_profiles()
        
        # Convert to Profile objects
        for profile_name in all_profiles:
            settings = self.settings_manager.get_profile_settings(profile_name)
            
            # Extract behaviors from settings (or create empty set)
            behaviors = set(settings.pop("behaviors", []))
            
            # Create Profile object
            self.profiles[profile_name] = Profile(profile_name, settings, behaviors)
        
        # Set active profile
        self.active_profile_name = self.settings_manager.get_current_profile()
        
        # Ensure default profile exists
        if "Default" not in self.profiles:
            self._create_default_profile()
    
    def _create_default_profile(self) -> None:
        """Create a default profile with standard behaviors."""
        default_settings = self.settings_manager.create_default_config()
        
        # Add standard behaviors to the default profile
        default_behaviors = {
            self.BEHAVIOR_SCREEN_RECORDING,
            self.BEHAVIOR_FILE_ACCESS,
            self.BEHAVIOR_UI_AUTOMATION,
            self.BEHAVIOR_INTERNET_ACCESS,
            self.BEHAVIOR_NOTIFICATIONS
        }
        
        # Create default profile and save it
        self.profiles["Default"] = Profile("Default", default_settings, default_behaviors)
        self._save_profile("Default")
    
    def _save_profile(self, profile_name: str) -> bool:
        """Save a profile to settings.
        
        Args:
            profile_name: Name of the profile to save
            
        Returns:
            True if the profile was saved successfully, False otherwise
        """
        if profile_name not in self.profiles:
            logger.warning(f"Cannot save non-existent profile: {profile_name}")
            return False
        
        profile = self.profiles[profile_name]
        
        # Create a copy of settings to add behaviors
        settings = profile.settings.copy()
        settings["behaviors"] = list(profile.behaviors)
        
        # Save to settings manager
        success = self.settings_manager.update_settings(settings, profile_name)
        if success:
            return self.settings_manager.save_settings()
        
        return False
    
    def create_profile(self, name: str, profile_type: str = PROFILE_TYPE_STANDARD) -> bool:
        """Create a new profile with predefined behaviors based on type.
        
        Args:
            name: Profile name
            profile_type: Type of profile to create
            
        Returns:
            True if the profile was created successfully, False otherwise
        """
        if name in self.profiles:
            logger.warning(f"Profile '{name}' already exists")
            return False
        
        # Base settings from default profile
        default_profile = self.profiles.get("Default")
        if not default_profile:
            self._create_default_profile()
            default_profile = self.profiles["Default"]
        
        settings = default_profile.settings.copy()
        
        # Set behaviors based on profile type
        behaviors = set()
        
        if profile_type == self.PROFILE_TYPE_STANDARD:
            behaviors = {
                self.BEHAVIOR_SCREEN_RECORDING,
                self.BEHAVIOR_FILE_ACCESS,
                self.BEHAVIOR_UI_AUTOMATION,
                self.BEHAVIOR_INTERNET_ACCESS,
                self.BEHAVIOR_NOTIFICATIONS
            }
        elif profile_type == self.PROFILE_TYPE_RESTRICTED:
            behaviors = {
                self.BEHAVIOR_NOTIFICATIONS
            }
        elif profile_type == self.PROFILE_TYPE_POWER_USER:
            behaviors = {
                self.BEHAVIOR_SCREEN_RECORDING,
                self.BEHAVIOR_FILE_ACCESS,
                self.BEHAVIOR_UI_AUTOMATION,
                self.BEHAVIOR_INTERNET_ACCESS,
                self.BEHAVIOR_USE_CLIPBOARD,
                self.BEHAVIOR_NOTIFICATIONS,
                self.BEHAVIOR_BACKGROUND_PROCESSING,
                self.BEHAVIOR_PROACTIVE_SUGGESTIONS
            }
        elif profile_type == self.PROFILE_TYPE_DEVELOPER:
            behaviors = {
                self.BEHAVIOR_SCREEN_RECORDING,
                self.BEHAVIOR_FILE_ACCESS,
                self.BEHAVIOR_UI_AUTOMATION,
                self.BEHAVIOR_INTERNET_ACCESS,
                self.BEHAVIOR_USE_CLIPBOARD,
                self.BEHAVIOR_NOTIFICATIONS,
                self.BEHAVIOR_BACKGROUND_PROCESSING
            }
            # Set developer-specific settings
            settings["verbosity"] = 4  # Maximum verbosity
        
        # Create the profile
        self.profiles[name] = Profile(name, settings, behaviors)
        
        # Save the profile
        return self._save_profile(name)
    
    def delete_profile(self, name: str) -> bool:
        """Delete a profile.
        
        Args:
            name: Name of the profile to delete
            
        Returns:
            True if the profile was deleted successfully, False otherwise
        """
        if name == "Default":
            logger.warning("Cannot delete Default profile")
            return False
        
        if name not in self.profiles:
            logger.warning(f"Profile '{name}' does not exist")
            return False
        
        # Delete from profiles dict
        del self.profiles[name]
        
        # Delete from settings manager
        success = self.settings_manager.delete_profile(name)
        
        # If active profile was deleted, switch to Default
        if self.active_profile_name == name:
            self.switch_profile("Default")
        
        return success
    
    def switch_profile(self, name: str) -> bool:
        """Switch to a different profile.
        
        Args:
            name: Name of the profile to switch to
            
        Returns:
            True if the profile was switched successfully, False otherwise
        """
        if name not in self.profiles:
            logger.warning(f"Cannot switch to non-existent profile: {name}")
            return False
        
        # Switch in settings manager
        success = self.settings_manager.switch_profile(name)
        
        if success:
            self.active_profile_name = name
            
            # Notify callbacks
            self._notify_profile_change(name)
        
        return success
    
    def get_active_profile(self) -> Profile:
        """Get the active profile.
        
        Returns:
            The active Profile instance
        """
        return self.profiles.get(self.active_profile_name, self.profiles["Default"])
    
    def get_active_profile_name(self) -> str:
        """Get the name of the active profile.
        
        Returns:
            Name of the active profile
        """
        return self.active_profile_name
    
    def get_profiles(self) -> Dict[str, Profile]:
        """Get all profiles.
        
        Returns:
            Dictionary of profile names to Profile instances
        """
        return self.profiles
    
    def get_profile_names(self) -> List[str]:
        """Get the names of all profiles.
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())
    
    def get_profile(self, name: str) -> Optional[Profile]:
        """Get a specific profile.
        
        Args:
            name: Profile name
            
        Returns:
            Profile instance or None if not found
        """
        return self.profiles.get(name)
    
    def update_profile_settings(self, name: str, settings: Dict[str, Any]) -> bool:
        """Update settings for a profile.
        
        Args:
            name: Profile name
            settings: New settings dictionary
            
        Returns:
            True if settings were updated successfully, False otherwise
        """
        if name not in self.profiles:
            logger.warning(f"Cannot update non-existent profile: {name}")
            return False
        
        profile = self.profiles[name]
        
        # Update settings
        profile.settings.update(settings)
        
        # Save changes
        return self._save_profile(name)
    
    def get_behavior(self, behavior: str) -> bool:
        """Check if a behavior is enabled in the active profile.
        
        Args:
            behavior: Behavior to check
            
        Returns:
            True if the behavior is enabled, False otherwise
        """
        active_profile = self.get_active_profile()
        return active_profile.has_behavior(behavior)
    
    def set_behavior(self, name: str, behavior: str, enabled: bool) -> bool:
        """Enable or disable a behavior for a profile.
        
        Args:
            name: Profile name
            behavior: Behavior to modify
            enabled: True to enable, False to disable
            
        Returns:
            True if the behavior was modified successfully, False otherwise
        """
        if name not in self.profiles:
            logger.warning(f"Cannot modify non-existent profile: {name}")
            return False
        
        profile = self.profiles[name]
        
        # Enable or disable the behavior
        if enabled:
            profile.enable_behavior(behavior)
        else:
            profile.disable_behavior(behavior)
        
        # Save changes
        return self._save_profile(name)
    
    def register_profile_change_callback(self, callback: Callable[[str], None]) -> None:
        """Register a callback for profile changes.
        
        Args:
            callback: Function to call when active profile changes
        """
        if callback not in self.profile_change_callbacks:
            self.profile_change_callbacks.append(callback)
    
    def unregister_profile_change_callback(self, callback: Callable[[str], None]) -> None:
        """Unregister a callback for profile changes.
        
        Args:
            callback: Previously registered callback function
        """
        if callback in self.profile_change_callbacks:
            self.profile_change_callbacks.remove(callback)
    
    def _notify_profile_change(self, profile_name: str) -> None:
        """Notify all registered callbacks about profile changes.
        
        Args:
            profile_name: Name of the new active profile
        """
        for callback in self.profile_change_callbacks:
            try:
                callback(profile_name)
            except Exception as e:
                logger.error(f"Error in profile change callback: {e}")
    
    def clone_profile(self, source_name: str, target_name: str) -> bool:
        """Clone a profile with a new name.
        
        Args:
            source_name: Name of the source profile
            target_name: Name for the cloned profile
            
        Returns:
            True if the profile was cloned successfully, False otherwise
        """
        if source_name not in self.profiles:
            logger.warning(f"Source profile '{source_name}' does not exist")
            return False
        
        if target_name in self.profiles:
            logger.warning(f"Target profile '{target_name}' already exists")
            return False
        
        # Get source profile
        source_profile = self.profiles[source_name]
        
        # Clone settings and behaviors
        settings = source_profile.settings.copy()
        behaviors = source_profile.behaviors.copy()
        
        # Create new profile
        self.profiles[target_name] = Profile(target_name, settings, behaviors)
        
        # Save the new profile
        return self._save_profile(target_name)
    
    def is_behavior_supported(self, behavior: str) -> bool:
        """Check if a behavior is supported by the system.
        
        Args:
            behavior: Behavior to check
            
        Returns:
            True if the behavior is supported, False otherwise
        """
        # List of all supported behaviors
        supported_behaviors = {
            self.BEHAVIOR_SCREEN_RECORDING,
            self.BEHAVIOR_FILE_ACCESS,
            self.BEHAVIOR_UI_AUTOMATION,
            self.BEHAVIOR_INTERNET_ACCESS,
            self.BEHAVIOR_USE_CLIPBOARD,
            self.BEHAVIOR_NOTIFICATIONS,
            self.BEHAVIOR_BACKGROUND_PROCESSING,
            self.BEHAVIOR_PROACTIVE_SUGGESTIONS
        }
        
        return behavior in supported_behaviors
    
    def get_supported_behaviors(self) -> List[str]:
        """Get a list of all supported behaviors.
        
        Returns:
            List of behavior identifiers
        """
        return [
            self.BEHAVIOR_SCREEN_RECORDING,
            self.BEHAVIOR_FILE_ACCESS,
            self.BEHAVIOR_UI_AUTOMATION,
            self.BEHAVIOR_INTERNET_ACCESS,
            self.BEHAVIOR_USE_CLIPBOARD,
            self.BEHAVIOR_NOTIFICATIONS,
            self.BEHAVIOR_BACKGROUND_PROCESSING,
            self.BEHAVIOR_PROACTIVE_SUGGESTIONS
        ]
    
    def get_profile_types(self) -> List[str]:
        """Get a list of all predefined profile types.
        
        Returns:
            List of profile type identifiers
        """
        return [
            self.PROFILE_TYPE_STANDARD,
            self.PROFILE_TYPE_RESTRICTED,
            self.PROFILE_TYPE_POWER_USER,
            self.PROFILE_TYPE_DEVELOPER
        ]
    
    def get_behavior_description(self, behavior: str) -> str:
        """Get the description of a behavior.
        
        Args:
            behavior: Behavior identifier
            
        Returns:
            Human-readable description of the behavior
        """
        descriptions = {
            self.BEHAVIOR_SCREEN_RECORDING: "Capture and analyze screen content",
            self.BEHAVIOR_FILE_ACCESS: "Access and modify files on your system",
            self.BEHAVIOR_UI_AUTOMATION: "Control UI elements and applications",
            self.BEHAVIOR_INTERNET_ACCESS: "Access the internet for information and services",
            self.BEHAVIOR_USE_CLIPBOARD: "Read from and write to the clipboard",
            self.BEHAVIOR_NOTIFICATIONS: "Display system notifications",
            self.BEHAVIOR_BACKGROUND_PROCESSING: "Process tasks in the background",
            self.BEHAVIOR_PROACTIVE_SUGGESTIONS: "Provide suggestions without explicit prompting"
        }
        
        return descriptions.get(behavior, "Unknown behavior")
    
    def get_profile_type_description(self, profile_type: str) -> str:
        """Get the description of a profile type.
        
        Args:
            profile_type: Profile type identifier
            
        Returns:
            Human-readable description of the profile type
        """
        descriptions = {
            self.PROFILE_TYPE_STANDARD: "Standard profile with basic capabilities",
            self.PROFILE_TYPE_RESTRICTED: "Restricted profile with minimal system access",
            self.PROFILE_TYPE_POWER_USER: "Power user profile with advanced features",
            self.PROFILE_TYPE_DEVELOPER: "Developer profile with debugging features"
        }
        
        return descriptions.get(profile_type, "Unknown profile type")
