"""
Settings Manager Module

This module provides functionality for managing application settings,
including support for multiple configuration profiles and user preferences.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

class SettingsManager:
    """
    Manages application settings for MacAgent.
    
    Features:
    - Load and save settings to disk
    - Support for multiple configuration profiles
    - Default settings fallback
    - Settings validation
    - Callbacks for settings changes
    """
    
    def __init__(self, config_dir: str = "~/.macagent"):
        """Initialize the settings manager.
        
        Args:
            config_dir: Directory to store configuration files
        """
        self.config_dir = os.path.expanduser(config_dir)
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.profiles_file = os.path.join(self.config_dir, "profiles.json")
        
        self.config_data: Dict[str, Any] = {}
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.current_profile: str = "Default"
        
        # Callbacks for settings changes
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Load settings
        self._ensure_config_dir()
        self.load_settings()
    
    def _ensure_config_dir(self) -> None:
        """Ensure the configuration directory exists."""
        os.makedirs(self.config_dir, exist_ok=True)
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "model": "GPT-4",
            "verbosity": 2,
            "api_key": "",
            "permissions": {
                "screen_recording": True,
                "file_access": True,
                "automation": True,
                "internet": True
            },
            "appearance": {
                "theme": "System Default",
                "primary_color": "#0078D7",
                "font_size": 12
            },
            "current_profile": "Default"
        }
    
    def load_settings(self) -> None:
        """Load settings from disk."""
        try:
            # Load main config
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config_data = json.load(f)
            else:
                self.config_data = self.create_default_config()
            
            # Load profiles
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, 'r') as f:
                    self.profiles = json.load(f)
            else:
                self.profiles = {"Default": self.create_default_config()}
            
            # Set current profile
            self.current_profile = self.config_data.get("current_profile", "Default")
            if self.current_profile not in self.profiles:
                logger.warning(f"Profile '{self.current_profile}' not found, using Default")
                self.current_profile = "Default"
                
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            self.config_data = self.create_default_config()
            self.profiles = {"Default": self.create_default_config()}
            self.current_profile = "Default"
    
    def save_settings(self) -> bool:
        """Save settings to disk.
        
        Returns:
            True if settings were saved successfully, False otherwise
        """
        try:
            # Update current profile in config
            self.config_data["current_profile"] = self.current_profile
            
            # Save profiles
            with open(self.profiles_file, 'w') as f:
                json.dump(self.profiles, f, indent=2)
            
            # Save main config
            with open(self.config_file, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            
            logger.info("Settings saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            The current configuration dictionary
        """
        return self.config_data
    
    def get_current_profile(self) -> str:
        """Get the name of the current profile.
        
        Returns:
            Current profile name
        """
        return self.current_profile
    
    def get_profiles(self) -> List[str]:
        """Get a list of available profiles.
        
        Returns:
            List of profile names
        """
        return list(self.profiles.keys())
    
    def get_profile_settings(self, profile_name: str) -> Dict[str, Any]:
        """Get settings for a specific profile.
        
        Args:
            profile_name: Name of the profile
            
        Returns:
            Profile settings dictionary or default config if profile not found
        """
        return self.profiles.get(profile_name, self.create_default_config())
    
    def switch_profile(self, profile_name: str) -> bool:
        """Switch to a different profile.
        
        Args:
            profile_name: Name of the profile to switch to
            
        Returns:
            True if profile was switched successfully, False otherwise
        """
        if profile_name not in self.profiles:
            logger.warning(f"Profile '{profile_name}' not found")
            return False
        
        self.current_profile = profile_name
        self.config_data = self.profiles[profile_name].copy()
        self.config_data["current_profile"] = profile_name
        
        # Notify callbacks
        self._notify_callbacks()
        
        return True
    
    def create_profile(self, profile_name: str, settings: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new profile.
        
        Args:
            profile_name: Name of the new profile
            settings: Settings for the new profile, or None to use current profile settings
            
        Returns:
            True if profile was created successfully, False otherwise
        """
        if profile_name in self.profiles:
            logger.warning(f"Profile '{profile_name}' already exists")
            return False
        
        if settings is None:
            # Use current profile settings
            settings = self.profiles[self.current_profile].copy()
        
        self.profiles[profile_name] = settings
        return True
    
    def delete_profile(self, profile_name: str) -> bool:
        """Delete a profile.
        
        Args:
            profile_name: Name of the profile to delete
            
        Returns:
            True if profile was deleted successfully, False otherwise
        """
        if profile_name == "Default":
            logger.warning("Cannot delete Default profile")
            return False
        
        if profile_name not in self.profiles:
            logger.warning(f"Profile '{profile_name}' not found")
            return False
        
        # Delete profile
        del self.profiles[profile_name]
        
        # If current profile was deleted, switch to Default
        if self.current_profile == profile_name:
            self.current_profile = "Default"
            self.config_data = self.profiles["Default"].copy()
            self.config_data["current_profile"] = "Default"
            self._notify_callbacks()
        
        return True
    
    def update_settings(self, settings: Dict[str, Any], profile_name: Optional[str] = None) -> bool:
        """Update settings for a profile.
        
        Args:
            settings: New settings dictionary
            profile_name: Name of the profile to update, or None for current profile
            
        Returns:
            True if settings were updated successfully, False otherwise
        """
        target_profile = profile_name or self.current_profile
        
        if target_profile not in self.profiles:
            logger.warning(f"Profile '{target_profile}' not found")
            return False
        
        # Update profile settings
        self.profiles[target_profile] = settings
        
        # If updating current profile, also update config_data
        if target_profile == self.current_profile:
            self.config_data = settings.copy()
            self.config_data["current_profile"] = self.current_profile
            self._notify_callbacks()
        
        return True
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value.
        
        Args:
            key: Setting key (can use dot notation for nested settings)
            default: Default value if setting is not found
            
        Returns:
            Setting value or default
        """
        if "." in key:
            # Handle nested settings with dot notation (e.g., "appearance.theme")
            parts = key.split(".")
            value = self.config_data
            
            for part in parts:
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
            
            return value
        
        return self.config_data.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """Set a specific setting value.
        
        Args:
            key: Setting key (can use dot notation for nested settings)
            value: New setting value
            
        Returns:
            True if setting was updated successfully, False otherwise
        """
        if "." in key:
            # Handle nested settings with dot notation (e.g., "appearance.theme")
            parts = key.split(".")
            target = self.config_data
            profile_target = self.profiles[self.current_profile]
            
            # Navigate to the correct nested dictionary, creating it if necessary
            for i, part in enumerate(parts[:-1]):
                if part not in target:
                    target[part] = {}
                    profile_target[part] = {}
                
                target = target[part]
                profile_target = profile_target[part]
            
            # Set the value
            target[parts[-1]] = value
            profile_target[parts[-1]] = value
        else:
            # Set top-level setting
            self.config_data[key] = value
            self.profiles[self.current_profile][key] = value
        
        self._notify_callbacks()
        return True
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback for settings changes.
        
        Args:
            callback: Function to call when settings change
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unregister a callback for settings changes.
        
        Args:
            callback: Previously registered callback function
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks about settings changes."""
        for callback in self.callbacks:
            try:
                callback(self.config_data)
            except Exception as e:
                logger.error(f"Error in settings callback: {e}")
    
    def export_settings(self, file_path: str) -> bool:
        """Export settings to a file.
        
        Args:
            file_path: Path to export settings to
            
        Returns:
            True if settings were exported successfully, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.profiles, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to export settings: {e}")
            return False
    
    def import_settings(self, file_path: str) -> bool:
        """Import settings from a file.
        
        Args:
            file_path: Path to import settings from
            
        Returns:
            True if settings were imported successfully, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                imported_profiles = json.load(f)
            
            if not isinstance(imported_profiles, dict):
                logger.error("Invalid settings format")
                return False
            
            # Merge imported profiles with existing profiles
            self.profiles.update(imported_profiles)
            
            # Notify about changes
            self._notify_callbacks()
            
            return True
        except Exception as e:
            logger.error(f"Failed to import settings: {e}")
            return False
    
    def reset_to_defaults(self, profile_name: Optional[str] = None) -> bool:
        """Reset settings to defaults.
        
        Args:
            profile_name: Name of the profile to reset, or None for current profile
            
        Returns:
            True if settings were reset successfully, False otherwise
        """
        target_profile = profile_name or self.current_profile
        
        if target_profile not in self.profiles:
            logger.warning(f"Profile '{target_profile}' not found")
            return False
        
        # Reset profile to defaults
        self.profiles[target_profile] = self.create_default_config()
        
        # If resetting current profile, also update config_data
        if target_profile == self.current_profile:
            self.config_data = self.profiles[target_profile].copy()
            self.config_data["current_profile"] = self.current_profile
            self._notify_callbacks()
        
        return True
