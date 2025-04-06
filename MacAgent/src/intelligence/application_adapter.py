"""
Application Adapter Module

This module provides specialized handling for different macOS applications,
implementing application-specific interaction patterns and fallback mechanisms.
"""

import os
import sys
import json
import importlib
import logging
import subprocess
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Type, Union, ClassVar

from AppKit import NSWorkspace

logger = logging.getLogger(__name__)


class ApplicationCategory(Enum):
    """Categories of applications."""
    PRODUCTIVITY = "productivity"
    BROWSER = "browser"
    DEVELOPMENT = "development"
    UTILITY = "utility"
    MEDIA = "media"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ApplicationFeature(Enum):
    """Common features applications may support."""
    DOCUMENT_EDITING = "document_editing"
    FILE_MANAGEMENT = "file_management"
    WEB_BROWSING = "web_browsing"
    MEDIA_PLAYBACK = "media_playback"
    TEXT_SEARCH = "text_search"
    IMAGE_EDITING = "image_editing"
    CODE_EDITING = "code_editing"
    TERMINAL_COMMANDS = "terminal_commands"
    EMAIL = "email"
    CALENDAR = "calendar"
    MESSAGING = "messaging"
    WINDOW_MANAGEMENT = "window_management"


class ApplicationCommand:
    """Represents a command that can be executed in an application."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        shortcut: Optional[str] = None,
        menu_path: Optional[List[str]] = None
    ):
        """Initialize an application command.
        
        Args:
            name: Name of the command
            description: Description of what the command does
            parameters: Dictionary of parameter names to descriptions/types
            shortcut: Keyboard shortcut for the command
            menu_path: Path to the command in the application menu
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.shortcut = shortcut
        self.menu_path = menu_path or []
    
    def __str__(self) -> str:
        """String representation of the command."""
        result = f"{self.name}: {self.description}"
        if self.shortcut:
            result += f" (Shortcut: {self.shortcut})"
        return result


class BaseAdapter(ABC):
    """Base class for all application adapters."""
    
    # Class variables
    name: ClassVar[str] = "Base Adapter"
    bundle_ids: ClassVar[List[str]] = []
    category: ClassVar[ApplicationCategory] = ApplicationCategory.UNKNOWN
    supported_features: ClassVar[List[ApplicationFeature]] = []
    default_commands: ClassVar[List[ApplicationCommand]] = []
    
    def __init__(self, bundle_id: str = ""):
        """Initialize the adapter.
        
        Args:
            bundle_id: Bundle ID of the application
        """
        self.bundle_id = bundle_id or (self.bundle_ids[0] if self.bundle_ids else "")
        self.commands: Dict[str, ApplicationCommand] = {}
        
        # Load default commands
        for command in self.default_commands:
            self.commands[command.name] = command
    
    @abstractmethod
    def launch(self, document_path: Optional[str] = None) -> bool:
        """Launch the application.
        
        Args:
            document_path: Optional path to a document to open
            
        Returns:
            True if application launched successfully
        """
        pass
    
    @abstractmethod
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a command in the application.
        
        Args:
            command_name: Name of the command to execute
            parameters: Parameters for the command
            
        Returns:
            Dictionary with the result of the command execution
        """
        pass
    
    def get_commands(self) -> Dict[str, ApplicationCommand]:
        """Get all available commands for this application.
        
        Returns:
            Dictionary of command names to ApplicationCommand objects
        """
        return self.commands
    
    def has_feature(self, feature: ApplicationFeature) -> bool:
        """Check if the application supports a specific feature.
        
        Args:
            feature: Feature to check
            
        Returns:
            True if the feature is supported
        """
        return feature in self.supported_features
    
    def get_supported_features(self) -> List[ApplicationFeature]:
        """Get all features supported by this application.
        
        Returns:
            List of supported features
        """
        return self.supported_features
    
    def get_category(self) -> ApplicationCategory:
        """Get the category of this application.
        
        Returns:
            Application category
        """
        return self.category
    
    def is_running(self) -> bool:
        """Check if the application is currently running.
        
        Returns:
            True if the application is running
        """
        workspace = NSWorkspace.sharedWorkspace()
        running_apps = workspace.runningApplications()
        
        for app in running_apps:
            if app.bundleIdentifier() == self.bundle_id:
                return True
        
        return False


class AppleScriptAdapter(BaseAdapter):
    """Base adapter for applications that support AppleScript."""
    
    def launch(self, document_path: Optional[str] = None) -> bool:
        """Launch the application using AppleScript.
        
        Args:
            document_path: Optional path to a document to open
            
        Returns:
            True if application launched successfully
        """
        app_name = self.name.split()[0]  # Extract first word of app name
        
        script = f'tell application "{app_name}"'
        script += '\n  activate'
        
        if document_path:
            # Escape double quotes in path
            safe_path = document_path.replace('"', '\\"')
            script += f'\n  open POSIX file "{safe_path}"'
        
        script += '\nend tell'
        
        return self._run_applescript(script)
    
    def _run_applescript(self, script: str) -> bool:
        """Run an AppleScript.
        
        Args:
            script: AppleScript to run
            
        Returns:
            True if script executed successfully
        """
        try:
            cmd = ['osascript', '-e', script]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to run AppleScript: {e}")
            return False
    
    def _run_applescript_with_result(self, script: str) -> Optional[str]:
        """Run an AppleScript and return its result.
        
        Args:
            script: AppleScript to run
            
        Returns:
            Result of the script or None if execution failed
        """
        try:
            cmd = ['osascript', '-e', script]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception as e:
            logger.error(f"Failed to run AppleScript with result: {e}")
            return None


class UniversalAdapter(AppleScriptAdapter):
    """Fallback adapter that works with any application."""
    
    name = "Universal Adapter"
    category = ApplicationCategory.UNKNOWN
    supported_features = [
        ApplicationFeature.WINDOW_MANAGEMENT
    ]
    default_commands = [
        ApplicationCommand(
            name="activate",
            description="Bring the application to the foreground",
            shortcut=None,
            menu_path=None
        ),
        ApplicationCommand(
            name="quit",
            description="Quit the application",
            shortcut="Cmd+Q",
            menu_path=["File", "Quit"]
        )
    ]
    
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a command in the application.
        
        Args:
            command_name: Name of the command to execute
            parameters: Parameters for the command
            
        Returns:
            Dictionary with the result of the command execution
        """
        parameters = parameters or {}
        
        if command_name == "activate":
            success = self.launch()
            return {"success": success}
        
        elif command_name == "quit":
            app_name = parameters.get("app_name", self.name.split()[0])
            script = f'tell application "{app_name}" to quit'
            success = self._run_applescript(script)
            return {"success": success}
        
        else:
            return {"success": False, "error": f"Command {command_name} not supported by Universal Adapter"}


class FinderAdapter(AppleScriptAdapter):
    """Adapter for the macOS Finder."""
    
    name = "Finder"
    bundle_ids = ["com.apple.finder"]
    category = ApplicationCategory.SYSTEM
    supported_features = [
        ApplicationFeature.FILE_MANAGEMENT,
        ApplicationFeature.WINDOW_MANAGEMENT
    ]
    default_commands = [
        ApplicationCommand(
            name="new_folder",
            description="Create a new folder",
            shortcut="Shift+Cmd+N",
            menu_path=["File", "New Folder"]
        ),
        ApplicationCommand(
            name="open",
            description="Open a file or folder",
            shortcut="Cmd+O",
            menu_path=["File", "Open"]
        ),
        ApplicationCommand(
            name="get_selection",
            description="Get the currently selected items",
            shortcut=None,
            menu_path=None
        ),
        ApplicationCommand(
            name="move_to_trash",
            description="Move selected items to trash",
            shortcut="Cmd+Delete",
            menu_path=["File", "Move to Trash"]
        )
    ]
    
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a command in Finder.
        
        Args:
            command_name: Name of the command to execute
            parameters: Parameters for the command
            
        Returns:
            Dictionary with the result of the command execution
        """
        parameters = parameters or {}
        
        if command_name == "new_folder":
            path = parameters.get("path", None)
            name = parameters.get("name", "New Folder")
            
            script = 'tell application "Finder"\n'
            
            if path:
                script += f'  set targetFolder to POSIX file "{path}" as alias\n'
                script += '  tell targetFolder\n'
                script += f'    make new folder with properties {{name:"{name}"}}\n'
                script += '  end tell\n'
            else:
                script += '  tell front window\n'
                script += f'    make new folder with properties {{name:"{name}"}}\n'
                script += '  end tell\n'
            
            script += 'end tell'
            
            success = self._run_applescript(script)
            return {"success": success}
        
        elif command_name == "open":
            path = parameters.get("path", None)
            
            if not path:
                return {"success": False, "error": "Path parameter is required"}
            
            script = 'tell application "Finder"\n'
            script += f'  open POSIX file "{path}"\n'
            script += 'end tell'
            
            success = self._run_applescript(script)
            return {"success": success}
        
        elif command_name == "get_selection":
            script = 'tell application "Finder"\n'
            script += '  set sel to selection\n'
            script += '  set paths to {}\n'
            script += '  repeat with i in sel\n'
            script += '    set paths to paths & (POSIX path of (i as alias))\n'
            script += '  end repeat\n'
            script += '  return paths\n'
            script += 'end tell'
            
            result = self._run_applescript_with_result(script)
            
            if result:
                paths = result.split(", ")
                return {"success": True, "paths": paths}
            else:
                return {"success": False, "error": "Failed to get selection"}
        
        elif command_name == "move_to_trash":
            paths = parameters.get("paths", [])
            
            if not paths:
                # Move current selection to trash
                script = 'tell application "Finder" to move selection to trash'
                success = self._run_applescript(script)
                return {"success": success}
            else:
                # Move specified paths to trash
                script = 'tell application "Finder"\n'
                script += '  set itemsToDelete to {}\n'
                
                for path in paths:
                    script += f'  set itemsToDelete to itemsToDelete & (POSIX file "{path}" as alias)\n'
                
                script += '  move itemsToDelete to trash\n'
                script += 'end tell'
                
                success = self._run_applescript(script)
                return {"success": success}
        
        else:
            # Fall back to universal commands
            return super().execute_command(command_name, parameters)


class SafariAdapter(AppleScriptAdapter):
    """Adapter for Safari web browser."""
    
    name = "Safari"
    bundle_ids = ["com.apple.Safari"]
    category = ApplicationCategory.BROWSER
    supported_features = [
        ApplicationFeature.WEB_BROWSING,
        ApplicationFeature.TEXT_SEARCH,
        ApplicationFeature.WINDOW_MANAGEMENT
    ]
    default_commands = [
        ApplicationCommand(
            name="open_url",
            description="Open a URL in Safari",
            parameters={"url": "URL to open"},
            shortcut=None,
            menu_path=None
        ),
        ApplicationCommand(
            name="get_current_url",
            description="Get the URL of the current page",
            shortcut=None,
            menu_path=None
        ),
        ApplicationCommand(
            name="search",
            description="Search for a term on the current page",
            parameters={"term": "Search term"},
            shortcut="Cmd+F",
            menu_path=["Edit", "Find", "Find..."]
        )
    ]
    
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a command in Safari.
        
        Args:
            command_name: Name of the command to execute
            parameters: Parameters for the command
            
        Returns:
            Dictionary with the result of the command execution
        """
        parameters = parameters or {}
        
        if command_name == "open_url":
            url = parameters.get("url", "")
            
            if not url:
                return {"success": False, "error": "URL parameter is required"}
            
            script = 'tell application "Safari"\n'
            script += '  activate\n'
            script += f'  open location "{url}"\n'
            script += 'end tell'
            
            success = self._run_applescript(script)
            return {"success": success}
        
        elif command_name == "get_current_url":
            script = 'tell application "Safari"\n'
            script += '  set currentURL to URL of current tab of front window\n'
            script += '  return currentURL\n'
            script += 'end tell'
            
            result = self._run_applescript_with_result(script)
            
            if result:
                return {"success": True, "url": result}
            else:
                return {"success": False, "error": "Failed to get current URL"}
        
        elif command_name == "search":
            term = parameters.get("term", "")
            
            if not term:
                return {"success": False, "error": "Search term parameter is required"}
            
            script = 'tell application "Safari"\n'
            script += '  activate\n'
            script += '  tell application "System Events"\n'
            script += '    keystroke "f" using command down\n'
            script += f'    keystroke "{term}"\n'
            script += '  end tell\n'
            script += 'end tell'
            
            success = self._run_applescript(script)
            return {"success": success}
        
        else:
            # Fall back to universal commands
            return super().execute_command(command_name, parameters)


class TerminalAdapter(AppleScriptAdapter):
    """Adapter for Terminal application."""
    
    name = "Terminal"
    bundle_ids = ["com.apple.Terminal"]
    category = ApplicationCategory.DEVELOPMENT
    supported_features = [
        ApplicationFeature.TERMINAL_COMMANDS,
        ApplicationFeature.WINDOW_MANAGEMENT
    ]
    default_commands = [
        ApplicationCommand(
            name="run_command",
            description="Run a shell command in Terminal",
            parameters={"command": "Command to run"},
            shortcut=None,
            menu_path=None
        ),
        ApplicationCommand(
            name="new_tab",
            description="Open a new tab",
            shortcut="Cmd+T",
            menu_path=["Shell", "New Tab", "Default"]
        )
    ]
    
    def execute_command(self, command_name: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a command in Terminal.
        
        Args:
            command_name: Name of the command to execute
            parameters: Parameters for the command
            
        Returns:
            Dictionary with the result of the command execution
        """
        parameters = parameters or {}
        
        if command_name == "run_command":
            command = parameters.get("command", "")
            
            if not command:
                return {"success": False, "error": "Command parameter is required"}
            
            # Escape double quotes in command
            safe_command = command.replace('"', '\\"')
            
            script = 'tell application "Terminal"\n'
            script += '  activate\n'
            script += '  do script "' + safe_command + '" in front window\n'
            script += 'end tell'
            
            success = self._run_applescript(script)
            return {"success": success}
        
        elif command_name == "new_tab":
            script = 'tell application "Terminal"\n'
            script += '  activate\n'
            script += '  tell application "System Events"\n'
            script += '    keystroke "t" using command down\n'
            script += '  end tell\n'
            script += 'end tell'
            
            success = self._run_applescript(script)
            return {"success": success}
        
        else:
            # Fall back to universal commands
            return super().execute_command(command_name, parameters)


class ApplicationAdapter:
    """
    Main adapter system that provides specialized handling for macOS applications.
    
    Features:
    - Specialized adapters for common applications
    - Fallback to universal adapter for unknown applications
    - Application-specific command execution
    - Extension system for custom adapters
    """
    
    def __init__(self, extensions_dir: Optional[Path] = None):
        """Initialize the application adapter system.
        
        Args:
            extensions_dir: Directory containing custom adapter extensions
        """
        # Register built-in adapters
        self._adapter_classes: Dict[str, Type[BaseAdapter]] = {
            "finder": FinderAdapter,
            "safari": SafariAdapter,
            "terminal": TerminalAdapter,
        }
        
        # Map bundle IDs to adapter classes
        self._bundle_id_map: Dict[str, str] = {}
        for adapter_name, adapter_class in self._adapter_classes.items():
            for bundle_id in adapter_class.bundle_ids:
                self._bundle_id_map[bundle_id] = adapter_name
        
        # Universal adapter for fallback
        self._universal_adapter = UniversalAdapter
        
        # Active adapter instances
        self._active_adapters: Dict[str, BaseAdapter] = {}
        
        # Load extensions if provided
        if extensions_dir:
            self._load_extensions(extensions_dir)
    
    def _load_extensions(self, extensions_dir: Path) -> None:
        """Load adapter extensions from a directory.
        
        Args:
            extensions_dir: Directory containing adapter extensions
        """
        if not extensions_dir.exists() or not extensions_dir.is_dir():
            logger.warning(f"Extensions directory not found: {extensions_dir}")
            return
        
        # Get all Python files in extensions directory
        extension_files = list(extensions_dir.glob("*.py"))
        
        for ext_file in extension_files:
            try:
                # Dynamically import the extension module
                module_name = ext_file.stem
                sys.path.insert(0, str(extensions_dir.parent))
                module = importlib.import_module(f"{extensions_dir.name}.{module_name}")
                
                # Find adapter classes in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    
                    # Check if this is an adapter class
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseAdapter) and 
                        attr is not BaseAdapter and
                        attr is not AppleScriptAdapter):
                        
                        # Register the adapter
                        adapter_name = attr_name.lower().replace("adapter", "")
                        self._adapter_classes[adapter_name] = attr
                        
                        # Map bundle IDs to this adapter
                        for bundle_id in attr.bundle_ids:
                            self._bundle_id_map[bundle_id] = adapter_name
                        
                        logger.info(f"Loaded extension adapter: {attr_name}")
                
            except Exception as e:
                logger.error(f"Failed to load extension {ext_file.name}: {e}")
    
    def get_adapter_for_app(self, app_name: str) -> BaseAdapter:
        """Get an adapter for an application by name.
        
        Args:
            app_name: Application name
            
        Returns:
            Appropriate adapter for the application
        """
        # Check if we already have an active adapter for this app
        if app_name.lower() in self._active_adapters:
            return self._active_adapters[app_name.lower()]
        
        # Try to find a matching adapter
        for adapter_name, adapter_class in self._adapter_classes.items():
            if adapter_name == app_name.lower() or adapter_class.name.lower() == app_name.lower():
                adapter = adapter_class()
                self._active_adapters[app_name.lower()] = adapter
                return adapter
        
        # Fall back to universal adapter
        logger.info(f"Using universal adapter for {app_name}")
        adapter = self._universal_adapter()
        self._active_adapters[app_name.lower()] = adapter
        return adapter
    
    def get_adapter_for_bundle_id(self, bundle_id: str) -> BaseAdapter:
        """Get an adapter for an application by bundle ID.
        
        Args:
            bundle_id: Application bundle ID
            
        Returns:
            Appropriate adapter for the application
        """
        # Check if we already have an active adapter for this bundle ID
        if bundle_id in self._active_adapters:
            return self._active_adapters[bundle_id]
        
        # Check if we have a registered adapter for this bundle ID
        if bundle_id in self._bundle_id_map:
            adapter_name = self._bundle_id_map[bundle_id]
            adapter_class = self._adapter_classes[adapter_name]
            adapter = adapter_class(bundle_id)
            self._active_adapters[bundle_id] = adapter
            return adapter
        
        # Fall back to universal adapter
        logger.info(f"Using universal adapter for bundle ID: {bundle_id}")
        adapter = self._universal_adapter(bundle_id)
        self._active_adapters[bundle_id] = adapter
        return adapter
    
    def register_adapter(self, adapter_class: Type[BaseAdapter]) -> None:
        """Register a custom adapter class.
        
        Args:
            adapter_class: Adapter class to register
        """
        adapter_name = adapter_class.__name__.lower().replace("adapter", "")
        self._adapter_classes[adapter_name] = adapter_class
        
        # Map bundle IDs to this adapter
        for bundle_id in adapter_class.bundle_ids:
            self._bundle_id_map[bundle_id] = adapter_name
        
        logger.info(f"Registered adapter: {adapter_class.__name__}")
    
    def execute_app_command(self, app_identifier: str, command: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a command in an application.
        
        Args:
            app_identifier: Application name or bundle ID
            command: Command to execute
            parameters: Command parameters
            
        Returns:
            Dictionary with the result of the command execution
        """
        # Determine if app_identifier is a name or bundle ID
        if "." in app_identifier:  # Likely a bundle ID
            adapter = self.get_adapter_for_bundle_id(app_identifier)
        else:  # Likely an app name
            adapter = self.get_adapter_for_app(app_identifier)
        
        # Check if the command is supported
        if command not in adapter.get_commands():
            return {"success": False, "error": f"Command {command} not supported by {adapter.name}"}
        
        # Execute the command
        return adapter.execute_command(command, parameters)
    
    def get_supported_apps(self) -> List[Dict[str, Any]]:
        """Get information about all supported applications.
        
        Returns:
            List of dictionaries with information about supported applications
        """
        supported_apps = []
        
        for adapter_name, adapter_class in self._adapter_classes.items():
            app_info = {
                "name": adapter_class.name,
                "adapter": adapter_name,
                "bundle_ids": adapter_class.bundle_ids,
                "category": adapter_class.category.value,
                "features": [feature.value for feature in adapter_class.supported_features],
                "commands": [cmd.name for cmd in adapter_class.default_commands]
            }
            supported_apps.append(app_info)
        
        return supported_apps
    
    def get_app_commands(self, app_identifier: str) -> Dict[str, Dict[str, Any]]:
        """Get all available commands for an application.
        
        Args:
            app_identifier: Application name or bundle ID
            
        Returns:
            Dictionary of command information
        """
        # Get the appropriate adapter
        if "." in app_identifier:  # Likely a bundle ID
            adapter = self.get_adapter_for_bundle_id(app_identifier)
        else:  # Likely an app name
            adapter = self.get_adapter_for_app(app_identifier)
        
        # Get commands from the adapter
        commands = adapter.get_commands()
        
        # Convert to dictionary of command info
        command_info = {}
        for cmd_name, cmd in commands.items():
            command_info[cmd_name] = {
                "description": cmd.description,
                "parameters": cmd.parameters,
                "shortcut": cmd.shortcut,
                "menu_path": cmd.menu_path
            }
        
        return command_info


# Example of usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create the adapter system
    adapter_system = ApplicationAdapter()
    
    # Get a Finder adapter
    finder_adapter = adapter_system.get_adapter_for_app("Finder")
    print(f"Got adapter: {finder_adapter.name}")
    
    # Execute a command
    result = adapter_system.execute_app_command(
        "Finder", 
        "get_selection", 
        {}
    )
    print(f"Command result: {result}")
    
    # Get all supported applications
    supported_apps = adapter_system.get_supported_apps()
    print(f"Supported applications: {len(supported_apps)}")
    for app in supported_apps:
        print(f"- {app['name']} ({app['adapter']})")
        print(f"  Commands: {', '.join(app['commands'])}")
    
    # Get commands for an app
    safari_commands = adapter_system.get_app_commands("safari")
    print("Safari commands:")
    for cmd_name, cmd_info in safari_commands.items():
        print(f"- {cmd_name}: {cmd_info['description']}")
        if cmd_info['shortcut']:
            print(f"  Shortcut: {cmd_info['shortcut']}")
