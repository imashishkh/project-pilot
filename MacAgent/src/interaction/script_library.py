"""
Script Library Module for MacAgent

This module provides a library of AppleScript snippets organized by application and function,
with support for parameterized templates and version control.
"""

import os
import re
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScriptLibrary:
    """
    A library of AppleScript snippets organized by application and function.
    
    This class provides access to a collection of useful AppleScript snippets,
    supports parameterized templates, and includes version control for scripts.
    """
    
    # Default scripts included with the library
    DEFAULT_SCRIPTS = {
        "system": {
            "get_frontmost_app": {
                "description": "Get the name of the frontmost application",
                "version": "1.0",
                "script": """
                tell application "System Events"
                    set frontApp to first application process whose frontmost is true
                    set frontAppName to name of frontApp
                    return frontAppName
                end tell
                """
            },
            "get_running_apps": {
                "description": "Get a list of running applications",
                "version": "1.0",
                "script": """
                tell application "System Events"
                    set appList to name of every application process whose background only is false
                    return appList
                end tell
                """
            },
            "get_screen_resolution": {
                "description": "Get the screen resolution",
                "version": "1.0",
                "script": """
                tell application "Finder"
                    set screenSize to bounds of window of desktop
                    return {item 3 of screenSize, item 4 of screenSize}
                end tell
                """
            },
            "show_notification": {
                "description": "Show a notification",
                "version": "1.0",
                "script": """
                display notification "$message$" with title "$title$"
                """
            },
            "show_dialog": {
                "description": "Show a dialog box with buttons",
                "version": "1.1",
                "script": """
                set dialogResult to display dialog "$message$" buttons {"$button1$",$button2_opt$} default button 1 with title "$title$"
                return button returned of dialogResult
                """,
                "parameters": {
                    "message": "Dialog message text",
                    "title": "Dialog title",
                    "button1": "Primary button text",
                    "button2_opt": "Optional second button"
                },
                "examples": [
                    {"message": "Do you want to continue?", "title": "Confirmation", "button1": "Yes", "button2_opt": "\"No\""}
                ]
            }
        },
        "finder": {
            "get_desktop_path": {
                "description": "Get the path to the desktop",
                "version": "1.0",
                "script": """
                tell application "Finder"
                    return POSIX path of (desktop as string)
                end tell
                """
            },
            "get_selected_files": {
                "description": "Get the selected files in Finder",
                "version": "1.0",
                "script": """
                tell application "Finder"
                    set selectedItems to selection
                    set fileList to {}
                    repeat with itemRef in selectedItems
                        set end of fileList to POSIX path of (itemRef as string)
                    end repeat
                    return fileList
                end tell
                """
            },
            "create_folder": {
                "description": "Create a new folder",
                "version": "1.0",
                "script": """
                tell application "Finder"
                    make new folder at (POSIX file "$path$") with properties {name:"$folder_name$"}
                    return true
                end tell
                """,
                "parameters": {
                    "path": "Path where the folder should be created",
                    "folder_name": "Name of the new folder"
                }
            }
        },
        "safari": {
            "get_url": {
                "description": "Get the URL of the current page in Safari",
                "version": "1.0",
                "script": """
                tell application "Safari"
                    return URL of current tab of window 1
                end tell
                """
            },
            "get_page_title": {
                "description": "Get the title of the current page in Safari",
                "version": "1.0",
                "script": """
                tell application "Safari"
                    return name of current tab of window 1
                end tell
                """
            },
            "open_url": {
                "description": "Open a URL in Safari",
                "version": "1.0",
                "script": """
                tell application "Safari"
                    open location "$url$"
                end tell
                """,
                "parameters": {
                    "url": "URL to open"
                }
            }
        },
        "chrome": {
            "get_url": {
                "description": "Get the URL of the current page in Chrome",
                "version": "1.0",
                "script": """
                tell application "Google Chrome"
                    return URL of active tab of window 1
                end tell
                """
            },
            "get_page_title": {
                "description": "Get the title of the current page in Chrome",
                "version": "1.0",
                "script": """
                tell application "Google Chrome"
                    return title of active tab of window 1
                end tell
                """
            },
            "open_url": {
                "description": "Open a URL in Chrome",
                "version": "1.0",
                "script": """
                tell application "Google Chrome"
                    open location "$url$"
                end tell
                """,
                "parameters": {
                    "url": "URL to open"
                }
            }
        },
        "terminal": {
            "run_command": {
                "description": "Run a command in Terminal",
                "version": "1.0",
                "script": """
                tell application "Terminal"
                    do script "$command$" in window 1
                end tell
                """,
                "parameters": {
                    "command": "Command to run"
                }
            },
            "new_window": {
                "description": "Open a new Terminal window",
                "version": "1.0",
                "script": """
                tell application "Terminal"
                    do script ""
                end tell
                """
            }
        },
        "mail": {
            "send_email": {
                "description": "Send an email",
                "version": "1.0",
                "script": """
                tell application "Mail"
                    set newMessage to make new outgoing message with properties {subject:"$subject$", content:"$body$", visible:true}
                    tell newMessage
                        make new to recipient at end of to recipients with properties {address:"$to_address$"}
                    end tell
                    send newMessage
                end tell
                """,
                "parameters": {
                    "to_address": "Recipient email address",
                    "subject": "Email subject",
                    "body": "Email body text"
                }
            }
        },
        "calendar": {
            "create_event": {
                "description": "Create a new calendar event",
                "version": "1.0",
                "script": """
                tell application "Calendar"
                    tell calendar "$calendar_name$"
                        make new event with properties {summary:"$event_title$", start date:date "$start_date$", end date:date "$end_date$", description:"$description$"}
                    end tell
                end tell
                """,
                "parameters": {
                    "calendar_name": "Name of the calendar",
                    "event_title": "Title of the event",
                    "start_date": "Start date/time (e.g., 'Friday, June 5, 2023 at 9:00:00 AM')",
                    "end_date": "End date/time",
                    "description": "Event description"
                }
            }
        }
    }
    
    def __init__(self, 
                 custom_script_path: Optional[str] = None,
                 load_defaults: bool = True):
        """
        Initialize the script library.
        
        Args:
            custom_script_path: Path to a JSON file with custom scripts
            load_defaults: Whether to load the default scripts
        """
        # Initialize the script collection
        self.scripts = {}
        
        # Load default scripts if requested
        if load_defaults:
            self.scripts.update(self.DEFAULT_SCRIPTS)
            logger.debug("Loaded default scripts")
        
        # Load custom scripts if path provided
        if custom_script_path and os.path.exists(custom_script_path):
            self.load_scripts_from_file(custom_script_path)
        
        logger.info(f"ScriptLibrary initialized with {self.count_scripts()} scripts")
    
    def count_scripts(self) -> int:
        """
        Count the total number of scripts in the library.
        
        Returns:
            Total number of scripts
        """
        count = 0
        for app in self.scripts:
            count += len(self.scripts[app])
        return count
    
    def get_script(self, app: str, script_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a script by application and name.
        
        Args:
            app: Application category
            script_name: Script name
            
        Returns:
            Script information dictionary or None if not found
        """
        # Ensure app exists in library
        if app not in self.scripts:
            logger.warning(f"Application category not found: {app}")
            return None
        
        # Ensure script exists for app
        if script_name not in self.scripts[app]:
            logger.warning(f"Script not found for {app}: {script_name}")
            return None
        
        # Return the script info
        return self.scripts[app][script_name]
    
    def get_script_content(self, app: str, script_name: str) -> Optional[str]:
        """
        Get the raw script content by application and name.
        
        Args:
            app: Application name
            script_name: Script name
            
        Returns:
            Script content or None if not found
        """
        script_data = self.get_script(app, script_name)
        return script_data["script"].strip() if script_data else None
    
    def get_parameters_for_script(self, app: str, script_name: str) -> Dict[str, str]:
        """
        Get the parameters for a script.
        
        Args:
            app: Application name
            script_name: Script name
            
        Returns:
            Dictionary of parameter names and descriptions
        """
        script_data = self.get_script(app, script_name)
        if script_data and "parameters" in script_data:
            return script_data["parameters"]
        return {}
    
    def render_script(self, 
                     app: str, 
                     script_name: str, 
                     parameters: Dict[str, Any] = None) -> Optional[str]:
        """
        Render a script template with the given parameters.
        
        Args:
            app: Application name
            script_name: Script name
            parameters: Dictionary of parameters to substitute
            
        Returns:
            Rendered script or None if script not found
        """
        script_info = self.get_script(app, script_name)
        if not script_info:
            logger.warning(f"Script not found: {app}/{script_name}")
            return None
            
        script_content = script_info.get('script', '')
        
        # Add AppleScript preamble to prevent syntax issues
        script_preamble = """
use scripting additions
use framework "Foundation"
use framework "AppKit"

# Initialize variables to prevent "variable not defined" errors
set roundedList to {}
"""
        script_content = script_preamble + script_content
        
        # If no parameters, return the script as is
        if not parameters:
            return script_content
            
        # Check required parameters
        required_params = script_info.get('parameters', {})
        missing_params = [p for p in required_params if p not in parameters]
        
        if missing_params:
            logger.warning(f"Missing required parameters for script {app}/{script_name}: {missing_params}")
            # We'll still proceed but with missing placeholders
        
        # Make a copy of the script content
        rendered_script = script_content
        
        # Replace all parameters
        for param_name, param_value in parameters.items():
            placeholder = f"${param_name}$"
            
            # Convert param_value to appropriate AppleScript format
            if param_value is None:
                as_value = "missing value"
            elif isinstance(param_value, bool):
                as_value = "true" if param_value else "false"
            elif isinstance(param_value, (int, float)):
                as_value = str(param_value)
            elif isinstance(param_value, list):
                # Ensure proper list comparison when nested lists are involved
                if len(param_value) == 1 and isinstance(param_value[0], list):
                    inner_items = [str(item) for item in param_value[0]]
                    as_value = "{{" + ", ".join(inner_items) + "}}"
                else:
                    items = [str(item) for item in param_value]
                    as_value = "{" + ", ".join(items) + "}"
            elif isinstance(param_value, str):
                # If it's already an AppleScript expression, use it directly
                if param_value.startswith('"') and param_value.endswith('"'):
                    as_value = param_value
                else:
                    # Escape quotes for AppleScript
                    escaped_value = param_value.replace('"', '\\"')
                    as_value = f'"{escaped_value}"'
            else:
                # Convert anything else to string
                as_value = f'"{str(param_value)}"'
            
            # Replace all occurrences of the placeholder
            rendered_script = rendered_script.replace(placeholder, as_value)
        
        return rendered_script
    
    def add_script(self, 
                  app: str, 
                  script_name: str, 
                  script_content: str,
                  description: str = "",
                  version: str = "1.0",
                  parameters: Dict[str, str] = None,
                  examples: List[Dict[str, Any]] = None) -> bool:
        """
        Add a new script to the library.
        
        Args:
            app: Application name
            script_name: Script name
            script_content: AppleScript code
            description: Description of the script
            version: Version string
            parameters: Dictionary of parameter names and descriptions
            examples: List of example parameter sets
            
        Returns:
            True if added successfully, False if it exists
        """
        app = app.lower()
        
        # Check if the script already exists
        if app in self.scripts and script_name in self.scripts[app]:
            logger.warning(f"Script already exists: {app}/{script_name}")
            return False
        
        # Ensure the app category exists
        if app not in self.scripts:
            self.scripts[app] = {}
        
        # Create the script entry
        script_data = {
            "description": description,
            "version": version,
            "script": script_content,
            "created": time.time()
        }
        
        # Add parameters if provided
        if parameters:
            script_data["parameters"] = parameters
        
        # Add examples if provided
        if examples:
            script_data["examples"] = examples
        
        # Add to the library
        self.scripts[app][script_name] = script_data
        
        logger.info(f"Added script: {app}/{script_name} (v{version})")
        return True
    
    def update_script(self, 
                     app: str, 
                     script_name: str, 
                     script_content: str = None,
                     description: str = None,
                     version: str = None,
                     parameters: Dict[str, str] = None,
                     examples: List[Dict[str, Any]] = None) -> bool:
        """
        Update an existing script in the library.
        
        Args:
            app: Application name
            script_name: Script name
            script_content: New AppleScript code (None to keep existing)
            description: New description (None to keep existing)
            version: New version string (None to keep existing)
            parameters: New parameters (None to keep existing)
            examples: New examples (None to keep existing)
            
        Returns:
            True if updated successfully, False if not found
        """
        app = app.lower()
        
        # Check if the script exists
        if app not in self.scripts or script_name not in self.scripts[app]:
            logger.warning(f"Script not found for update: {app}/{script_name}")
            return False
        
        # Get the existing script data
        script_data = self.scripts[app][script_name]
        
        # Store the old version for history
        old_version = script_data.get("version", "1.0")
        
        # Update fields if provided
        if script_content is not None:
            script_data["script"] = script_content
        
        if description is not None:
            script_data["description"] = description
        
        if version is not None:
            script_data["version"] = version
        else:
            # Auto-increment version if content changed but version not specified
            if script_content is not None:
                version_parts = old_version.split('.')
                if len(version_parts) >= 2 and version_parts[-1].isdigit():
                    minor = int(version_parts[-1]) + 1
                    version_parts[-1] = str(minor)
                    script_data["version"] = '.'.join(version_parts)
        
        if parameters is not None:
            script_data["parameters"] = parameters
        
        if examples is not None:
            script_data["examples"] = examples
        
        # Add update timestamp
        script_data["updated"] = time.time()
        
        # Add history entry if version changed
        if "version" in script_data and script_data["version"] != old_version:
            if "history" not in script_data:
                script_data["history"] = []
            
            script_data["history"].append({
                "version": old_version,
                "timestamp": time.time()
            })
        
        logger.info(f"Updated script: {app}/{script_name} (v{script_data['version']})")
        return True
    
    def remove_script(self, app: str, script_name: str) -> bool:
        """
        Remove a script from the library.
        
        Args:
            app: Application name
            script_name: Script name
            
        Returns:
            True if removed successfully, False if not found
        """
        app = app.lower()
        
        # Check if the script exists
        if app not in self.scripts or script_name not in self.scripts[app]:
            logger.warning(f"Script not found for removal: {app}/{script_name}")
            return False
        
        # Remove the script
        del self.scripts[app][script_name]
        
        # Remove the app category if empty
        if not self.scripts[app]:
            del self.scripts[app]
        
        logger.info(f"Removed script: {app}/{script_name}")
        return True
    
    def list_applications(self) -> List[str]:
        """
        Get a list of all application categories.
        
        Returns:
            List of application names
        """
        return sorted(list(self.scripts.keys()))
    
    def list_scripts_for_app(self, app: str) -> List[Dict[str, Any]]:
        """
        Get a list of all scripts for an application.
        
        Args:
            app: Application name
            
        Returns:
            List of script metadata dictionaries
        """
        app = app.lower()
        
        if app not in self.scripts:
            return []
        
        result = []
        for script_name, script_data in self.scripts[app].items():
            result.append({
                "name": script_name,
                "description": script_data.get("description", ""),
                "version": script_data.get("version", "1.0"),
                "has_parameters": "parameters" in script_data
            })
        
        return sorted(result, key=lambda x: x["name"])
    
    def search_scripts(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for scripts by keyword.
        
        Args:
            query: Search query
            
        Returns:
            List of matching script metadata
        """
        query = query.lower()
        results = []
        
        for app in self.scripts:
            for script_name, script_data in self.scripts[app].items():
                # Search in name, description, and content
                if (query in script_name.lower() or 
                    query in script_data.get("description", "").lower() or 
                    query in script_data.get("script", "").lower()):
                    
                    results.append({
                        "app": app,
                        "name": script_name,
                        "description": script_data.get("description", ""),
                        "version": script_data.get("version", "1.0")
                    })
        
        return results
    
    def save_scripts_to_file(self, file_path: str) -> bool:
        """
        Save the script library to a JSON file.
        
        Args:
            file_path: Path to the output file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.scripts, f, indent=2)
            
            logger.info(f"Saved script library to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving scripts to file: {e}")
            return False
    
    def load_scripts_from_file(self, file_path: str) -> bool:
        """
        Load scripts from a JSON file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                loaded_scripts = json.load(f)
            
            # Validate the loaded data
            if not isinstance(loaded_scripts, dict):
                logger.error(f"Invalid script file format: {file_path}")
                return False
            
            # Merge with existing scripts
            for app, scripts in loaded_scripts.items():
                if app not in self.scripts:
                    self.scripts[app] = {}
                
                for script_name, script_data in scripts.items():
                    self.scripts[app][script_name] = script_data
            
            logger.info(f"Loaded scripts from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading scripts from file: {e}")
            return False
    
    def get_version_history(self, app: str, script_name: str) -> List[Dict[str, Any]]:
        """
        Get the version history for a script.
        
        Args:
            app: Application name
            script_name: Script name
            
        Returns:
            List of version history entries
        """
        script_data = self.get_script(app, script_name)
        if not script_data or "history" not in script_data:
            return []
        
        # Include current version
        history = script_data["history"].copy()
        history.append({
            "version": script_data["version"],
            "timestamp": script_data.get("updated", script_data.get("created", time.time())),
            "current": True
        })
        
        # Sort by timestamp (newest first)
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)
