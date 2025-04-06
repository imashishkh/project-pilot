"""
Application Controller Module for MacAgent

This module provides high-level control of Mac applications using AppleScript,
with capabilities for launching, quitting, switching, and managing applications.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .applescript_bridge import AppleScriptBridge
from .script_library import ScriptLibrary

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ApplicationController:
    """
    Controller for Mac applications using AppleScript.
    
    This class provides methods for launching, quitting, switching between applications,
    retrieving application state, and controlling application windows.
    """
    
    # Basic application control scripts
    APP_SCRIPTS = {
        "launch_app": """
            tell application "$app_name$"
                launch
                $activate_opt$
            end tell
        """,
        "quit_app": """
            tell application "$app_name$"
                quit
            end tell
        """,
        "activate_app": """
            tell application "$app_name$"
                activate
            end tell
        """,
        "is_running": """
            tell application "System Events"
                return (count of (every process whose name is "$app_name$")) > 0
            end tell
        """,
        "get_app_info": """
            tell application "System Events"
                if exists process "$app_name$" then
                    tell process "$app_name$"
                        set appInfo to {
                            name:name,
                            frontmost:frontmost,
                            visible:visible
                        }
                        return appInfo
                    end tell
                else
                    return {name:"$app_name$", frontmost:false, visible:false}
                end if
            end tell
        """,
        "get_window_list": """
            tell application "System Events"
                if exists process "$app_name$" then
                    tell process "$app_name$"
                        set windowList to {}
                        repeat with w in windows
                            set windowInfo to {
                                title:name,
                                id:id,
                                position:{position},
                                size:{size},
                                minimized:minimized,
                                index:index
                            }
                            set end of windowList to windowInfo
                        end repeat
                        return windowList
                    end tell
                else
                    return {}
                end if
            end tell
        """,
        "get_frontmost_window": """
            tell application "System Events"
                if exists process "$app_name$" then
                    tell process "$app_name$"
                        if (count of windows) > 0 then
                            set w to window 1
                            set windowInfo to {
                                title:name,
                                id:id,
                                position:{position},
                                size:{size},
                                minimized:minimized,
                                index:index
                            }
                            return windowInfo
                        else
                            return missing value
                        end if
                    end tell
                else
                    return missing value
                end if
            end tell
        """,
        "move_window": """
            tell application "System Events"
                tell process "$app_name$"
                    set position of window $window_id$ to {$x$, $y$}
                end tell
            end tell
        """,
        "resize_window": """
            tell application "System Events"
                tell process "$app_name$"
                    set size of window $window_id$ to {$width$, $height$}
                end tell
            end tell
        """,
        "minimize_window": """
            tell application "System Events"
                tell process "$app_name$"
                    set minimized of window $window_id$ to true
                end tell
            end tell
        """,
        "unminimize_window": """
            tell application "System Events"
                tell process "$app_name$"
                    set minimized of window $window_id$ to false
                end tell
            end tell
        """,
        "close_window": """
            tell application "System Events"
                tell process "$app_name$"
                    click button 1 of window $window_id$
                end tell
            end tell
        """,
        "get_menu_items": """
            tell application "System Events"
                tell process "$app_name$"
                    set menuList to {}
                    repeat with m in menu bars
                        repeat with mi in menu bar items of m
                            set menuName to name of mi
                            set menuEntry to {
                                title:menuName,
                                items:{}
                            }
                            
                            try
                                tell mi
                                    repeat with submi in menu items
                                        set itemName to name of submi
                                        if itemName is not "" then
                                            set end of items of menuEntry to itemName
                                        end if
                                    end repeat
                                end tell
                            end try
                            
                            set end of menuList to menuEntry
                        end repeat
                    end repeat
                    return menuList
                end tell
            end tell
        """,
        "select_menu_item": """
            tell application "System Events"
                tell process "$app_name$"
                    tell menu bar 1
                        tell menu bar item "$menu_name$"
                            tell menu 1
                                click menu item "$item_name$"
                            end tell
                        end tell
                    end tell
                end tell
            end tell
        """,
        "get_running_apps": """
            tell application "System Events"
                set appList to name of every application process whose background only is false
                return appList
            end tell
        """,
        "get_frontmost_app": """
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set frontAppName to name of frontApp
                return frontAppName
            end tell
        """,
        "take_screenshot": """
            set region_param to "$region_opt$"
            set file_param to "$file_opt$"
            
            set cmd to "screencapture "
            
            if region_param is not "" then
                set cmd to cmd & region_param & " "
            end if
            
            if file_param is not "" then
                set cmd to cmd & file_param
            else
                set cmd to cmd & "-c"
            end if
            
            try
                do shell script cmd
                return true
            on error errMsg
                log errMsg
                return false
            end try
        """
    }
    
    def __init__(self, 
                 bridge: AppleScriptBridge = None,
                 script_library: ScriptLibrary = None,
                 default_timeout: float = 10.0):
        """
        Initialize the application controller.
        
        Args:
            bridge: AppleScriptBridge instance or None to create a new one
            script_library: ScriptLibrary instance or None to create a new one
            default_timeout: Default timeout for operations in seconds
        """
        # Create bridge and script library if not provided
        self.bridge = bridge if bridge else AppleScriptBridge(default_timeout=default_timeout)
        self.script_library = script_library if script_library else ScriptLibrary()
        
        self.default_timeout = default_timeout
        
        # Add application control scripts to the script library
        self._register_app_scripts()
        
        # Cache for application information
        self.app_cache = {}
        self.cache_expiry = 5.0  # Cache expiry time in seconds
        
        logger.info(f"ApplicationController initialized with timeout={default_timeout}s")
    
    def _register_app_scripts(self):
        """Register application control scripts with the script library."""
        app_category = "app_controller"
        
        for script_name, script_content in self.APP_SCRIPTS.items():
            # Check if the script already exists in the library
            existing = self.script_library.get_script(app_category, script_name)
            if not existing:
                # Add the script to the library
                self.script_library.add_script(
                    app_category,
                    script_name,
                    script_content,
                    description=f"Application control: {script_name}"
                )
    
    def _run_app_script(self, 
                       script_name: str, 
                       parameters: Dict[str, Any] = None, 
                       timeout: float = None) -> Tuple[bool, Any]:
        """
        Run an application control script.
        
        Args:
            script_name: Name of the script
            parameters: Script parameters
            timeout: Timeout in seconds (None to use default)
            
        Returns:
            Tuple of (success, result)
        """
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout
            
        # Initialize parameters if None
        if parameters is None:
            parameters = {}
            
        # Check for required parameters based on script name
        required_params = {
            "is_running": ["app_name"],
            "launch_app": ["app_name"],
            "quit_app": ["app_name"],
            "activate_app": ["app_name"],
            "get_app_info": ["app_name"],
            "get_window_list": ["app_name"],
            "get_frontmost_window": ["app_name"],
            "move_window": ["app_name", "window_id", "x", "y"],
            "resize_window": ["app_name", "window_id", "width", "height"],
            "minimize_window": ["app_name", "window_id"],
            "unminimize_window": ["app_name", "window_id"],
            "close_window": ["app_name", "window_id"],
            "select_menu_item": ["app_name", "menu_name", "item_name"]
        }
        
        # Check if this script has required parameters
        if script_name in required_params:
            # Verify all required parameters are provided
            missing_params = []
            for param in required_params[script_name]:
                if param not in parameters:
                    missing_params.append(param)
                    
            if missing_params:
                error_msg = f"Missing required parameter(s) for script {script_name}: {', '.join(missing_params)}"
                logger.error(error_msg)
                return False, error_msg
        
        # Render the script
        script = self.script_library.render_script("app_controller", script_name, parameters)
        if not script:
            error_msg = f"Script not found: app_controller/{script_name}"
            logger.error(error_msg)
            return False, error_msg
        
        # Run the script
        return self.bridge.run_script(script, timeout=timeout)
    
    def launch_application(self, 
                          app_name: str, 
                          activate: bool = True, 
                          wait_timeout: float = 5.0) -> bool:
        """
        Launch an application.
        
        Args:
            app_name: Name of the application
            activate: Whether to bring the application to the front
            wait_timeout: Time to wait for the app to launch
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is already running
        is_running, running = self._run_app_script("is_running", {"app_name": app_name})
        if is_running and running:
            logger.debug(f"Application already running: {app_name}")
            
            # If requested, activate the app
            if activate:
                return self.activate_application(app_name)
            
            return True
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "activate_opt": "activate" if activate else ""
        }
        
        # Launch the app
        success, result = self._run_app_script("launch_app", params)
        if not success:
            logger.error(f"Failed to launch application: {app_name} - {result}")
            return False
        
        # Wait for the application to launch
        if wait_timeout > 0:
            start_time = time.time()
            while time.time() - start_time < wait_timeout:
                # Check if the app is now running
                is_running, running = self._run_app_script("is_running", {"app_name": app_name})
                if is_running and running:
                    logger.debug(f"Application launched: {app_name}")
                    return True
                
                # Wait a bit before checking again
                time.sleep(0.5)
            
            logger.warning(f"Timed out waiting for application to launch: {app_name}")
            return False
        
        return True
    
    def quit_application(self, 
                        app_name: str, 
                        force: bool = False,
                        wait_timeout: float = 5.0) -> bool:
        """
        Quit an application.
        
        Args:
            app_name: Name of the application
            force: Whether to force quit if normal quit fails
            wait_timeout: Time to wait for the app to quit
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        is_running, running = self._run_app_script("is_running", {"app_name": app_name})
        if not (is_running and running):
            logger.debug(f"Application not running: {app_name}")
            return True
        
        # Prepare script parameters
        params = {"app_name": app_name}
        
        # Quit the app
        success, result = self._run_app_script("quit_app", params)
        
        # Wait for the application to quit
        if wait_timeout > 0:
            start_time = time.time()
            while time.time() - start_time < wait_timeout:
                # Check if the app is still running
                is_running, running = self._run_app_script("is_running", {"app_name": app_name})
                if is_running and not running:
                    logger.debug(f"Application quit: {app_name}")
                    
                    # Clear app cache
                    if app_name in self.app_cache:
                        del self.app_cache[app_name]
                    
                    return True
                
                # Wait a bit before checking again
                time.sleep(0.5)
            
            logger.warning(f"Timed out waiting for application to quit: {app_name}")
            
            # Force quit if requested and normal quit failed
            if force:
                logger.info(f"Force quitting application: {app_name}")
                
                # Force quit using System Events
                force_quit_script = """
                    tell application "System Events"
                        set appList to every process whose name is "$app_name$"
                        repeat with thisApp in appList
                            do shell script "kill -9 " & (unix id of thisApp as text)
                        end repeat
                    end tell
                """
                
                force_success, _ = self.bridge.run_script(
                    force_quit_script.replace("$app_name$", app_name)
                )
                
                # Clear app cache
                if app_name in self.app_cache:
                    del self.app_cache[app_name]
                
                return force_success
            
            return False
        
        return success
    
    def activate_application(self, app_name: str) -> bool:
        """
        Activate (bring to front) an application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        is_running, running = self._run_app_script("is_running", {"app_name": app_name})
        if not (is_running and running):
            logger.warning(f"Cannot activate: Application not running: {app_name}")
            return False
        
        # Prepare script parameters
        params = {"app_name": app_name}
        
        # Activate the app
        success, result = self._run_app_script("activate_app", params)
        if not success:
            logger.error(f"Failed to activate application: {app_name} - {result}")
            return False
        
        logger.debug(f"Application activated: {app_name}")
        return True
    
    def is_application_running(self, app_name: str) -> bool:
        """
        Check if an application is running.
        
        Args:
            app_name: Name of the application
            
        Returns:
            True if running, False otherwise
        """
        if not app_name:
            logger.error("Application name cannot be empty")
            return False
            
        # Use cached info if available and fresh
        if app_name in self.app_cache:
            cache_entry = self.app_cache[app_name]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry:
                return cache_entry.get("running", False)
        
        # Get fresh info
        is_running, running = self._run_app_script("is_running", {"app_name": app_name})
        
        # Update cache
        if app_name not in self.app_cache:
            self.app_cache[app_name] = {}
            
        self.app_cache[app_name]["running"] = running if is_running else False
        self.app_cache[app_name]["timestamp"] = time.time()
        
        logger.debug(f"Application {app_name} running status: {running if is_running else False}")
        return running if is_running else False
    
    def get_application_info(self, app_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get information about an application.
        
        Args:
            app_name: Name of the application
            use_cache: Whether to use cached information if available
            
        Returns:
            Dictionary of application information or None if not running
        """
        # Use cached info if requested and available
        if use_cache and app_name in self.app_cache:
            cache_entry = self.app_cache[app_name]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry and "info" in cache_entry:
                return cache_entry["info"]
        
        # Check if the app is running
        if not self.is_application_running(app_name):
            logger.debug(f"Cannot get info: Application not running: {app_name}")
            return None
        
        # Get app info
        params = {"app_name": app_name}
        success, result = self._run_app_script("get_app_info", params)
        
        if not success:
            logger.error(f"Failed to get info for application: {app_name}")
            return None
            
        # Ensure we got valid data
        if not result or not isinstance(result, dict):
            logger.error(f"Invalid application info for {app_name}: {result}")
            # Return minimal information
            result = {
                "name": app_name,
                "frontmost": False,
                "visible": False
            }
        
        # Update cache
        if app_name not in self.app_cache:
            self.app_cache[app_name] = {}
            
        self.app_cache[app_name]["info"] = result
        self.app_cache[app_name]["timestamp"] = time.time()
        
        return result
    
    def get_application_windows(self, 
                               app_name: str, 
                               use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get a list of windows for an application.
        
        Args:
            app_name: Name of the application
            use_cache: Whether to use cached information if available
            
        Returns:
            List of window information dictionaries
        """
        # Use cached info if requested and available
        if use_cache and app_name in self.app_cache:
            cache_entry = self.app_cache[app_name]
            if time.time() - cache_entry["timestamp"] < self.cache_expiry and "windows" in cache_entry:
                return cache_entry["windows"]
        
        # Check if the app is running
        if not self.is_application_running(app_name):
            return []
        
        # Get window list
        params = {"app_name": app_name}
        success, result = self._run_app_script("get_window_list", params)
        
        if not success or not result:
            logger.error(f"Failed to get windows for application: {app_name}")
            return []
        
        # Update cache
        self.app_cache.setdefault(app_name, {})
        self.app_cache[app_name]["windows"] = result
        self.app_cache[app_name]["timestamp"] = time.time()
        
        return result
    
    def get_frontmost_window(self, app_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the frontmost window of an application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            Window information dictionary or None if not found
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return None
        
        # Get frontmost window
        params = {"app_name": app_name}
        success, result = self._run_app_script("get_frontmost_window", params)
        
        if not success or not result:
            logger.error(f"Failed to get frontmost window for application: {app_name}")
            return None
        
        return result
    
    def move_window(self, 
                   app_name: str, 
                   window_id: Union[int, str], 
                   x: int, 
                   y: int) -> bool:
        """
        Move an application window to a new position.
        
        Args:
            app_name: Name of the application
            window_id: Window ID or index
            x: New x coordinate
            y: New y coordinate
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return False
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "window_id": window_id,
            "x": x,
            "y": y
        }
        
        # Move the window
        success, result = self._run_app_script("move_window", params)
        
        # Clear window cache
        if app_name in self.app_cache and "windows" in self.app_cache[app_name]:
            del self.app_cache[app_name]["windows"]
        
        return success
    
    def resize_window(self, 
                     app_name: str, 
                     window_id: Union[int, str], 
                     width: int, 
                     height: int) -> bool:
        """
        Resize an application window.
        
        Args:
            app_name: Name of the application
            window_id: Window ID or index
            width: New width
            height: New height
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return False
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "window_id": window_id,
            "width": width,
            "height": height
        }
        
        # Resize the window
        success, result = self._run_app_script("resize_window", params)
        
        # Clear window cache
        if app_name in self.app_cache and "windows" in self.app_cache[app_name]:
            del self.app_cache[app_name]["windows"]
        
        return success
    
    def minimize_window(self, app_name: str, window_id: Union[int, str]) -> bool:
        """
        Minimize an application window.
        
        Args:
            app_name: Name of the application
            window_id: Window ID or index
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return False
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "window_id": window_id
        }
        
        # Minimize the window
        success, result = self._run_app_script("minimize_window", params)
        
        # Clear window cache
        if app_name in self.app_cache and "windows" in self.app_cache[app_name]:
            del self.app_cache[app_name]["windows"]
        
        return success
    
    def unminimize_window(self, app_name: str, window_id: Union[int, str]) -> bool:
        """
        Unminimize (restore) an application window.
        
        Args:
            app_name: Name of the application
            window_id: Window ID or index
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return False
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "window_id": window_id
        }
        
        # Unminimize the window
        success, result = self._run_app_script("unminimize_window", params)
        
        # Clear window cache
        if app_name in self.app_cache and "windows" in self.app_cache[app_name]:
            del self.app_cache[app_name]["windows"]
        
        return success
    
    def close_window(self, app_name: str, window_id: Union[int, str]) -> bool:
        """
        Close an application window.
        
        Args:
            app_name: Name of the application
            window_id: Window ID or index
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return False
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "window_id": window_id
        }
        
        # Close the window
        success, result = self._run_app_script("close_window", params)
        
        # Clear window cache
        if app_name in self.app_cache and "windows" in self.app_cache[app_name]:
            del self.app_cache[app_name]["windows"]
        
        return success
    
    def get_menu_items(self, app_name: str) -> List[Dict[str, Any]]:
        """
        Get the menu structure of an application.
        
        Args:
            app_name: Name of the application
            
        Returns:
            List of menu information dictionaries
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return []
        
        # Prepare script parameters
        params = {"app_name": app_name}
        
        # Get menu items
        success, result = self._run_app_script("get_menu_items", params)
        
        if not success or not result:
            logger.error(f"Failed to get menu items for application: {app_name}")
            return []
        
        return result
    
    def select_menu_item(self, 
                        app_name: str, 
                        menu_name: str, 
                        item_name: str) -> bool:
        """
        Select a menu item in an application.
        
        Args:
            app_name: Name of the application
            menu_name: Name of the menu
            item_name: Name of the menu item
            
        Returns:
            True if successful, False otherwise
        """
        # Check if the app is running
        if not self.is_application_running(app_name):
            return False
        
        # Ensure the app is active
        if not self.activate_application(app_name):
            return False
        
        # Prepare script parameters
        params = {
            "app_name": app_name,
            "menu_name": menu_name,
            "item_name": item_name
        }
        
        # Select the menu item
        success, result = self._run_app_script("select_menu_item", params)
        
        return success
    
    def run_application_script(self, 
                             app_name: str, 
                             script: str, 
                             timeout: float = None) -> Tuple[bool, Any]:
        """
        Run a custom AppleScript for an application.
        
        Args:
            app_name: Name of the application
            script: AppleScript code
            timeout: Timeout in seconds (None to use default)
            
        Returns:
            Tuple of (success, result)
        """
        # Use default timeout if not specified
        if timeout is None:
            timeout = self.default_timeout
        
        # Wrap the script in a tell block if it doesn't already have one
        if not script.strip().startswith("tell application"):
            script = f"""
                tell application "{app_name}"
                    {script}
                end tell
            """
        
        # Run the script
        return self.bridge.run_script(script, timeout=timeout)
    
    def get_running_applications(self) -> List[str]:
        """
        Get a list of running applications.
        
        Returns:
            List of application names
        """
        script = """
            tell application "System Events"
                set appList to name of every application process whose background only is false
                return appList
            end tell
        """
        
        success, result = self.bridge.run_script(script)
        
        return result if success and result else []
    
    def get_frontmost_application(self) -> Optional[str]:
        """
        Get the name of the frontmost application.
        
        Returns:
            Application name or None if not found
        """
        script = """
            tell application "System Events"
                set frontApp to first application process whose frontmost is true
                set frontAppName to name of frontApp
                return frontAppName
            end tell
        """
        
        success, result = self.bridge.run_script(script)
        
        return result if success and result else None
    
    def take_screenshot(self, 
                       file_path: str = None, 
                       region: Tuple[int, int, int, int] = None) -> bool:
        """
        Take a screenshot.
        
        Args:
            file_path: Path to save the screenshot (None for clipboard)
            region: Region to capture (x, y, width, height) or None for full screen
            
        Returns:
            True if successful, False otherwise
        """
        if file_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Build the command parameters
        params = {}
        
        # Handle region parameter
        if region:
            x, y, width, height = region
            params["region_opt"] = f"-R {x},{y},{width},{height}"
        else:
            params["region_opt"] = ""
        
        # Handle file path parameter    
        if file_path:
            # We need to quote the path for shell
            params["file_opt"] = file_path
        else:
            params["file_opt"] = ""
        
        # Run the screenshot script
        success, result = self._run_app_script("take_screenshot", params)
        
        if not success:
            logger.error(f"Failed to take screenshot: {result}")
            
        return success and result
    
    def clear_cache(self):
        """Clear the application cache."""
        self.app_cache = {}
        logger.debug("Application cache cleared") 