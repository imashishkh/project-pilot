import json
import os
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple

class Permission:
    """Represents a single permission in the system."""
    
    def __init__(self, name: str, description: str, risk_level: str = "medium"):
        """
        Initialize a Permission object.
        
        Args:
            name: Unique identifier for the permission
            description: Human-readable description of what this permission allows
            risk_level: Risk level of this permission ("low", "medium", "high", "critical")
        """
        self.name = name
        self.description = description
        self.risk_level = risk_level

class PermissionManager:
    """
    Manages permissions for an AI agent, providing granular control over
    what actions the agent can perform on the system.
    """
    
    # Standard permission categories
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    NETWORK_ACCESS = "network_access"
    EXECUTE_COMMAND = "execute_command"
    SYSTEM_SETTINGS = "system_settings"
    CLIPBOARD_ACCESS = "clipboard_access"
    USER_INTERACTION = "user_interaction"
    DATABASE_ACCESS = "database_access"
    CAMERA_ACCESS = "camera_access"
    MICROPHONE_ACCESS = "microphone_access"
    LOCATION_ACCESS = "location_access"
    CONTACTS_ACCESS = "contacts_access"
    CALENDAR_ACCESS = "calendar_access"
    
    def __init__(self, config_path: str = "config/permissions.json", 
                 logs_path: str = "logs/permissions"):
        """
        Initialize the PermissionManager.
        
        Args:
            config_path: Path to the permissions configuration file
            logs_path: Path to store permission logs
        """
        self.config_path = config_path
        self.logs_path = logs_path
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(logs_path), exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("permission_manager")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        os.makedirs(logs_path, exist_ok=True)
        log_file = os.path.join(logs_path, f"permissions_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to logger
        self.logger.addHandler(file_handler)
        
        # Load or create permissions config
        self._load_permissions_config()
        
        # Current temporarily elevated permissions with expiration times
        self.temporary_permissions: Dict[str, datetime] = {}
        
        # Permission request history
        self.request_history: List[Dict[str, Any]] = []
    
    def _load_permissions_config(self) -> None:
        """Load the permissions configuration from the config file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            else:
                # Create default permission configuration
                config = self._create_default_config()
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
            
            # Initialize permission objects
            self.permissions: Dict[str, Permission] = {}
            for perm_name, perm_data in config["permissions"].items():
                self.permissions[perm_name] = Permission(
                    name=perm_name,
                    description=perm_data["description"],
                    risk_level=perm_data["risk_level"]
                )
            
            # Load granted permissions
            self.granted_permissions: Set[str] = set(config.get("granted_permissions", []))
            
            # Load permission mappings (what type of action requires what permission)
            self.permission_mappings: Dict[str, List[str]] = config.get("permission_mappings", {})
            
            # Load sensitive paths
            self.sensitive_paths: List[str] = config.get("sensitive_paths", [])
            
        except Exception as e:
            self.logger.error(f"Error loading permission config: {str(e)}")
            # Create a basic default config
            config = self._create_default_config()
            self.permissions = {name: Permission(name, data["description"], data["risk_level"]) 
                             for name, data in config["permissions"].items()}
            self.granted_permissions = set()
            self.permission_mappings = config.get("permission_mappings", {})
            self.sensitive_paths = config.get("sensitive_paths", [])
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default permission configuration."""
        return {
            "permissions": {
                self.FILE_READ: {
                    "description": "Read files from the disk",
                    "risk_level": "low"
                },
                self.FILE_WRITE: {
                    "description": "Write or modify files on the disk",
                    "risk_level": "medium"
                },
                self.FILE_DELETE: {
                    "description": "Delete files from the disk",
                    "risk_level": "high"
                },
                self.NETWORK_ACCESS: {
                    "description": "Access the internet or network resources",
                    "risk_level": "medium"
                },
                self.EXECUTE_COMMAND: {
                    "description": "Execute system commands or scripts",
                    "risk_level": "high"
                },
                self.SYSTEM_SETTINGS: {
                    "description": "Modify system settings",
                    "risk_level": "high"
                },
                self.CLIPBOARD_ACCESS: {
                    "description": "Read from or write to the clipboard",
                    "risk_level": "medium"
                },
                self.USER_INTERACTION: {
                    "description": "Interact with or prompt the user",
                    "risk_level": "low"
                },
                self.DATABASE_ACCESS: {
                    "description": "Access local databases",
                    "risk_level": "medium"
                },
                self.CAMERA_ACCESS: {
                    "description": "Access device camera",
                    "risk_level": "high"
                },
                self.MICROPHONE_ACCESS: {
                    "description": "Access device microphone",
                    "risk_level": "high"
                },
                self.LOCATION_ACCESS: {
                    "description": "Access device location",
                    "risk_level": "high"
                },
                self.CONTACTS_ACCESS: {
                    "description": "Access user contacts",
                    "risk_level": "high"
                },
                self.CALENDAR_ACCESS: {
                    "description": "Access user calendar data",
                    "risk_level": "high"
                }
            },
            "granted_permissions": [
                self.FILE_READ,
                self.USER_INTERACTION
            ],
            "permission_mappings": {
                "file": {
                    "read": [self.FILE_READ],
                    "write": [self.FILE_WRITE],
                    "delete": [self.FILE_DELETE]
                },
                "network": {
                    "http_request": [self.NETWORK_ACCESS],
                    "api_call": [self.NETWORK_ACCESS]
                },
                "system": {
                    "execute_command": [self.EXECUTE_COMMAND],
                    "modify_settings": [self.SYSTEM_SETTINGS]
                },
                "user": {
                    "clipboard": [self.CLIPBOARD_ACCESS],
                    "prompt": [self.USER_INTERACTION]
                }
            },
            "sensitive_paths": [
                "/private",
                "/System",
                "~/Library/Keychains",
                "~/Library/Cookies",
                "~/Library/Preferences",
                "~/.ssh"
            ]
        }
    
    def save_config(self) -> None:
        """Save the current permission configuration to the config file."""
        config = {
            "permissions": {
                name: {
                    "description": perm.description,
                    "risk_level": perm.risk_level
                } for name, perm in self.permissions.items()
            },
            "granted_permissions": list(self.granted_permissions),
            "permission_mappings": self.permission_mappings,
            "sensitive_paths": self.sensitive_paths
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving permission config: {str(e)}")
    
    def request_permission(self, permission_name: str, reason: str = None, 
                          context: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Request a permission, either from cache or by prompting the user.
        
        Args:
            permission_name: The permission being requested
            reason: Why the permission is needed
            context: Additional contextual information about the request
            
        Returns:
            (granted, message): Whether permission was granted and a message
        """
        if permission_name not in self.permissions:
            error_msg = f"Unknown permission: {permission_name}"
            self.logger.error(error_msg)
            return False, error_msg
        
        # Check if permission is already granted
        if permission_name in self.granted_permissions:
            self._log_permission_use(permission_name, True, reason, context)
            return True, f"Permission '{permission_name}' already granted"
        
        # Check if permission is temporarily elevated
        if permission_name in self.temporary_permissions:
            if datetime.now() < self.temporary_permissions[permission_name]:
                self._log_permission_use(permission_name, True, reason, context)
                return True, f"Temporary permission '{permission_name}' is active"
            else:
                # Expired temporary permission
                del self.temporary_permissions[permission_name]
        
        # Need to request permission from user
        permission = self.permissions[permission_name]
        prompt = self._generate_permission_prompt(permission, reason, context)
        
        # In a real implementation, this would prompt the user
        # For now, we'll simulate rejection for high-risk permissions
        granted = permission.risk_level not in ["high", "critical"]
        
        # Log the request
        self._log_permission_use(permission_name, granted, reason, context)
        
        if granted:
            # Add to granted permissions if the user approves
            self.granted_permissions.add(permission_name)
            self.save_config()
            return True, f"Permission '{permission_name}' granted"
        else:
            return False, f"Permission '{permission_name}' denied"
    
    def elevate_permission_temporarily(self, permission_name: str, duration_minutes: int, 
                                     reason: str = None) -> Tuple[bool, str]:
        """
        Temporarily elevate a permission for a specified duration.
        
        Args:
            permission_name: The permission to temporarily grant
            duration_minutes: How long the permission should be granted (minutes)
            reason: Why the permission elevation is needed
            
        Returns:
            (success, message): Whether elevation succeeded and a message
        """
        if permission_name not in self.permissions:
            error_msg = f"Unknown permission: {permission_name}"
            self.logger.error(error_msg)
            return False, error_msg
        
        # Calculate expiration time
        expiration = datetime.now() + timedelta(minutes=duration_minutes)
        
        # In a real implementation, this would prompt the user
        # For now, we'll simulate rejection for critical-risk permissions
        granted = self.permissions[permission_name].risk_level != "critical"
        
        if granted:
            self.temporary_permissions[permission_name] = expiration
            self.logger.info(f"Temporarily elevated permission '{permission_name}' "
                           f"for {duration_minutes} minutes. Reason: {reason}")
            return True, f"Permission '{permission_name}' temporarily elevated for {duration_minutes} minutes"
        else:
            self.logger.warning(f"Temporary elevation of '{permission_name}' denied. Reason: {reason}")
            return False, f"Temporary elevation of permission '{permission_name}' denied"
    
    def check_permission(self, permission_name: str, context: Dict[str, Any] = None) -> bool:
        """
        Check if a permission is currently granted.
        
        Args:
            permission_name: The permission to check
            context: Additional contextual information about the check
            
        Returns:
            is_granted: Whether the permission is currently granted
        """
        if permission_name not in self.permissions:
            self.logger.error(f"Unknown permission checked: {permission_name}")
            return False
        
        # Check regular permissions
        if permission_name in self.granted_permissions:
            return True
        
        # Check temporary permissions
        if permission_name in self.temporary_permissions:
            if datetime.now() < self.temporary_permissions[permission_name]:
                return True
            else:
                # Clean up expired permission
                del self.temporary_permissions[permission_name]
        
        return False
    
    def check_action_permission(self, resource_type: str, action: str, 
                              resource_path: str = None) -> Tuple[bool, Optional[str]]:
        """
        Check if a specific action on a resource is permitted.
        
        Args:
            resource_type: Type of resource (e.g., "file", "network")
            action: Action to perform (e.g., "read", "write")
            resource_path: Path or identifier of the specific resource
            
        Returns:
            (permitted, required_permission): Whether action is permitted and the required permission if not
        """
        # Find required permissions for this action
        required_permissions = []
        
        if resource_type in self.permission_mappings and action in self.permission_mappings[resource_type]:
            required_permissions = self.permission_mappings[resource_type][action]
        
        # Check if any of the required permissions are granted
        for permission in required_permissions:
            if self.check_permission(permission):
                # Check for sensitive paths for file operations
                if resource_type == "file" and resource_path:
                    for sensitive_path in self.sensitive_paths:
                        if resource_path.startswith(os.path.expanduser(sensitive_path)):
                            self.logger.warning(f"Attempted access to sensitive path: {resource_path}")
                            return False, permission
                
                # Permission granted
                return True, None
        
        # No required permission is granted
        return False, required_permissions[0] if required_permissions else None
    
    def revoke_permission(self, permission_name: str) -> bool:
        """
        Revoke a previously granted permission.
        
        Args:
            permission_name: The permission to revoke
            
        Returns:
            success: Whether the permission was successfully revoked
        """
        if permission_name not in self.permissions:
            self.logger.error(f"Attempted to revoke unknown permission: {permission_name}")
            return False
        
        # Remove from granted permissions
        if permission_name in self.granted_permissions:
            self.granted_permissions.remove(permission_name)
            self.save_config()
            self.logger.info(f"Permission '{permission_name}' revoked")
            return True
        
        # Remove from temporary permissions
        if permission_name in self.temporary_permissions:
            del self.temporary_permissions[permission_name]
            self.logger.info(f"Temporary permission '{permission_name}' revoked")
            return True
        
        self.logger.warning(f"Attempted to revoke non-granted permission: {permission_name}")
        return False
    
    def _generate_permission_prompt(self, permission: Permission, reason: str, 
                                  context: Dict[str, Any]) -> str:
        """
        Generate a user-friendly permission request prompt.
        
        Args:
            permission: The permission being requested
            reason: Why the permission is needed
            context: Additional contextual information
            
        Returns:
            prompt: A user-friendly permission request
        """
        risk_indicators = {
            "low": "⚪",
            "medium": "🟡",
            "high": "🟠",
            "critical": "🔴"
        }
        
        risk_indicator = risk_indicators.get(permission.risk_level, "⚪")
        
        prompt = f"""
Permission Request {risk_indicator}

The application is requesting permission to:
{permission.description}

Risk Level: {permission.risk_level.upper()}

Reason: {reason or "Not specified"}
"""
        
        if context:
            prompt += "\nAdditional Context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
        
        prompt += "\nDo you want to grant this permission? (Yes/No/Always)"
        
        return prompt
    
    def _log_permission_use(self, permission_name: str, granted: bool, 
                          reason: str = None, context: Dict[str, Any] = None) -> None:
        """
        Log permission usage or violations.
        
        Args:
            permission_name: The permission being used
            granted: Whether the permission was granted
            reason: Why the permission was needed
            context: Additional contextual information
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "permission": permission_name,
            "granted": granted,
            "reason": reason,
            "context": context
        }
        
        self.request_history.append(log_entry)
        
        if granted:
            self.logger.info(f"Permission '{permission_name}' used. Reason: {reason}")
        else:
            self.logger.warning(f"Permission '{permission_name}' denied. Reason: {reason}")
    
    def get_permission_history(self, permission_name: str = None, 
                             start_time: datetime = None,
                             end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        Get history of permission requests and usage.
        
        Args:
            permission_name: Filter by specific permission
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            history: List of permission history entries
        """
        filtered_history = self.request_history.copy()
        
        if permission_name:
            filtered_history = [entry for entry in filtered_history 
                             if entry["permission"] == permission_name]
        
        if start_time:
            filtered_history = [entry for entry in filtered_history 
                             if datetime.fromisoformat(entry["timestamp"]) >= start_time]
        
        if end_time:
            filtered_history = [entry for entry in filtered_history 
                             if datetime.fromisoformat(entry["timestamp"]) <= end_time]
        
        return filtered_history
