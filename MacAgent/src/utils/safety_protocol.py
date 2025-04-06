import os
import json
import time
import shutil
import logging
import threading
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple, Set, Union

# Import permission manager to integrate with permission system
from .permission_manager import PermissionManager

class UndoOperation:
    """Represents an operation that can be undone."""
    
    def __init__(self, operation_id: str, operation_type: str, description: str,
                undo_function: Callable, undo_args: List[Any] = None, 
                undo_kwargs: Dict[str, Any] = None):
        """
        Initialize an undo operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation (file, network, system, etc.)
            description: Human-readable description of the operation
            undo_function: Function to call to undo this operation
            undo_args: Positional arguments for the undo function
            undo_kwargs: Keyword arguments for the undo function
        """
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.description = description
        self.timestamp = datetime.now()
        self.undo_function = undo_function
        self.undo_args = undo_args or []
        self.undo_kwargs = undo_kwargs or {}
        self.undone = False
    
    def execute_undo(self) -> bool:
        """
        Execute the undo operation.
        
        Returns:
            success: Whether the undo operation was successful
        """
        if self.undone:
            return False
        
        try:
            self.undo_function(*self.undo_args, **self.undo_kwargs)
            self.undone = True
            return True
        except Exception as e:
            return False

class SafetyProtocol:
    """
    Implements safety mechanisms for AI agent operations including
    destructive operation detection, confirmation mechanisms,
    simulations, undo capabilities, and emergency stop.
    """
    
    # Potentially destructive operation types
    FILE_DELETE = "file_delete"
    FILE_OVERWRITE = "file_overwrite"
    SYSTEM_MODIFICATION = "system_modification"
    NETWORK_DATA_SEND = "network_data_send"
    PROCESS_TERMINATION = "process_termination"
    
    # Risk levels
    LOW_RISK = "low"
    MEDIUM_RISK = "medium"
    HIGH_RISK = "high"
    CRITICAL_RISK = "critical"
    
    def __init__(self, permission_manager: PermissionManager, 
                config_path: str = "config/safety.json",
                logs_path: str = "logs/safety"):
        """
        Initialize the SafetyProtocol.
        
        Args:
            permission_manager: PermissionManager instance to check permissions
            config_path: Path to the safety configuration file
            logs_path: Path to store safety logs
        """
        self.permission_manager = permission_manager
        self.config_path = config_path
        self.logs_path = logs_path
        
        # Ensure logs directory exists
        os.makedirs(logs_path, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("safety_protocol")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        log_file = os.path.join(logs_path, f"safety_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to logger
        self.logger.addHandler(file_handler)
        
        # Load or create safety config
        self._load_safety_config()
        
        # Undo history
        self.undo_history: List[UndoOperation] = []
        
        # Operation history
        self.operation_history: List[Dict[str, Any]] = []
        
        # Emergency stop flag
        self.emergency_stop_flag = threading.Event()
        
        # Create emergency stop file if needed
        self.emergency_stop_file = os.path.join(os.path.dirname(self.config_path), "EMERGENCY_STOP")
        
        # Start monitoring thread for emergency stop file
        self._start_emergency_stop_monitor()
    
    def _load_safety_config(self) -> None:
        """Load the safety configuration from the config file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default safety configuration
                self.config = self._create_default_config()
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error loading safety config: {str(e)}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default safety configuration."""
        return {
            "confirmation_required": {
                self.FILE_DELETE: True,
                self.FILE_OVERWRITE: True,
                self.SYSTEM_MODIFICATION: True,
                self.NETWORK_DATA_SEND: True,
                self.PROCESS_TERMINATION: True
            },
            "simulation_required": {
                self.FILE_DELETE: True,
                self.FILE_OVERWRITE: True,
                self.SYSTEM_MODIFICATION: True,
                self.NETWORK_DATA_SEND: False,
                self.PROCESS_TERMINATION: False
            },
            "high_risk_paths": [
                "/System",
                "/private",
                "/usr",
                "/var",
                "/Library",
                "~/.ssh",
                "~/Library/Keychains"
            ],
            "high_risk_processes": [
                "kernel",
                "launchd",
                "systemd",
                "init",
                "WindowServer",
                "Finder",
                "loginwindow",
                "Dock"
            ],
            "undo_history_limit": 100,
            "simulation_timeout_seconds": 10
        }
    
    def save_config(self) -> None:
        """Save the current safety configuration to the config file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving safety config: {str(e)}")
    
    def is_destructive_operation(self, operation_type: str, 
                               resource_path: str = None) -> Tuple[bool, str]:
        """
        Check if an operation is potentially destructive.
        
        Args:
            operation_type: Type of operation
            resource_path: Path or identifier of the resource being operated on
            
        Returns:
            (is_destructive, risk_level): Whether the operation is destructive and risk level
        """
        # Check if operation type is inherently destructive
        inherently_destructive = operation_type in [
            self.FILE_DELETE,
            self.FILE_OVERWRITE,
            self.SYSTEM_MODIFICATION,
            self.PROCESS_TERMINATION
        ]
        
        # Default risk level
        risk_level = self.LOW_RISK
        
        # Evaluate risk level based on path or target
        if resource_path:
            # Check if path is in high risk paths
            for high_risk_path in self.config.get("high_risk_paths", []):
                expanded_path = os.path.expanduser(high_risk_path)
                if resource_path.startswith(expanded_path):
                    risk_level = self.HIGH_RISK
                    break
            
            # Check if it's a system directory
            system_dirs = ["/System", "/usr", "/bin", "/sbin", "/var", "/etc", "/Library"]
            if any(resource_path.startswith(d) for d in system_dirs):
                risk_level = self.HIGH_RISK
            
            # Check if it's a user configuration directory
            if resource_path.startswith(os.path.expanduser("~/.config")):
                risk_level = self.MEDIUM_RISK
            
            # Check if it's a process name in high risk processes
            if operation_type == self.PROCESS_TERMINATION:
                if resource_path in self.config.get("high_risk_processes", []):
                    risk_level = self.CRITICAL_RISK
        
        return inherently_destructive, risk_level
    
    def request_confirmation(self, operation_type: str, resource_path: str,
                          risk_level: str, details: Dict[str, Any] = None) -> bool:
        """
        Request confirmation for a potentially destructive operation.
        
        Args:
            operation_type: Type of operation
            resource_path: Path or identifier of the resource being operated on
            risk_level: Risk level of the operation
            details: Additional details about the operation
            
        Returns:
            confirmed: Whether the operation was confirmed
        """
        # Check if confirmation is required for this operation type
        confirmation_required = self.config.get("confirmation_required", {}).get(operation_type, False)
        
        if not confirmation_required and risk_level == self.LOW_RISK:
            return True
        
        # Format a confirmation message
        operation_descriptions = {
            self.FILE_DELETE: "delete file",
            self.FILE_OVERWRITE: "overwrite file",
            self.SYSTEM_MODIFICATION: "modify system settings",
            self.NETWORK_DATA_SEND: "send data over network",
            self.PROCESS_TERMINATION: "terminate process"
        }
        
        operation_desc = operation_descriptions.get(operation_type, operation_type)
        
        risk_symbols = {
            self.LOW_RISK: "⚪",
            self.MEDIUM_RISK: "🟡",
            self.HIGH_RISK: "🟠",
            self.CRITICAL_RISK: "🔴"
        }
        
        risk_symbol = risk_symbols.get(risk_level, "⚪")
        
        message = f"""
Safety Confirmation {risk_symbol}

The application is requesting to {operation_desc}:
{resource_path}

Risk Level: {risk_level.upper()}
"""
        
        if details:
            message += "\nDetails:\n"
            for key, value in details.items():
                message += f"- {key}: {value}\n"
        
        message += "\nDo you confirm this operation? (Yes/No)"
        
        # In a real implementation, this would prompt the user
        # For now, we'll simulate rejection for high/critical-risk operations
        confirmed = risk_level not in [self.HIGH_RISK, self.CRITICAL_RISK]
        
        # Log the confirmation request
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "resource_path": resource_path,
            "risk_level": risk_level,
            "details": details,
            "confirmed": confirmed
        }
        
        self.operation_history.append(log_entry)
        
        if confirmed:
            self.logger.info(f"Operation '{operation_type}' on '{resource_path}' confirmed")
        else:
            self.logger.warning(f"Operation '{operation_type}' on '{resource_path}' rejected")
        
        return confirmed
    
    def simulate_operation(self, operation_type: str, operation_function: Callable,
                         args: List[Any] = None, kwargs: Dict[str, Any] = None,
                         resource_path: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Simulate an operation to preview its effects.
        
        Args:
            operation_type: Type of operation
            operation_function: Function that performs the operation
            args: Positional arguments for the operation function
            kwargs: Keyword arguments for the operation function
            resource_path: Path or identifier of the resource being operated on
            
        Returns:
            (success, results): Whether the simulation was successful and the results
        """
        args = args or []
        kwargs = kwargs or {}
        
        # Check if simulation is required for this operation type
        simulation_required = self.config.get("simulation_required", {}).get(operation_type, False)
        
        if not simulation_required:
            return True, {"message": "Simulation not required for this operation"}
        
        # Create a temporary directory for simulation
        temp_dir = tempfile.mkdtemp(prefix="safety_sim_")
        
        try:
            # Prepare simulation environment based on operation type
            simulation_results = {}
            
            if operation_type == self.FILE_DELETE:
                # For file deletion, check if file exists and is important
                if resource_path and os.path.exists(resource_path):
                    file_stats = os.stat(resource_path)
                    simulation_results["file_exists"] = True
                    simulation_results["file_size"] = file_stats.st_size
                    simulation_results["last_modified"] = datetime.fromtimestamp(
                        file_stats.st_mtime).isoformat()
                    
                    # Check if it's a directory with contents
                    if os.path.isdir(resource_path):
                        file_count = sum(len(files) for _, _, files in os.walk(resource_path))
                        dir_count = sum(len(dirs) for _, dirs, _ in os.walk(resource_path))
                        simulation_results["is_directory"] = True
                        simulation_results["contains_files"] = file_count
                        simulation_results["contains_subdirectories"] = dir_count
                else:
                    simulation_results["file_exists"] = False
            
            elif operation_type == self.FILE_OVERWRITE:
                # For file overwrite, check if destination exists and differs
                if resource_path and os.path.exists(resource_path):
                    file_stats = os.stat(resource_path)
                    simulation_results["file_exists"] = True
                    simulation_results["file_size"] = file_stats.st_size
                    simulation_results["last_modified"] = datetime.fromtimestamp(
                        file_stats.st_mtime).isoformat()
                    
                    # If the operation involves new content, we could compare it
                    # For now, just note that the file will be modified
                    simulation_results["will_be_modified"] = True
                else:
                    simulation_results["file_exists"] = False
                    simulation_results["will_be_created"] = True
            
            elif operation_type == self.SYSTEM_MODIFICATION:
                # For system modification, we can't easily simulate
                # Just provide basic information
                simulation_results["note"] = "System modifications may affect system stability"
                simulation_results["can_be_undone"] = False
            
            # Add timeout to prevent infinite loops or hanging
            timeout = self.config.get("simulation_timeout_seconds", 10)
            simulation_results["timeout_seconds"] = timeout
            
            self.logger.info(f"Simulated operation '{operation_type}' on '{resource_path}'")
            return True, simulation_results
            
        except Exception as e:
            self.logger.error(f"Simulation error for '{operation_type}': {str(e)}")
            return False, {"error": str(e)}
        
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def register_undo_operation(self, operation_type: str, resource_path: str,
                              undo_function: Callable, undo_args: List[Any] = None,
                              undo_kwargs: Dict[str, Any] = None,
                              description: str = None) -> str:
        """
        Register an operation that can be undone.
        
        Args:
            operation_type: Type of operation
            resource_path: Path or identifier of the resource being operated on
            undo_function: Function to call to undo this operation
            undo_args: Positional arguments for the undo function
            undo_kwargs: Keyword arguments for the undo function
            description: Human-readable description of the operation
            
        Returns:
            operation_id: Unique identifier for the undo operation
        """
        # Generate a description if not provided
        if description is None:
            operation_descriptions = {
                self.FILE_DELETE: f"Deleted {resource_path}",
                self.FILE_OVERWRITE: f"Modified {resource_path}",
                self.SYSTEM_MODIFICATION: f"Modified system setting: {resource_path}",
                self.NETWORK_DATA_SEND: f"Sent data to {resource_path}",
                self.PROCESS_TERMINATION: f"Terminated process {resource_path}"
            }
            description = operation_descriptions.get(operation_type, f"{operation_type}: {resource_path}")
        
        # Create an undo operation
        operation_id = f"undo_{int(time.time())}_{hash(resource_path) % 10000}"
        undo_op = UndoOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            description=description,
            undo_function=undo_function,
            undo_args=undo_args,
            undo_kwargs=undo_kwargs
        )
        
        # Add to undo history
        self.undo_history.append(undo_op)
        
        # Enforce history limit
        history_limit = self.config.get("undo_history_limit", 100)
        if len(self.undo_history) > history_limit:
            self.undo_history = self.undo_history[-history_limit:]
        
        self.logger.info(f"Registered undo operation: {description}")
        return operation_id
    
    def undo_operation(self, operation_id: str) -> Tuple[bool, str]:
        """
        Undo a previously registered operation.
        
        Args:
            operation_id: Identifier of the operation to undo
            
        Returns:
            (success, message): Whether the undo was successful and a message
        """
        # Find the operation in history
        for undo_op in reversed(self.undo_history):
            if undo_op.operation_id == operation_id:
                if undo_op.undone:
                    return False, "Operation already undone"
                
                try:
                    success = undo_op.execute_undo()
                    if success:
                        self.logger.info(f"Undid operation: {undo_op.description}")
                        return True, f"Successfully undid: {undo_op.description}"
                    else:
                        self.logger.error(f"Failed to undo operation: {undo_op.description}")
                        return False, f"Failed to undo: {undo_op.description}"
                except Exception as e:
                    self.logger.error(f"Error undoing operation {operation_id}: {str(e)}")
                    return False, f"Error undoing operation: {str(e)}"
        
        return False, f"Operation {operation_id} not found in history"
    
    def get_undo_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of operations that can be undone.
        
        Returns:
            history: List of undo operations
        """
        return [
            {
                "operation_id": op.operation_id,
                "operation_type": op.operation_type,
                "description": op.description,
                "timestamp": op.timestamp.isoformat(),
                "undone": op.undone
            }
            for op in self.undo_history
        ]
    
    def emergency_stop(self) -> None:
        """
        Trigger an emergency stop for all operations.
        """
        self.logger.warning("Emergency stop triggered")
        self.emergency_stop_flag.set()
        
        # Create emergency stop file as a backup mechanism
        try:
            with open(self.emergency_stop_file, 'w') as f:
                f.write(f"Emergency stop triggered at {datetime.now().isoformat()}")
        except Exception as e:
            self.logger.error(f"Failed to create emergency stop file: {str(e)}")
    
    def clear_emergency_stop(self) -> None:
        """
        Clear the emergency stop flag.
        """
        self.logger.info("Emergency stop cleared")
        self.emergency_stop_flag.clear()
        
        # Remove emergency stop file if it exists
        if os.path.exists(self.emergency_stop_file):
            try:
                os.remove(self.emergency_stop_file)
            except Exception as e:
                self.logger.error(f"Failed to remove emergency stop file: {str(e)}")
    
    def is_emergency_stop_active(self) -> bool:
        """
        Check if emergency stop is active.
        
        Returns:
            active: Whether emergency stop is active
        """
        # Check flag
        if self.emergency_stop_flag.is_set():
            return True
        
        # Also check file as backup
        return os.path.exists(self.emergency_stop_file)
    
    def _start_emergency_stop_monitor(self) -> None:
        """Start a thread to monitor the emergency stop file."""
        def monitor_emergency_stop():
            while True:
                if os.path.exists(self.emergency_stop_file) and not self.emergency_stop_flag.is_set():
                    self.logger.warning("Emergency stop file detected, activating emergency stop")
                    self.emergency_stop_flag.set()
                
                time.sleep(1)
        
        monitor_thread = threading.Thread(target=monitor_emergency_stop, daemon=True)
        monitor_thread.start()
    
    def safe_execute(self, operation_type: str, resource_path: str,
                   operation_function: Callable, args: List[Any] = None,
                   kwargs: Dict[str, Any] = None, undo_function: Callable = None,
                   undo_args: List[Any] = None, undo_kwargs: Dict[str, Any] = None,
                   description: str = None) -> Tuple[bool, Any, str]:
        """
        Safely execute an operation with all safety protocols.
        
        Args:
            operation_type: Type of operation
            resource_path: Path or identifier of the resource
            operation_function: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            undo_function: Function to undo the operation
            undo_args: Positional arguments for the undo function
            undo_kwargs: Keyword arguments for the undo function
            description: Human-readable description of the operation
            
        Returns:
            (success, result, message): Whether operation succeeded, its result, and a message
        """
        args = args or []
        kwargs = kwargs or {}
        
        # Check for emergency stop
        if self.is_emergency_stop_active():
            return False, None, "Emergency stop is active"
        
        # Check if operation is destructive
        is_destructive, risk_level = self.is_destructive_operation(operation_type, resource_path)
        
        # Check permissions for this operation
        resource_type = operation_type.split("_")[0] if "_" in operation_type else "system"
        action = operation_type.split("_")[1] if "_" in operation_type else operation_type
        
        permitted, required_permission = self.permission_manager.check_action_permission(
            resource_type, action, resource_path
        )
        
        if not permitted:
            # Try to request permission
            if required_permission:
                granted, message = self.permission_manager.request_permission(
                    required_permission,
                    f"Needed for {operation_type} on {resource_path}",
                    {"risk_level": risk_level}
                )
                
                if not granted:
                    return False, None, f"Permission denied: {message}"
            else:
                return False, None, "Permission denied and no permission available to request"
        
        # If destructive, request confirmation
        if is_destructive:
            details = kwargs.copy()
            details["operation_function"] = operation_function.__name__
            
            confirmed = self.request_confirmation(
                operation_type, resource_path, risk_level, details
            )
            
            if not confirmed:
                return False, None, "Operation not confirmed by user"
        
        # Simulate operation if needed
        if self.config.get("simulation_required", {}).get(operation_type, False):
            simulation_success, simulation_results = self.simulate_operation(
                operation_type, operation_function, args, kwargs, resource_path
            )
            
            if not simulation_success:
                return False, None, f"Simulation failed: {simulation_results.get('error', 'Unknown error')}"
        
        # Execute the operation
        try:
            result = operation_function(*args, **kwargs)
            
            # Register undo operation if provided
            if undo_function:
                operation_id = self.register_undo_operation(
                    operation_type, resource_path, undo_function,
                    undo_args, undo_kwargs, description
                )
            
            self.logger.info(f"Successfully executed {operation_type} on {resource_path}")
            return True, result, "Operation completed successfully"
            
        except Exception as e:
            self.logger.error(f"Error executing {operation_type} on {resource_path}: {str(e)}")
            return False, None, f"Operation failed: {str(e)}"
