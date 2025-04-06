import os
import sys
import json
import time
import signal
import logging
import threading
import importlib
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Type
from datetime import datetime

# Import core components
from ..utils.permission_manager import PermissionManager
from ..utils.safety_protocol import SafetyProtocol
from ..utils.security_monitor import SecurityMonitor
from ..utils.diagnostic_system import DiagnosticSystem

class ComponentStatus:
    """Represents the status of a system component."""
    
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"
    RECOVERING = "recovering"
    
    def __init__(self, name: str, status: str = INITIALIZING, 
                details: Dict[str, Any] = None):
        """
        Initialize a component status.
        
        Args:
            name: Name of the component
            status: Current status
            details: Additional status details
        """
        self.name = name
        self.status = status
        self.details = details or {}
        self.last_updated = datetime.now()
        self.error = None
        self.recovery_attempts = 0
    
    def update(self, status: str, details: Dict[str, Any] = None,
              error: Exception = None) -> None:
        """
        Update the component status.
        
        Args:
            status: New status
            details: Additional status details
            error: Error that occurred, if any
        """
        self.status = status
        
        if details:
            self.details.update(details)
        
        self.last_updated = datetime.now()
        
        if error:
            self.error = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc()
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the status to a dictionary.
        
        Returns:
            Dictionary representation of the status
        """
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "last_updated": self.last_updated.isoformat(),
            "error": self.error,
            "recovery_attempts": self.recovery_attempts
        }

class SystemOrchestrator:
    """
    Coordinates all subsystems of the AI agent, manages component lifecycle,
    handles configurations, and provides error recovery mechanisms.
    """
    
    def __init__(self, config_path: str = "config/system.json"):
        """
        Initialize the SystemOrchestrator.
        
        Args:
            config_path: Path to the system configuration file
        """
        # Set up basic logging until proper logging is configured
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("system_orchestrator")
        
        self.config_path = config_path
        self.components = {}
        self.component_status = {}
        self.shutting_down = False
        self.startup_complete = False
        
        # Load system configuration
        self._load_config()
        
        # Configure system-wide logging
        self._configure_logging()
        
        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Register for automatic cleanup on exit
        import atexit
        atexit.register(self.shutdown)
    
    def _load_config(self) -> None:
        """Load the system configuration."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default configuration
                self.config = self._create_default_config()
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                    
            self.logger.info(f"Loaded system configuration from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading system configuration: {str(e)}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default system configuration."""
        return {
            "system_name": "MacAgent",
            "version": "1.0.0",
            "logging": {
                "level": "INFO",
                "log_dir": "logs",
                "max_log_size_mb": 10,
                "backup_count": 5,
                "log_to_console": True
            },
            "components": {
                "permission_manager": {
                    "enabled": True,
                    "config_path": "config/permissions.json",
                    "logs_path": "logs/permissions"
                },
                "safety_protocol": {
                    "enabled": True,
                    "config_path": "config/safety.json",
                    "logs_path": "logs/safety"
                },
                "security_monitor": {
                    "enabled": True,
                    "config_path": "config/security.json",
                    "logs_path": "logs/security",
                    "secure_storage_path": "data/secure"
                },
                "diagnostic_system": {
                    "enabled": True,
                    "config_path": "config/diagnostics.json",
                    "logs_path": "logs/diagnostics",
                    "check_interval_minutes": 10
                }
            },
            "dependencies": {
                "permission_manager": [],
                "safety_protocol": ["permission_manager"],
                "security_monitor": ["permission_manager", "safety_protocol"],
                "diagnostic_system": []
            },
            "recovery": {
                "max_recovery_attempts": 3,
                "recovery_timeout_seconds": 30,
                "critical_components": ["permission_manager", "safety_protocol"]
            }
        }
    
    def _configure_logging(self) -> None:
        """Configure system-wide logging."""
        try:
            log_config = self.config.get("logging", {})
            log_level_name = log_config.get("level", "INFO")
            log_level = getattr(logging, log_level_name)
            
            log_dir = log_config.get("log_dir", "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Root logger configuration
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Clear existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            
            # Configure file handler
            log_file = os.path.join(log_dir, f"system_{datetime.now().strftime('%Y%m%d')}.log")
            max_bytes = log_config.get("max_log_size_mb", 10) * 1024 * 1024
            backup_count = log_config.get("backup_count", 5)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            
            # Configure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            # Add console handler if enabled
            if log_config.get("log_to_console", True):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
            
            self.logger.info("Configured system-wide logging")
            
        except Exception as e:
            self.logger.error(f"Error configuring logging: {str(e)}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.info("Set up signal handlers for graceful shutdown")
        except Exception as e:
            self.logger.error(f"Error setting up signal handlers: {str(e)}")
    
    def _signal_handler(self, sig, frame) -> None:
        """
        Handle termination signals.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(sig).name
        self.logger.info(f"Received {signal_name} signal, initiating shutdown")
        self.shutdown()
    
    def startup(self) -> bool:
        """
        Start all system components in the correct order.
        
        Returns:
            success: Whether startup was successful
        """
        self.logger.info(f"Starting {self.config.get('system_name', 'MacAgent')} v{self.config.get('version', '1.0.0')}")
        
        try:
            # Initialize components based on dependency order
            component_order = self._resolve_dependency_order()
            
            if not component_order:
                self.logger.error("Failed to resolve component dependencies")
                return False
                
            self.logger.info(f"Component initialization order: {component_order}")
            
            # Initialize components in order
            for component_name in component_order:
                if not self._initialize_component(component_name):
                    self.logger.error(f"Failed to initialize {component_name}")
                    
                    # If this is a critical component, we cannot continue
                    critical_components = self.config.get("recovery", {}).get("critical_components", [])
                    if component_name in critical_components:
                        self.logger.critical(f"Critical component {component_name} failed to initialize, cannot continue")
                        return False
            
            # Start any background tasks or services after all components are initialized
            self._start_background_tasks()
            
            self.startup_complete = True
            self.logger.info("System startup completed successfully")
            return True
            
        except Exception as e:
            self.logger.critical(f"System startup failed: {str(e)}")
            self.logger.exception("Startup exception details:")
            return False
    
    def _resolve_dependency_order(self) -> List[str]:
        """
        Resolve the initialization order of components based on dependencies.
        
        Returns:
            ordered_components: List of component names in initialization order
        """
        dependencies = self.config.get("dependencies", {})
        components_config = self.config.get("components", {})
        
        # Only include enabled components
        enabled_components = [
            name for name, config in components_config.items() 
            if config.get("enabled", True)
        ]
        
        # Create a filtered dependency map
        filtered_deps = {
            name: [dep for dep in deps if dep in enabled_components]
            for name, deps in dependencies.items()
            if name in enabled_components
        }
        
        # Check for missing components in dependency map
        for component in enabled_components:
            if component not in filtered_deps:
                filtered_deps[component] = []
        
        # Topological sort
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(component):
            if component in temp_visited:
                self.logger.error(f"Circular dependency detected involving {component}")
                return False
                
            if component in visited:
                return True
                
            temp_visited.add(component)
            
            for dependency in filtered_deps.get(component, []):
                if not visit(dependency):
                    return False
            
            temp_visited.remove(component)
            visited.add(component)
            result.append(component)
            return True
        
        # Visit all components
        for component in filtered_deps:
            if component not in visited:
                if not visit(component):
                    return []
        
        return result
    
    def _initialize_component(self, component_name: str) -> bool:
        """
        Initialize a system component.
        
        Args:
            component_name: Name of the component to initialize
            
        Returns:
            success: Whether initialization was successful
        """
        self.logger.info(f"Initializing component: {component_name}")
        
        component_config = self.config.get("components", {}).get(component_name, {})
        if not component_config.get("enabled", True):
            self.logger.info(f"Component {component_name} is disabled, skipping")
            return True
        
        # Update component status
        self.component_status[component_name] = ComponentStatus(component_name)
        
        try:
            # Initialize the component based on its type
            if component_name == "permission_manager":
                config_path = component_config.get("config_path", "config/permissions.json")
                logs_path = component_config.get("logs_path", "logs/permissions")
                
                self.components[component_name] = PermissionManager(
                    config_path=config_path,
                    logs_path=logs_path
                )
                
            elif component_name == "safety_protocol":
                # Get the permission manager dependency
                permission_manager = self.components.get("permission_manager")
                if not permission_manager:
                    raise ValueError("Safety Protocol requires Permission Manager")
                
                config_path = component_config.get("config_path", "config/safety.json")
                logs_path = component_config.get("logs_path", "logs/safety")
                
                self.components[component_name] = SafetyProtocol(
                    permission_manager=permission_manager,
                    config_path=config_path,
                    logs_path=logs_path
                )
                
            elif component_name == "security_monitor":
                # Get dependencies
                permission_manager = self.components.get("permission_manager")
                safety_protocol = self.components.get("safety_protocol")
                
                if not permission_manager or not safety_protocol:
                    raise ValueError("Security Monitor requires Permission Manager and Safety Protocol")
                
                config_path = component_config.get("config_path", "config/security.json")
                logs_path = component_config.get("logs_path", "logs/security")
                secure_storage_path = component_config.get("secure_storage_path", "data/secure")
                
                self.components[component_name] = SecurityMonitor(
                    permission_manager=permission_manager,
                    safety_protocol=safety_protocol,
                    config_path=config_path,
                    logs_path=logs_path,
                    secure_storage_path=secure_storage_path
                )
                
            elif component_name == "diagnostic_system":
                config_path = component_config.get("config_path", "config/diagnostics.json")
                logs_path = component_config.get("logs_path", "logs/diagnostics")
                check_interval = component_config.get("check_interval_minutes", 10)
                
                # The diagnostic system needs access to all components and the orchestrator
                self.components[component_name] = DiagnosticSystem(
                    system_orchestrator=self,
                    config_path=config_path,
                    logs_path=logs_path,
                    check_interval_minutes=check_interval
                )
            
            # Update component status to running
            self.component_status[component_name].update(ComponentStatus.RUNNING)
            self.logger.info(f"Component {component_name} initialized successfully")
            return True
            
        except Exception as e:
            # Update component status to error
            self.component_status[component_name].update(
                ComponentStatus.ERROR,
                error=e
            )
            self.logger.error(f"Error initializing {component_name}: {str(e)}")
            self.logger.exception("Initialization exception details:")
            
            # Try to recover the component
            if self._try_recover_component(component_name):
                return True
                
            return False
    
    def _start_background_tasks(self) -> None:
        """Start any background tasks or services."""
        self.logger.info("Starting background tasks")
        
        # Add background task initialization here
        # For example, starting the diagnostic system's health check thread
        diagnostic_system = self.get_component("diagnostic_system")
        if diagnostic_system:
            diagnostic_system.start_health_monitoring()
    
    def shutdown(self) -> None:
        """Gracefully shutdown all system components."""
        if self.shutting_down:
            return
            
        self.shutting_down = True
        self.logger.info("Initiating system shutdown")
        
        # Shutdown components in reverse dependency order
        component_order = self._resolve_dependency_order()
        if component_order:
            for component_name in reversed(component_order):
                self._shutdown_component(component_name)
        
        self.logger.info("System shutdown complete")
    
    def _shutdown_component(self, component_name: str) -> None:
        """
        Shutdown a system component.
        
        Args:
            component_name: Name of the component to shutdown
        """
        component = self.components.get(component_name)
        if not component:
            return
            
        self.logger.info(f"Shutting down component: {component_name}")
        
        try:
            # Handle shutdown differently based on component type
            if hasattr(component, "shutdown"):
                component.shutdown()
            
            # Update component status
            if component_name in self.component_status:
                self.component_status[component_name].update(ComponentStatus.STOPPED)
                
            self.logger.info(f"Component {component_name} shutdown successfully")
            
        except Exception as e:
            self.logger.error(f"Error shutting down {component_name}: {str(e)}")
            self.logger.exception("Shutdown exception details:")
    
    def get_component(self, component_name: str) -> Any:
        """
        Get a system component by name.
        
        Args:
            component_name: Name of the component
            
        Returns:
            component: The component or None if not found
        """
        return self.components.get(component_name)
    
    def get_component_status(self, component_name: str = None) -> Dict[str, Any]:
        """
        Get the status of a component or all components.
        
        Args:
            component_name: Name of the component or None for all
            
        Returns:
            status: Component status information
        """
        if component_name:
            if component_name in self.component_status:
                return self.component_status[component_name].to_dict()
            return None
        
        return {name: status.to_dict() for name, status in self.component_status.items()}
    
    def _try_recover_component(self, component_name: str) -> bool:
        """
        Attempt to recover a failed component.
        
        Args:
            component_name: Name of the component to recover
            
        Returns:
            success: Whether recovery was successful
        """
        status = self.component_status.get(component_name)
        if not status:
            return False
            
        if status.status != ComponentStatus.ERROR:
            return True
            
        recovery_config = self.config.get("recovery", {})
        max_attempts = recovery_config.get("max_recovery_attempts", 3)
        
        if status.recovery_attempts >= max_attempts:
            self.logger.error(f"Maximum recovery attempts reached for {component_name}")
            return False
            
        self.logger.info(f"Attempting to recover component {component_name} "
                       f"(attempt {status.recovery_attempts + 1}/{max_attempts})")
        
        status.update(ComponentStatus.RECOVERING)
        status.recovery_attempts += 1
        
        try:
            # Remove the component if it exists
            if component_name in self.components:
                del self.components[component_name]
            
            # Re-initialize the component
            result = self._initialize_component(component_name)
            
            if result:
                self.logger.info(f"Successfully recovered component {component_name}")
                return True
            else:
                self.logger.error(f"Failed to recover component {component_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error recovering component {component_name}: {str(e)}")
            self.logger.exception("Recovery exception details:")
            return False
    
    def update_config(self, new_config: Dict[str, Any], save: bool = True) -> bool:
        """
        Update the system configuration.
        
        Args:
            new_config: New configuration to apply
            save: Whether to save the configuration to disk
            
        Returns:
            success: Whether the update was successful
        """
        try:
            # Merge the new configuration with existing
            self._merge_config(self.config, new_config)
            
            if save:
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            self.logger.info("Updated system configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {str(e)}")
            return False
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge a source config into a target config.
        
        Args:
            target: Target configuration to update
            source: Source configuration to merge in
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def restart_component(self, component_name: str) -> bool:
        """
        Restart a system component.
        
        Args:
            component_name: Name of the component to restart
            
        Returns:
            success: Whether the restart was successful
        """
        self.logger.info(f"Restarting component: {component_name}")
        
        # Shutdown the component
        self._shutdown_component(component_name)
        
        # Re-initialize the component
        return self._initialize_component(component_name)
    
    def handle_error(self, component_name: str, error: Exception, 
                   context: Dict[str, Any] = None) -> bool:
        """
        Handle an error in a system component.
        
        Args:
            component_name: Name of the component where the error occurred
            error: The error that occurred
            context: Additional context about the error
            
        Returns:
            handled: Whether the error was handled
        """
        self.logger.error(f"Error in component {component_name}: {str(error)}")
        
        if context:
            self.logger.error(f"Error context: {context}")
        
        # Update component status
        if component_name in self.component_status:
            self.component_status[component_name].update(
                ComponentStatus.ERROR,
                details=context,
                error=error
            )
        
        # Check if component is critical
        recovery_config = self.config.get("recovery", {})
        critical_components = recovery_config.get("critical_components", [])
        
        if component_name in critical_components:
            # Attempt to recover critical component
            self.logger.warning(f"Critical component {component_name} encountered an error, attempting recovery")
            return self._try_recover_component(component_name)
        else:
            # For non-critical components, log but don't automatically recover
            self.logger.info(f"Non-critical component {component_name} encountered an error")
            return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the system.
        
        Returns:
            system_info: System information
        """
        import platform
        import psutil
        
        try:
            # Get basic system information
            system_info = {
                "system_name": self.config.get("system_name", "MacAgent"),
                "version": self.config.get("version", "1.0.0"),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                },
                "python": {
                    "version": platform.python_version(),
                    "implementation": platform.python_implementation(),
                    "compiler": platform.python_compiler()
                },
                "startup_time": datetime.now().isoformat() if not self.startup_complete else None,
                "uptime_seconds": None,
                "components": self.get_component_status(),
                "resources": {
                    "cpu_percent": psutil.cpu_percent(interval=0.1),
                    "memory": {
                        "total": psutil.virtual_memory().total,
                        "available": psutil.virtual_memory().available,
                        "percent": psutil.virtual_memory().percent
                    },
                    "disk": {
                        "total": psutil.disk_usage('/').total,
                        "free": psutil.disk_usage('/').free,
                        "percent": psutil.disk_usage('/').percent
                    }
                }
            }
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"Error getting system info: {str(e)}")
            return {
                "error": str(e),
                "components": self.get_component_status()
            }
