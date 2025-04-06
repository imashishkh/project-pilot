import os
import json
import time
import socket
import platform
import logging
import threading
import subprocess
import traceback
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Callable

class SystemTest:
    """Represents a diagnostic test for a system component."""
    
    def __init__(self, name: str, test_function: Callable, 
                component: str = None, critical: bool = False,
                description: str = None):
        """
        Initialize a system test.
        
        Args:
            name: Name of the test
            test_function: Function that performs the test
            component: Name of the component being tested (or None for system-wide)
            critical: Whether the test is critical for system operation
            description: Description of what the test checks
        """
        self.name = name
        self.test_function = test_function
        self.component = component
        self.critical = critical
        self.description = description
        self.last_run = None
        self.last_result = None
        self.last_duration = None
        self.error = None
    
    def run(self) -> bool:
        """
        Run the test.
        
        Returns:
            passed: Whether the test passed
        """
        self.last_run = datetime.now()
        start_time = time.time()
        self.error = None
        
        try:
            result = self.test_function()
            self.last_result = bool(result)
        except Exception as e:
            self.last_result = False
            self.error = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        
        self.last_duration = time.time() - start_time
        return self.last_result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test to a dictionary.
        
        Returns:
            Dictionary representation of the test
        """
        return {
            "name": self.name,
            "component": self.component,
            "critical": self.critical,
            "description": self.description,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_result": self.last_result,
            "last_duration": self.last_duration,
            "error": self.error
        }

class DiagnosticSystem:
    """
    Implements system-wide health monitoring, diagnostics, and self-testing
    for the AI agent system.
    """
    
    def __init__(self, system_orchestrator, config_path: str = "config/diagnostics.json",
                logs_path: str = "logs/diagnostics", check_interval_minutes: int = 10):
        """
        Initialize the DiagnosticSystem.
        
        Args:
            system_orchestrator: SystemOrchestrator instance
            config_path: Path to the diagnostics configuration file
            logs_path: Path to store diagnostic logs
            check_interval_minutes: Interval between automatic health checks
        """
        self.system_orchestrator = system_orchestrator
        self.config_path = config_path
        self.logs_path = logs_path
        self.check_interval_minutes = check_interval_minutes
        
        # Ensure logs directory exists
        os.makedirs(logs_path, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("diagnostic_system")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        log_file = os.path.join(logs_path, f"diagnostics_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Add formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to logger
        self.logger.addHandler(file_handler)
        
        # Load or create diagnostics config
        self._load_config()
        
        # Test registry
        self.tests: Dict[str, SystemTest] = {}
        
        # Health monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Health history
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = self.config.get("max_history_size", 1000)
        
        # Register standard tests
        self._register_standard_tests()
    
    def _load_config(self) -> None:
        """Load the diagnostics configuration from the config file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Create default diagnostics configuration
                self.config = self._create_default_config()
                with open(self.config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error loading diagnostics config: {str(e)}")
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default diagnostics configuration."""
        return {
            "enabled": True,
            "check_interval_minutes": self.check_interval_minutes,
            "max_history_size": 1000,
            "remote_diagnostics": {
                "enabled": False,
                "host": "localhost",
                "port": 8080,
                "require_auth": True
            },
            "alerts": {
                "enabled": True,
                "log_critical_issues": True,
                "notify_on_critical": True
            },
            "tests": {
                "permission_manager": {
                    "enabled": True,
                    "critical": True
                },
                "safety_protocol": {
                    "enabled": True,
                    "critical": True
                },
                "security_monitor": {
                    "enabled": True,
                    "critical": True
                },
                "system_resources": {
                    "enabled": True,
                    "critical": False,
                    "thresholds": {
                        "cpu_percent": 90,
                        "memory_percent": 90,
                        "disk_percent": 95
                    }
                },
                "network_connectivity": {
                    "enabled": True,
                    "critical": False,
                    "targets": ["8.8.8.8", "1.1.1.1"]
                }
            }
        }
    
    def _register_standard_tests(self) -> None:
        """Register standard system tests."""
        self.logger.info("Registering standard diagnostic tests")
        tests_config = self.config.get("tests", {})
        
        # Permission Manager Test
        if tests_config.get("permission_manager", {}).get("enabled", True):
            self.register_test(
                "permission_manager_status",
                self._test_permission_manager,
                component="permission_manager",
                critical=tests_config.get("permission_manager", {}).get("critical", True),
                description="Checks if the Permission Manager is running and accessible"
            )
        
        # Safety Protocol Test
        if tests_config.get("safety_protocol", {}).get("enabled", True):
            self.register_test(
                "safety_protocol_status",
                self._test_safety_protocol,
                component="safety_protocol",
                critical=tests_config.get("safety_protocol", {}).get("critical", True),
                description="Checks if the Safety Protocol is running and accessible"
            )
        
        # Security Monitor Test
        if tests_config.get("security_monitor", {}).get("enabled", True):
            self.register_test(
                "security_monitor_status",
                self._test_security_monitor,
                component="security_monitor",
                critical=tests_config.get("security_monitor", {}).get("critical", True),
                description="Checks if the Security Monitor is running and accessible"
            )
        
        # System Resources Test
        if tests_config.get("system_resources", {}).get("enabled", True):
            self.register_test(
                "system_resources",
                self._test_system_resources,
                component=None,
                critical=tests_config.get("system_resources", {}).get("critical", False),
                description="Checks if system resources (CPU, memory, disk) are within acceptable thresholds"
            )
        
        # Network Connectivity Test
        if tests_config.get("network_connectivity", {}).get("enabled", True):
            self.register_test(
                "network_connectivity",
                self._test_network_connectivity,
                component=None,
                critical=tests_config.get("network_connectivity", {}).get("critical", False),
                description="Checks if the system has network connectivity"
            )
        
        # Configuration Files Test
        self.register_test(
            "configuration_files",
            self._test_configuration_files,
            component=None,
            critical=False,
            description="Checks if all configuration files exist and are valid JSON"
        )
        
        # Log Files Test
        self.register_test(
            "log_files",
            self._test_log_files,
            component=None,
            critical=False,
            description="Checks if log files are being written to and not too large"
        )
        
        self.logger.info(f"Registered {len(self.tests)} diagnostic tests")
    
    def register_test(self, name: str, test_function: Callable,
                    component: str = None, critical: bool = False,
                    description: str = None) -> None:
        """
        Register a new system test.
        
        Args:
            name: Name of the test
            test_function: Function that performs the test
            component: Name of the component being tested (or None for system-wide)
            critical: Whether the test is critical for system operation
            description: Description of what the test checks
        """
        self.tests[name] = SystemTest(
            name=name,
            test_function=test_function,
            component=component,
            critical=critical,
            description=description
        )
        self.logger.debug(f"Registered test: {name}")
    
    def run_test(self, test_name: str) -> bool:
        """
        Run a single test by name.
        
        Args:
            test_name: Name of the test to run
            
        Returns:
            passed: Whether the test passed
        """
        if test_name not in self.tests:
            self.logger.error(f"Test '{test_name}' not found")
            return False
            
        test = self.tests[test_name]
        self.logger.info(f"Running test: {test_name}")
        
        result = test.run()
        
        if result:
            self.logger.info(f"Test '{test_name}' passed")
        else:
            self.logger.warning(f"Test '{test_name}' failed")
            if test.error:
                self.logger.error(f"Test error: {test.error.get('message')}")
        
        return result
    
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run all registered tests.
        
        Returns:
            results: Dictionary of test names to test results
        """
        self.logger.info("Running all diagnostic tests")
        results = {}
        
        for test_name, test in self.tests.items():
            results[test_name] = self.run_test(test_name)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        self.logger.info(f"Test run complete: {passed}/{total} tests passed")
        return results
    
    def run_component_tests(self, component_name: str) -> Dict[str, bool]:
        """
        Run all tests for a specific component.
        
        Args:
            component_name: Name of the component to test
            
        Returns:
            results: Dictionary of test names to test results
        """
        self.logger.info(f"Running tests for component: {component_name}")
        results = {}
        
        for test_name, test in self.tests.items():
            if test.component == component_name:
                results[test_name] = self.run_test(test_name)
        
        if not results:
            self.logger.warning(f"No tests found for component {component_name}")
            
        return results
    
    def start_health_monitoring(self) -> None:
        """Start the health monitoring thread."""
        if not self.config.get("enabled", True):
            self.logger.info("Health monitoring is disabled in config")
            return
            
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.info("Health monitoring is already running")
            return
            
        self.logger.info("Starting health monitoring")
        self.stop_monitoring.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_health_monitoring(self) -> None:
        """Stop the health monitoring thread."""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            return
            
        self.logger.info("Stopping health monitoring")
        self.stop_monitoring.set()
        
        # Wait for thread to finish
        self.monitoring_thread.join(timeout=5)
        
        if self.monitoring_thread.is_alive():
            self.logger.warning("Health monitoring thread did not terminate gracefully")
    
    def _health_monitoring_loop(self) -> None:
        """Main loop for health monitoring."""
        self.logger.info(f"Health monitoring started with interval of {self.check_interval_minutes} minutes")
        
        try:
            while not self.stop_monitoring.is_set():
                # Run a system health check
                self.check_system_health()
                
                # Sleep for the configured interval
                for _ in range(self.check_interval_minutes * 60):
                    if self.stop_monitoring.is_set():
                        break
                    time.sleep(1)
        
        except Exception as e:
            self.logger.error(f"Error in health monitoring loop: {str(e)}")
            self.logger.exception("Health monitoring exception details:")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform a system health check.
        
        Returns:
            health_status: Health status information
        """
        self.logger.info("Performing system health check")
        
        # Run critical tests first
        critical_tests = {name: test for name, test in self.tests.items() if test.critical}
        critical_results = {}
        
        for test_name, test in critical_tests.items():
            critical_results[test_name] = self.run_test(test_name)
        
        # Then run non-critical tests
        non_critical_tests = {name: test for name, test in self.tests.items() if not test.critical}
        non_critical_results = {}
        
        for test_name, test in non_critical_tests.items():
            non_critical_results[test_name] = self.run_test(test_name)
        
        # Combine results
        all_results = {**critical_results, **non_critical_results}
        
        # Get component statuses
        component_statuses = self.system_orchestrator.get_component_status()
        
        # Check for critical failures
        critical_failures = [name for name, result in critical_results.items() if not result]
        has_critical_failures = len(critical_failures) > 0
        
        # Create health record
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "degraded" if has_critical_failures else "healthy",
            "tests": {name: test.to_dict() for name, test in self.tests.items()},
            "components": component_statuses,
            "critical_failures": critical_failures,
            "system_resources": self._get_system_resources()
        }
        
        # Add to history
        self.health_history.append(health_status)
        
        # Trim history if needed
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
        
        # Log critical failures
        if has_critical_failures and self.config.get("alerts", {}).get("log_critical_issues", True):
            self.logger.error(f"Critical test failures detected: {', '.join(critical_failures)}")
            
            # Here you would implement additional alerting (email, notifications, etc.)
            if self.config.get("alerts", {}).get("notify_on_critical", True):
                self._send_critical_alert(critical_failures)
        
        return health_status
    
    def generate_system_report(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Generate a comprehensive system status report.
        
        Args:
            include_history: Whether to include historical health data
            
        Returns:
            report: System status report
        """
        self.logger.info("Generating system report")
        
        # Get system info from orchestrator
        system_info = self.system_orchestrator.get_system_info()
        
        # Get latest test results
        test_results = {name: test.to_dict() for name, test in self.tests.items()}
        
        # Create report
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_info": system_info,
            "tests": test_results,
            "resources": self._get_system_resources(),
            "python_info": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler()
            }
        }
        
        # Include history if requested
        if include_history:
            report["health_history"] = self.health_history
        
        return report
    
    def _test_permission_manager(self) -> bool:
        """Test if the Permission Manager is running correctly."""
        permission_manager = self.system_orchestrator.get_component("permission_manager")
        if not permission_manager:
            return False
        
        # Check if it has required attributes
        required_attrs = ["permissions", "granted_permissions", "request_permission"]
        for attr in required_attrs:
            if not hasattr(permission_manager, attr):
                return False
        
        # Check basic functionality
        try:
            # Try to access a property 
            permissions = permission_manager.permissions
            return True
        except Exception:
            return False
    
    def _test_safety_protocol(self) -> bool:
        """Test if the Safety Protocol is running correctly."""
        safety_protocol = self.system_orchestrator.get_component("safety_protocol")
        if not safety_protocol:
            return False
        
        # Check if it has required attributes
        required_attrs = ["config", "is_emergency_stop_active", "request_confirmation"]
        for attr in required_attrs:
            if not hasattr(safety_protocol, attr):
                return False
        
        # Check emergency stop status
        try:
            # Make sure emergency stop is not active
            if safety_protocol.is_emergency_stop_active():
                return False
            return True
        except Exception:
            return False
    
    def _test_security_monitor(self) -> bool:
        """Test if the Security Monitor is running correctly."""
        security_monitor = self.system_orchestrator.get_component("security_monitor")
        if not security_monitor:
            return False
        
        # Check if it has required attributes
        required_attrs = ["config", "log_security_event", "is_suspicious_activity"]
        for attr in required_attrs:
            if not hasattr(security_monitor, attr):
                return False
        
        # Check basic functionality
        try:
            # Try to check if an activity is suspicious (should return False for non-suspicious)
            result, _ = security_monitor.is_suspicious_activity("test", "harmless_resource")
            # If non-suspicious, the result should be False
            return not result
        except Exception:
            return False
    
    def _test_system_resources(self) -> bool:
        """Test if system resources are within acceptable thresholds."""
        resources = self._get_system_resources()
        thresholds = self.config.get("tests", {}).get("system_resources", {}).get("thresholds", {})
        
        # Check CPU usage
        if resources["cpu_percent"] > thresholds.get("cpu_percent", 90):
            return False
        
        # Check memory usage
        if resources["memory_percent"] > thresholds.get("memory_percent", 90):
            return False
        
        # Check disk usage
        if resources["disk_percent"] > thresholds.get("disk_percent", 95):
            return False
        
        return True
    
    def _test_network_connectivity(self) -> bool:
        """Test if the system has network connectivity."""
        targets = self.config.get("tests", {}).get("network_connectivity", {}).get("targets", ["8.8.8.8", "1.1.1.1"])
        
        for target in targets:
            try:
                # Try to create a socket connection to the target
                socket.create_connection((target, 53), timeout=3)
                return True
            except OSError:
                continue
                
        return False
    
    def _test_configuration_files(self) -> bool:
        """Test if configuration files exist and are valid JSON."""
        config_files = [
            self.system_orchestrator.config_path,
            self.config_path
        ]
        
        components = self.system_orchestrator.config.get("components", {})
        for component, config in components.items():
            if "config_path" in config:
                config_files.append(config["config_path"])
        
        for config_file in config_files:
            if not os.path.exists(config_file):
                return False
            
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError:
                return False
                
        return True
    
    def _test_log_files(self) -> bool:
        """Test if log files are being written to and not too large."""
        log_dir = self.system_orchestrator.config.get("logging", {}).get("log_dir", "logs")
        
        if not os.path.exists(log_dir):
            return False
            
        # Check if there are any log files
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        if not log_files:
            return False
            
        # Check if the most recent log file is being updated
        latest_log = max([os.path.join(log_dir, f) for f in log_files], 
                       key=os.path.getmtime)
        
        # Check if file was modified in the last hour
        if time.time() - os.path.getmtime(latest_log) > 3600:
            return False
            
        return True
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource usage information."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "memory_total_mb": psutil.virtual_memory().total / (1024 * 1024),
            "disk_percent": psutil.disk_usage('/').percent,
            "disk_free_gb": psutil.disk_usage('/').free / (1024 * 1024 * 1024),
            "disk_total_gb": psutil.disk_usage('/').total / (1024 * 1024 * 1024)
        }
    
    def _send_critical_alert(self, failures: List[str]) -> None:
        """
        Send an alert for critical failures.
        
        Args:
            failures: List of failed critical tests
        """
        self.logger.critical(f"ALERT: Critical system failures detected: {', '.join(failures)}")
        # In a real implementation, this would send emails, SMS, or other notifications
    
    def shutdown(self) -> None:
        """Shut down the diagnostic system."""
        self.logger.info("Shutting down diagnostic system")
        self.stop_health_monitoring()
