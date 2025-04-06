#!/usr/bin/env python3
"""
ResourceMonitor module for MacAgent.

Monitors and manages system resources to ensure optimal agent performance.
Provides mechanisms for adaptive resource usage and prevents resource exhaustion.
"""

import os
import sys
import psutil
import logging
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np


class ResourceAlert:
    """Resource alert notification."""
    
    def __init__(self, resource_type: str, threshold: float, current_value: float,
                 severity: str, timestamp: float, message: str):
        """
        Initialize a resource alert.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk, etc.)
            threshold: Alert threshold value
            current_value: Current value that triggered the alert
            severity: Alert severity (warning, critical)
            timestamp: Alert timestamp
            message: Alert message
        """
        self.resource_type = resource_type
        self.threshold = threshold
        self.current_value = current_value
        self.severity = severity
        self.timestamp = timestamp
        self.message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "resource_type": self.resource_type,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "message": self.message,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of the alert."""
        dt = datetime.fromtimestamp(self.timestamp)
        return (f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}] {self.severity.upper()} {self.resource_type}: "
                f"{self.message} ({self.current_value:.1f}% > {self.threshold:.1f}%)")


class ResourceThresholds:
    """Thresholds for resource monitoring."""
    
    def __init__(self, 
                 cpu_warning: float = 80.0,
                 cpu_critical: float = 90.0,
                 memory_warning: float = 75.0,
                 memory_critical: float = 85.0,
                 disk_warning: float = 85.0,
                 disk_critical: float = 95.0,
                 io_warning: float = 70.0,
                 io_critical: float = 85.0):
        """
        Initialize resource thresholds.
        
        Args:
            cpu_warning: CPU usage warning threshold (%)
            cpu_critical: CPU usage critical threshold (%)
            memory_warning: Memory usage warning threshold (%)
            memory_critical: Memory usage critical threshold (%)
            disk_warning: Disk usage warning threshold (%)
            disk_critical: Disk usage critical threshold (%)
            io_warning: IO usage warning threshold (%)
            io_critical: IO usage critical threshold (%)
        """
        self.cpu_warning = cpu_warning
        self.cpu_critical = cpu_critical
        self.memory_warning = memory_warning
        self.memory_critical = memory_critical
        self.disk_warning = disk_warning
        self.disk_critical = disk_critical
        self.io_warning = io_warning
        self.io_critical = io_critical
    
    def update(self, thresholds: Dict[str, float]) -> None:
        """
        Update thresholds from a dictionary.
        
        Args:
            thresholds: Dictionary of threshold values
        """
        for name, value in thresholds.items():
            if hasattr(self, name) and isinstance(value, (int, float)):
                setattr(self, name, float(value))


class ResourceManager:
    """Interface for adaptive resource management."""
    
    def __init__(self):
        """Initialize the resource manager."""
        pass
    
    def adjust_resource_usage(self, resource_type: str, usage_level: float) -> Dict[str, Any]:
        """
        Adjust resource usage based on current levels.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk, etc.)
            usage_level: Current usage level percentage
            
        Returns:
            Dictionary of adjustment parameters
        """
        # Default implementation - should be overridden by subclasses
        return {}
    
    def get_optimal_parallelism(self, cpu_usage: float) -> int:
        """
        Get optimal level of parallelism based on CPU usage.
        
        Args:
            cpu_usage: Current CPU usage percentage
            
        Returns:
            Optimal number of parallel processes/threads
        """
        total_cpus = os.cpu_count() or 4
        
        if cpu_usage < 50.0:
            # Low usage, can use more parallelism
            return max(1, total_cpus - 1)
        elif cpu_usage < 70.0:
            # Moderate usage, use half available CPUs
            return max(1, total_cpus // 2)
        elif cpu_usage < 85.0:
            # High usage, use quarter of available CPUs
            return max(1, total_cpus // 4)
        else:
            # Very high usage, minimal parallelism
            return 1


class ResourceMonitor:
    """
    Monitors system resources and provides adaptive resource management.
    
    Features:
    - Tracks system resource availability
    - Implements adaptive resource usage
    - Prevents resource exhaustion
    - Optimizes parallel operations based on available resources
    - Provides alerts for resource constraints
    """
    
    def __init__(self, 
                 storage_dir: str = "memory/resource_monitor",
                 logging_level: int = logging.INFO,
                 sampling_interval: float = 1.0,
                 alert_callback: Optional[Callable[[ResourceAlert], None]] = None,
                 resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the resource monitor.
        
        Args:
            storage_dir: Directory to store resource monitoring data
            logging_level: Level for logging
            sampling_interval: Interval for sampling system metrics in seconds
            alert_callback: Callback function for resource alerts
            resource_manager: Custom resource manager implementation
        """
        self.storage_dir = storage_dir
        self.sampling_interval = sampling_interval
        self.alert_callback = alert_callback
        
        # Ensure storage directory exists
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("ResourceMonitor")
        self.logger.setLevel(logging_level)
        
        # Resource thresholds
        self.thresholds = ResourceThresholds()
        
        # Resource manager
        self.resource_manager = resource_manager or ResourceManager()
        
        # Resource data storage
        self.samples = []
        self.alerts = []
        self.resource_history = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "io": [],
            "network": []
        }
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Thread pool for parallel operations
        self.thread_pool = None
        
        # Load configuration if available
        config_path = os.path.join(storage_dir, "config.json")
        if os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._create_default_config(config_path)
        
        # Initialize for macOS-specific monitoring
        self._init_mac_specific()
        
        self.logger.info("Resource monitor initialized")
    
    def _init_mac_specific(self):
        """Initialize macOS-specific monitoring."""
        self.is_mac = sys.platform == 'darwin'
        if self.is_mac:
            try:
                # Check if we can access macOS-specific performance data
                self.has_mac_perf = True
                self.logger.info("macOS-specific monitoring enabled")
            except Exception:
                self.has_mac_perf = False
                self.logger.info("macOS-specific monitoring not available")
    
    def _create_default_config(self, config_path: str) -> None:
        """Create a default configuration file if none exists."""
        default_config = {
            "sampling_interval": self.sampling_interval,
            "thresholds": {
                "cpu_warning": self.thresholds.cpu_warning,
                "cpu_critical": self.thresholds.cpu_critical,
                "memory_warning": self.thresholds.memory_warning,
                "memory_critical": self.thresholds.memory_critical,
                "disk_warning": self.thresholds.disk_warning,
                "disk_critical": self.thresholds.disk_critical,
                "io_warning": self.thresholds.io_warning,
                "io_critical": self.thresholds.io_critical
            }
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Default configuration created at {config_path}")
        except Exception as e:
            self.logger.error(f"Error creating default configuration: {e}")
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if configuration was loaded successfully
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Update sampling interval
            if "sampling_interval" in config:
                self.sampling_interval = float(config["sampling_interval"])
            
            # Update thresholds
            if "thresholds" in config:
                self.thresholds.update(config["thresholds"])
            
            self.logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring of system resources."""
        if self._monitoring:
            self.logger.warning("Resource monitoring is already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            daemon=True,
            name="ResourceMonitorThread"
        )
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring of system resources."""
        if not self._monitoring:
            self.logger.warning("Resource monitoring is not running")
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self) -> None:
        """Continuously monitor system resources at the specified interval."""
        while self._monitoring:
            try:
                sample = self._sample_resources()
                self._check_thresholds(sample)
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                # Continue monitoring even if there's an error
                time.sleep(self.sampling_interval)
    
    def _sample_resources(self) -> Dict[str, Any]:
        """
        Take a sample of current resource usage.
        
        Returns:
            Dictionary of resource metrics
        """
        timestamp = time.time()
        
        # Get CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_stats = {
            "percent": cpu_percent,
            "count": cpu_count,
            "freq_current": cpu_freq.current if cpu_freq else None,
            "freq_max": cpu_freq.max if cpu_freq else None
        }
        
        # Get memory metrics
        memory = psutil.virtual_memory()
        memory_stats = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        
        # Get disk metrics
        disk = psutil.disk_usage('/')
        disk_stats = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        
        # Get I/O metrics
        try:
            disk_io = psutil.disk_io_counters()
            disk_io_stats = {
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "read_count": disk_io.read_count,
                "write_count": disk_io.write_count
            }
        except Exception:
            disk_io_stats = {
                "read_bytes": 0,
                "write_bytes": 0,
                "read_count": 0,
                "write_count": 0
            }
        
        # Get network metrics
        try:
            net_io = psutil.net_io_counters()
            net_io_stats = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            }
        except Exception:
            net_io_stats = {
                "bytes_sent": 0,
                "bytes_recv": 0,
                "packets_sent": 0,
                "packets_recv": 0
            }
        
        # MacOS specific metrics
        mac_stats = {}
        if self.is_mac and self.has_mac_perf:
            try:
                # Add macOS-specific metrics here if needed
                pass
            except Exception:
                pass
        
        # Create and store the sample
        sample = {
            "timestamp": timestamp,
            "cpu": cpu_stats,
            "memory": memory_stats,
            "disk": disk_stats,
            "io": disk_io_stats,
            "network": net_io_stats,
            "mac": mac_stats
        }
        
        # Store sample and update history
        self.samples.append(sample)
        
        # Keep only the last 1000 samples
        if len(self.samples) > 1000:
            self.samples.pop(0)
        
        # Update resource history for trending
        self._update_resource_history(sample)
        
        return sample
    
    def _update_resource_history(self, sample: Dict[str, Any]) -> None:
        """
        Update resource history with the latest sample.
        
        Args:
            sample: Latest resource sample
        """
        self.resource_history["cpu"].append((sample["timestamp"], sample["cpu"]["percent"]))
        self.resource_history["memory"].append((sample["timestamp"], sample["memory"]["percent"]))
        self.resource_history["disk"].append((sample["timestamp"], sample["disk"]["percent"]))
        
        # Keep history at a reasonable size
        max_history = 10000
        for resource_type in self.resource_history:
            if len(self.resource_history[resource_type]) > max_history:
                # Remove oldest entries
                self.resource_history[resource_type] = self.resource_history[resource_type][-max_history:]
    
    def _check_thresholds(self, sample: Dict[str, Any]) -> None:
        """
        Check if any resource thresholds are exceeded.
        
        Args:
            sample: Resource sample to check
        """
        timestamp = sample["timestamp"]
        
        # Check CPU threshold
        cpu_percent = sample["cpu"]["percent"]
        if cpu_percent > self.thresholds.cpu_critical:
            self._create_alert("cpu", self.thresholds.cpu_critical, cpu_percent, 
                              "critical", timestamp, "CPU usage critically high")
        elif cpu_percent > self.thresholds.cpu_warning:
            self._create_alert("cpu", self.thresholds.cpu_warning, cpu_percent, 
                              "warning", timestamp, "CPU usage high")
        
        # Check memory threshold
        memory_percent = sample["memory"]["percent"]
        if memory_percent > self.thresholds.memory_critical:
            self._create_alert("memory", self.thresholds.memory_critical, memory_percent, 
                              "critical", timestamp, "Memory usage critically high")
        elif memory_percent > self.thresholds.memory_warning:
            self._create_alert("memory", self.thresholds.memory_warning, memory_percent, 
                              "warning", timestamp, "Memory usage high")
        
        # Check disk threshold
        disk_percent = sample["disk"]["percent"]
        if disk_percent > self.thresholds.disk_critical:
            self._create_alert("disk", self.thresholds.disk_critical, disk_percent, 
                              "critical", timestamp, "Disk usage critically high")
        elif disk_percent > self.thresholds.disk_warning:
            self._create_alert("disk", self.thresholds.disk_warning, disk_percent, 
                              "warning", timestamp, "Disk usage high")
    
    def _create_alert(self, resource_type: str, threshold: float, current_value: float,
                     severity: str, timestamp: float, message: str) -> None:
        """
        Create and process a resource alert.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk, etc.)
            threshold: Alert threshold value
            current_value: Current value that triggered the alert
            severity: Alert severity (warning, critical)
            timestamp: Alert timestamp
            message: Alert message
        """
        alert = ResourceAlert(
            resource_type=resource_type,
            threshold=threshold,
            current_value=current_value,
            severity=severity,
            timestamp=timestamp,
            message=message
        )
        
        # Store the alert
        self.alerts.append(alert)
        
        # Log the alert
        if severity == "critical":
            self.logger.error(str(alert))
        else:
            self.logger.warning(str(alert))
        
        # Call the alert callback if available
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage percentages.
        
        Returns:
            Dictionary of resource usage percentages
        """
        if not self.samples:
            # Get a fresh sample if none available
            sample = self._sample_resources()
        else:
            sample = self.samples[-1]
        
        return {
            "cpu": sample["cpu"]["percent"],
            "memory": sample["memory"]["percent"],
            "disk": sample["disk"]["percent"]
        }
    
    def get_resource_details(self) -> Dict[str, Any]:
        """
        Get detailed information about system resources.
        
        Returns:
            Dictionary with detailed resource information
        """
        if not self.samples:
            # Get a fresh sample if none available
            sample = self._sample_resources()
        else:
            sample = self.samples[-1]
            
        # Format memory values in more readable form
        memory = sample["memory"]
        memory_formatted = {
            "total_gb": memory["total"] / (1024 ** 3),
            "available_gb": memory["available"] / (1024 ** 3),
            "used_gb": memory["used"] / (1024 ** 3),
            "percent": memory["percent"]
        }
        
        # Format disk values
        disk = sample["disk"]
        disk_formatted = {
            "total_gb": disk["total"] / (1024 ** 3),
            "free_gb": disk["free"] / (1024 ** 3),
            "used_gb": disk["used"] / (1024 ** 3),
            "percent": disk["percent"]
        }
        
        return {
            "timestamp": datetime.fromtimestamp(sample["timestamp"]).isoformat(),
            "cpu": sample["cpu"],
            "memory": memory_formatted,
            "disk": disk_formatted,
            "process_count": len(psutil.pids())
        }
    
    def predict_resource_exhaustion(self, resource_type: str = "memory") -> Optional[Dict[str, Any]]:
        """
        Predict when a resource might be exhausted based on trend analysis.
        
        Args:
            resource_type: Type of resource to analyze (cpu, memory, disk)
            
        Returns:
            Dictionary with prediction details or None if prediction not possible
        """
        if resource_type not in self.resource_history or len(self.resource_history[resource_type]) < 10:
            return None
        
        history = self.resource_history[resource_type]
        
        # Extract timestamps and values
        timestamps = [entry[0] for entry in history]
        values = [entry[1] for entry in history]
        
        # Normalize timestamps to seconds from start
        start_time = timestamps[0]
        norm_timestamps = [(t - start_time) for t in timestamps]
        
        try:
            # Fit a linear regression to predict trend
            coeffs = np.polyfit(norm_timestamps, values, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # If resource usage is decreasing or stable, no risk of exhaustion
            if slope <= 0:
                return {
                    "resource_type": resource_type,
                    "trend": "stable_or_decreasing",
                    "current_value": values[-1],
                    "slope": slope,
                    "exhaustion_risk": "none"
                }
            
            # Calculate time to reach critical threshold
            current_value = values[-1]
            if resource_type == "cpu":
                threshold = self.thresholds.cpu_critical
            elif resource_type == "memory":
                threshold = self.thresholds.memory_critical
            else:  # disk
                threshold = self.thresholds.disk_critical
            
            if current_value >= threshold:
                # Already at or above threshold
                time_to_threshold = 0
            else:
                # Calculate time to reach threshold in seconds
                time_to_threshold = (threshold - current_value) / slope
            
            # Convert to more readable format
            if time_to_threshold < 60:
                time_str = f"{time_to_threshold:.0f} seconds"
                risk = "immediate"
            elif time_to_threshold < 3600:
                time_str = f"{time_to_threshold / 60:.1f} minutes"
                risk = "high" if time_to_threshold < 600 else "medium"
            else:
                time_str = f"{time_to_threshold / 3600:.1f} hours"
                risk = "low"
            
            return {
                "resource_type": resource_type,
                "trend": "increasing",
                "current_value": current_value,
                "critical_threshold": threshold,
                "slope": slope,
                "time_to_threshold": time_to_threshold,
                "time_to_threshold_str": time_str,
                "exhaustion_risk": risk
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting resource exhaustion: {e}")
            return None
    
    def get_optimal_thread_count(self) -> int:
        """
        Get optimal number of threads/processes for parallel operations.
        
        Returns:
            Optimal thread count based on current system load
        """
        current_usage = self.get_current_usage()
        cpu_usage = current_usage.get("cpu", 0)
        
        return self.resource_manager.get_optimal_parallelism(cpu_usage)
    
    def create_thread_pool(self, max_workers: Optional[int] = None) -> concurrent.futures.ThreadPoolExecutor:
        """
        Create a thread pool with optimal size based on current system resources.
        
        Args:
            max_workers: Maximum number of workers (None for auto-detection)
            
        Returns:
            ThreadPoolExecutor with optimal worker count
        """
        if max_workers is None:
            max_workers = self.get_optimal_thread_count()
        
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ResourceMonitor"
        )
        
        return self.thread_pool
    
    def get_alerts(self, severity: Optional[str] = None, 
                  resource_type: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get resource alerts with optional filtering.
        
        Args:
            severity: Filter by severity (warning, critical)
            resource_type: Filter by resource type (cpu, memory, disk)
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        filtered_alerts = []
        
        for alert in reversed(self.alerts):
            if severity and alert.severity != severity:
                continue
            if resource_type and alert.resource_type != resource_type:
                continue
            
            filtered_alerts.append(alert.to_dict())
            
            if len(filtered_alerts) >= limit:
                break
        
        return filtered_alerts
    
    def clear_alerts(self) -> int:
        """
        Clear all stored alerts.
        
        Returns:
            Number of alerts cleared
        """
        count = len(self.alerts)
        self.alerts = []
        return count
    
    def adaptive_sleep(self, base_delay: float) -> float:
        """
        Sleep for an adaptive amount of time based on system load.
        
        Args:
            base_delay: Base delay in seconds
            
        Returns:
            Actual sleep time in seconds
        """
        current_usage = self.get_current_usage()
        cpu_usage = current_usage.get("cpu", 0)
        
        # Adjust delay based on CPU usage
        if cpu_usage > 90:
            multiplier = 2.0  # Double delay when CPU very high
        elif cpu_usage > 75:
            multiplier = 1.5  # 50% longer delay when CPU high
        elif cpu_usage < 30:
            multiplier = 0.75  # Shorter delay when CPU low
        else:
            multiplier = 1.0  # Normal delay
        
        adjusted_delay = base_delay * multiplier
        time.sleep(adjusted_delay)
        
        return adjusted_delay
    
    def should_throttle(self) -> bool:
        """
        Determine if operations should be throttled based on resource usage.
        
        Returns:
            True if operations should be throttled
        """
        current_usage = self.get_current_usage()
        
        # Check if any resource is above warning threshold
        if (current_usage.get("cpu", 0) > self.thresholds.cpu_warning or
            current_usage.get("memory", 0) > self.thresholds.memory_warning):
            return True
        
        return False
    
    def generate_resource_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive resource usage report.
        
        Returns:
            Dictionary containing the resource report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "current_usage": self.get_resource_details(),
            "alerts": {
                "critical": self.get_alerts(severity="critical", limit=5),
                "warning": self.get_alerts(severity="warning", limit=5)
            },
            "predictions": {}
        }
        
        # Add predictions
        for resource_type in ["cpu", "memory", "disk"]:
            prediction = self.predict_resource_exhaustion(resource_type)
            if prediction:
                report["predictions"][resource_type] = prediction
        
        # Add optimization recommendations
        report["recommendations"] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Generate resource optimization recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        current_usage = self.get_current_usage()
        
        # CPU recommendations
        cpu_usage = current_usage.get("cpu", 0)
        if cpu_usage > self.thresholds.cpu_warning:
            recommendations.append(f"Reduce parallel operations - CPU usage is high ({cpu_usage:.1f}%)")
            recommendations.append("Consider limiting background tasks")
        
        # Memory recommendations
        memory_usage = current_usage.get("memory", 0)
        if memory_usage > self.thresholds.memory_warning:
            recommendations.append(f"Reduce memory-intensive operations - Memory usage is high ({memory_usage:.1f}%)")
            recommendations.append("Consider implementing more aggressive caching of results")
        
        # Add general recommendations if none specific
        if not recommendations:
            thread_count = self.get_optimal_thread_count()
            recommendations.append(f"Optimal thread count for operations: {thread_count}")
        
        return recommendations
    
    def save_resource_history(self, filename: Optional[str] = None) -> str:
        """
        Save resource history to a file.
        
        Args:
            filename: Filename to save to, or None for auto-generation
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"resource_history_{timestamp}.json"
        
        filepath = os.path.join(self.storage_dir, filename)
        
        # Prepare data for serialization
        history_data = {}
        for resource_type, history in self.resource_history.items():
            history_data[resource_type] = [
                {"timestamp": entry[0], "value": entry[1]}
                for entry in history
            ]
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"Resource history saved to {filepath}")
        return filepath
    
    def plot_resource_usage(self, resource_type: str = "all", 
                           duration_minutes: Optional[float] = None,
                           filename: Optional[str] = None) -> Optional[str]:
        """
        Generate a plot of resource usage.
        
        Args:
            resource_type: Type of resource to plot (cpu, memory, disk, all)
            duration_minutes: Time window to plot in minutes, or None for all data
            filename: Filename to save plot to, or None for auto-generation
            
        Returns:
            Path to the saved plot file or None if plotting failed
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"resource_usage_{resource_type}_{timestamp}.png"
            
            filepath = os.path.join(self.storage_dir, filename)
            
            # Set up the plot
            plt.figure(figsize=(12, 8))
            
            # Filter by duration if specified
            current_time = time.time()
            
            # Plot each resource type
            if resource_type in ["cpu", "all"]:
                self._plot_resource("cpu", duration_minutes, current_time, "CPU Usage (%)", "tab:red")
            
            if resource_type in ["memory", "all"]:
                self._plot_resource("memory", duration_minutes, current_time, "Memory Usage (%)", "tab:blue")
            
            if resource_type in ["disk", "all"]:
                self._plot_resource("disk", duration_minutes, current_time, "Disk Usage (%)", "tab:green")
            
            # Add thresholds if not plotting 'all'
            if resource_type != "all":
                if resource_type == "cpu":
                    plt.axhline(y=self.thresholds.cpu_warning, color='orange', linestyle='--', alpha=0.7, label="Warning")
                    plt.axhline(y=self.thresholds.cpu_critical, color='red', linestyle='--', alpha=0.7, label="Critical")
                elif resource_type == "memory":
                    plt.axhline(y=self.thresholds.memory_warning, color='orange', linestyle='--', alpha=0.7, label="Warning")
                    plt.axhline(y=self.thresholds.memory_critical, color='red', linestyle='--', alpha=0.7, label="Critical")
                elif resource_type == "disk":
                    plt.axhline(y=self.thresholds.disk_warning, color='orange', linestyle='--', alpha=0.7, label="Warning")
                    plt.axhline(y=self.thresholds.disk_critical, color='red', linestyle='--', alpha=0.7, label="Critical")
            
            # Finalize plot
            plt.title(f"Resource Usage Over Time")
            plt.xlabel("Time")
            plt.ylabel("Usage (%)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(filepath)
            plt.close()
            
            self.logger.info(f"Resource usage plot saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error generating resource usage plot: {e}")
            return None
    
    def _plot_resource(self, resource_type: str, duration_minutes: Optional[float],
                      current_time: float, label: str, color: str) -> None:
        """
        Plot a specific resource's usage history.
        
        Args:
            resource_type: Type of resource to plot
            duration_minutes: Time window in minutes
            current_time: Current timestamp
            label: Label for the plot
            color: Color for the plot line
        """
        if resource_type not in self.resource_history or not self.resource_history[resource_type]:
            return
        
        history = self.resource_history[resource_type]
        
        # Filter by duration if specified
        if duration_minutes is not None:
            cutoff_time = current_time - (duration_minutes * 60)
            history = [entry for entry in history if entry[0] >= cutoff_time]
        
        if not history:
            return
        
        # Extract timestamps and values
        timestamps = [entry[0] for entry in history]
        values = [entry[1] for entry in history]
        
        # Convert timestamps to datetime for better x-axis labels
        datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Plot the data
        plt.plot(datetimes, values, label=label, color=color) 