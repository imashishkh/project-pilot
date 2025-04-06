#!/usr/bin/env python3
"""
Performance Optimization System for MacAgent.

This module integrates all components of the performance optimization system,
including the PerformanceProfiler, OptimizationManager, and ResourceMonitor.
It provides a unified interface for performance monitoring and optimization.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum, auto
from pathlib import Path

from MacAgent.src.utils.performance_profiler import PerformanceProfiler
from MacAgent.src.utils.optimization_manager import OptimizationManager, OptimizationLevel
from MacAgent.src.utils.resource_monitor import ResourceMonitor, ResourceAlert


class PerformanceSystem:
    """
    Unified performance optimization system for MacAgent.
    
    Integrates all performance optimization components and provides a centralized
    interface for monitoring, profiling, and optimizing agent performance.
    """
    
    class State(Enum):
        """Performance system state."""
        INACTIVE = auto()
        ACTIVE = auto()
        MONITORING = auto()
        OPTIMIZING = auto()
    
    def __init__(self, 
                 base_dir: str = "memory/performance",
                 logging_level: int = logging.INFO,
                 auto_start: bool = True,
                 default_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        """
        Initialize the performance system.
        
        Args:
            base_dir: Base directory for storing performance data
            logging_level: Logging level
            auto_start: Whether to automatically start the system
            default_optimization_level: Default optimization level
        """
        self.base_dir = base_dir
        
        # Ensure base directory exists
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("PerformanceSystem")
        self.logger.setLevel(logging_level)
        
        if not self.logger.handlers:
            # Add console handler if none exists
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Add file handler
            log_file = os.path.join(base_dir, "performance_system.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Component directories
        profiler_dir = os.path.join(base_dir, "profiler")
        optimizer_dir = os.path.join(base_dir, "optimizer")
        monitor_dir = os.path.join(base_dir, "monitor")
        
        # Initialize components
        self.profiler = PerformanceProfiler(storage_dir=profiler_dir, logging_level=logging_level)
        self.optimizer = OptimizationManager(
            config_path=os.path.join(optimizer_dir, "config.json"),
            cache_dir=os.path.join(optimizer_dir, "cache"),
            profiler=self.profiler,
            logging_level=logging_level
        )
        self.monitor = ResourceMonitor(
            storage_dir=monitor_dir,
            logging_level=logging_level,
            alert_callback=self._handle_resource_alert
        )
        
        # System state
        self.state = self.State.INACTIVE
        
        # Performance metrics
        self.performance_metrics = {
            "startup_time": None,
            "system_uptime": 0,
            "optimization_count": 0,
            "profiling_sessions": 0,
            "alerts": []
        }
        
        # Start the system if auto_start is True
        if auto_start:
            self.start(default_optimization_level)
    
    def start(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> bool:
        """
        Start the performance system.
        
        Args:
            optimization_level: Initial optimization level
            
        Returns:
            True if started successfully
        """
        if self.state != self.State.INACTIVE:
            self.logger.warning("Performance system is already active")
            return False
        
        self.logger.info("Starting performance system")
        
        try:
            # Start resource monitoring
            self.monitor.start_monitoring()
            
            # Set optimization level
            self.optimizer.set_optimization_level(optimization_level)
            
            # Record startup time
            self.performance_metrics["startup_time"] = time.time()
            
            # Update state
            self.state = self.State.ACTIVE
            
            self.logger.info(f"Performance system started with optimization level: {optimization_level.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting performance system: {e}")
            self.stop()  # Cleanup in case of partial start
            return False
    
    def stop(self) -> bool:
        """
        Stop the performance system.
        
        Returns:
            True if stopped successfully
        """
        if self.state == self.State.INACTIVE:
            self.logger.warning("Performance system is already inactive")
            return False
        
        self.logger.info("Stopping performance system")
        
        try:
            # Stop resource monitoring
            self.monitor.stop_monitoring()
            
            # Update uptime
            if self.performance_metrics["startup_time"]:
                self.performance_metrics["system_uptime"] += time.time() - self.performance_metrics["startup_time"]
            
            # Generate final reports
            self._save_final_reports()
            
            # Update state
            self.state = self.State.INACTIVE
            
            self.logger.info("Performance system stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping performance system: {e}")
            # Force state to inactive even if there's an error
            self.state = self.State.INACTIVE
            return False
    
    def _save_final_reports(self) -> None:
        """Save final performance reports."""
        try:
            # Generate summary report
            report = self.generate_performance_report()
            
            # Save report
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_path = os.path.join(self.base_dir, f"performance_report_{timestamp}.json")
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Final performance report saved to {report_path}")
            
            # Save resource history
            self.monitor.save_resource_history()
            
            # Save profiling data
            self.profiler.save_profiling_data()
            
        except Exception as e:
            self.logger.error(f"Error saving final reports: {e}")
    
    def _handle_resource_alert(self, alert: ResourceAlert) -> None:
        """
        Handle resource alerts from the monitor.
        
        Args:
            alert: Resource alert
        """
        # Log the alert
        alert_str = str(alert)
        if alert.severity == "critical":
            self.logger.error(f"CRITICAL RESOURCE ALERT: {alert_str}")
        else:
            self.logger.warning(f"Resource alert: {alert_str}")
        
        # Store the alert
        self.performance_metrics["alerts"].append(alert.to_dict())
        
        # Take action based on alert
        if alert.severity == "critical":
            # For critical alerts, adjust optimization level to conserve resources
            if alert.resource_type == "memory":
                self.logger.info("Adjusting to SPEED optimization level due to memory constraints")
                self.optimizer.set_optimization_level(OptimizationLevel.SPEED)
            elif alert.resource_type == "cpu":
                self.logger.info("Adjusting to ACCURACY optimization level due to CPU constraints")
                self.optimizer.set_optimization_level(OptimizationLevel.ACCURACY)
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """
        Profile a function and return its result.
        
        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result of the function
        """
        self.performance_metrics["profiling_sessions"] += 1
        return self.profiler.profile_function(func, *args, **kwargs)
    
    def start_block_profiling(self, block_name: str) -> None:
        """
        Start profiling a block of code.
        
        Args:
            block_name: Name of the code block
        """
        self.profiler.start_block(block_name)
    
    def end_block_profiling(self, block_name: str) -> Dict[str, Any]:
        """
        End profiling a block of code.
        
        Args:
            block_name: Name of the code block
            
        Returns:
            Profiling results for the block
        """
        result = self.profiler.end_block(block_name)
        self.performance_metrics["profiling_sessions"] += 1
        return result
    
    def optimize(self, 
                domain: str, 
                context: Dict[str, Any] = None, 
                force_level: Optional[OptimizationLevel] = None) -> Dict[str, Any]:
        """
        Apply optimizations for a specific domain.
        
        Args:
            domain: Optimization domain (vision, intelligence, interaction, core)
            context: Context information for optimization
            force_level: Override the current optimization level
            
        Returns:
            Optimization parameters
        """
        self.state = self.State.OPTIMIZING
        self.performance_metrics["optimization_count"] += 1
        
        try:
            # Apply the optimization
            params = self.optimizer.optimize_domain(domain, context, force_level)
            
            # Add resource context
            resource_context = self.monitor.get_current_usage()
            if not params.get("resource_context"):
                params["resource_context"] = resource_context
            
            # Add adaptive throttling if needed
            if self.monitor.should_throttle():
                params["throttle"] = True
                params["throttle_factor"] = 0.5 if resource_context.get("cpu", 0) > 90 else 0.8
            
            return params
            
        finally:
            self.state = self.State.ACTIVE
    
    def cache_result(self, domain: str, key: str, result: Any, ttl: int = 3600) -> bool:
        """
        Cache a result for future use.
        
        Args:
            domain: Domain the result belongs to
            key: Cache key
            result: Result to cache
            ttl: Time to live in seconds
            
        Returns:
            True if caching was successful
        """
        return self.optimizer.cache_result(domain, key, result, ttl)
    
    def get_cached_result(self, domain: str, key: str) -> Optional[Any]:
        """
        Get a cached result.
        
        Args:
            domain: Domain the result belongs to
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        return self.optimizer.get_cached_result(domain, key)
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Analyze performance bottlenecks.
        
        Returns:
            List of identified bottlenecks
        """
        return self.profiler.identify_bottlenecks()
    
    def get_resource_state(self) -> Dict[str, Any]:
        """
        Get current resource state.
        
        Returns:
            Dictionary with resource state information
        """
        return {
            "resource_usage": self.monitor.get_resource_details(),
            "should_throttle": self.monitor.should_throttle(),
            "optimal_thread_count": self.monitor.get_optimal_thread_count(),
            "alerts": self.monitor.get_alerts(limit=5)
        }
    
    def get_optimization_state(self) -> Dict[str, Any]:
        """
        Get current optimization state.
        
        Returns:
            Dictionary with optimization state information
        """
        return {
            "current_level": self.optimizer.current_level.name,
            "domain_settings": {
                "vision": self.optimizer.get_domain_settings("vision"),
                "intelligence": self.optimizer.get_domain_settings("intelligence"),
                "interaction": self.optimizer.get_domain_settings("interaction"),
                "core": self.optimizer.get_domain_settings("core")
            },
            "cache_stats": self.optimizer.get_cache_stats()
        }
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get summary of profiling data.
        
        Returns:
            Dictionary with profiling summary
        """
        return self.profiler.get_summary()
    
    def suggest_optimizations(self) -> List[str]:
        """
        Suggest optimizations based on profiling and resource data.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Get bottlenecks
        bottlenecks = self.analyze_bottlenecks()
        
        # Get resource recommendations
        resource_report = self.monitor.generate_resource_report()
        resource_recommendations = resource_report.get("recommendations", [])
        
        # Add bottleneck-specific suggestions
        for bottleneck in bottlenecks[:3]:  # Focus on top 3 bottlenecks
            name = bottleneck.get("name", "Unknown")
            impact = bottleneck.get("impact", 0)
            category = bottleneck.get("category", "Unknown")
            
            if impact > 30:  # High impact bottleneck
                if category == "cpu":
                    suggestions.append(f"Optimize {name}: Consider caching results or reducing complexity")
                elif category == "memory":
                    suggestions.append(f"Optimize {name}: Consider reducing memory usage or implementing streaming")
                elif category == "io":
                    suggestions.append(f"Optimize {name}: Consider asynchronous I/O or bulk operations")
            elif impact > 10:  # Medium impact bottleneck
                suggestions.append(f"Consider optimizing {name} (medium impact: {impact:.1f}%)")
        
        # Add resource recommendations
        suggestions.extend(resource_recommendations)
        
        # Add optimization level suggestions
        resource_usage = self.monitor.get_current_usage()
        current_level = self.optimizer.current_level
        
        if resource_usage.get("cpu", 0) > 85 and current_level != OptimizationLevel.ACCURACY:
            suggestions.append("Consider switching to ACCURACY optimization mode to reduce CPU usage")
        elif resource_usage.get("memory", 0) > 80 and current_level != OptimizationLevel.SPEED:
            suggestions.append("Consider switching to SPEED optimization mode to reduce memory usage")
        elif (resource_usage.get("cpu", 0) < 30 and 
              resource_usage.get("memory", 0) < 50 and 
              current_level != OptimizationLevel.ULTRA_SPEED):
            suggestions.append("Resources available: Could switch to ULTRA_SPEED for maximum performance")
        
        return suggestions
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing the performance report
        """
        # Calculate uptime
        current_uptime = self.performance_metrics["system_uptime"]
        if self.state != self.State.INACTIVE and self.performance_metrics["startup_time"]:
            current_uptime += time.time() - self.performance_metrics["startup_time"]
        
        report = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_state": self.state.name,
            "uptime_seconds": current_uptime,
            "uptime_formatted": self._format_duration(current_uptime),
            "profiling_sessions": self.performance_metrics["profiling_sessions"],
            "optimization_count": self.performance_metrics["optimization_count"],
            "resource_state": self.get_resource_state(),
            "optimization_state": self.get_optimization_state(),
            "bottlenecks": self.analyze_bottlenecks(),
            "profiling_summary": self.get_profiling_summary(),
            "suggestions": self.suggest_optimizations(),
            "recent_alerts": self.performance_metrics["alerts"][-10:] if self.performance_metrics["alerts"] else []
        }
        
        return report
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to a human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{int(minutes)} minutes {int(seconds % 60)} seconds"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{int(hours)} hours {int(minutes)} minutes"


# Singleton instance for easy access
_performance_system_instance = None

def get_performance_system(
    base_dir: str = "memory/performance",
    logging_level: int = logging.INFO,
    auto_start: bool = True,
    default_optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
) -> PerformanceSystem:
    """
    Get or create the global performance system instance.
    
    Args:
        base_dir: Base directory for storing performance data
        logging_level: Logging level
        auto_start: Whether to automatically start the system
        default_optimization_level: Default optimization level
        
    Returns:
        Global PerformanceSystem instance
    """
    global _performance_system_instance
    
    if _performance_system_instance is None:
        _performance_system_instance = PerformanceSystem(
            base_dir=base_dir,
            logging_level=logging_level,
            auto_start=auto_start,
            default_optimization_level=default_optimization_level
        )
    
    return _performance_system_instance


# Context managers for easy profiling
class ProfileBlock:
    """Context manager for profiling a block of code."""
    
    def __init__(self, block_name: str, performance_system: Optional[PerformanceSystem] = None):
        """
        Initialize a profiling block.
        
        Args:
            block_name: Name of the block
            performance_system: PerformanceSystem instance (uses global instance if None)
        """
        self.block_name = block_name
        self.performance_system = performance_system or get_performance_system()
    
    def __enter__(self):
        """Start profiling the block."""
        self.performance_system.start_block_profiling(self.block_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling the block."""
        self.performance_system.end_block_profiling(self.block_name)
        return False  # Don't suppress exceptions


def profile(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator for profiling functions.
    
    Can be used as:
    @profile
    def my_function():
        pass
    
    Or with custom name:
    @profile(name="Custom Name")
    def my_function():
        pass
    
    Args:
        func: Function to profile
        name: Custom name for the function in profiling results
        
    Returns:
        Decorated function
    """
    def decorator(f):
        def wrapped(*args, **kwargs):
            perf_system = get_performance_system()
            return perf_system.profile_function(f, *args, **kwargs)
        
        # Preserve function metadata
        wrapped.__name__ = f.__name__
        wrapped.__doc__ = f.__doc__
        wrapped.__module__ = f.__module__
        
        # Set custom name if provided
        if name:
            wrapped.__name__ = name
        
        return wrapped
    
    # Handle both @profile and @profile(name="...")
    if func is None:
        return decorator
    return decorator(func) 