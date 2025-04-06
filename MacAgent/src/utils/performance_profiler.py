#!/usr/bin/env python3
"""
PerformanceProfiler module for MacAgent.

Tracks execution time, resource usage, and performance metrics for the agent's components.
Enables identification of bottlenecks and provides detailed performance analysis.
"""

import time
import os
import psutil
import json
import logging
import functools
import traceback
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from collections import defaultdict
import threading
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class PerformanceProfiler:
    """
    Profiles the performance of MacAgent components.
    
    Features:
    - Measures execution time of functions and code blocks
    - Identifies performance bottlenecks through timing analysis
    - Tracks resource usage (CPU, memory, network)
    - Generates detailed performance reports
    - Provides real-time and historical performance analysis
    """
    
    def __init__(self, profile_name: str = "default", 
                 storage_dir: str = "memory/performance_profiles",
                 enabled: bool = True,
                 logging_level: int = logging.INFO,
                 sampling_interval: float = 0.5):
        """
        Initialize the performance profiler.
        
        Args:
            profile_name: Name of this performance profile
            storage_dir: Directory to store performance data
            enabled: Whether profiling is enabled
            logging_level: Level for logging
            sampling_interval: Interval for sampling system metrics in seconds
        """
        self.profile_name = profile_name
        self.storage_dir = storage_dir
        self.enabled = enabled
        self.sampling_interval = sampling_interval
        
        # Ensure storage directory exists
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(f"PerformanceProfiler.{profile_name}")
        self.logger.setLevel(logging_level)
        
        # Performance data storage
        self.function_timings = defaultdict(list)
        self.block_timings = defaultdict(list)
        self.resource_samples = []
        self.sample_timestamps = []
        self.network_usage = []
        self.bottlenecks = []
        
        # For active monitoring
        self._monitoring = False
        self._monitor_thread = None
        
        # Context manager stack for nested profiling
        self.context_stack = []
        
        # Process for resource monitoring
        self.process = psutil.Process(os.getpid())
        
        self.logger.info(f"Performance profiler '{profile_name}' initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous monitoring of system resources."""
        if self._monitoring:
            self.logger.warning("Resource monitoring is already running")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            daemon=True
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
                self._sample_resources()
                time.sleep(self.sampling_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                break
    
    def _sample_resources(self) -> Dict[str, float]:
        """Take a sample of current resource usage."""
        if not self.enabled:
            return {}
        
        try:
            # Get CPU and memory usage for our process
            cpu_percent = self.process.cpu_percent(interval=0.1)
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Get system-wide metrics
            system_cpu = psutil.cpu_percent(interval=None)
            system_memory = psutil.virtual_memory()
            
            # Get disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Get network I/O
            net_io = psutil.net_io_counters()
            
            # Timestamp
            timestamp = time.time()
            
            # Create sample
            sample = {
                "timestamp": timestamp,
                "process_cpu_percent": cpu_percent,
                "process_memory_rss": memory_info.rss / (1024 * 1024),  # MB
                "process_memory_percent": memory_percent,
                "system_cpu_percent": system_cpu,
                "system_memory_percent": system_memory.percent,
                "system_memory_available": system_memory.available / (1024 * 1024),  # MB
                "disk_read_bytes": disk_io.read_bytes,
                "disk_write_bytes": disk_io.write_bytes,
                "net_sent_bytes": net_io.bytes_sent,
                "net_recv_bytes": net_io.bytes_recv
            }
            
            # Store the sample
            self.resource_samples.append(sample)
            self.sample_timestamps.append(timestamp)
            
            # Calculate network usage rate if we have previous samples
            if len(self.resource_samples) > 1:
                prev_sample = self.resource_samples[-2]
                time_diff = sample["timestamp"] - prev_sample["timestamp"]
                if time_diff > 0:
                    net_sent_rate = (sample["net_sent_bytes"] - prev_sample["net_sent_bytes"]) / time_diff
                    net_recv_rate = (sample["net_recv_bytes"] - prev_sample["net_recv_bytes"]) / time_diff
                    
                    network_usage = {
                        "timestamp": timestamp,
                        "sent_bytes_per_sec": net_sent_rate,
                        "recv_bytes_per_sec": net_recv_rate
                    }
                    self.network_usage.append(network_usage)
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error sampling resources: {e}")
            return {}
    
    def profile_function(self, name: Optional[str] = None) -> Callable:
        """
        Decorator to profile a function's execution time.
        
        Args:
            name: Optional custom name for the profiled function
            
        Returns:
            Decorated function with profiling
        """
        def decorator(func):
            # Use provided name or function name
            func_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                # Measure execution time
                start_time = time.time()
                start_resources = self._sample_resources()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    self.logger.error(f"Exception in profiled function {func_name}: {e}")
                    traceback.print_exc()
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    end_resources = self._sample_resources()
                    
                    # Record timing data
                    timing_data = {
                        "function": func_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "success": success
                    }
                    
                    # Calculate resource usage delta
                    if start_resources and end_resources:
                        resource_delta = {}
                        for key in start_resources:
                            if key in end_resources and key != "timestamp":
                                resource_delta[f"{key}_delta"] = end_resources[key] - start_resources[key]
                        timing_data.update(resource_delta)
                    
                    self.function_timings[func_name].append(timing_data)
                    
                    # Log timing information
                    self.logger.debug(f"Function {func_name} took {duration:.6f} seconds")
                    
                    # Check if this might be a bottleneck
                    if duration > 1.0:  # Threshold for potential bottleneck
                        self.logger.info(f"Potential bottleneck: Function {func_name} took {duration:.6f} seconds")
                        self.bottlenecks.append({
                            "type": "function",
                            "name": func_name,
                            "duration": duration,
                            "timestamp": end_time
                        })
                
                return result
            
            return wrapper
        
        return decorator
    
    def profile_block(self, block_name: str) -> "BlockProfiler":
        """
        Create a context manager to profile a block of code.
        
        Args:
            block_name: Name of the code block to profile
            
        Returns:
            Context manager for the profiled block
        """
        return BlockProfiler(self, block_name)
    
    def record_block_timing(self, block_name: str, start_time: float, 
                            end_time: float, success: bool,
                            start_resources: Dict, end_resources: Dict) -> None:
        """
        Record timing data for a profiled code block.
        
        Args:
            block_name: Name of the profiled block
            start_time: Block execution start time
            end_time: Block execution end time
            success: Whether the block executed successfully
            start_resources: Resource usage at start
            end_resources: Resource usage at end
        """
        if not self.enabled:
            return
        
        duration = end_time - start_time
        
        # Record timing data
        timing_data = {
            "block": block_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "success": success,
            "context_depth": len(self.context_stack)
        }
        
        # Calculate resource usage delta
        if start_resources and end_resources:
            resource_delta = {}
            for key in start_resources:
                if key in end_resources and key != "timestamp":
                    resource_delta[f"{key}_delta"] = end_resources[key] - start_resources[key]
            timing_data.update(resource_delta)
        
        self.block_timings[block_name].append(timing_data)
        
        # Log timing information
        self.logger.debug(f"Block {block_name} took {duration:.6f} seconds")
        
        # Check if this might be a bottleneck
        if duration > 1.0:  # Threshold for potential bottleneck
            self.logger.info(f"Potential bottleneck: Block {block_name} took {duration:.6f} seconds")
            self.bottlenecks.append({
                "type": "block",
                "name": block_name,
                "duration": duration,
                "timestamp": end_time
            })
    
    def get_function_stats(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific function or all functions.
        
        Args:
            function_name: Name of function to get stats for, or None for all
            
        Returns:
            Dictionary of function timing statistics
        """
        stats = {}
        
        if function_name:
            if function_name in self.function_timings:
                timings = [t["duration"] for t in self.function_timings[function_name]]
                stats[function_name] = self._calculate_stats(timings)
        else:
            for func_name, timings_list in self.function_timings.items():
                timings = [t["duration"] for t in timings_list]
                stats[func_name] = self._calculate_stats(timings)
        
        return stats
    
    def get_block_stats(self, block_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a specific code block or all blocks.
        
        Args:
            block_name: Name of block to get stats for, or None for all
            
        Returns:
            Dictionary of block timing statistics
        """
        stats = {}
        
        if block_name:
            if block_name in self.block_timings:
                timings = [t["duration"] for t in self.block_timings[block_name]]
                stats[block_name] = self._calculate_stats(timings)
        else:
            for blk_name, timings_list in self.block_timings.items():
                timings = [t["duration"] for t in timings_list]
                stats[blk_name] = self._calculate_stats(timings)
        
        return stats
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of timing values."""
        if not values:
            return {
                "count": 0,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "std_dev": 0
            }
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def identify_bottlenecks(self, threshold_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks based on execution time.
        
        Args:
            threshold_seconds: Minimum time to consider as a bottleneck
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Check function bottlenecks
        for func_name, timings in self.function_timings.items():
            stats = self._calculate_stats([t["duration"] for t in timings])
            if stats["mean"] > threshold_seconds:
                bottlenecks.append({
                    "type": "function",
                    "name": func_name,
                    "mean_duration": stats["mean"],
                    "max_duration": stats["max"],
                    "call_count": stats["count"],
                    "severity": "high" if stats["mean"] > threshold_seconds * 3 else "medium"
                })
        
        # Check block bottlenecks
        for block_name, timings in self.block_timings.items():
            stats = self._calculate_stats([t["duration"] for t in timings])
            if stats["mean"] > threshold_seconds:
                bottlenecks.append({
                    "type": "block",
                    "name": block_name,
                    "mean_duration": stats["mean"],
                    "max_duration": stats["max"],
                    "call_count": stats["count"],
                    "severity": "high" if stats["mean"] > threshold_seconds * 3 else "medium"
                })
        
        # Sort by mean duration (descending)
        bottlenecks.sort(key=lambda x: x["mean_duration"], reverse=True)
        
        return bottlenecks
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """
        Get statistics on resource usage.
        
        Returns:
            Dictionary of resource usage statistics
        """
        if not self.resource_samples:
            return {}
        
        # Initialize result dictionary
        stats = {
            "cpu": {},
            "memory": {},
            "disk": {},
            "network": {}
        }
        
        # Extract series for each metric
        process_cpu = [s["process_cpu_percent"] for s in self.resource_samples]
        system_cpu = [s["system_cpu_percent"] for s in self.resource_samples]
        process_mem = [s["process_memory_percent"] for s in self.resource_samples]
        system_mem = [s["system_memory_percent"] for s in self.resource_samples]
        
        # Calculate statistics
        stats["cpu"]["process"] = self._calculate_stats(process_cpu)
        stats["cpu"]["system"] = self._calculate_stats(system_cpu)
        stats["memory"]["process"] = self._calculate_stats(process_mem)
        stats["memory"]["system"] = self._calculate_stats(system_mem)
        
        # Network statistics if available
        if self.network_usage:
            sent_rates = [n["sent_bytes_per_sec"] for n in self.network_usage]
            recv_rates = [n["recv_bytes_per_sec"] for n in self.network_usage]
            stats["network"]["sent_bytes_per_sec"] = self._calculate_stats(sent_rates)
            stats["network"]["recv_bytes_per_sec"] = self._calculate_stats(recv_rates)
        
        return stats
    
    def generate_report(self, include_plots: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            include_plots: Whether to generate and include plot images
            
        Returns:
            Dictionary containing the full performance report
        """
        report = {
            "profile_name": self.profile_name,
            "timestamp": datetime.now().isoformat(),
            "duration": {
                "start": self.sample_timestamps[0] if self.sample_timestamps else None,
                "end": self.sample_timestamps[-1] if self.sample_timestamps else None,
                "total_seconds": self.sample_timestamps[-1] - self.sample_timestamps[0] if len(self.sample_timestamps) > 1 else 0
            },
            "function_stats": self.get_function_stats(),
            "block_stats": self.get_block_stats(),
            "resource_stats": self.get_resource_usage_stats(),
            "bottlenecks": self.identify_bottlenecks()
        }
        
        # Generate plots if requested
        if include_plots and self.resource_samples:
            plots_dir = os.path.join(self.storage_dir, "plots")
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            
            try:
                # Generate CPU usage plot
                cpu_plot_path = os.path.join(plots_dir, f"{self.profile_name}_cpu_usage.png")
                self._plot_resource_usage("cpu", cpu_plot_path)
                report["plots"] = {"cpu": cpu_plot_path}
                
                # Generate memory usage plot
                mem_plot_path = os.path.join(plots_dir, f"{self.profile_name}_memory_usage.png")
                self._plot_resource_usage("memory", mem_plot_path)
                report["plots"]["memory"] = mem_plot_path
                
                # Generate network usage plot if available
                if self.network_usage:
                    net_plot_path = os.path.join(plots_dir, f"{self.profile_name}_network_usage.png")
                    self._plot_resource_usage("network", net_plot_path)
                    report["plots"]["network"] = net_plot_path
                
                # Generate bottleneck analysis plot
                if self.bottlenecks:
                    bottleneck_plot_path = os.path.join(plots_dir, f"{self.profile_name}_bottlenecks.png")
                    self._plot_bottlenecks(bottleneck_plot_path)
                    report["plots"]["bottlenecks"] = bottleneck_plot_path
            
            except Exception as e:
                self.logger.error(f"Error generating plots: {e}")
                report["plots_error"] = str(e)
        
        # Save the report
        report_path = os.path.join(self.storage_dir, f"{self.profile_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report generated and saved to {report_path}")
        
        return report
    
    def _plot_resource_usage(self, resource_type: str, filename: str) -> None:
        """
        Generate a plot for a specific resource type.
        
        Args:
            resource_type: Type of resource to plot (cpu, memory, network)
            filename: Path to save the plot
        """
        if not self.resource_samples:
            return
        
        plt.figure(figsize=(12, 6))
        
        timestamps = [s["timestamp"] for s in self.resource_samples]
        relative_times = [(t - timestamps[0]) / 60 for t in timestamps]  # Convert to minutes
        
        if resource_type == "cpu":
            process_cpu = [s["process_cpu_percent"] for s in self.resource_samples]
            system_cpu = [s["system_cpu_percent"] for s in self.resource_samples]
            
            plt.plot(relative_times, process_cpu, label="Process CPU %")
            plt.plot(relative_times, system_cpu, label="System CPU %")
            plt.title("CPU Usage Over Time")
            plt.ylabel("CPU Usage (%)")
            
        elif resource_type == "memory":
            process_mem = [s["process_memory_rss"] for s in self.resource_samples]
            system_mem = [s["system_memory_percent"] for s in self.resource_samples]
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel("Process Memory (MB)", color="tab:blue")
            ax1.plot(relative_times, process_mem, color="tab:blue", label="Process Memory (MB)")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            
            ax2 = ax1.twinx()
            ax2.set_ylabel("System Memory (%)", color="tab:red")
            ax2.plot(relative_times, system_mem, color="tab:red", label="System Memory (%)")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            
            plt.title("Memory Usage Over Time")
            fig.tight_layout()
            fig.savefig(filename)
            plt.close(fig)
            return
            
        elif resource_type == "network":
            if not self.network_usage:
                return
                
            net_timestamps = [n["timestamp"] for n in self.network_usage]
            relative_net_times = [(t - timestamps[0]) / 60 for t in net_timestamps]  # Convert to minutes
            
            sent_rates = [n["sent_bytes_per_sec"] / 1024 for n in self.network_usage]  # KB/s
            recv_rates = [n["recv_bytes_per_sec"] / 1024 for n in self.network_usage]  # KB/s
            
            plt.plot(relative_net_times, sent_rates, label="Upload (KB/s)")
            plt.plot(relative_net_times, recv_rates, label="Download (KB/s)")
            plt.title("Network Usage Over Time")
            plt.ylabel("Data Rate (KB/s)")
        
        plt.xlabel("Time (minutes)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _plot_bottlenecks(self, filename: str) -> None:
        """
        Generate a plot showing the identified bottlenecks.
        
        Args:
            filename: Path to save the plot
        """
        if not self.bottlenecks:
            return
        
        # Sort bottlenecks by duration
        sorted_bottlenecks = sorted(self.bottlenecks, key=lambda x: x["duration"])
        
        # Extract names and durations
        names = [f"{b['type']}: {b['name']}" for b in sorted_bottlenecks]
        durations = [b["duration"] for b in sorted_bottlenecks]
        
        # Limit to top 15 for readability
        if len(names) > 15:
            names = names[-15:]
            durations = durations[-15:]
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, max(6, len(names) * 0.4)))
        bars = plt.barh(names, durations, color="tab:red")
        
        # Add duration values at the end of each bar
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{durations[i]:.2f}s",
                va="center"
            )
        
        plt.xlabel("Duration (seconds)")
        plt.title("Performance Bottlenecks")
        plt.grid(True, linestyle="--", alpha=0.7, axis="x")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def save_data(self) -> str:
        """
        Save all performance data to disk.
        
        Returns:
            Path to the saved data file
        """
        data = {
            "profile_name": self.profile_name,
            "timestamp": datetime.now().isoformat(),
            "function_timings": dict(self.function_timings),
            "block_timings": dict(self.block_timings),
            "resource_samples": self.resource_samples,
            "network_usage": self.network_usage,
            "bottlenecks": self.bottlenecks
        }
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.profile_name}_{timestamp}.json"
        filepath = os.path.join(self.storage_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Performance data saved to {filepath}")
        return filepath
    
    def load_data(self, filepath: str) -> bool:
        """
        Load performance data from a file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            True if data was loaded successfully, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.function_timings = defaultdict(list, data.get("function_timings", {}))
            self.block_timings = defaultdict(list, data.get("block_timings", {}))
            self.resource_samples = data.get("resource_samples", [])
            self.sample_timestamps = [s["timestamp"] for s in self.resource_samples]
            self.network_usage = data.get("network_usage", [])
            self.bottlenecks = data.get("bottlenecks", [])
            
            self.logger.info(f"Performance data loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
            return False
    
    def reset(self) -> None:
        """Reset all performance data."""
        self.function_timings = defaultdict(list)
        self.block_timings = defaultdict(list)
        self.resource_samples = []
        self.sample_timestamps = []
        self.network_usage = []
        self.bottlenecks = []
        self.logger.info("Performance data reset")
    
    def compare_profiles(self, other_profile: "PerformanceProfiler") -> Dict[str, Any]:
        """
        Compare this profile with another profile to identify improvements or regressions.
        
        Args:
            other_profile: Another PerformanceProfiler instance to compare with
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            "profile_a": self.profile_name,
            "profile_b": other_profile.profile_name,
            "functions": {},
            "blocks": {},
            "resources": {}
        }
        
        # Compare function timings
        all_functions = set(list(self.function_timings.keys()) + list(other_profile.function_timings.keys()))
        for func in all_functions:
            stats_a = self.get_function_stats(func).get(func, {})
            stats_b = other_profile.get_function_stats(func).get(func, {})
            
            if stats_a and stats_b:
                mean_diff = stats_b["mean"] - stats_a["mean"]
                mean_diff_percent = (mean_diff / stats_a["mean"]) * 100 if stats_a["mean"] > 0 else 0
                
                comparison["functions"][func] = {
                    "mean_a": stats_a["mean"],
                    "mean_b": stats_b["mean"],
                    "mean_diff": mean_diff,
                    "mean_diff_percent": mean_diff_percent,
                    "improved": mean_diff < 0
                }
        
        # Compare block timings
        all_blocks = set(list(self.block_timings.keys()) + list(other_profile.block_timings.keys()))
        for block in all_blocks:
            stats_a = self.get_block_stats(block).get(block, {})
            stats_b = other_profile.get_block_stats(block).get(block, {})
            
            if stats_a and stats_b:
                mean_diff = stats_b["mean"] - stats_a["mean"]
                mean_diff_percent = (mean_diff / stats_a["mean"]) * 100 if stats_a["mean"] > 0 else 0
                
                comparison["blocks"][block] = {
                    "mean_a": stats_a["mean"],
                    "mean_b": stats_b["mean"],
                    "mean_diff": mean_diff,
                    "mean_diff_percent": mean_diff_percent,
                    "improved": mean_diff < 0
                }
        
        # Compare resource usage
        resource_stats_a = self.get_resource_usage_stats()
        resource_stats_b = other_profile.get_resource_usage_stats()
        
        if resource_stats_a and resource_stats_b:
            for resource_type in ["cpu", "memory", "network"]:
                if resource_type in resource_stats_a and resource_type in resource_stats_b:
                    comparison["resources"][resource_type] = {}
                    
                    for metric in resource_stats_a[resource_type]:
                        if metric in resource_stats_b[resource_type]:
                            stats_a = resource_stats_a[resource_type][metric]
                            stats_b = resource_stats_b[resource_type][metric]
                            
                            mean_diff = stats_b["mean"] - stats_a["mean"]
                            mean_diff_percent = (mean_diff / stats_a["mean"]) * 100 if stats_a["mean"] > 0 else 0
                            
                            comparison["resources"][resource_type][metric] = {
                                "mean_a": stats_a["mean"],
                                "mean_b": stats_b["mean"],
                                "mean_diff": mean_diff,
                                "mean_diff_percent": mean_diff_percent,
                                "improved": mean_diff < 0
                            }
        
        return comparison


class BlockProfiler:
    """Context manager for profiling blocks of code."""
    
    def __init__(self, profiler: PerformanceProfiler, block_name: str):
        """
        Initialize the block profiler.
        
        Args:
            profiler: The parent PerformanceProfiler instance
            block_name: Name of the code block to profile
        """
        self.profiler = profiler
        self.block_name = block_name
        self.start_time = None
        self.start_resources = None
        
    def __enter__(self) -> "BlockProfiler":
        """Enter the profiling context."""
        if not self.profiler.enabled:
            return self
        
        self.start_time = time.time()
        self.start_resources = self.profiler._sample_resources()
        self.profiler.context_stack.append(self.block_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the profiling context and record timing."""
        if not self.profiler.enabled:
            return
        
        end_time = time.time()
        end_resources = self.profiler._sample_resources()
        
        # Remove this block from context stack
        if self.profiler.context_stack and self.profiler.context_stack[-1] == self.block_name:
            self.profiler.context_stack.pop()
        
        # Record timing data
        self.profiler.record_block_timing(
            self.block_name,
            self.start_time,
            end_time,
            exc_type is None,
            self.start_resources,
            end_resources
        )
