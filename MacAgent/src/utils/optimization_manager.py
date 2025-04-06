#!/usr/bin/env python3
"""
OptimizationManager module for MacAgent.

Provides performance optimization strategies, dynamic parameter adjustment,
caching mechanisms, and various performance profiles for the Mac agent.
"""

import os
import json
import logging
import time
import functools
import pickle
import hashlib
import sys
import psutil
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from enum import Enum
from pathlib import Path
import threading
import concurrent.futures
from datetime import datetime, timedelta
import weakref

# Import from MacAgent utils
from MacAgent.src.utils.performance_profiler import PerformanceProfiler


class OptimizationLevel(Enum):
    """Optimization levels for the agent."""
    ACCURACY = 0   # Prioritize accuracy over speed
    BALANCED = 1   # Balance between accuracy and speed
    SPEED = 2      # Prioritize speed over accuracy
    ULTRA_SPEED = 3  # Maximum speed, may reduce quality


class OptimizationDomain(Enum):
    """Domains for optimization."""
    VISION = "vision"
    INTERACTION = "interaction"
    INTELLIGENCE = "intelligence"
    CORE = "core"
    ALL = "all"


class CacheStrategy(Enum):
    """Caching strategies for different types of data."""
    MEMORY = "memory"     # In-memory caching only
    DISK = "disk"         # Disk-based caching
    HYBRID = "hybrid"     # Memory with disk fallback
    NONE = "none"         # No caching


class OptimizationManager:
    """
    Manages performance optimization for the Mac agent.
    
    Features:
    - Implements various optimization strategies
    - Dynamically adjusts system parameters for optimal performance
    - Balances speed and accuracy based on context
    - Implements caching and memoization
    - Provides configuration options for different performance profiles
    """
    
    def __init__(self, 
                 config_path: Optional[str] = "config/optimization.json",
                 cache_dir: str = "memory/cache",
                 profiler: Optional[PerformanceProfiler] = None,
                 logging_level: int = logging.INFO):
        """
        Initialize the optimization manager.
        
        Args:
            config_path: Path to the optimization configuration file
            cache_dir: Directory for disk cache
            profiler: Performance profiler instance
            logging_level: Level for logging
        """
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("OptimizationManager")
        self.logger.setLevel(logging_level)
        
        # Performance profiler
        self.profiler = profiler
        
        # Default optimization settings
        self.optimization_level = OptimizationLevel.BALANCED
        self.domain_settings = {domain: {} for domain in OptimizationDomain}
        
        # Cache settings and storage
        self.memory_cache = {}
        self.memory_cache_stats = {"hits": 0, "misses": 0, "size": 0}
        self.disk_cache_stats = {"hits": 0, "misses": 0, "size": 0}
        self.cache_lock = threading.RLock()
        self.max_memory_cache_size = 100 * 1024 * 1024  # 100 MB default
        self.memory_cache_ttl = 3600  # 1 hour default
        
        # Thread pool for parallel optimizations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count(),
            thread_name_prefix="OptimizationThread"
        )
        
        # Load configuration
        if config_path:
            self.load_config(config_path)
        
        self.logger.info("Optimization manager initialized")
    
    def load_config(self, config_path: str) -> bool:
        """
        Load optimization configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if configuration was loaded successfully
        """
        try:
            if not os.path.exists(config_path):
                self.logger.warning(f"Configuration file not found: {config_path}")
                self._create_default_config(config_path)
                return False
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Set optimization level
            level_name = config.get("optimization_level", "BALANCED")
            try:
                self.optimization_level = OptimizationLevel[level_name]
            except KeyError:
                self.logger.warning(f"Invalid optimization level: {level_name}, using BALANCED")
                self.optimization_level = OptimizationLevel.BALANCED
            
            # Set domain-specific settings
            for domain_name, settings in config.get("domains", {}).items():
                try:
                    domain = OptimizationDomain(domain_name)
                    self.domain_settings[domain] = settings
                except ValueError:
                    self.logger.warning(f"Unknown optimization domain: {domain_name}")
            
            # Set cache settings
            cache_config = config.get("cache", {})
            self.max_memory_cache_size = cache_config.get("max_memory_size", 100) * 1024 * 1024
            self.memory_cache_ttl = cache_config.get("memory_ttl", 3600)
            
            self.logger.info(f"Optimization configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading optimization configuration: {e}")
            return False
    
    def _create_default_config(self, config_path: str) -> None:
        """Create a default configuration file if none exists."""
        default_config = {
            "optimization_level": "BALANCED",
            "domains": {
                "vision": {
                    "image_quality": 0.8,
                    "detection_threshold": 0.7,
                    "max_elements": 50
                },
                "interaction": {
                    "action_delay": 0.1,
                    "retry_attempts": 3,
                    "timeout": 10.0
                },
                "intelligence": {
                    "model_precision": "medium",
                    "response_tokens": 1024,
                    "context_window": 8192
                },
                "core": {
                    "parallel_tasks": os.cpu_count(),
                    "throttle_interval": 0.05
                }
            },
            "cache": {
                "max_memory_size": 100,  # MB
                "memory_ttl": 3600,  # seconds
                "disk_ttl": 86400  # seconds
            }
        }
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            self.logger.info(f"Default configuration created at {config_path}")
        except Exception as e:
            self.logger.error(f"Error creating default configuration: {e}")
    
    def set_optimization_level(self, level: OptimizationLevel) -> None:
        """
        Set the global optimization level.
        
        Args:
            level: The optimization level to set
        """
        self.optimization_level = level
        self.logger.info(f"Optimization level set to {level.name}")
        
        # Adjust domain settings based on the new level
        self._update_domain_settings()
    
    def _update_domain_settings(self) -> None:
        """Update domain-specific settings based on the current optimization level."""
        level = self.optimization_level
        
        # Vision adjustments
        vision_settings = self.domain_settings.get(OptimizationDomain.VISION, {}).copy()
        if level == OptimizationLevel.ACCURACY:
            vision_settings.update({
                "image_quality": 1.0,
                "detection_threshold": 0.8,
                "max_elements": 100
            })
        elif level == OptimizationLevel.SPEED:
            vision_settings.update({
                "image_quality": 0.6,
                "detection_threshold": 0.6,
                "max_elements": 30
            })
        elif level == OptimizationLevel.ULTRA_SPEED:
            vision_settings.update({
                "image_quality": 0.4,
                "detection_threshold": 0.5,
                "max_elements": 15
            })
        self.domain_settings[OptimizationDomain.VISION] = vision_settings
        
        # Intelligence adjustments
        intelligence_settings = self.domain_settings.get(OptimizationDomain.INTELLIGENCE, {}).copy()
        if level == OptimizationLevel.ACCURACY:
            intelligence_settings.update({
                "model_precision": "high",
                "response_tokens": 2048,
                "context_window": 16384
            })
        elif level == OptimizationLevel.SPEED:
            intelligence_settings.update({
                "model_precision": "low",
                "response_tokens": 512,
                "context_window": 4096
            })
        elif level == OptimizationLevel.ULTRA_SPEED:
            intelligence_settings.update({
                "model_precision": "lowest",
                "response_tokens": 256,
                "context_window": 2048
            })
        self.domain_settings[OptimizationDomain.INTELLIGENCE] = intelligence_settings
        
        # Interaction adjustments
        interaction_settings = self.domain_settings.get(OptimizationDomain.INTERACTION, {}).copy()
        if level == OptimizationLevel.ACCURACY:
            interaction_settings.update({
                "action_delay": 0.2,
                "retry_attempts": 5,
                "timeout": 15.0
            })
        elif level == OptimizationLevel.SPEED:
            interaction_settings.update({
                "action_delay": 0.05,
                "retry_attempts": 2,
                "timeout": 5.0
            })
        elif level == OptimizationLevel.ULTRA_SPEED:
            interaction_settings.update({
                "action_delay": 0.01,
                "retry_attempts": 1,
                "timeout": 3.0
            })
        self.domain_settings[OptimizationDomain.INTERACTION] = interaction_settings
    
    def get_setting(self, domain: OptimizationDomain, setting_name: str, default: Any = None) -> Any:
        """
        Get a specific optimization setting for a domain.
        
        Args:
            domain: The optimization domain
            setting_name: Name of the setting to retrieve
            default: Default value if setting is not found
            
        Returns:
            The setting value or default if not found
        """
        domain_settings = self.domain_settings.get(domain, {})
        return domain_settings.get(setting_name, default)
    
    def optimize_vision(self, image_quality: Optional[float] = None) -> Dict[str, Any]:
        """
        Get optimization settings for vision operations.
        
        Args:
            image_quality: Override the default image quality
            
        Returns:
            Dictionary of vision optimization settings
        """
        vision_settings = self.domain_settings.get(OptimizationDomain.VISION, {}).copy()
        
        if image_quality is not None:
            vision_settings["image_quality"] = max(0.1, min(1.0, image_quality))
        
        return vision_settings
    
    def optimize_intelligence(self, model_precision: Optional[str] = None) -> Dict[str, Any]:
        """
        Get optimization settings for intelligence operations.
        
        Args:
            model_precision: Override the default model precision
            
        Returns:
            Dictionary of intelligence optimization settings
        """
        intelligence_settings = self.domain_settings.get(OptimizationDomain.INTELLIGENCE, {}).copy()
        
        if model_precision is not None:
            valid_precisions = ["high", "medium", "low", "lowest"]
            if model_precision in valid_precisions:
                intelligence_settings["model_precision"] = model_precision
        
        return intelligence_settings
    
    def optimize_interaction(self, action_delay: Optional[float] = None) -> Dict[str, Any]:
        """
        Get optimization settings for interaction operations.
        
        Args:
            action_delay: Override the default action delay
            
        Returns:
            Dictionary of interaction optimization settings
        """
        interaction_settings = self.domain_settings.get(OptimizationDomain.INTERACTION, {}).copy()
        
        if action_delay is not None:
            interaction_settings["action_delay"] = max(0.0, action_delay)
        
        return interaction_settings
    
    def memoize(self, 
                cache_strategy: CacheStrategy = CacheStrategy.MEMORY,
                ttl: Optional[int] = None,
                max_size: Optional[int] = None) -> Callable:
        """
        Decorator to memoize function results.
        
        Args:
            cache_strategy: Strategy to use for caching
            ttl: Time-to-live in seconds (None for no expiration)
            max_size: Maximum cache size (ignored for memory strategy)
            
        Returns:
            Decorated function with memoization
        """
        def decorator(func):
            cache_key_prefix = f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create a cache key from function arguments
                key_parts = [cache_key_prefix]
                key_parts.extend([str(arg) for arg in args])
                key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
                
                # Use specified TTL or default
                cache_ttl = ttl if ttl is not None else self.memory_cache_ttl
                
                # Try to get from cache
                cached_result = self._get_from_cache(key, cache_strategy)
                if cached_result is not None:
                    return cached_result[0]  # Return the cached value
                
                # Call the function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Cache the result
                self._store_in_cache(key, result, cache_strategy, cache_ttl, execution_time)
                
                return result
            
            return wrapper
        
        return decorator
    
    def _get_from_cache(self, key: str, strategy: CacheStrategy) -> Optional[Tuple[Any, float]]:
        """
        Get an item from cache.
        
        Args:
            key: Cache key
            strategy: Caching strategy to use
            
        Returns:
            Tuple of (cached_value, timestamp) or None if not found
        """
        if strategy == CacheStrategy.NONE:
            return None
        
        # Check memory cache first
        if strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            with self.cache_lock:
                if key in self.memory_cache:
                    value, timestamp, _ = self.memory_cache[key]
                    # Check if expired
                    if self.memory_cache_ttl > 0 and time.time() - timestamp > self.memory_cache_ttl:
                        del self.memory_cache[key]
                        self.memory_cache_stats["misses"] += 1
                        self.logger.debug(f"Memory cache miss (expired): {key}")
                    else:
                        self.memory_cache_stats["hits"] += 1
                        self.logger.debug(f"Memory cache hit: {key}")
                        return (value, timestamp)
                else:
                    self.memory_cache_stats["misses"] += 1
                    self.logger.debug(f"Memory cache miss: {key}")
        
        # Check disk cache if memory cache missed
        if strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        value, timestamp = pickle.load(f)
                    
                    # Check if expired
                    if self.memory_cache_ttl > 0 and time.time() - timestamp > self.memory_cache_ttl:
                        os.remove(cache_file)
                        self.disk_cache_stats["misses"] += 1
                        self.logger.debug(f"Disk cache miss (expired): {key}")
                    else:
                        self.disk_cache_stats["hits"] += 1
                        self.logger.debug(f"Disk cache hit: {key}")
                        
                        # Store in memory cache for faster access next time
                        if strategy == CacheStrategy.HYBRID:
                            self._store_in_memory_cache(key, value, timestamp, 0)
                        
                        return (value, timestamp)
                except Exception as e:
                    self.logger.warning(f"Error reading from disk cache: {e}")
                    self.disk_cache_stats["misses"] += 1
            else:
                self.disk_cache_stats["misses"] += 1
                self.logger.debug(f"Disk cache miss: {key}")
        
        return None
    
    def _store_in_cache(self, key: str, value: Any, strategy: CacheStrategy, 
                      ttl: int, execution_time: float) -> None:
        """
        Store an item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            strategy: Caching strategy to use
            ttl: Time-to-live in seconds
            execution_time: Execution time of the function
        """
        if strategy == CacheStrategy.NONE:
            return
        
        timestamp = time.time()
        
        # Store in memory cache
        if strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            self._store_in_memory_cache(key, value, timestamp, execution_time)
        
        # Store in disk cache
        if strategy in [CacheStrategy.DISK, CacheStrategy.HYBRID]:
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump((value, timestamp), f)
                
                # Update disk cache stats
                file_size = os.path.getsize(cache_file)
                self.disk_cache_stats["size"] += file_size
                
                self.logger.debug(f"Stored in disk cache: {key} ({file_size} bytes)")
            except Exception as e:
                self.logger.warning(f"Error writing to disk cache: {e}")
    
    def _store_in_memory_cache(self, key: str, value: Any, timestamp: float, execution_time: float) -> None:
        """Store an item in memory cache with size management."""
        with self.cache_lock:
            # Estimate size of the value (very rough approximation)
            try:
                value_size = len(pickle.dumps(value))
            except:
                value_size = sys.getsizeof(value)
            
            # Check if we need to make room
            if self.memory_cache_stats["size"] + value_size > self.max_memory_cache_size:
                self._clean_memory_cache(value_size)
            
            # Store in cache
            self.memory_cache[key] = (value, timestamp, execution_time)
            self.memory_cache_stats["size"] += value_size
            self.logger.debug(f"Stored in memory cache: {key} ({value_size} bytes)")
    
    def _clean_memory_cache(self, needed_size: int) -> None:
        """
        Clean memory cache to make room for a new item.
        
        Args:
            needed_size: Size needed for the new item
        """
        with self.cache_lock:
            if not self.memory_cache:
                return
            
            # Sort cache items by last access time (oldest first)
            items = list(self.memory_cache.items())
            items.sort(key=lambda x: x[1][1])  # Sort by timestamp
            
            # Remove items until we have enough space
            removed = 0
            freed_size = 0
            for key, (value, _, _) in items:
                try:
                    value_size = len(pickle.dumps(value))
                except:
                    value_size = sys.getsizeof(value)
                
                del self.memory_cache[key]
                freed_size += value_size
                removed += 1
                
                if freed_size >= needed_size:
                    break
            
            self.memory_cache_stats["size"] -= freed_size
            self.logger.debug(f"Cleaned memory cache: removed {removed} items, freed {freed_size} bytes")
    
    def clear_cache(self, strategy: Optional[CacheStrategy] = None) -> None:
        """
        Clear the cache.
        
        Args:
            strategy: Strategy to clear, or None for all
        """
        # Clear memory cache
        if strategy in [None, CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
            with self.cache_lock:
                self.memory_cache.clear()
                self.memory_cache_stats = {"hits": 0, "misses": 0, "size": 0}
            self.logger.info("Memory cache cleared")
        
        # Clear disk cache
        if strategy in [None, CacheStrategy.DISK, CacheStrategy.HYBRID]:
            try:
                cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.cache')]
                for file in cache_files:
                    os.remove(os.path.join(self.cache_dir, file))
                
                self.disk_cache_stats = {"hits": 0, "misses": 0, "size": 0}
                self.logger.info(f"Disk cache cleared: removed {len(cache_files)} files")
            except Exception as e:
                self.logger.error(f"Error clearing disk cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.cache_lock:
            stats = {
                "memory": self.memory_cache_stats.copy(),
                "disk": self.disk_cache_stats.copy()
            }
            
            # Calculate hit ratio
            for cache_type in ["memory", "disk"]:
                total = stats[cache_type]["hits"] + stats[cache_type]["misses"]
                if total > 0:
                    stats[cache_type]["hit_ratio"] = (stats[cache_type]["hits"] / total) * 100
                else:
                    stats[cache_type]["hit_ratio"] = 0
            
            return stats
    
    def parallel_execute(self, func_list: List[Tuple[Callable, List, Dict]]) -> List[Any]:
        """
        Execute multiple functions in parallel.
        
        Args:
            func_list: List of (function, args, kwargs) tuples
            
        Returns:
            List of results in the same order as input functions
        """
        if not func_list:
            return []
        
        max_workers = self.get_setting(
            OptimizationDomain.CORE, 
            "parallel_tasks", 
            os.cpu_count()
        )
        
        results = []
        
        # Submit all tasks
        futures = []
        for func, args, kwargs in func_list:
            future = self.thread_pool.submit(func, *args, **kwargs)
            futures.append(future)
        
        # Collect results in order
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in parallel execution: {e}")
                results.append(None)
        
        return results
    
    def adaptive_throttle(self, load_threshold: float = 0.8) -> bool:
        """
        Throttle execution based on system load.
        
        Args:
            load_threshold: CPU load threshold for throttling
            
        Returns:
            True if throttling was applied, False otherwise
        """
        # Get current CPU load
        try:
            cpu_load = psutil.cpu_percent(interval=0.1) / 100.0
            
            if cpu_load > load_threshold:
                # Calculate adaptive delay based on load
                excess_load = cpu_load - load_threshold
                delay = excess_load * 0.1  # 0-20ms delay range
                
                throttle_interval = self.get_setting(
                    OptimizationDomain.CORE, 
                    "throttle_interval", 
                    0.05
                )
                
                # Apply throttling
                time.sleep(max(throttle_interval, delay))
                self.logger.debug(f"Throttling applied: CPU load {cpu_load:.2f}, delay {delay:.3f}s")
                return True
        except Exception as e:
            self.logger.warning(f"Error in adaptive throttling: {e}")
        
        return False
    
    def create_optimization_profile(self, profile_name: str,
                                    level: OptimizationLevel,
                                    settings: Dict[str, Any]) -> bool:
        """
        Create a custom optimization profile.
        
        Args:
            profile_name: Name for the new profile
            level: Base optimization level
            settings: Custom settings for the profile
            
        Returns:
            True if profile was created successfully
        """
        try:
            # Use existing domain settings as base
            profile = {
                "name": profile_name,
                "optimization_level": level.name,
                "domains": {}
            }
            
            # Start with current domain settings
            for domain in OptimizationDomain:
                if domain != OptimizationDomain.ALL:
                    profile["domains"][domain.value] = self.domain_settings.get(domain, {}).copy()
            
            # Apply custom settings
            for domain_name, domain_settings in settings.items():
                try:
                    domain = OptimizationDomain(domain_name)
                    if domain != OptimizationDomain.ALL:
                        if domain not in profile["domains"]:
                            profile["domains"][domain.value] = {}
                        profile["domains"][domain.value].update(domain_settings)
                except ValueError:
                    self.logger.warning(f"Unknown domain in profile: {domain_name}")
            
            # Save profile
            profiles_dir = os.path.join(os.path.dirname(self.cache_dir), "profiles")
            Path(profiles_dir).mkdir(parents=True, exist_ok=True)
            
            profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            self.logger.info(f"Optimization profile '{profile_name}' created at {profile_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating optimization profile: {e}")
            return False
    
    def apply_optimization_profile(self, profile_name: str) -> bool:
        """
        Apply a saved optimization profile.
        
        Args:
            profile_name: Name of the profile to apply
            
        Returns:
            True if profile was applied successfully
        """
        try:
            profiles_dir = os.path.join(os.path.dirname(self.cache_dir), "profiles")
            profile_path = os.path.join(profiles_dir, f"{profile_name}.json")
            
            if not os.path.exists(profile_path):
                self.logger.warning(f"Optimization profile not found: {profile_name}")
                return False
            
            with open(profile_path, 'r') as f:
                profile = json.load(f)
            
            # Set optimization level
            level_name = profile.get("optimization_level", "BALANCED")
            try:
                self.optimization_level = OptimizationLevel[level_name]
            except KeyError:
                self.logger.warning(f"Invalid optimization level in profile: {level_name}")
                self.optimization_level = OptimizationLevel.BALANCED
            
            # Apply domain settings
            for domain_name, settings in profile.get("domains", {}).items():
                try:
                    domain = OptimizationDomain(domain_name)
                    self.domain_settings[domain] = settings
                except ValueError:
                    self.logger.warning(f"Unknown domain in profile: {domain_name}")
            
            self.logger.info(f"Applied optimization profile: {profile_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying optimization profile: {e}")
            return False
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a report on current optimization settings and performance.
        
        Returns:
            Dictionary containing the optimization report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "optimization_level": self.optimization_level.name,
            "domain_settings": {},
            "cache_stats": self.get_cache_stats()
        }
        
        # Add domain settings
        for domain in OptimizationDomain:
            if domain != OptimizationDomain.ALL:
                report["domain_settings"][domain.value] = self.domain_settings.get(domain, {})
        
        # Add profiler data if available
        if self.profiler:
            report["performance"] = {
                "bottlenecks": self.profiler.identify_bottlenecks(),
                "resource_stats": self.profiler.get_resource_usage_stats()
            }
        
        return report
