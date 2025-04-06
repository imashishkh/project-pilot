#!/usr/bin/env python3
"""
Performance Optimization System Example

This script demonstrates how to use the MacAgent performance optimization system
to monitor, profile, and optimize code execution.
"""

import time
import random
import os
import sys
import logging
from typing import List, Dict, Any

# Add the parent directory to the path to import MacAgent modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the performance system
from MacAgent.src.utils.performance_system import (
    get_performance_system, 
    ProfileBlock, 
    profile
)
from MacAgent.src.utils.optimization_manager import OptimizationLevel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PerformanceExample")


# Example functions that we'll profile
@profile
def cpu_intensive_task(size: int = 1000) -> List[int]:
    """A CPU-intensive sorting task."""
    data = [random.randint(1, 10000) for _ in range(size)]
    return sorted(data)


@profile(name="Memory Task")
def memory_intensive_task(size: int = 1000000) -> Dict[int, int]:
    """A memory-intensive task that creates a large dictionary."""
    result = {}
    for i in range(size):
        result[i] = i * i
    return result


def io_intensive_task(iterations: int = 5) -> None:
    """An I/O-intensive task that writes and reads files."""
    with ProfileBlock("IO Operations"):
        # Create a temporary directory
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Write files
        for i in range(iterations):
            with open(os.path.join(tmp_dir, f"test_{i}.txt"), "w") as f:
                f.write("a" * 1000000)  # Write 1MB of data
            
            # Simulate network delay
            time.sleep(0.1)
        
        # Read files
        for i in range(iterations):
            with open(os.path.join(tmp_dir, f"test_{i}.txt"), "r") as f:
                data = f.read()
            
            # Simulate network delay
            time.sleep(0.1)
        
        # Clean up
        for i in range(iterations):
            os.remove(os.path.join(tmp_dir, f"test_{i}.txt"))
        
        os.rmdir(tmp_dir)


def optimize_vision_example() -> None:
    """Example of optimizing vision operations."""
    # Get the performance system
    perf_system = get_performance_system()
    
    # Get optimization parameters for vision tasks
    with ProfileBlock("Vision Optimization"):
        # First with default settings
        default_params = perf_system.optimize("vision")
        logger.info(f"Default vision optimization parameters: {default_params}")
        
        # Then with a specific context
        context = {"image_size": (1920, 1080), "is_webcam": False, "detail_required": "high"}
        detailed_params = perf_system.optimize("vision", context)
        logger.info(f"Context-aware vision optimization parameters: {detailed_params}")
        
        # Finally with forced optimization level
        ultra_speed_params = perf_system.optimize("vision", context, OptimizationLevel.ULTRA_SPEED)
        logger.info(f"Ultra-speed vision optimization parameters: {ultra_speed_params}")


def dynamic_optimization_example() -> None:
    """Example of dynamic optimization based on resource usage."""
    # Get the performance system
    perf_system = get_performance_system()
    
    logger.info("Starting dynamic optimization example...")
    
    # First, let's generate some load
    for _ in range(3):
        # Run a CPU-intensive task
        cpu_intensive_task(size=50000)
        
        # Check the resource state
        resource_state = perf_system.get_resource_state()
        usage = resource_state["resource_usage"]
        logger.info(f"CPU: {usage['cpu']['percent']}%, Memory: {usage['memory']['percent']}%")
        
        # Get optimization suggestions
        suggestions = perf_system.suggest_optimizations()
        if suggestions:
            logger.info(f"Optimization suggestions: {suggestions}")
        
        # Sleep a bit
        time.sleep(1)
    
    # Now run some memory-intensive tasks
    memory_intensive_task(size=2000000)
    
    # Check the optimization state
    opt_state = perf_system.get_optimization_state()
    logger.info(f"Current optimization level: {opt_state['current_level']}")
    
    # Generate a performance report
    report = perf_system.generate_performance_report()
    logger.info(f"System uptime: {report['uptime_formatted']}")
    logger.info(f"Profiling sessions: {report['profiling_sessions']}")
    logger.info(f"Optimization count: {report['optimization_count']}")


def caching_example() -> None:
    """Example of using the caching system."""
    # Get the performance system
    perf_system = get_performance_system()
    
    # Define a function to cache
    def fibonacci(n: int) -> int:
        """Calculate the nth Fibonacci number."""
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    # First execution without caching (will be slow for larger values)
    start_time = time.time()
    result1 = fibonacci(30)
    duration1 = time.time() - start_time
    logger.info(f"Fibonacci(30) = {result1}, took {duration1:.2f} seconds")
    
    # Now implement a memoized version using our caching system
    def cached_fibonacci(n: int) -> int:
        """Calculate the nth Fibonacci number with caching."""
        # Check if result is in cache
        cache_key = f"fib_{n}"
        cached_result = perf_system.get_cached_result("math", cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for Fibonacci({n})")
            return cached_result
        
        # Not in cache, calculate
        if n <= 1:
            result = n
        else:
            result = cached_fibonacci(n-1) + cached_fibonacci(n-2)
        
        # Store in cache
        perf_system.cache_result("math", cache_key, result)
        return result
    
    # First execution of cached version (will build up the cache)
    start_time = time.time()
    result2 = cached_fibonacci(30)
    duration2 = time.time() - start_time
    logger.info(f"Cached Fibonacci(30) = {result2}, took {duration2:.2f} seconds")
    
    # Second execution of cached version (should be much faster)
    start_time = time.time()
    result3 = cached_fibonacci(30)
    duration3 = time.time() - start_time
    logger.info(f"Cached Fibonacci(30) second call = {result3}, took {duration3:.2f} seconds")
    
    # Check cache stats
    cache_stats = perf_system.optimizer.get_cache_stats()
    logger.info(f"Cache stats: {cache_stats}")


def main() -> None:
    """Main function to run the performance optimization examples."""
    logger.info("=== Performance Optimization System Example ===")
    
    # Initialize the performance system with custom settings
    perf_system = get_performance_system(
        base_dir="memory/performance_example",
        logging_level=logging.INFO,
        auto_start=True,
        default_optimization_level=OptimizationLevel.BALANCED
    )
    
    try:
        # Run CPU-intensive task example
        logger.info("\n--- CPU-Intensive Task Example ---")
        cpu_intensive_task(size=10000)
        cpu_intensive_task(size=50000)
        
        # Run memory-intensive task example
        logger.info("\n--- Memory-Intensive Task Example ---")
        memory_intensive_task(size=500000)
        
        # Run I/O-intensive task example
        logger.info("\n--- I/O-Intensive Task Example ---")
        io_intensive_task(iterations=3)
        
        # Run vision optimization example
        logger.info("\n--- Vision Optimization Example ---")
        optimize_vision_example()
        
        # Run dynamic optimization example
        logger.info("\n--- Dynamic Optimization Example ---")
        dynamic_optimization_example()
        
        # Run caching example
        logger.info("\n--- Caching Example ---")
        caching_example()
        
        # Generate and display a final performance report
        logger.info("\n--- Final Performance Report ---")
        report = perf_system.generate_performance_report()
        logger.info(f"Total profiling sessions: {report['profiling_sessions']}")
        logger.info(f"Identified bottlenecks: {len(report['bottlenecks'])}")
        if report['bottlenecks']:
            for bottleneck in report['bottlenecks'][:3]:
                logger.info(f"Bottleneck: {bottleneck['name']} - Impact: {bottleneck['impact']:.2f}%")
        
        logger.info(f"Optimization suggestions:")
        for suggestion in report['suggestions']:
            logger.info(f"- {suggestion}")
        
    finally:
        # Stop the performance system
        perf_system.stop()
        logger.info("Performance system stopped")


if __name__ == "__main__":
    main() 