# MacAgent Performance Optimization System

A comprehensive performance optimization system for monitoring, profiling, and enhancing the performance of the MacAgent system.

## Overview

The Performance Optimization System consists of three main components:

1. **PerformanceProfiler**: Measures execution time, identifies bottlenecks, and generates reports
2. **OptimizationManager**: Implements optimization strategies, caching, and parameter adjustments
3. **ResourceMonitor**: Tracks system resources and prevents resource exhaustion

These components work together through the unified `PerformanceSystem` interface to provide a complete performance optimization solution.

## Components

### PerformanceProfiler

The `PerformanceProfiler` tracks and analyzes code execution, providing insights into:

- Function execution times
- Resource usage during execution
- Performance bottlenecks
- Historical performance trends
- Comparative analytics

Features:
- Function profiling via decorators
- Code block profiling via context managers
- Bottleneck identification
- Performance visualization
- Performance reports

### OptimizationManager

The `OptimizationManager` implements various optimization strategies:

- Different optimization levels (ACCURACY, BALANCED, SPEED, ULTRA_SPEED)
- Domain-specific optimizations (vision, intelligence, interaction, core)
- Caching/memoization of results
- Dynamic parameter adjustments

Features:
- Multiple optimization profiles
- Intelligent caching system
- Parameter tuning based on context
- Adaptive optimization based on resource constraints

### ResourceMonitor

The `ResourceMonitor` tracks system resources and prevents exhaustion:

- CPU usage monitoring
- Memory usage tracking
- Disk space management
- I/O monitoring
- Adaptive throttling

Features:
- Real-time resource monitoring
- Resource alerts and warnings
- Predictive resource exhaustion
- Adaptive parallel execution
- Resource usage reports and visualizations

### PerformanceSystem

The `PerformanceSystem` provides a unified interface for:

- Accessing all performance components
- Starting/stopping monitoring
- Applying optimizations
- Generating reports
- Managing optimization levels

## Installation

The Performance Optimization System is included as part of the MacAgent codebase.

Dependencies:
- Python 3.6+
- psutil
- matplotlib
- numpy

## Usage

### Basic Usage

```python
from MacAgent.src.utils.performance_system import get_performance_system, profile, ProfileBlock

# Get the global performance system instance
perf_system = get_performance_system()

# Profile a function using the decorator
@profile
def my_function():
    # Function code
    pass

# Profile a block of code using the context manager
with ProfileBlock("My Code Block"):
    # Code to profile
    pass

# Get performance metrics
bottlenecks = perf_system.analyze_bottlenecks()
suggestions = perf_system.suggest_optimizations()
report = perf_system.generate_performance_report()
```

### Optimization

```python
from MacAgent.src.utils.performance_system import get_performance_system
from MacAgent.src.utils.optimization_manager import OptimizationLevel

# Get the performance system
perf_system = get_performance_system()

# Set global optimization level
perf_system.optimizer.set_optimization_level(OptimizationLevel.SPEED)

# Get domain-specific optimization parameters
vision_params = perf_system.optimize("vision")

# Get domain-specific optimization with context
context = {"image_size": (1920, 1080), "is_webcam": True}
vision_params = perf_system.optimize("vision", context)

# Cache and retrieve results
perf_system.cache_result("domain", "key", result)
cached_result = perf_system.get_cached_result("domain", "key")
```

### Resource Monitoring

```python
from MacAgent.src.utils.performance_system import get_performance_system

# Get the performance system
perf_system = get_performance_system()

# Start resource monitoring
perf_system.monitor.start_monitoring()

# Get current resource usage
usage = perf_system.monitor.get_current_usage()
print(f"CPU: {usage['cpu']}%, Memory: {usage['memory']}%")

# Check if operations should be throttled
if perf_system.monitor.should_throttle():
    # Implement throttling logic
    pass

# Get optimal thread count for parallel operations
thread_count = perf_system.monitor.get_optimal_thread_count()

# Generate resource report
resource_report = perf_system.monitor.generate_resource_report()

# Stop resource monitoring
perf_system.monitor.stop_monitoring()
```

## Example

Here's a complete example showing how to use the Performance Optimization System:

```python
import time
from MacAgent.src.utils.performance_system import get_performance_system, profile, ProfileBlock
from MacAgent.src.utils.optimization_manager import OptimizationLevel

# Initialize the performance system
perf_system = get_performance_system(
    base_dir="memory/performance",
    logging_level=logging.INFO,
    auto_start=True,
    default_optimization_level=OptimizationLevel.BALANCED
)

try:
    # Profile a function
    @profile
    def example_function(iterations=1000000):
        result = 0
        for i in range(iterations):
            result += i
        return result
    
    # Run the function
    result = example_function()
    
    # Profile a block of code
    with ProfileBlock("Example Block"):
        time.sleep(1)  # Simulate work
    
    # Get optimization parameters
    params = perf_system.optimize("vision")
    
    # Generate a performance report
    report = perf_system.generate_performance_report()
    print(f"Profiling sessions: {report['profiling_sessions']}")
    print(f"Bottlenecks: {report['bottlenecks']}")
    print(f"Suggestions: {report['suggestions']}")
    
finally:
    # Always stop the performance system
    perf_system.stop()
```

For a more detailed example, see the `MacAgent/examples/performance_example.py` file.

## Configuration

The Performance Optimization System can be configured through:

1. **PerformanceSystem initialization parameters**:
   - `base_dir`: Directory for storing performance data
   - `logging_level`: Level of logging detail
   - `auto_start`: Whether to start automatically on initialization
   - `default_optimization_level`: Default optimization level

2. **Configuration files**:
   - The `OptimizationManager` uses a JSON configuration file for optimization settings
   - The `ResourceMonitor` uses a JSON configuration file for resource thresholds

## Extending the System

### Custom Optimization Strategies

You can extend the `OptimizationManager` to implement custom optimization strategies:

```python
from MacAgent.src.utils.optimization_manager import OptimizationManager

class CustomOptimizationManager(OptimizationManager):
    def optimize_vision(self, context=None, force_level=None):
        # Custom vision optimization logic
        params = super().optimize_vision(context, force_level)
        # Add custom parameters
        params["custom_setting"] = True
        return params
```

### Custom Resource Management

You can extend the `ResourceMonitor` for custom resource management:

```python
from MacAgent.src.utils.resource_monitor import ResourceMonitor, ResourceManager

class CustomResourceManager(ResourceManager):
    def get_optimal_parallelism(self, cpu_usage):
        # Custom logic for determining optimal parallelism
        return max(1, os.cpu_count() // 2)

# Use the custom resource manager
monitor = ResourceMonitor(resource_manager=CustomResourceManager())
``` 