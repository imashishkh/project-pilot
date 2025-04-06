"""
Pipeline Manager Module for MacAgent.

This module provides a flexible pipeline management system that coordinates
the perception-planning-action pipeline. It handles stage execution, 
parallelism, monitoring, diagnostics, and error recovery.
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import traceback
from collections import defaultdict

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage."""
    calls: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None
    last_run_time: Optional[float] = None
    
    @property
    def avg_duration(self) -> float:
        """Average duration of the stage in seconds."""
        return self.total_duration / self.calls if self.calls > 0 else 0.0


@dataclass
class PipelineMetrics:
    """Metrics for the entire pipeline."""
    executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    stage_metrics: Dict[str, StageMetrics] = field(default_factory=dict)
    
    @property
    def avg_duration(self) -> float:
        """Average duration of the pipeline execution in seconds."""
        return self.total_duration / self.executions if self.executions > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        return (self.successful_executions / self.executions * 100) if self.executions > 0 else 0.0


class PipelineManager:
    """
    PipelineManager coordinates the execution of the perception-planning-action pipeline.
    
    This class is responsible for:
    1. Registering and managing pipeline stages
    2. Determining execution order based on dependencies
    3. Executing the pipeline with proper error handling
    4. Collecting performance metrics for each stage
    5. Implementing recovery mechanisms for pipeline failures
    """
    
    def __init__(
        self,
        default_timeout: float = 30.0,
        enable_parallelism: bool = True,
        max_workers: int = 4,
        execution_mode: str = "async",
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            default_timeout: Default timeout for stage execution in seconds
            enable_parallelism: Whether to enable parallel execution of independent stages
            max_workers: Maximum number of worker threads for parallel execution
            execution_mode: Execution mode, either "async" or "sync"
            max_retries: Maximum number of retries for failed stages
            retry_delay: Delay between retries in seconds
        """
        self.stages: Dict[str, Callable] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.default_timeout = default_timeout
        self.enable_parallelism = enable_parallelism
        self.execution_mode = execution_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.running = False
        
        # Set up executor for parallel execution
        if enable_parallelism:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize metrics
        self.pipeline_metrics = PipelineMetrics()
        self.current_pipeline_id = 0
        
        # Pipeline validation status
        self.is_valid = False
        self.validation_errors = []
        
        # Active pipeline executions
        self.active_executions: Dict[int, Dict[str, Any]] = {}
        
        logger.info(f"PipelineManager initialized with mode={execution_mode}, parallelism={enable_parallelism}")
    
    def register_stage(
        self, 
        name: str, 
        stage_func: Callable, 
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Register a pipeline stage.
        
        Args:
            name: Stage name
            stage_func: Stage function that takes a context dict and returns an updated context
            dependencies: List of stage names that must be executed before this stage
        """
        if name in self.stages:
            logger.warning(f"Overwriting existing stage: {name}")
            
        self.stages[name] = stage_func
        self.dependencies[name] = dependencies or []
        
        # Initialize metrics for this stage
        if name not in self.pipeline_metrics.stage_metrics:
            self.pipeline_metrics.stage_metrics[name] = StageMetrics()
        
        # Invalidate pipeline validation
        self.is_valid = False
        
        logger.debug(f"Registered pipeline stage: {name} with dependencies: {dependencies}")
    
    def unregister_stage(self, name: str) -> bool:
        """
        Unregister a pipeline stage.
        
        Args:
            name: Stage name
            
        Returns:
            True if the stage was removed, False otherwise
        """
        if name not in self.stages:
            logger.warning(f"Stage not found: {name}")
            return False
            
        # Remove the stage
        del self.stages[name]
        del self.dependencies[name]
        
        # Remove this stage from other stage dependencies
        for stage, deps in self.dependencies.items():
            if name in deps:
                self.dependencies[stage].remove(name)
        
        # Invalidate pipeline validation
        self.is_valid = False
        
        logger.debug(f"Unregistered pipeline stage: {name}")
        return True
    
    def _validate_pipeline(self) -> bool:
        """
        Validate the pipeline configuration.
        
        Returns:
            True if the pipeline is valid, False otherwise
        """
        if not self.stages:
            self.validation_errors = ["No stages registered"]
            return False
            
        # Check for circular dependencies
        visited = set()
        path = []
        on_path = set()
        
        def dfs(node):
            if node in on_path:
                self.validation_errors.append(f"Circular dependency detected: {' -> '.join(path + [node])}")
                return False
                
            if node in visited:
                return True
                
            visited.add(node)
            on_path.add(node)
            path.append(node)
            
            for dep in self.dependencies.get(node, []):
                if dep not in self.stages:
                    self.validation_errors.append(f"Stage '{node}' depends on non-existent stage '{dep}'")
                    return False
                    
                if not dfs(dep):
                    return False
                    
            path.pop()
            on_path.remove(node)
            return True
            
        for stage in self.stages:
            if not dfs(stage):
                return False
                
        self.is_valid = True
        self.validation_errors = []
        return True
    
    def _get_execution_order(self) -> List[List[str]]:
        """
        Determine the execution order of stages based on dependencies.
        
        Returns:
            List of lists, where each inner list contains stages that can be executed in parallel
        """
        if not self.is_valid and not self._validate_pipeline():
            raise ValueError(f"Invalid pipeline configuration: {', '.join(self.validation_errors)}")
            
        # Implementation of topological sort with parallelism awareness
        in_degree = {stage: 0 for stage in self.stages}
        for stage in self.stages:
            for dep in self.dependencies[stage]:
                in_degree[dep] += 1
                
        # Start with stages that have no dependencies
        queue = [stage for stage, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            if self.enable_parallelism:
                # Add all stages in the current queue level to a parallel execution group
                parallel_group = queue.copy()
                execution_order.append(parallel_group)
                
                # Process all stages in the current level
                for stage in parallel_group:
                    # Decrease in-degree for all stages that depend on current stage
                    for dependent, deps in self.dependencies.items():
                        if stage in deps:
                            in_degree[dependent] -= 1
                            
                            # If all dependencies are satisfied, add to queue
                            if in_degree[dependent] == 0:
                                queue.append(dependent)
                                
                # Remove processed stages from queue
                for stage in parallel_group:
                    queue.remove(stage)
            else:
                # Process one stage at a time
                stage = queue.pop(0)
                execution_order.append([stage])
                
                # Decrease in-degree for all stages that depend on current stage
                for dependent, deps in self.dependencies.items():
                    if stage in deps:
                        in_degree[dependent] -= 1
                        
                        # If all dependencies are satisfied, add to queue
                        if in_degree[dependent] == 0:
                            queue.append(dependent)
                            
        return execution_order
    
    async def _execute_stage(
        self, 
        stage_name: str, 
        context: Dict[str, Any], 
        pipeline_id: int
    ) -> Dict[str, Any]:
        """
        Execute a single pipeline stage with error handling and metrics collection.
        
        Args:
            stage_name: Stage name
            context: Pipeline execution context
            pipeline_id: Pipeline execution ID
            
        Returns:
            Updated pipeline context
        """
        stage_func = self.stages[stage_name]
        metrics = self.pipeline_metrics.stage_metrics[stage_name]
        
        start_time = time.time()
        metrics.calls += 1
        metrics.last_run_time = start_time
        
        try:
            # Execute the stage with timeout
            if asyncio.iscoroutinefunction(stage_func):
                result = await asyncio.wait_for(stage_func(context), self.default_timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: stage_func(context)), 
                    self.default_timeout
                )
                
            # Update metrics
            duration = time.time() - start_time
            metrics.total_duration += duration
            metrics.min_duration = min(metrics.min_duration, duration)
            metrics.max_duration = max(metrics.max_duration, duration)
            
            logger.debug(f"Pipeline {pipeline_id} - Stage {stage_name} completed in {duration:.3f}s")
            
            # Update diagnostics for this execution
            self.active_executions[pipeline_id]["stages"][stage_name] = {
                "status": "completed",
                "duration": duration,
                "timestamp": time.time()
            }
            
            return result
            
        except asyncio.TimeoutError:
            # Handle timeout
            duration = time.time() - start_time
            metrics.error_count += 1
            metrics.last_error = "Timeout"
            
            logger.error(f"Pipeline {pipeline_id} - Stage {stage_name} timed out after {duration:.3f}s")
            
            # Update diagnostics
            self.active_executions[pipeline_id]["stages"][stage_name] = {
                "status": "timeout",
                "duration": duration,
                "timestamp": time.time(),
                "error": "Timeout"
            }
            
            # Add error to context
            if "errors" not in context:
                context["errors"] = []
                
            context["errors"].append({
                "stage": stage_name,
                "error": "Stage execution timed out",
                "timestamp": time.time()
            })
            
            return context
            
        except Exception as e:
            # Handle other errors
            duration = time.time() - start_time
            metrics.error_count += 1
            metrics.last_error = str(e)
            
            logger.error(f"Pipeline {pipeline_id} - Stage {stage_name} failed after {duration:.3f}s: {str(e)}")
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Update diagnostics
            self.active_executions[pipeline_id]["stages"][stage_name] = {
                "status": "error",
                "duration": duration,
                "timestamp": time.time(),
                "error": str(e)
            }
            
            # Add error to context
            if "errors" not in context:
                context["errors"] = []
                
            context["errors"].append({
                "stage": stage_name,
                "error": str(e),
                "timestamp": time.time()
            })
            
            return context
    
    async def _execute_parallel_group(
        self, 
        group: List[str], 
        context: Dict[str, Any], 
        pipeline_id: int
    ) -> Dict[str, Any]:
        """
        Execute a group of stages in parallel.
        
        Args:
            group: List of stage names to execute in parallel
            context: Pipeline execution context
            pipeline_id: Pipeline execution ID
            
        Returns:
            Updated pipeline context
        """
        if len(group) == 1:
            # Only one stage, execute directly
            return await self._execute_stage(group[0], context, pipeline_id)
            
        # Multiple stages, execute in parallel
        tasks = []
        for stage_name in group:
            # Create a copy of the context for each parallel stage
            stage_context = context.copy()
            tasks.append(self._execute_stage(stage_name, stage_context, pipeline_id))
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Merge the results back into the main context
        for result in results:
            for key, value in result.items():
                if key == "errors":
                    # Merge error lists
                    if "errors" not in context:
                        context["errors"] = []
                    context["errors"].extend(value)
                else:
                    # Overwrite other values
                    context[key] = value
                    
        return context
    
    async def execute_pipeline(self, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the entire pipeline.
        
        Args:
            initial_context: Initial pipeline context
            
        Returns:
            Final pipeline context after execution
        """
        if not self.is_valid and not self._validate_pipeline():
            raise ValueError(f"Invalid pipeline configuration: {', '.join(self.validation_errors)}")
            
        # Generate pipeline ID
        self.current_pipeline_id += 1
        pipeline_id = self.current_pipeline_id
        
        # Initialize context
        context = initial_context or {}
        if "_pipeline" not in context:
            context["_pipeline"] = {
                "id": pipeline_id,
                "start_time": time.time(),
                "stages_completed": []
            }
            
        # Set up pipeline execution tracking
        self.active_executions[pipeline_id] = {
            "status": "running",
            "start_time": time.time(),
            "context": context,
            "stages": {}
        }
        
        logger.info(f"Pipeline {pipeline_id} - Execution started")
        
        # Update metrics
        self.pipeline_metrics.executions += 1
        start_time = time.time()
        
        try:
            # Get execution order
            execution_order = self._get_execution_order()
            
            # Execute stages in order
            for group in execution_order:
                context = await self._execute_parallel_group(group, context, pipeline_id)
                
                # Update stages completed
                context["_pipeline"]["stages_completed"].extend(group)
                
            # Pipeline completed successfully
            duration = time.time() - start_time
            self.pipeline_metrics.successful_executions += 1
            self.pipeline_metrics.total_duration += duration
            self.pipeline_metrics.min_duration = min(self.pipeline_metrics.min_duration, duration)
            self.pipeline_metrics.max_duration = max(self.pipeline_metrics.max_duration, duration)
            
            # Update execution tracking
            self.active_executions[pipeline_id]["status"] = "completed"
            self.active_executions[pipeline_id]["duration"] = duration
            self.active_executions[pipeline_id]["end_time"] = time.time()
            
            logger.info(f"Pipeline {pipeline_id} - Execution completed in {duration:.3f}s")
            
            return context
            
        except Exception as e:
            # Pipeline failed
            duration = time.time() - start_time
            self.pipeline_metrics.failed_executions += 1
            
            # Update execution tracking
            self.active_executions[pipeline_id]["status"] = "failed"
            self.active_executions[pipeline_id]["duration"] = duration
            self.active_executions[pipeline_id]["end_time"] = time.time()
            self.active_executions[pipeline_id]["error"] = str(e)
            
            logger.error(f"Pipeline {pipeline_id} - Execution failed after {duration:.3f}s: {str(e)}")
            
            # Add error to context
            if "errors" not in context:
                context["errors"] = []
                
            context["errors"].append({
                "stage": "pipeline",
                "error": str(e),
                "timestamp": time.time()
            })
            
            return context
        finally:
            # Clean up old executions (keep only last 10)
            execution_ids = sorted(self.active_executions.keys())
            if len(execution_ids) > 10:
                for old_id in execution_ids[:-10]:
                    del self.active_executions[old_id]
    
    def get_pipeline_status(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the status of a pipeline execution.
        
        Args:
            pipeline_id: Pipeline execution ID
            
        Returns:
            Status information or None if the pipeline ID is not found
        """
        return self.active_executions.get(pipeline_id)
    
    def get_active_pipelines(self) -> List[int]:
        """
        Get the IDs of all active pipeline executions.
        
        Returns:
            List of pipeline IDs
        """
        return [pid for pid, info in self.active_executions.items() if info["status"] == "running"]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the pipeline.
        
        Returns:
            Dictionary with pipeline metrics
        """
        result = {
            "executions": self.pipeline_metrics.executions,
            "successful_executions": self.pipeline_metrics.successful_executions,
            "failed_executions": self.pipeline_metrics.failed_executions,
            "success_rate": self.pipeline_metrics.success_rate,
            "avg_duration": self.pipeline_metrics.avg_duration,
            "min_duration": self.pipeline_metrics.min_duration,
            "max_duration": self.pipeline_metrics.max_duration,
            "stages": {}
        }
        
        # Add stage metrics
        for stage_name, metrics in self.pipeline_metrics.stage_metrics.items():
            result["stages"][stage_name] = {
                "calls": metrics.calls,
                "avg_duration": metrics.avg_duration,
                "min_duration": metrics.min_duration,
                "max_duration": metrics.max_duration,
                "error_count": metrics.error_count,
                "last_error": metrics.last_error,
                "last_run_time": metrics.last_run_time
            }
            
        return result
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.pipeline_metrics = PipelineMetrics()
        for stage_name in self.stages:
            self.pipeline_metrics.stage_metrics[stage_name] = StageMetrics()
            
        logger.info("Pipeline metrics reset")
    
    def start(self) -> None:
        """Start the pipeline manager."""
        if self.running:
            logger.warning("Pipeline manager already running")
            return
            
        self.running = True
        logger.info("Pipeline manager started")
    
    def stop(self) -> None:
        """Stop the pipeline manager."""
        if not self.running:
            logger.warning("Pipeline manager not running")
            return
            
        self.running = False
        
        if self.enable_parallelism:
            self.executor.shutdown(wait=False)
            
        logger.info("Pipeline manager stopped")
    
    def get_pipeline_visualization(self) -> Dict[str, Any]:
        """
        Generate a visualization of the pipeline structure.
        
        Returns:
            Dictionary with pipeline structure information
        """
        execution_order = self._get_execution_order()
        
        return {
            "stages": list(self.stages.keys()),
            "dependencies": self.dependencies,
            "execution_order": execution_order,
            "parallelism_enabled": self.enable_parallelism,
            "execution_mode": self.execution_mode,
            "metrics": self.get_performance_metrics()
        }
    
    def recover_pipeline(self, pipeline_id: int) -> Optional[Dict[str, Any]]:
        """
        Attempt to recover a failed pipeline execution.
        
        Args:
            pipeline_id: Pipeline execution ID
            
        Returns:
            Updated pipeline context or None if recovery failed
        """
        if pipeline_id not in self.active_executions:
            logger.warning(f"Pipeline {pipeline_id} not found")
            return None
            
        execution_info = self.active_executions[pipeline_id]
        if execution_info["status"] not in ["failed", "timeout"]:
            logger.warning(f"Pipeline {pipeline_id} is not in a failed state")
            return None
            
        logger.info(f"Attempting to recover pipeline {pipeline_id}")
        
        # Copy the original context
        context = execution_info["context"].copy()
        
        # Find failed stages
        failed_stages = []
        for stage_name, stage_info in execution_info["stages"].items():
            if stage_info["status"] in ["error", "timeout"]:
                failed_stages.append(stage_name)
                
        if not failed_stages:
            logger.warning(f"No failed stages found in pipeline {pipeline_id}")
            return None
            
        # Retry failed stages
        for stage_name in failed_stages:
            try:
                # Remove stage from completed list
                if "_pipeline" in context and "stages_completed" in context["_pipeline"]:
                    if stage_name in context["_pipeline"]["stages_completed"]:
                        context["_pipeline"]["stages_completed"].remove(stage_name)
                
                # Retry the stage
                logger.info(f"Retrying stage {stage_name} in pipeline {pipeline_id}")
                context = asyncio.run(self._execute_stage(stage_name, context, pipeline_id))
                
                # Add back to completed list
                if "_pipeline" in context and "stages_completed" in context["_pipeline"]:
                    context["_pipeline"]["stages_completed"].append(stage_name)
                    
            except Exception as e:
                logger.error(f"Recovery failed for stage {stage_name}: {str(e)}")
                return None
                
        # Update execution status
        execution_info["status"] = "recovered"
        execution_info["recovery_time"] = time.time()
        
        logger.info(f"Pipeline {pipeline_id} recovered successfully")
        return context
    
    def is_pipeline_complete(self, pipeline_id: int) -> bool:
        """
        Check if a pipeline execution is complete.
        
        Args:
            pipeline_id: Pipeline execution ID
            
        Returns:
            True if the pipeline is complete, False otherwise
        """
        if pipeline_id not in self.active_executions:
            return False
            
        return self.active_executions[pipeline_id]["status"] in ["completed", "recovered"]
    
    def is_pipeline_failed(self, pipeline_id: int) -> bool:
        """
        Check if a pipeline execution has failed.
        
        Args:
            pipeline_id: Pipeline execution ID
            
        Returns:
            True if the pipeline has failed, False otherwise
        """
        if pipeline_id not in self.active_executions:
            return False
            
        return self.active_executions[pipeline_id]["status"] in ["failed", "timeout"]
