import asyncio
from typing import Dict, Any, Optional, List, Union
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enum representing task status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

class TaskManager:
    """Manages and tracks asyncio tasks."""
    
    def __init__(self):
        """Initialize task tracking dictionaries and lock."""
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, Any] = {}
        self._statuses: Dict[str, TaskStatus] = {}
        self._exceptions: Dict[str, Exception] = {}
        self._lock = asyncio.Lock()
    
    async def create_task(self, name: str, coro) -> asyncio.Task:
        """
        Create and track a new task.
        
        Args:
            name: A unique name for the task
            coro: The coroutine to run as a task
            
        Returns:
            The created asyncio Task object
        """
        async with self._lock:
            if name in self._tasks:
                raise ValueError(f"Task with name '{name}' already exists")
            
            wrapped_coro = self._wrapped_coro(name, coro)
            task = asyncio.create_task(wrapped_coro, name=name)
            
            self._tasks[name] = task
            self._statuses[name] = TaskStatus.PENDING
            
            logger.debug(f"Created task: {name}")
            return task
    
    async def _wrapped_coro(self, name: str, coro):
        """
        Wrapper to track task completion and results.
        
        Args:
            name: The task name
            coro: The original coroutine
            
        Returns:
            The result of the coroutine
        """
        try:
            async with self._lock:
                self._statuses[name] = TaskStatus.RUNNING
            
            result = await coro
            
            async with self._lock:
                self._results[name] = result
                self._statuses[name] = TaskStatus.COMPLETED
                
            logger.debug(f"Task completed successfully: {name}")
            return result
            
        except asyncio.CancelledError:
            async with self._lock:
                self._statuses[name] = TaskStatus.CANCELLED
            logger.debug(f"Task cancelled: {name}")
            raise
            
        except Exception as e:
            async with self._lock:
                self._exceptions[name] = e
                self._statuses[name] = TaskStatus.FAILED
            
            logger.error(f"Task failed: {name}", exc_info=e)
            raise
    
    async def wait_for_task(self, name: str, timeout: Optional[float] = None) -> Any:
        """
        Wait for a specific task to complete.
        
        Args:
            name: The task name
            timeout: Optional timeout in seconds
            
        Returns:
            The task result
            
        Raises:
            KeyError: If task doesn't exist
            TimeoutError: If timeout occurs
            Exception: If task failed with exception
        """
        async with self._lock:
            if name not in self._tasks:
                raise KeyError(f"No task found with name '{name}'")
            
            task = self._tasks[name]
        
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for task: {name}")
            raise TimeoutError(f"Timeout waiting for task: {name}")
        
        async with self._lock:
            if self._statuses[name] == TaskStatus.FAILED:
                raise self._exceptions[name]
            return self._results.get(name)
    
    async def wait_for_all_tasks(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Wait for all tasks to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary of task names to results
            
        Raises:
            TimeoutError: If timeout occurs
        """
        async with self._lock:
            tasks = list(self._tasks.values())
        
        if not tasks:
            return {}
        
        try:
            done, pending = await asyncio.wait(
                [asyncio.shield(task) for task in tasks],
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            if pending:
                logger.warning(f"Timeout waiting for {len(pending)} tasks")
                raise TimeoutError(f"Timeout waiting for {len(pending)} tasks")
            
        except asyncio.CancelledError:
            logger.warning("Wait for all tasks cancelled")
            raise
        
        async with self._lock:
            return {name: self._results[name] 
                   for name in self._tasks 
                   if name in self._results and self._statuses[name] == TaskStatus.COMPLETED}
    
    def get_task_status(self, name: str) -> TaskStatus:
        """
        Get the status of a specific task.
        
        Args:
            name: The task name
            
        Returns:
            TaskStatus enum value
            
        Raises:
            KeyError: If task doesn't exist
        """
        if name not in self._statuses:
            raise KeyError(f"No task found with name '{name}'")
        
        return self._statuses[name]
    
    def get_all_task_statuses(self) -> Dict[str, TaskStatus]:
        """
        Get the status of all tasks.
        
        Returns:
            Dictionary of task names to TaskStatus values
        """
        return self._statuses.copy()
    
    def cancel_task(self, name: str) -> bool:
        """
        Cancel a specific task.
        
        Args:
            name: The task name
            
        Returns:
            True if task was cancelled, False if already done
            
        Raises:
            KeyError: If task doesn't exist
        """
        if name not in self._tasks:
            raise KeyError(f"No task found with name '{name}'")
        
        task = self._tasks[name]
        
        if task.done():
            return False
        
        task.cancel()
        self._statuses[name] = TaskStatus.CANCELLED
        logger.debug(f"Cancelled task: {name}")
        return True
    
    def cancel_all_tasks(self) -> List[str]:
        """
        Cancel all running tasks.
        
        Returns:
            List of names of tasks that were cancelled
        """
        cancelled = []
        
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                self._statuses[name] = TaskStatus.CANCELLED
                cancelled.append(name)
        
        if cancelled:
            logger.debug(f"Cancelled {len(cancelled)} tasks")
        
        return cancelled
