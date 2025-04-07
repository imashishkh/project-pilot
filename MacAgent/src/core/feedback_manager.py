import time
import logging
import uuid
from typing import Dict, Any, List, Optional
from enum import Enum, auto
from datetime import datetime

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Status of an execution or step."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()

class FeedbackManager:
    """Manages feedback for agent actions and execution."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize feedback tracking.
        
        Args:
            verbose: Whether to log detailed feedback
        """
        # Configuration
        self.verbose = verbose
        
        # Tracking data structures
        self.executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[str] = []
        self.current_execution_id: Optional[str] = None
        self.next_step_id = 1
    
    def start_execution(self, instruction: str) -> str:
        """
        Start tracking a new execution.
        
        Args:
            instruction: The instruction being executed
            
        Returns:
            Execution ID string
        """
        # Generate a unique ID for this execution
        execution_id = str(uuid.uuid4())
        
        # Record start time
        start_time = time.time()
        
        # Create execution record
        execution = {
            "id": execution_id,
            "instruction": instruction,
            "status": ExecutionStatus.IN_PROGRESS,
            "start_time": start_time,
            "start_time_formatted": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "duration": 0,
            "steps": {},
            "steps_order": [],
            "result": None,
            "success": None,
            "error": None
        }
        
        # Store execution and set as current
        self.executions[execution_id] = execution
        self.execution_history.append(execution_id)
        self.current_execution_id = execution_id
        
        if self.verbose:
            logger.info(f"Started execution [{execution_id}]: {instruction}")
        
        return execution_id
    
    def add_step(self, description: str, action_type: str) -> int:
        """
        Add a new step to the current execution.
        
        Args:
            description: Description of the step
            action_type: Type of action being performed
            
        Returns:
            Step ID integer
            
        Raises:
            RuntimeError: If no execution is in progress
        """
        if not self.current_execution_id:
            raise RuntimeError("No execution in progress. Call start_execution first.")
        
        # Generate step ID
        step_id = self.next_step_id
        self.next_step_id += 1
        
        # Record start time
        start_time = time.time()
        
        # Create step record
        step = {
            "id": step_id,
            "description": description,
            "action_type": action_type,
            "status": ExecutionStatus.PENDING,
            "start_time": start_time,
            "start_time_formatted": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "duration": 0,
            "result": None,
            "error": None
        }
        
        # Store step
        current_execution = self.executions[self.current_execution_id]
        current_execution["steps"][step_id] = step
        current_execution["steps_order"].append(step_id)
        
        if self.verbose:
            logger.info(f"Added step {step_id} [{action_type}]: {description}")
        
        return step_id
    
    def update_step(self, step_id: int, status: str, result: Dict[str, Any] = None) -> None:
        """
        Update the status and result of a step.
        
        Args:
            step_id: ID of the step to update
            status: New status (e.g., 'completed', 'failed', 'in_progress')
            result: Optional result data
            
        Raises:
            RuntimeError: If no execution is in progress
            ValueError: If step_id is invalid
            ValueError: If status is invalid
        """
        if not self.current_execution_id:
            raise RuntimeError("No execution in progress.")
        
        current_execution = self.executions[self.current_execution_id]
        
        if step_id not in current_execution["steps"]:
            raise ValueError(f"Invalid step ID: {step_id}")
        
        # Get the step
        step = current_execution["steps"][step_id]
        
        # Map string status to enum
        try:
            if isinstance(status, str):
                status_enum = ExecutionStatus[status.upper()]
            elif isinstance(status, ExecutionStatus):
                status_enum = status
            else:
                raise ValueError(f"Invalid status type: {type(status)}")
        except (KeyError, AttributeError):
            raise ValueError(f"Invalid status: {status}. Valid values are: {', '.join([s.name.lower() for s in ExecutionStatus])}")
        
        # Update step status
        step["status"] = status_enum
        
        # Update result if provided
        if result:
            step["result"] = result
            if "error" in result and not result.get("success", False):
                step["error"] = result["error"]
        
        # If step is in a terminal state, update end time and duration
        if status_enum in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED, ExecutionStatus.TIMEOUT]:
            step["end_time"] = time.time()
            step["duration"] = step["end_time"] - step["start_time"]
        
        if self.verbose:
            status_message = f"Step {step_id} {status_enum.name.lower()}"
            if status_enum == ExecutionStatus.FAILED and step.get("error"):
                status_message += f": {step['error']}"
            logger.info(status_message)
    
    def complete_execution(self, success: bool, result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete the current execution.
        
        Args:
            success: Whether the execution was successful
            result: Optional result data
            
        Returns:
            Execution summary
            
        Raises:
            RuntimeError: If no execution is in progress
        """
        if not self.current_execution_id:
            raise RuntimeError("No execution in progress.")
        
        current_execution = self.executions[self.current_execution_id]
        
        # Set completion time and status
        current_execution["end_time"] = time.time()
        current_execution["duration"] = current_execution["end_time"] - current_execution["start_time"]
        current_execution["success"] = success
        current_execution["status"] = ExecutionStatus.COMPLETED if success else ExecutionStatus.FAILED
        
        # Store result if provided
        if result:
            current_execution["result"] = result
            if "error" in result and not success:
                current_execution["error"] = result["error"]
        
        # Create summary
        summary = {
            "id": current_execution["id"],
            "instruction": current_execution["instruction"],
            "success": success,
            "duration": current_execution["duration"],
            "steps_total": len(current_execution["steps"]),
            "steps_completed": len([s for s in current_execution["steps"].values() 
                                   if s["status"] == ExecutionStatus.COMPLETED]),
            "steps_failed": len([s for s in current_execution["steps"].values() 
                                if s["status"] == ExecutionStatus.FAILED]),
            "result": result
        }
        
        if self.verbose:
            status_message = f"Execution {current_execution['id']} {'succeeded' if success else 'failed'}"
            if not success and current_execution.get("error"):
                status_message += f": {current_execution['error']}"
            logger.info(status_message)
        
        # Clear current execution
        execution_id = self.current_execution_id
        self.current_execution_id = None
        
        return summary
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the execution history.
        
        Returns:
            List of execution summaries
        """
        history = []
        for execution_id in self.execution_history:
            execution = self.executions[execution_id]
            history.append({
                "id": execution["id"],
                "instruction": execution["instruction"],
                "status": execution["status"].name,
                "start_time": execution["start_time_formatted"],
                "duration": execution["duration"],
                "success": execution["success"],
                "steps_total": len(execution["steps"]),
                "steps_completed": len([s for s in execution["steps"].values() 
                                      if s["status"] == ExecutionStatus.COMPLETED]),
                "steps_failed": len([s for s in execution["steps"].values() 
                                   if s["status"] == ExecutionStatus.FAILED])
            })
        return history
    
    def get_last_execution(self) -> Optional[Dict[str, Any]]:
        """
        Get the last execution.
        
        Returns:
            Last execution details or None if no executions
        """
        if not self.execution_history:
            return None
        
        last_execution_id = self.execution_history[-1]
        execution = self.executions[last_execution_id]
        
        # Build detailed execution report
        steps_info = []
        for step_id in execution["steps_order"]:
            step = execution["steps"][step_id]
            steps_info.append({
                "id": step["id"],
                "description": step["description"],
                "action_type": step["action_type"],
                "status": step["status"].name,
                "duration": step["duration"],
                "error": step.get("error"),
                "result": step.get("result")
            })
        
        return {
            "id": execution["id"],
            "instruction": execution["instruction"],
            "status": execution["status"].name,
            "start_time": execution["start_time_formatted"],
            "end_time": datetime.fromtimestamp(execution["end_time"]).strftime("%Y-%m-%d %H:%M:%S") if execution["end_time"] else None,
            "duration": execution["duration"],
            "success": execution["success"],
            "error": execution.get("error"),
            "result": execution.get("result"),
            "steps": steps_info
        }
    
    def cancel_execution(self, reason: str = "User cancelled") -> Dict[str, Any]:
        """
        Cancel the current execution.
        
        Args:
            reason: Reason for cancellation
            
        Returns:
            Execution summary
        """
        if not self.current_execution_id:
            raise RuntimeError("No execution in progress.")
        
        current_execution = self.executions[self.current_execution_id]
        
        # Set cancellation time and status
        current_execution["end_time"] = time.time()
        current_execution["duration"] = current_execution["end_time"] - current_execution["start_time"]
        current_execution["success"] = False
        current_execution["status"] = ExecutionStatus.CANCELLED
        current_execution["error"] = reason
        
        # Cancel any pending steps
        for step in current_execution["steps"].values():
            if step["status"] == ExecutionStatus.PENDING or step["status"] == ExecutionStatus.IN_PROGRESS:
                step["status"] = ExecutionStatus.CANCELLED
                step["end_time"] = time.time()
                step["duration"] = step["end_time"] - step["start_time"]
                step["error"] = "Cancelled due to execution cancellation"
        
        if self.verbose:
            logger.info(f"Execution {current_execution['id']} cancelled: {reason}")
        
        # Create summary
        summary = {
            "id": current_execution["id"],
            "instruction": current_execution["instruction"],
            "success": False,
            "cancelled": True,
            "duration": current_execution["duration"],
            "reason": reason,
            "steps_total": len(current_execution["steps"]),
            "steps_completed": len([s for s in current_execution["steps"].values() 
                                  if s["status"] == ExecutionStatus.COMPLETED]),
            "steps_cancelled": len([s for s in current_execution["steps"].values() 
                                  if s["status"] == ExecutionStatus.CANCELLED])
        }
        
        # Clear current execution
        self.current_execution_id = None
        
        return summary
