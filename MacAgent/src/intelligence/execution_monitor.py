"""
Execution Monitoring Module for MacAgent.

This module provides execution monitoring capabilities for the MacAgent, including:
- Tracking task execution and completion status
- Detecting deviations from expected outcomes
- Providing feedback on task progress and results
- Adapting to execution environment changes
"""

import enum
import json
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field, asdict

from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider
from MacAgent.src.intelligence.prompt_manager import PromptManager, PromptStrategy

# Configure logging
logger = logging.getLogger(__name__)


class DeviationType(enum.Enum):
    """Types of deviations that can occur during execution."""
    NONE = "none"  # No deviation
    ERROR = "error"  # Execution error
    TIMEOUT = "timeout"  # Execution timeout
    UNEXPECTED_RESULT = "unexpected_result"  # Result does not match expected outcome
    PARTIAL_COMPLETION = "partial_completion"  # Task partially completed
    RESOURCE_LIMITATION = "resource_limitation"  # Resource limitations prevented completion
    PERMISSION_DENIED = "permission_denied"  # Permission issues
    DEPENDENCY_FAILURE = "dependency_failure"  # Dependency task failed
    USER_INTERRUPTION = "user_interruption"  # User interrupted the task
    SYSTEM_CHANGE = "system_change"  # System state changed unexpectedly


@dataclass
class ExpectedOutcome:
    """Represents an expected outcome for a task."""
    id: str  # Unique identifier
    description: str  # Description of the expected outcome
    success_criteria: List[str]  # Criteria for considering the outcome successful
    verification_method: str  # How to verify the outcome
    parameters: Dict[str, Any] = field(default_factory=dict)  # Parameters for verification
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert expected outcome to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExpectedOutcome':
        """Create expected outcome from dictionary."""
        return cls(**data)


@dataclass
class ExecutionResult:
    """Represents the result of a task execution."""
    id: str  # Unique identifier
    task_id: str  # ID of the task that was executed
    status: str  # Status (completed, failed, timeout, interrupted)
    start_time: float  # When execution started
    end_time: Optional[float] = None  # When execution ended
    output: Any = None  # Output of the execution
    error_message: Optional[str] = None  # Error message if failed
    success: bool = False  # Whether execution was successful
    expected_outcome_id: Optional[str] = None  # ID of the expected outcome
    deviation_type: DeviationType = DeviationType.NONE  # Type of deviation
    deviation_details: Optional[Dict[str, Any]] = None  # Details about the deviation
    context: Dict[str, Any] = field(default_factory=dict)  # Execution context
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert execution result to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "output": self.output,
            "error_message": self.error_message,
            "success": self.success,
            "expected_outcome_id": self.expected_outcome_id,
            "deviation_type": self.deviation_type.value if self.deviation_type else None,
            "deviation_details": self.deviation_details,
            "context": self.context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionResult':
        """Create execution result from dictionary."""
        # Convert deviation_type string to enum if present
        if "deviation_type" in data and data["deviation_type"]:
            data["deviation_type"] = DeviationType(data["deviation_type"])
        return cls(**data)
    
    def get_duration(self) -> Optional[float]:
        """Get the execution duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


class ExecutionMonitor:
    """
    Execution Monitor for MacAgent system.
    
    Tracks task execution, detects deviations, and provides feedback.
    """
    
    def __init__(
        self,
        llm_connector: LLMConnector,
        prompt_manager: PromptManager,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        default_model: str = "gpt-3.5-turbo",
        results_dir: str = "data/execution_results"
    ):
        """
        Initialize the ExecutionMonitor.
        
        Args:
            llm_connector: The LLM connector for analyzing outcomes
            prompt_manager: The prompt manager for generating prompts
            default_provider: Default LLM provider to use
            default_model: Default model to use
            results_dir: Directory to store execution results
        """
        self.llm_connector = llm_connector
        self.prompt_manager = prompt_manager
        self.default_provider = default_provider
        self.default_model = default_model
        self.results_dir = results_dir
        
        # Ensure results directory exists
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize collections
        self.expected_outcomes: Dict[str, ExpectedOutcome] = {}
        self.execution_results: Dict[str, ExecutionResult] = self._load_results()
        
        logger.info(f"ExecutionMonitor initialized with {len(self.execution_results)} existing results")
    
    def _load_results(self) -> Dict[str, ExecutionResult]:
        """Load existing execution results from disk."""
        import os
        results = {}
        result_files = [f for f in os.listdir(self.results_dir) 
                       if f.endswith('.json') and os.path.isfile(os.path.join(self.results_dir, f))]
        
        for file_name in result_files:
            file_path = os.path.join(self.results_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    result_data = json.load(f)
                    result = ExecutionResult.from_dict(result_data)
                    results[result.id] = result
            except Exception as e:
                logger.error(f"Error loading execution result from {file_path}: {e}")
        
        return results
    
    def _save_result(self, result: ExecutionResult) -> None:
        """Save execution result to disk."""
        import os
        file_path = os.path.join(self.results_dir, f"{result.id}.json")
        with open(file_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    async def generate_expected_outcome(
        self,
        task_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> ExpectedOutcome:
        """
        Generate an expected outcome for a task.
        
        Args:
            task_id: ID of the task
            task_description: Description of the task
            context: Additional context for the task
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            An ExpectedOutcome object
        """
        # Use default values if not provided
        context = context or {}
        provider = provider or self.default_provider
        model = model or self.default_model
        
        # Prepare prompt for outcome generation
        template_name = "expected_outcome_generation"
        prompt_context = {
            "task_id": task_id,
            "task_description": task_description,
            "context": json.dumps(context, indent=2)
        }
        
        # Get the prompt
        prompt = await self.prompt_manager.get_prompt(
            template_name=template_name,
            strategy=PromptStrategy.STRUCTURED_OUTPUT,
            context=prompt_context
        )
        
        # Generate with LLM
        response = await self.llm_connector.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            json_response=True,
            temperature=0.3
        )
        
        # Parse response
        try:
            outcome_data = response.get('json', {})
            
            # Create outcome ID
            outcome_id = str(uuid.uuid4())
            
            # Create expected outcome
            expected_outcome = ExpectedOutcome(
                id=outcome_id,
                description=outcome_data.get("description", ""),
                success_criteria=outcome_data.get("success_criteria", []),
                verification_method=outcome_data.get("verification_method", "manual"),
                parameters=outcome_data.get("parameters", {}),
                metadata=outcome_data.get("metadata", {})
            )
            
            # Store the expected outcome
            self.expected_outcomes[outcome_id] = expected_outcome
            
            logger.info(f"Generated expected outcome {outcome_id} for task {task_id}")
            return expected_outcome
            
        except Exception as e:
            logger.error(f"Error generating expected outcome: {e}")
            raise ValueError(f"Failed to generate expected outcome: {e}")
    
    def start_execution(
        self,
        task_id: str,
        expected_outcome_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Start monitoring the execution of a task.
        
        Args:
            task_id: ID of the task being executed
            expected_outcome_id: ID of the expected outcome
            context: Execution context
            
        Returns:
            An ExecutionResult object with the initial state
        """
        # Create result ID
        result_id = str(uuid.uuid4())
        
        # Create execution result
        execution_result = ExecutionResult(
            id=result_id,
            task_id=task_id,
            status="in_progress",
            start_time=time.time(),
            expected_outcome_id=expected_outcome_id,
            context=context or {}
        )
        
        # Store the execution result
        self.execution_results[result_id] = execution_result
        self._save_result(execution_result)
        
        logger.info(f"Started monitoring execution of task {task_id} with result ID {result_id}")
        return execution_result
    
    def complete_execution(
        self,
        result_id: str,
        success: bool,
        output: Any = None,
        error_message: Optional[str] = None
    ) -> ExecutionResult:
        """
        Complete the execution of a task.
        
        Args:
            result_id: ID of the execution result
            success: Whether the execution was successful
            output: Output of the execution
            error_message: Error message if failed
            
        Returns:
            Updated ExecutionResult
        """
        if result_id not in self.execution_results:
            raise ValueError(f"Execution result with ID {result_id} not found")
        
        execution_result = self.execution_results[result_id]
        
        # Update execution result
        execution_result.end_time = time.time()
        execution_result.success = success
        execution_result.output = output
        execution_result.error_message = error_message
        execution_result.status = "completed" if success else "failed"
        
        # Save the execution result
        self._save_result(execution_result)
        
        logger.info(f"Completed execution of task {execution_result.task_id} with result {success}")
        return execution_result
    
    async def verify_outcome(
        self,
        result_id: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> ExecutionResult:
        """
        Verify if the execution result matches the expected outcome.
        
        Args:
            result_id: ID of the execution result
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Updated ExecutionResult with verification results
        """
        if result_id not in self.execution_results:
            raise ValueError(f"Execution result with ID {result_id} not found")
        
        execution_result = self.execution_results[result_id]
        
        # Skip if no expected outcome
        if not execution_result.expected_outcome_id:
            logger.warning(f"No expected outcome for execution result {result_id}")
            return execution_result
        
        # Get the expected outcome
        expected_outcome = self.expected_outcomes.get(execution_result.expected_outcome_id)
        if not expected_outcome:
            logger.warning(f"Expected outcome {execution_result.expected_outcome_id} not found")
            return execution_result
        
        # Prepare prompt for verification
        template_name = "outcome_verification"
        prompt_context = {
            "task_id": execution_result.task_id,
            "execution_result": {
                "status": execution_result.status,
                "success": execution_result.success,
                "output": execution_result.output,
                "error_message": execution_result.error_message,
                "duration": execution_result.get_duration()
            },
            "expected_outcome": {
                "description": expected_outcome.description,
                "success_criteria": expected_outcome.success_criteria,
                "verification_method": expected_outcome.verification_method,
                "parameters": expected_outcome.parameters
            },
            "context": json.dumps(execution_result.context, indent=2)
        }
        
        # Get the prompt
        prompt = await self.prompt_manager.get_prompt(
            template_name=template_name,
            strategy=PromptStrategy.STRUCTURED_OUTPUT,
            context=prompt_context
        )
        
        # Generate with LLM
        provider = provider or self.default_provider
        model = model or self.default_model
        
        response = await self.llm_connector.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            json_response=True,
            temperature=0.2
        )
        
        # Parse response
        try:
            verification = response.get('json', {})
            
            # Update execution result
            if "deviation_type" in verification and verification["deviation_type"]:
                execution_result.deviation_type = DeviationType(verification["deviation_type"])
            else:
                execution_result.deviation_type = DeviationType.NONE
            
            execution_result.deviation_details = verification.get("deviation_details", {})
            
            # If there's a deviation and success was initially true, update it
            if execution_result.deviation_type != DeviationType.NONE and execution_result.success:
                execution_result.success = False
            
            # Save the execution result
            self._save_result(execution_result)
            
            logger.info(f"Verified outcome for task {execution_result.task_id}, deviation: {execution_result.deviation_type.value}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Error verifying outcome: {e}")
            
            # Set a default deviation
            execution_result.deviation_type = DeviationType.ERROR
            execution_result.deviation_details = {"error": str(e)}
            
            # Save the execution result
            self._save_result(execution_result)
            
            return execution_result
    
    def get_execution_result(self, result_id: str) -> ExecutionResult:
        """
        Get an execution result by ID.
        
        Args:
            result_id: ID of the execution result
            
        Returns:
            ExecutionResult object
        """
        if result_id not in self.execution_results:
            raise ValueError(f"Execution result with ID {result_id} not found")
        
        return self.execution_results[result_id]
    
    def get_all_execution_results(self) -> List[ExecutionResult]:
        """
        Get all execution results.
        
        Returns:
            List of all ExecutionResult objects
        """
        return list(self.execution_results.values())
    
    def get_task_execution_results(self, task_id: str) -> List[ExecutionResult]:
        """
        Get all execution results for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of ExecutionResult objects for the task
        """
        return [
            result for result in self.execution_results.values()
            if result.task_id == task_id
        ]
    
    async def analyze_failure(
        self,
        result_id: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a failed execution to determine cause and suggest remediation.
        
        Args:
            result_id: ID of the execution result
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Analysis of the failure
        """
        if result_id not in self.execution_results:
            raise ValueError(f"Execution result with ID {result_id} not found")
        
        execution_result = self.execution_results[result_id]
        
        # Skip if successful
        if execution_result.success:
            return {"message": "Execution was successful, no failure to analyze"}
        
        # Get the expected outcome if available
        expected_outcome = None
        if execution_result.expected_outcome_id:
            expected_outcome = self.expected_outcomes.get(execution_result.expected_outcome_id)
        
        # Prepare prompt for analysis
        template_name = "failure_analysis"
        prompt_context = {
            "task_id": execution_result.task_id,
            "execution_result": {
                "status": execution_result.status,
                "output": execution_result.output,
                "error_message": execution_result.error_message,
                "duration": execution_result.get_duration(),
                "deviation_type": execution_result.deviation_type.value if execution_result.deviation_type else None,
                "deviation_details": execution_result.deviation_details
            },
            "expected_outcome": None if not expected_outcome else {
                "description": expected_outcome.description,
                "success_criteria": expected_outcome.success_criteria,
                "verification_method": expected_outcome.verification_method,
                "parameters": expected_outcome.parameters
            },
            "context": json.dumps(execution_result.context, indent=2)
        }
        
        # Get the prompt
        prompt = await self.prompt_manager.get_prompt(
            template_name=template_name,
            strategy=PromptStrategy.STRUCTURED_OUTPUT,
            context=prompt_context
        )
        
        # Generate with LLM
        provider = provider or self.default_provider
        model = model or self.default_model
        
        response = await self.llm_connector.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            json_response=True,
            temperature=0.3
        )
        
        # Parse response
        try:
            analysis = response.get('json', {})
            
            # Store analysis in execution result metadata
            execution_result.metadata["failure_analysis"] = analysis
            
            # Save the execution result
            self._save_result(execution_result)
            
            logger.info(f"Analyzed failure for task {execution_result.task_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing failure: {e}")
            raise ValueError(f"Failed to analyze failure: {e}")
    
    async def generate_progress_report(
        self,
        task_ids: List[str],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a progress report for multiple tasks.
        
        Args:
            task_ids: IDs of the tasks to include in the report
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Progress report
        """
        # Get the latest execution results for each task
        task_results = {}
        for task_id in task_ids:
            results = self.get_task_execution_results(task_id)
            if results:
                # Sort by start time (descending) and take the latest
                latest_result = sorted(results, key=lambda r: r.start_time, reverse=True)[0]
                task_results[task_id] = latest_result
        
        if not task_results:
            return {"message": "No execution results found for the specified tasks"}
        
        # Prepare prompt for report generation
        template_name = "progress_report"
        prompt_context = {
            "task_results": {
                task_id: {
                    "status": result.status,
                    "success": result.success,
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "duration": result.get_duration(),
                    "deviation_type": result.deviation_type.value if result.deviation_type else None,
                    "deviation_details": result.deviation_details
                } for task_id, result in task_results.items()
            }
        }
        
        # Get the prompt
        prompt = await self.prompt_manager.get_prompt(
            template_name=template_name,
            strategy=PromptStrategy.STRUCTURED_OUTPUT,
            context=prompt_context
        )
        
        # Generate with LLM
        provider = provider or self.default_provider
        model = model or self.default_model
        
        response = await self.llm_connector.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            json_response=True,
            temperature=0.3
        )
        
        # Parse response
        try:
            report = response.get('json', {})
            logger.info(f"Generated progress report for {len(task_ids)} tasks")
            return report
            
        except Exception as e:
            logger.error(f"Error generating progress report: {e}")
            raise ValueError(f"Failed to generate progress report: {e}")
    
    def track_execution_metrics(self, task_id: str) -> Dict[str, Any]:
        """
        Calculate execution metrics for a task across multiple executions.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Metrics for the task
        """
        results = self.get_task_execution_results(task_id)
        
        if not results:
            return {"message": f"No execution results found for task {task_id}"}
        
        # Calculate metrics
        total_executions = len(results)
        successful_executions = sum(1 for r in results if r.success)
        failed_executions = total_executions - successful_executions
        
        # Calculate success rate
        success_rate = (successful_executions / total_executions) * 100 if total_executions > 0 else 0
        
        # Calculate average duration for successful executions
        durations = [r.get_duration() for r in results if r.success and r.get_duration() is not None]
        avg_duration = sum(durations) / len(durations) if durations else None
        
        # Count deviation types
        deviation_counts = {}
        for result in results:
            if result.deviation_type:
                deviation_type = result.deviation_type.value
                deviation_counts[deviation_type] = deviation_counts.get(deviation_type, 0) + 1
        
        metrics = {
            "task_id": task_id,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "deviation_counts": deviation_counts,
            "first_execution": min(r.start_time for r in results),
            "last_execution": max(r.start_time for r in results)
        }
        
        logger.info(f"Calculated execution metrics for task {task_id}")
        return metrics 