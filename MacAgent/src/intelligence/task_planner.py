"""
Task Planning Module for MacAgent.

This module provides task planning capabilities for the MacAgent, including:
- Creating structured task plans from high-level goals
- Managing task dependencies and priorities
- Supporting different planning strategies (sequential, hierarchical)
- Adapting plans based on new information or constraints
"""

import os
import json
import uuid
import enum
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict

from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider
from MacAgent.src.intelligence.prompt_manager import PromptManager, PromptStrategy

# Configure logging
logger = logging.getLogger(__name__)


class PlanningStrategy(enum.Enum):
    """Enumeration of available planning strategies."""
    SEQUENTIAL = "sequential"  # Tasks executed in strict sequence
    HIERARCHICAL = "hierarchical"  # Tasks organized in hierarchical structure
    PARALLEL = "parallel"  # Maximize parallel task execution


@dataclass
class Task:
    """Represents a single task in a task plan."""
    id: str  # Unique identifier for the task
    name: str  # Short name for the task
    description: str  # Detailed description of what the task entails
    estimated_duration: Optional[float] = None  # Estimated duration in minutes
    dependencies: List[str] = field(default_factory=list)  # IDs of tasks this task depends on
    priority: int = 1  # Priority level (1-5, with 5 being highest)
    status: str = "pending"  # Status of the task (pending, in_progress, completed, failed)
    parent_id: Optional[str] = None  # Parent task ID if hierarchical
    subtasks: List[str] = field(default_factory=list)  # IDs of child tasks
    tags: List[str] = field(default_factory=list)  # Tags for categorizing tasks
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(**data)


@dataclass
class TaskPlan:
    """Represents a complete task plan."""
    id: str  # Unique identifier for the plan
    name: str  # Name of the plan
    description: str  # Description of what the plan achieves
    goal: str  # High-level goal the plan aims to accomplish
    tasks: Dict[str, Task] = field(default_factory=dict)  # Map of task ID to Task
    strategy: PlanningStrategy = PlanningStrategy.SEQUENTIAL
    created_at: float = field(default_factory=time.time)  # Creation timestamp
    updated_at: float = field(default_factory=time.time)  # Last update timestamp
    context: Dict[str, Any] = field(default_factory=dict)  # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "strategy": self.strategy.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "context": self.context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskPlan':
        """Create plan from dictionary."""
        # Convert strategy string to enum
        if isinstance(data.get("strategy"), str):
            data["strategy"] = PlanningStrategy(data["strategy"])
        
        # Convert tasks dict to Task objects
        tasks_dict = data.pop("tasks", {})
        plan = cls(**data)
        plan.tasks = {task_id: Task.from_dict(task_data) 
                      for task_id, task_data in tasks_dict.items()}
        return plan
    
    def get_next_tasks(self) -> List[Task]:
        """Get tasks that are ready to be executed (all dependencies completed)."""
        ready_tasks = []
        completed_task_ids = {
            task_id for task_id, task in self.tasks.items() 
            if task.status == "completed"
        }
        
        for task_id, task in self.tasks.items():
            if task.status == "pending":
                # Check if all dependencies are completed
                if all(dep_id in completed_task_ids for dep_id in task.dependencies):
                    ready_tasks.append(task)
        
        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks
    
    def get_task_depth(self, task_id: str) -> int:
        """Get the depth of a task in the hierarchy."""
        task = self.tasks.get(task_id)
        if not task:
            return 0
        
        depth = 0
        current_id = task.parent_id
        while current_id:
            depth += 1
            parent = self.tasks.get(current_id)
            if not parent:
                break
            current_id = parent.parent_id
        
        return depth
    
    def is_completed(self) -> bool:
        """Check if the plan is completed."""
        return all(task.status == "completed" for task in self.tasks.values())
    
    def get_completion_percentage(self) -> float:
        """Get the completion percentage of the plan."""
        if not self.tasks:
            return 0.0
        
        completed = sum(1 for task in self.tasks.values() if task.status == "completed")
        return (completed / len(self.tasks)) * 100.0


class TaskPlanner:
    """
    Task Planner for MacAgent system.
    
    Generates, manages, and adapts task plans for complex user requests.
    """
    
    def __init__(
        self,
        llm_connector: LLMConnector,
        prompt_manager: PromptManager,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        default_model: str = "gpt-3.5-turbo",
        plans_dir: str = "data/plans"
    ):
        """
        Initialize the TaskPlanner.
        
        Args:
            llm_connector: The LLM connector for generating plans
            prompt_manager: The prompt manager for generating prompts
            default_provider: Default LLM provider to use
            default_model: Default model to use
            plans_dir: Directory to store plan files
        """
        self.llm_connector = llm_connector
        self.prompt_manager = prompt_manager
        self.default_provider = default_provider
        self.default_model = default_model
        self.plans_dir = plans_dir
        
        # Ensure plans directory exists
        os.makedirs(self.plans_dir, exist_ok=True)
        
        # Load existing plans
        self.plans: Dict[str, TaskPlan] = self._load_plans()
        
        logger.info(f"TaskPlanner initialized with {len(self.plans)} existing plans")
    
    def _load_plans(self) -> Dict[str, TaskPlan]:
        """Load existing plans from disk."""
        plans = {}
        plan_files = [f for f in os.listdir(self.plans_dir) 
                     if f.endswith('.json') and os.path.isfile(os.path.join(self.plans_dir, f))]
        
        for file_name in plan_files:
            file_path = os.path.join(self.plans_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    plan_data = json.load(f)
                    plan = TaskPlan.from_dict(plan_data)
                    plans[plan.id] = plan
            except Exception as e:
                logger.error(f"Error loading plan from {file_path}: {e}")
        
        return plans
    
    def _save_plan(self, plan: TaskPlan) -> None:
        """Save plan to disk."""
        file_path = os.path.join(self.plans_dir, f"{plan.id}.json")
        with open(file_path, 'w') as f:
            json.dump(plan.to_dict(), f, indent=2)
    
    async def _generate_plan_with_llm(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: PlanningStrategy = PlanningStrategy.SEQUENTIAL,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a plan using LLM."""
        provider = provider or self.default_provider
        model = model or self.default_model
        context = context or {}
        
        # Prepare prompt
        template_name = "task_planning"
        prompt_context = {
            "goal": goal,
            "context": json.dumps(context, indent=2),
            "strategy": strategy.value,
            "current_time": datetime.now().isoformat(),
            "output_format_description": """
            {
                "name": "Plan name",
                "description": "Plan description",
                "tasks": [
                    {
                        "name": "Task 1 name",
                        "description": "Detailed description of task 1",
                        "estimated_duration": 15,  # minutes
                        "dependencies": [],  # task names this depends on
                        "priority": 3  # 1-5 scale
                    },
                    ...
                ]
            }
            """
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
            temperature=0.2
        )
        
        # Parse response
        try:
            plan_data = response.get('json', {})
            return plan_data
        except Exception as e:
            logger.error(f"Error parsing plan from LLM response: {e}")
            raise ValueError(f"Failed to generate valid plan: {e}")
    
    async def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: PlanningStrategy = PlanningStrategy.SEQUENTIAL,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> TaskPlan:
        """
        Create a new task plan based on a goal.
        
        Args:
            goal: High-level goal to accomplish
            context: Additional context for the plan
            strategy: Planning strategy to use
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            A new TaskPlan object
        """
        # Generate plan data with LLM
        plan_data = await self._generate_plan_with_llm(
            goal=goal,
            context=context,
            strategy=strategy,
            provider=provider,
            model=model
        )
        
        # Create plan ID and tasks
        plan_id = str(uuid.uuid4())
        
        # Create tasks with IDs
        tasks = {}
        task_name_to_id = {}
        
        # First pass: create tasks with IDs
        for i, task_data in enumerate(plan_data.get("tasks", [])):
            task_id = f"{plan_id}_task_{i}"
            task_name_to_id[task_data["name"]] = task_id
            
            # Create task (without dependencies for now)
            task = Task(
                id=task_id,
                name=task_data["name"],
                description=task_data.get("description", ""),
                estimated_duration=task_data.get("estimated_duration"),
                priority=task_data.get("priority", 1),
                tags=task_data.get("tags", []),
                metadata=task_data.get("metadata", {})
            )
            tasks[task_id] = task
        
        # Second pass: set up dependencies
        for i, task_data in enumerate(plan_data.get("tasks", [])):
            task_id = f"{plan_id}_task_{i}"
            task = tasks[task_id]
            
            # Map dependency names to IDs
            dependencies = []
            for dep_name in task_data.get("dependencies", []):
                if dep_name in task_name_to_id:
                    dependencies.append(task_name_to_id[dep_name])
            
            task.dependencies = dependencies
            
            # For hierarchical plans, set up parent/child relationships
            if strategy == PlanningStrategy.HIERARCHICAL and "parent" in task_data:
                parent_name = task_data["parent"]
                if parent_name in task_name_to_id:
                    parent_id = task_name_to_id[parent_name]
                    task.parent_id = parent_id
                    
                    # Add this task as a subtask of parent
                    parent_task = tasks[parent_id]
                    parent_task.subtasks.append(task_id)
        
        # Create the plan
        plan = TaskPlan(
            id=plan_id,
            name=plan_data.get("name", f"Plan for: {goal[:50]}"),
            description=plan_data.get("description", ""),
            goal=goal,
            tasks=tasks,
            strategy=strategy,
            context=context or {},
            metadata=plan_data.get("metadata", {})
        )
        
        # Save the plan
        self.plans[plan_id] = plan
        self._save_plan(plan)
        
        logger.info(f"Created plan {plan_id} with {len(tasks)} tasks")
        return plan
    
    async def update_plan(
        self,
        plan_id: str,
        context: Optional[Dict[str, Any]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> TaskPlan:
        """
        Update a plan based on new context or feedback.
        
        Args:
            plan_id: ID of the plan to update
            context: New context information
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Updated TaskPlan
        """
        if plan_id not in self.plans:
            raise ValueError(f"Plan with ID {plan_id} not found")
        
        # Get existing plan
        existing_plan = self.plans[plan_id]
        
        # Merge contexts
        updated_context = {**existing_plan.context}
        if context:
            updated_context.update(context)
        
        # Generate updated plan data
        provider = provider or self.default_provider
        model = model or self.default_model
        
        # Prepare prompt for plan update
        template_name = "task_planning_update"
        prompt_context = {
            "goal": existing_plan.goal,
            "existing_plan": json.dumps(existing_plan.to_dict(), indent=2),
            "new_context": json.dumps(context, indent=2) if context else "{}",
            "current_time": datetime.now().isoformat()
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
            updated_plan_data = response.get('json', {})
            
            # Update existing plan
            existing_plan.updated_at = time.time()
            existing_plan.context = updated_context
            
            if "name" in updated_plan_data:
                existing_plan.name = updated_plan_data["name"]
            
            if "description" in updated_plan_data:
                existing_plan.description = updated_plan_data["description"]
            
            # Process tasks
            if "tasks" in updated_plan_data:
                # Existing task map
                existing_task_ids = set(existing_plan.tasks.keys())
                
                # Process updated tasks
                task_name_to_id = {task.name: task_id for task_id, task in existing_plan.tasks.items()}
                
                # Track which tasks to remove
                tasks_to_keep = set()
                
                # First pass: update existing tasks and create new ones
                for i, task_data in enumerate(updated_plan_data["tasks"]):
                    task_name = task_data["name"]
                    
                    # Check if this is an existing task
                    if task_name in task_name_to_id:
                        # Update existing task
                        task_id = task_name_to_id[task_name]
                        task = existing_plan.tasks[task_id]
                        
                        # Update fields
                        task.description = task_data.get("description", task.description)
                        task.estimated_duration = task_data.get("estimated_duration", task.estimated_duration)
                        task.priority = task_data.get("priority", task.priority)
                        task.tags = task_data.get("tags", task.tags)
                        task.metadata.update(task_data.get("metadata", {}))
                        
                        # Track this task
                        tasks_to_keep.add(task_id)
                    else:
                        # Create new task
                        task_id = f"{plan_id}_task_{len(existing_plan.tasks)}"
                        task_name_to_id[task_name] = task_id
                        
                        task = Task(
                            id=task_id,
                            name=task_name,
                            description=task_data.get("description", ""),
                            estimated_duration=task_data.get("estimated_duration"),
                            priority=task_data.get("priority", 1),
                            tags=task_data.get("tags", []),
                            metadata=task_data.get("metadata", {})
                        )
                        existing_plan.tasks[task_id] = task
                        tasks_to_keep.add(task_id)
                
                # Second pass: update dependencies
                for task_data in updated_plan_data["tasks"]:
                    task_name = task_data["name"]
                    task_id = task_name_to_id[task_name]
                    task = existing_plan.tasks[task_id]
                    
                    # Update dependencies
                    dependencies = []
                    for dep_name in task_data.get("dependencies", []):
                        if dep_name in task_name_to_id:
                            dependencies.append(task_name_to_id[dep_name])
                    
                    task.dependencies = dependencies
                    
                    # For hierarchical plans, update parent/child relationships
                    if existing_plan.strategy == PlanningStrategy.HIERARCHICAL and "parent" in task_data:
                        parent_name = task_data["parent"]
                        if parent_name in task_name_to_id:
                            parent_id = task_name_to_id[parent_name]
                            task.parent_id = parent_id
                
                # Remove tasks that are no longer needed
                tasks_to_remove = existing_task_ids - tasks_to_keep
                for task_id in tasks_to_remove:
                    if existing_plan.tasks[task_id].status != "completed":
                        # Only remove non-completed tasks
                        del existing_plan.tasks[task_id]
            
            # Save the updated plan
            self._save_plan(existing_plan)
            
            logger.info(f"Updated plan {plan_id} with {len(existing_plan.tasks)} tasks")
            return existing_plan
            
        except Exception as e:
            logger.error(f"Error updating plan: {e}")
            raise ValueError(f"Failed to update plan: {e}")
    
    def get_plan(self, plan_id: str) -> TaskPlan:
        """
        Get a plan by ID.
        
        Args:
            plan_id: ID of the plan to retrieve
            
        Returns:
            TaskPlan object
        """
        if plan_id not in self.plans:
            raise ValueError(f"Plan with ID {plan_id} not found")
        
        return self.plans[plan_id]
    
    def get_all_plans(self) -> List[TaskPlan]:
        """
        Get all plans.
        
        Returns:
            List of all TaskPlan objects
        """
        return list(self.plans.values())
    
    def delete_plan(self, plan_id: str) -> None:
        """
        Delete a plan.
        
        Args:
            plan_id: ID of the plan to delete
        """
        if plan_id not in self.plans:
            raise ValueError(f"Plan with ID {plan_id} not found")
        
        # Remove from memory
        del self.plans[plan_id]
        
        # Remove from disk
        file_path = os.path.join(self.plans_dir, f"{plan_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Deleted plan {plan_id}")
    
    def update_task_status(self, plan_id: str, task_id: str, status: str) -> Task:
        """
        Update the status of a task in a plan.
        
        Args:
            plan_id: ID of the plan
            task_id: ID of the task
            status: New status for the task
            
        Returns:
            Updated Task object
        """
        if plan_id not in self.plans:
            raise ValueError(f"Plan with ID {plan_id} not found")
        
        plan = self.plans[plan_id]
        if task_id not in plan.tasks:
            raise ValueError(f"Task with ID {task_id} not found in plan {plan_id}")
        
        # Update status
        task = plan.tasks[task_id]
        task.status = status
        
        # Update plan
        plan.updated_at = time.time()
        
        # Save the plan
        self._save_plan(plan)
        
        logger.info(f"Updated task {task_id} status to {status} in plan {plan_id}")
        return task
    
    async def generate_action_plan(
        self, 
        plan_id: str, 
        task_id: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a detailed action plan for a specific task.
        
        Args:
            plan_id: ID of the plan
            task_id: ID of the task
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Detailed action plan
        """
        if plan_id not in self.plans:
            raise ValueError(f"Plan with ID {plan_id} not found")
        
        plan = self.plans[plan_id]
        if task_id not in plan.tasks:
            raise ValueError(f"Task with ID {task_id} not found in plan {plan_id}")
        
        task = plan.tasks[task_id]
        
        # Prepare prompt
        template_name = "task_action_plan"
        prompt_context = {
            "plan_goal": plan.goal,
            "task_name": task.name,
            "task_description": task.description,
            "plan_context": json.dumps(plan.context, indent=2),
            "current_time": datetime.now().isoformat()
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
            action_plan = response.get('json', {})
            
            # Store action plan in task metadata
            task.metadata["action_plan"] = action_plan
            
            # Update plan
            plan.updated_at = time.time()
            
            # Save the plan
            self._save_plan(plan)
            
            logger.info(f"Generated action plan for task {task_id} in plan {plan_id}")
            return action_plan
        
        except Exception as e:
            logger.error(f"Error generating action plan: {e}")
            raise ValueError(f"Failed to generate action plan: {e}")
    
    async def reprioritize_tasks(
        self,
        plan_id: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> TaskPlan:
        """
        Reprioritize tasks in a plan based on current context.
        
        Args:
            plan_id: ID of the plan to reprioritize
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Updated TaskPlan
        """
        if plan_id not in self.plans:
            raise ValueError(f"Plan with ID {plan_id} not found")
        
        plan = self.plans[plan_id]
        
        # Prepare prompt
        template_name = "task_reprioritization"
        prompt_context = {
            "plan": json.dumps(plan.to_dict(), indent=2),
            "current_time": datetime.now().isoformat()
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
            priorities_data = response.get('json', {})
            
            # Update priorities
            for task_id, priority in priorities_data.get("task_priorities", {}).items():
                if task_id in plan.tasks:
                    plan.tasks[task_id].priority = priority
            
            # Update plan
            plan.updated_at = time.time()
            
            # Save the plan
            self._save_plan(plan)
            
            logger.info(f"Reprioritized tasks in plan {plan_id}")
            return plan
        
        except Exception as e:
            logger.error(f"Error reprioritizing tasks: {e}")
            raise ValueError(f"Failed to reprioritize tasks: {e}")
