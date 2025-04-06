"""
Intelligence module for the MacAgent system.

This module provides the AI intelligence components for the MacAgent system,
including LLM integration, prompt management, task planning, decision making,
and execution monitoring.
"""

# Re-export key classes and constants
from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig
from MacAgent.src.intelligence.prompt_manager import PromptManager, PromptStrategy
from MacAgent.src.intelligence.task_planner import TaskPlanner, Task, TaskPlan, PlanningStrategy
from MacAgent.src.intelligence.decision_maker import DecisionMaker, Decision, Option, DecisionCriteria
from MacAgent.src.intelligence.execution_monitor import ExecutionMonitor, ExecutionResult, ExpectedOutcome, DeviationType

__all__ = [
    'LLMConnector',
    'LLMProvider',
    'ModelConfig',
    'PromptManager',
    'PromptStrategy',
    'TaskPlanner',
    'Task',
    'TaskPlan',
    'PlanningStrategy',
    'DecisionMaker',
    'Decision',
    'Option',
    'DecisionCriteria',
    'ExecutionMonitor',
    'ExecutionResult',
    'ExpectedOutcome',
    'DeviationType'
]
