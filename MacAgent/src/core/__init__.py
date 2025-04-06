"""
Core package for MacAgent.

This package contains the core components of the MacAgent system:
- Agent: Main agent loop and coordination logic
- Perception: Screen capture and analysis
- Action: Mouse and keyboard interactions
- Planning: Instruction processing and action planning
- Memory: State storage and retrieval
- SystemIntegrator: Connects vision, interaction, and intelligence components
- PipelineManager: Coordinates the perception-planning-action pipeline
- TestingFramework: Implements automated integration tests
"""

from MacAgent.src.core.agent import AgentLoop, AgentConfig
from MacAgent.src.core.perception import PerceptionModule
from MacAgent.src.core.action import ActionModule
from MacAgent.src.core.planning import PlanningModule, PlanStatus, Plan, PlanStep
from MacAgent.src.core.memory import MemorySystem, UIState, AgentAction
from MacAgent.src.core.system_integrator import SystemIntegrator, SystemConfig, ComponentConfig
from MacAgent.src.core.pipeline_manager import PipelineManager, StageMetrics, PipelineMetrics
from MacAgent.src.core.testing_framework import TestingFramework, TestCase, TestSuite, TestReport, UserSimulator

__all__ = [
    'AgentLoop',
    'AgentConfig',
    'PerceptionModule',
    'ActionModule',
    'PlanningModule',
    'PlanStatus',
    'Plan',
    'PlanStep',
    'MemorySystem',
    'UIState',
    'AgentAction',
    'SystemIntegrator',
    'SystemConfig',
    'ComponentConfig',
    'PipelineManager',
    'StageMetrics',
    'PipelineMetrics',
    'TestingFramework',
    'TestCase',
    'TestSuite',
    'TestReport',
    'UserSimulator',
]
