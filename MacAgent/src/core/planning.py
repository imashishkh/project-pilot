"""
Planning module for instruction processing and action planning.

This module handles the planning and decision-making capabilities of the agent,
including processing instructions, generating plans, and deciding what actions to take.
"""

import logging
import time
import asyncio
import uuid
import os
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, field

# Import InstructionProcessor for advanced NLP
try:
    from MacAgent.src.intelligence.instruction_processor import InstructionProcessor, Instruction, TaskBreakdown
    from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider, ModelConfig
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    INTELLIGENCE_AVAILABLE = False
    # Create dummy classes for type hints
    class InstructionProcessor:
        pass
    class Instruction:
        pass
    class TaskBreakdown:
        pass

# Set up logging
logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Status of a plan or step."""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class PlanStep:
    """Represents a single step in a plan."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    action_type: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    status: PlanStatus = PlanStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def mark_started(self) -> None:
        """Mark this step as started."""
        self.status = PlanStatus.IN_PROGRESS
        self.start_time = time.time()
        logger.debug(f"Started step {self.step_id}: {self.description}")
    
    def mark_completed(self, result: Any = None) -> None:
        """Mark this step as completed."""
        self.status = PlanStatus.COMPLETED
        self.result = result
        self.end_time = time.time()
        duration = self.end_time - (self.start_time or self.end_time)
        logger.debug(f"Completed step {self.step_id} in {duration:.2f}s: {self.description}")
    
    def mark_failed(self, error: str) -> None:
        """Mark this step as failed."""
        self.status = PlanStatus.FAILED
        self.error = error
        self.end_time = time.time()
        duration = self.end_time - (self.start_time or self.end_time)
        logger.error(f"Failed step {self.step_id} after {duration:.2f}s: {self.description} - {error}")

    def is_complete(self) -> bool:
        """Check if this step is complete."""
        return self.status in [PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED]


@dataclass
class Plan:
    """Represents a complete plan with multiple steps."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING
    current_step_index: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def add_step(self, step: PlanStep) -> None:
        """Add a step to this plan."""
        self.steps.append(step)
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_current_step(self) -> Optional[PlanStep]:
        """Get the current active step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def mark_started(self) -> None:
        """Mark this plan as started."""
        self.status = PlanStatus.IN_PROGRESS
        self.start_time = time.time()
        logger.info(f"Started plan {self.plan_id} for instruction: {self.instruction}")
    
    def update_status(self) -> None:
        """Update this plan's status based on its steps."""
        if self.status == PlanStatus.CANCELLED:
            return
        
        if not self.steps:
            self.status = PlanStatus.COMPLETED
            self.end_time = time.time()
            return
        
        # Check if all steps are completed
        all_completed = all(step.status == PlanStatus.COMPLETED for step in self.steps)
        any_failed = any(step.status == PlanStatus.FAILED for step in self.steps)
        
        if all_completed:
            self.status = PlanStatus.COMPLETED
            self.end_time = time.time()
            duration = self.end_time - (self.start_time or self.end_time)
            logger.info(f"Completed plan {self.plan_id} in {duration:.2f}s")
        elif any_failed:
            self.status = PlanStatus.FAILED
            self.end_time = time.time()
            duration = self.end_time - (self.start_time or self.end_time)
            logger.error(f"Plan {self.plan_id} failed after {duration:.2f}s")
    
    def next_step(self) -> Optional[PlanStep]:
        """Move to the next step and return it."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return self.get_current_step()
        return None
    
    def cancel(self) -> None:
        """Cancel this plan."""
        self.status = PlanStatus.CANCELLED
        self.end_time = time.time()
        duration = self.end_time - (self.start_time or self.end_time)
        logger.info(f"Cancelled plan {self.plan_id} after {duration:.2f}s")
        
        # Mark any pending steps as cancelled
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                step.status = PlanStatus.CANCELLED


class PlanningModule:
    """
    Planning module that handles planning and decision-making for the agent.
    
    This module is responsible for:
    - Processing instructions and generating plans
    - Deciding what actions to take
    - Adapting plans based on observations
    """
    
    def __init__(self):
        """Initialize the planning module."""
        self.action_handlers: Dict[str, Callable] = {}
        self.current_plan: Optional[Plan] = None
        
        # Initialize instruction processor if intelligence module is available
        self.instruction_processor = None
        self.initialize_instruction_processor()
        
        logger.info("Initialized PlanningModule")
        
    def initialize_instruction_processor(self):
        """Initialize the instruction processor if intelligence module is available."""
        if INTELLIGENCE_AVAILABLE:
            try:
                # Load API config from file
                config_path = "config/api_keys.json"
                if not os.path.exists(config_path):
                    # Try the absolute path
                    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config/api_keys.json")
                
                llm_connector = LLMConnector(config_path=config_path)
                self.instruction_processor = InstructionProcessor(llm_connector)
                logger.info("Initialized InstructionProcessor for advanced NLP capabilities")
            except Exception as e:
                logger.warning(f"Failed to initialize InstructionProcessor: {e}")
                self.instruction_processor = None
        else:
            logger.info("Intelligence module not available, using basic instruction processing")
    
    def register_action_handler(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type."""
        self.action_handlers[action_type] = handler
        logger.debug(f"Registered handler for action type: {action_type}")
    
    async def create_plan_from_instruction(self, instruction: str) -> Plan:
        """
        Create a plan from an instruction.
        
        Args:
            instruction: The instruction to create a plan for.
            
        Returns:
            A plan with steps to execute the instruction.
        """
        plan = Plan(instruction=instruction)
        
        # Use advanced NLP if available
        if self.instruction_processor:
            try:
                logger.info("Using InstructionProcessor for advanced instruction parsing")
                # Parse the instruction
                parsed_instruction = await self.instruction_processor.parse_instruction(instruction)
                
                # Break down into steps if complex
                task_breakdown = await self.instruction_processor.break_down_complex_task(parsed_instruction)
                
                if task_breakdown and task_breakdown.steps:
                    # Convert task steps to plan steps
                    for task_step in task_breakdown.steps:
                        # Extract actionable parameters
                        action_type, params = self._convert_instruction_to_action(task_step.instruction)
                        
                        plan.add_step(PlanStep(
                            description=task_step.description,
                            action_type=action_type,
                            params=params
                        ))
                    
                    logger.info(f"Created plan with {len(plan.steps)} steps using advanced NLP")
                    return plan
                else:
                    logger.warning("Advanced processing failed to break down task, falling back to basic parsing")
            except Exception as e:
                logger.error(f"Error in advanced instruction processing: {e}")
                logger.info("Falling back to basic instruction parsing")
        
        # Fall back to the basic parsing if advanced parsing fails or is unavailable
        # Process the instruction to create a sequence of executable steps
        instruction_lower = instruction.lower()
        
        # Handle screenshot instructions
        if "screenshot" in instruction_lower or "take a picture" in instruction_lower:
            plan.add_step(PlanStep(
                description="Take a screenshot",
                action_type="take_screenshot",
                params={}
            ))
        
        # Handle opening applications
        elif "open" in instruction_lower and ("app" in instruction_lower or "application" in instruction_lower):
            # Extract application name
            app_name = None
            common_apps = {
                "finder": "Finder", 
                "safari": "Safari", 
                "chrome": "Google Chrome",
                "terminal": "Terminal",
                "mail": "Mail",
                "messages": "Messages",
                "calendar": "Calendar",
                "notes": "Notes",
                "photos": "Photos",
                "music": "Music",
                "system preferences": "System Preferences",
                "settings": "System Settings"
            }
            
            for app_keyword, app_full_name in common_apps.items():
                if app_keyword in instruction_lower:
                    app_name = app_full_name
                    break
            
            if app_name:
                plan.add_step(PlanStep(
                    description=f"Open {app_name}",
                    action_type="press_key",
                    params={"key_combination": ["command", "space"]}
                ))
                plan.add_step(PlanStep(
                    description=f"Type application name",
                    action_type="type_text",
                    params={"text": app_name}
                ))
                plan.add_step(PlanStep(
                    description="Press Enter to launch",
                    action_type="press_key",
                    params={"key": "return"}
                ))
            else:
                # Generic application opening
                plan.add_step(PlanStep(
                    description="Open Spotlight Search",
                    action_type="press_key",
                    params={"key_combination": ["command", "space"]}
                ))
                plan.add_step(PlanStep(
                    description="Wait for Spotlight to appear",
                    action_type="wait",
                    params={"seconds": 0.5}
                ))
                plan.add_step(PlanStep(
                    description="Type application name from instruction",
                    action_type="type_text",
                    params={"text": instruction.split("open")[1].split("app")[0].strip()}
                ))
                plan.add_step(PlanStep(
                    description="Press Enter to launch",
                    action_type="press_key",
                    params={"key": "return"}
                ))
        
        # Handle mouse click instructions
        elif "click" in instruction_lower:
            # Try to parse x,y coordinates if provided
            import re
            coords = re.findall(r'(\d+)[,\s]+(\d+)', instruction_lower)
            
            if coords and len(coords[0]) == 2:
                x, y = int(coords[0][0]), int(coords[0][1])
                plan.add_step(PlanStep(
                    description=f"Click at coordinates ({x}, {y})",
                    action_type="click_at",
                    params={"x": x, "y": y}
                ))
            elif "on" in instruction_lower or "at" in instruction_lower:
                # Try to find a named element to click on
                element_text = None
                if "on" in instruction_lower:
                    element_text = instruction_lower.split("on")[1].strip().split()[0]
                elif "at" in instruction_lower:
                    element_text = instruction_lower.split("at")[1].strip().split()[0]
                
                plan.add_step(PlanStep(
                    description=f"Find and click on element with text: {element_text}",
                    action_type="click_ui_element",
                    params={"text": element_text}
                ))
            else:
                # Default to center of screen if no specific target
                plan.add_step(PlanStep(
                    description="Click at current mouse position",
                    action_type="click",
                    params={}
                ))
        
        # Handle typing instructions
        elif "type" in instruction_lower or "write" in instruction_lower or "enter" in instruction_lower:
            text_to_type = ""
            
            # Extract text between quotes if present
            import re
            quoted_text = re.findall(r'"([^"]*)"', instruction)
            if quoted_text:
                text_to_type = quoted_text[0]
            else:
                # Try to extract text after "type" or "write" or "enter"
                for keyword in ["type", "write", "enter"]:
                    if keyword in instruction_lower:
                        parts = instruction_lower.split(keyword, 1)
                        if len(parts) > 1:
                            text_to_type = parts[1].strip()
                            break
            
            if text_to_type:
                plan.add_step(PlanStep(
                    description=f"Type text: {text_to_type}",
                    action_type="type_text",
                    params={"text": text_to_type}
                ))
            else:
                plan.add_step(PlanStep(
                    description="Type text from instruction",
                    action_type="type_text",
                    params={"text": instruction.split("type")[1].strip() if "type" in instruction_lower else 
                            instruction.split("write")[1].strip() if "write" in instruction_lower else
                            instruction.split("enter")[1].strip()}
                ))
        
        # Handle navigation to folder/directory
        elif any(keyword in instruction_lower for keyword in ["navigate to", "go to", "open folder", "open directory"]):
            folder_path = None
            common_locations = {
                "downloads": "~/Downloads",
                "documents": "~/Documents",
                "desktop": "~/Desktop",
                "pictures": "~/Pictures",
                "music": "~/Music",
                "movies": "~/Movies",
                "applications": "/Applications",
                "home": "~"
            }
            
            for location_name, path in common_locations.items():
                if location_name in instruction_lower:
                    folder_path = path
                    break
            
            if folder_path:
                # First make sure Finder is open
                plan.add_step(PlanStep(
                    description="Open Finder",
                    action_type="press_key",
                    params={"key_combination": ["command", "space"]}
                ))
                plan.add_step(PlanStep(
                    description="Type Finder",
                    action_type="type_text",
                    params={"text": "Finder"}
                ))
                plan.add_step(PlanStep(
                    description="Launch Finder",
                    action_type="press_key",
                    params={"key": "return"}
                ))
                plan.add_step(PlanStep(
                    description="Wait for Finder to open",
                    action_type="wait",
                    params={"seconds": 1.0}
                ))
                # Use Go To Folder
                plan.add_step(PlanStep(
                    description="Open Go To Folder dialog",
                    action_type="press_key",
                    params={"key_combination": ["shift", "command", "g"]}
                ))
                plan.add_step(PlanStep(
                    description=f"Type folder path: {folder_path}",
                    action_type="type_text",
                    params={"text": folder_path}
                ))
                plan.add_step(PlanStep(
                    description="Confirm folder path",
                    action_type="press_key",
                    params={"key": "return"}
                ))
        
        # Default fallback for unrecognized instructions
        else:
            plan.add_step(PlanStep(
                description="Process instruction (fallback)",
                action_type="unknown",
                params={"instruction": instruction}
            ))
            logger.warning(f"Using fallback for unrecognized instruction: {instruction}")
        
        logger.info(f"Created plan with {len(plan.steps)} steps for instruction: {instruction}")
        return plan
    
    def _convert_instruction_to_action(self, instruction: Instruction) -> Tuple[str, Dict[str, Any]]:
        """
        Convert a parsed instruction to an action type and parameters.
        
        Args:
            instruction: Parsed instruction
            
        Returns:
            Tuple of action type and parameters dictionary
        """
        # Default values
        action_type = "unknown"
        params = {"instruction": instruction.raw_text}
        
        try:
            # Map instruction intents to action types
            intent_to_action = {
                "navigation": "press_key",  # For opening apps and navigating
                "search": "type_text",      # For searching
                "create": "unknown",        # Complex, handled case by case
                "edit": "unknown",          # Complex, handled case by case
                "delete": "unknown",        # Complex, handled case by case
                "execute": "unknown",       # Complex, handled case by case
                "query": "unknown",         # Handle differently 
                "confirm": "press_key",     # Press Enter/Return
                "cancel": "press_key",      # Press Escape
                "unknown": "unknown"
            }
            
            # Get action type from intent
            intent_str = instruction.intent.value if hasattr(instruction.intent, 'value') else str(instruction.intent)
            action_type = intent_to_action.get(intent_str.lower(), "unknown")
            
            # Extract parameters from instruction
            params = {}
            for param_name, param in instruction.parameters.items():
                params[param_name] = param.value
                
            # Special case handling
            if "click" in instruction.raw_text.lower():
                if "coordinates" in params:
                    try:
                        coords = params["coordinates"].split(",")
                        x, y = int(coords[0]), int(coords[1])
                        action_type = "click_at"
                        params = {"x": x, "y": y}
                    except:
                        action_type = "click"
                        params = {}
                elif "element" in params or "text" in params:
                    action_type = "click_ui_element"
                    params = {"text": params.get("element", params.get("text", ""))}
                else:
                    action_type = "click"
                    params = {}
                    
            elif "type" in instruction.raw_text.lower() and "text" in params:
                action_type = "type_text"
                params = {"text": params["text"]}
                
            elif "screenshot" in instruction.raw_text.lower():
                action_type = "take_screenshot"
                params = {}
                
        except Exception as e:
            logger.error(f"Error converting instruction to action: {e}")
            
        return action_type, params
    
    async def execute_step(self, step: PlanStep) -> None:
        """
        Execute a single step in a plan.
        
        Args:
            step: The step to execute.
        """
        if step.status != PlanStatus.PENDING:
            logger.warning(f"Attempting to execute step {step.step_id} with status {step.status}")
            return
        
        step.mark_started()
        
        try:
            handler = self.action_handlers.get(step.action_type)
            if handler:
                result = await handler(**step.params)
                step.mark_completed(result)
            else:
                step.mark_failed(f"No handler registered for action type: {step.action_type}")
        except Exception as e:
            step.mark_failed(f"Error executing step: {str(e)}")
    
    async def execute_plan(self, plan: Plan) -> None:
        """
        Execute a plan step by step.
        
        Args:
            plan: The plan to execute.
        """
        self.current_plan = plan
        plan.mark_started()
        
        current_step = plan.get_current_step()
        while current_step and plan.status == PlanStatus.IN_PROGRESS:
            await self.execute_step(current_step)
            plan.update_status()
            
            if plan.status != PlanStatus.IN_PROGRESS:
                break
                
            current_step = plan.next_step()
            
            # Small delay between steps for stability
            await asyncio.sleep(0.1)
    
    async def adapt_plan_based_on_observation(self, plan: Plan, observation: Any) -> Plan:
        """
        Adapt a plan based on an observation.
        
        Args:
            plan: The current plan.
            observation: The observation to adapt to.
            
        Returns:
            The adapted plan.
        """
        # This is a placeholder for a more sophisticated adaptation mechanism
        # In a real implementation, this would involve analyzing the observation
        # and modifying the plan accordingly
        
        logger.info(f"Adapting plan based on observation (placeholder implementation)")
        return plan
    
    def cancel_current_plan(self) -> None:
        """Cancel the current plan."""
        if self.current_plan:
            self.current_plan.cancel()
            logger.info(f"Cancelled current plan: {self.current_plan.plan_id}")
        else:
            logger.warning("No current plan to cancel")
