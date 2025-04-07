"""
Agent Module for MacAgent.

This module contains the main agent loop and coordination logic,
binding together perception, planning, action, and memory components.
"""

import logging
import asyncio
import signal
import time
import traceback
import os
import random
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .perception import PerceptionModule
from .action import ActionModule
from .planning import PlanningModule, Plan, PlanStatus, PlanStep
from .memory import MemorySystem, AgentAction
from .service_registry import ServiceRegistry

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    # Loop timing
    tick_rate: float = 0.1  # Seconds per tick
    
    # Performance monitoring
    monitor_performance: bool = True
    perf_sample_rate: int = 100  # Sample performance every N ticks
    
    # Logging
    log_level: int = logging.INFO
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Perception
    screenshot_interval: float = 0.2  # Seconds between screenshots
    capture_region: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height) or None for full screen
    ui_detection_interval: int = 5  # Run UI detection every N ticks
    
    # Action
    move_speed: float = 0.3  # Mouse movement duration
    click_delay: float = 0.1  # Delay between clicks
    
    # Memory
    memory_dir: Optional[str] = "memory"
    max_actions: int = 100  # Max actions to keep in memory
    
    # Vision-Interaction Connection
    enable_ui_interaction: bool = True  # Enable direct interaction with detected UI elements
    ui_confidence_threshold: float = 0.7  # Confidence threshold for UI element detection
    
    # Intelligence-Action Connection
    plan_refinement_steps: int = 2  # Number of refinement steps for plans
    action_feedback_enabled: bool = True  # Enable feedback from actions to planning
    dynamic_replanning: bool = True  # Enable dynamic replanning on failures
    context_aware_actions: bool = True  # Enable context-aware action execution


@dataclass
class UIElement:
    """Represents a detected UI element on screen."""
    element_id: str
    element_type: str  # button, text_field, checkbox, etc.
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    text: Optional[str] = None
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center coordinates of the element."""
        x, y, w, h = self.bounds
        return (x + w // 2, y + h // 2)


@dataclass
class ActionContext:
    """Context information for action execution."""
    screenshot: Optional[Any] = None
    ui_elements: List[UIElement] = field(default_factory=list)
    current_instruction: Optional[str] = None
    previous_actions: List[Dict[str, Any]] = field(default_factory=list)
    environment_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentLoop:
    """
    Main agent loop that coordinates all components.
    
    This class is responsible for the execution flow of the agent,
    coordinating perception, planning, action, and memory.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the agent loop with components.
        
        Args:
            config: Optional configuration for the agent
        """
        # Use default config if none provided
        self.config = config or AgentConfig()
        
        # Set up logging
        self._configure_logging()
        
        # Initialize service registry
        self._services = ServiceRegistry()
        logger.info("Initializing agent components...")
        
        # Initialize perception module
        perception = PerceptionModule(
            capture_region=self.config.capture_region,
            screenshot_interval=self.config.screenshot_interval
        )
        self._services.register("perception", perception)
        
        # Initialize action module
        action = ActionModule(
            move_speed=self.config.move_speed,
            click_delay=self.config.click_delay
        )
        
        # Enable debug mode for troubleshooting
        action.set_debug_mode(True)
        logger.info("Debug mode enabled for troubleshooting")
        self._services.register("action", action)
        
        # Initialize planning module
        planning = PlanningModule()
        self._services.register("planning", planning)
        
        # Initialize memory system
        memory_dir = self.config.memory_dir
        if memory_dir and not os.path.exists(memory_dir):
            os.makedirs(memory_dir, exist_ok=True)
            
        memory = MemorySystem(
            storage_dir=memory_dir,
            max_actions=self.config.max_actions
        )
        self._services.register("memory", memory)
        
        # Add task manager
        from MacAgent.src.core.task_manager import TaskManager
        task_manager = TaskManager()
        self._services.register("task_manager", task_manager)
        
        # Add feedback manager
        from MacAgent.src.core.feedback_manager import FeedbackManager
        feedback = FeedbackManager(verbose=True)
        self._services.register("feedback", feedback)
        
        # State variables
        self.running: bool = False
        self.paused: bool = False
        self.current_plan: Optional[Plan] = None
        self._shutdown_requested: bool = False
        
        # Tracked UI elements
        self.detected_ui_elements: List[UIElement] = []
        self.last_ui_detection_time: float = 0
        
        # Current action context
        self.current_action_context = ActionContext()
        
        # Action execution history
        self.action_history: List[Dict[str, Any]] = []
        self.last_action_result: Optional[Dict[str, Any]] = None
        
        # Performance monitoring
        self.tick_count: int = 0
        self.perf_stats: Dict[str, List[float]] = {
            'loop_time': [],
            'perception_time': [],
            'planning_time': [],
            'action_time': [],
            'memory_time': [],
            'ui_detection_time': [],
        }
        
        # Register action handlers
        self._register_action_handlers()
        
        # Define critical actions that require retry
        self.critical_actions = {
            'click', 'click_at', 'click_ui_element', 'type_text', 
            'find_ui_element', 'press_key', 'hotkey', 'execute_plan'
        }
        
        logger.info("AgentLoop initialized successfully")
    
    # Property getters for backward compatibility
    @property
    def perception(self) -> PerceptionModule:
        """Get the perception module."""
        return self._services.get("perception")
    
    @property
    def action(self) -> ActionModule:
        """Get the action module."""
        return self._services.get("action")
    
    @property
    def planning(self) -> PlanningModule:
        """Get the planning module."""
        return self._services.get("planning")
    
    @property
    def memory(self) -> MemorySystem:
        """Get the memory system."""
        return self._services.get("memory")
    
    @property
    def task_manager(self):
        """Get the task manager."""
        return self._services.get("task_manager")
    
    @property
    def feedback(self):
        """Get the feedback manager."""
        return self._services.get("feedback")
    
    def _configure_logging(self) -> None:
        """Configure the logging system."""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level)
        
        # Remove existing handlers to avoid duplicates
        while root_logger.handlers:
            root_logger.removeHandler(root_logger.handlers[0])
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Create file handler if enabled
        if self.config.log_to_file:
            log_dir = self.config.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"macagent_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
    
    def _register_action_handlers(self) -> None:
        """Register handlers for different action types."""
        # Register basic actions that connect planning to action module
        self.planning.register_action_handler('move_mouse', self.action.move_to)
        self.planning.register_action_handler('click', self.action.click)
        self.planning.register_action_handler('click_at', self.action.click_at)
        self.planning.register_action_handler('type_text', self.action.type_text)
        self.planning.register_action_handler('press_key', self.action.press_key)
        self.planning.register_action_handler('hotkey', self.action.perform_hotkey)
        self.planning.register_action_handler('drag', self.action.drag_to)
        self.planning.register_action_handler('scroll', self.action.scroll)
        
        # Register screenshot actions
        self.planning.register_action_handler('take_screenshot', self._handle_take_screenshot)
        
        # Register UI interaction actions
        self.planning.register_action_handler('find_ui_element', self._handle_find_ui_element)
        self.planning.register_action_handler('click_ui_element', self._handle_click_ui_element)
        self.planning.register_action_handler('get_ui_elements', self._handle_get_ui_elements)
        
        # Register intelligence-action connection handlers
        self.planning.register_action_handler('execute_plan', self._handle_execute_plan)
        
        # Register utility actions
        self.planning.register_action_handler('wait', self._handle_wait)
        
        # Register compound actions
        self.planning.register_action_handler('unknown', self._handle_unknown_action)
        
        logger.debug("Action handlers registered")
    
    async def _handle_take_screenshot(self, **kwargs) -> Dict[str, Any]:
        """
        Handle a take_screenshot action.
        
        Returns:
            Dictionary with result information
        """
        try:
            logger.info("Taking screenshot")
            screenshot = await self.perception.capture_screen()
            
            # Store in memory
            timestamp = int(time.time())
            screenshot_key = f"screenshot_{timestamp}"
            self.memory.store_observation(screenshot_key, {"timestamp": timestamp, "has_image": True})
            
            return {"success": True, "key": screenshot_key}
        except Exception as e:
            logger.error(f"Error taking screenshot: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_unknown_action(self, instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Handle an unknown action (fallback).
        
        Args:
            instruction: The original instruction
            
        Returns:
            Dictionary with result information
        """
        logger.warning(f"Received unknown action type for instruction: {instruction}")
        await asyncio.sleep(0.5)  # Pause briefly
        return {
            "success": False,
            "message": "Unknown action type",
            "instruction": instruction
        }
    
    async def _handle_find_ui_element(self, element_type: Optional[str] = None, 
                                     text: Optional[str] = None, 
                                     **kwargs) -> Dict[str, Any]:
        """
        Find a UI element by type and/or text.
        
        Args:
            element_type: Optional type of UI element to find
            text: Optional text content to search for
            
        Returns:
            Dictionary with result information
        """
        try:
            # Make sure UI elements are detected
            if not self.detected_ui_elements:
                await self._detect_ui_elements()
                
            if not self.detected_ui_elements:
                return {
                    "success": False,
                    "message": "No UI elements detected"
                }
                
            # Filter elements by criteria
            matching_elements = []
            for element in self.detected_ui_elements:
                # Skip elements with low confidence
                if element.confidence < self.config.ui_confidence_threshold:
                    continue
                    
                # Match element type if specified
                if element_type and element.element_type != element_type:
                    continue
                    
                # Match text if specified
                if text and (not element.text or text.lower() not in element.text.lower()):
                    continue
                    
                matching_elements.append(element)
                
            if not matching_elements:
                return {
                    "success": False,
                    "message": f"No matching UI elements found for type={element_type}, text={text}"
                }
                
            # Return the best match (highest confidence)
            best_match = max(matching_elements, key=lambda e: e.confidence)
            return {
                "success": True,
                "element_id": best_match.element_id,
                "element_type": best_match.element_type,
                "bounds": best_match.bounds,
                "center": best_match.center,
                "text": best_match.text,
                "confidence": best_match.confidence
            }
                
        except Exception as e:
            logger.error(f"Error finding UI element: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_click_ui_element(self, element_id: Optional[str] = None,
                                      element_type: Optional[str] = None,
                                      text: Optional[str] = None,
                                      **kwargs) -> Dict[str, Any]:
        """
        Click on a UI element identified by ID, type, or text.
        
        Args:
            element_id: Optional element ID to click on
            element_type: Optional element type to search for
            text: Optional text content to search for
            
        Returns:
            Dictionary with result information
        """
        try:
            # Find the element first
            if element_id:
                # Find by ID
                element = next((e for e in self.detected_ui_elements if e.element_id == element_id), None)
                if not element:
                    return {
                        "success": False,
                        "message": f"UI element with ID {element_id} not found"
                    }
            else:
                # Find by type/text
                result = await self._handle_find_ui_element(element_type=element_type, text=text)
                if not result["success"]:
                    return result
                    
                element_id = result["element_id"]
                element = next((e for e in self.detected_ui_elements if e.element_id == element_id), None)
                
            if not element:
                return {
                    "success": False,
                    "message": "UI element not found"
                }
                
            # Click on the element's center
            center_x, center_y = element.center
            await self.action.click_at(center_x, center_y)
            
            logger.info(f"Clicked UI element {element.element_id} ({element.element_type}) at {center_x}, {center_y}")
            
            return {
                "success": True,
                "element_id": element.element_id,
                "element_type": element.element_type,
                "position": (center_x, center_y)
            }
                
        except Exception as e:
            logger.error(f"Error clicking UI element: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _handle_get_ui_elements(self, refresh: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Get all detected UI elements.
        
        Args:
            refresh: Whether to force a refresh of UI detection
            
        Returns:
            Dictionary with result information
        """
        try:
            if refresh or not self.detected_ui_elements:
                await self._detect_ui_elements()
                
            elements_data = []
            for element in self.detected_ui_elements:
                if element.confidence >= self.config.ui_confidence_threshold:
                    elements_data.append({
                        "element_id": element.element_id,
                        "element_type": element.element_type,
                        "bounds": element.bounds,
                        "center": element.center,
                        "text": element.text,
                        "confidence": element.confidence
                    })
                    
            return {
                "success": True,
                "elements": elements_data,
                "count": len(elements_data)
            }
                
        except Exception as e:
            logger.error(f"Error getting UI elements: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _detect_ui_elements(self) -> None:
        """Detect UI elements in the current screenshot."""
        detection_start = time.time()
        
        try:
            # Ensure we have a current screenshot
            await self.perception.capture_screen()
            
            # Get UI elements from perception module
            raw_elements = await self.perception.get_ui_elements()
            
            # Convert to UIElement objects
            self.detected_ui_elements = []
            for i, elem_data in enumerate(raw_elements):
                # Extract element data
                element_type = elem_data.get("type", "unknown")
                bounds = elem_data.get("bounds", (0, 0, 0, 0))
                text = elem_data.get("text")
                confidence = elem_data.get("confidence", 1.0)
                
                # Create UIElement object
                element = UIElement(
                    element_id=f"ui_{i}_{int(time.time())}",
                    element_type=element_type,
                    bounds=bounds,
                    text=text,
                    confidence=confidence,
                    attributes=elem_data.get("attributes", {})
                )
                
                self.detected_ui_elements.append(element)
                
            self.last_ui_detection_time = time.time()
            
            logger.debug(f"Detected {len(self.detected_ui_elements)} UI elements")
            
            # Update performance stats
            detection_time = time.time() - detection_start
            self.perf_stats['ui_detection_time'].append(detection_time)
            
        except Exception as e:
            logger.error(f"Error detecting UI elements: {str(e)}")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_shutdown_signal)
            
        logger.debug("Signal handlers registered")
    
    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received shutdown signal {signum}, stopping agent loop")
        self._shutdown_requested = True
    
    async def process_instruction(self, instruction: str) -> Optional[Plan]:
        """
        Process a new instruction and create a plan.
        
        Args:
            instruction: Natural language instruction to process
            
        Returns:
            The created plan, or None if failed
        """
        try:
            logger.info(f"Processing instruction: {instruction}")
            
            # Create a plan from the instruction
            plan = await self.planning.create_plan_from_instruction(instruction)
            
            # Store the current plan
            self.current_plan = plan
            
            # Log the plan details
            step_descriptions = [step.description for step in plan.steps]
            logger.info(f"Created plan with {len(plan.steps)} steps: {step_descriptions}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error processing instruction: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    async def _handle_execute_plan(self, plan_id: Optional[str] = None, 
                                  instruction: Optional[str] = None, 
                                  **kwargs) -> Dict[str, Any]:
        """
        Execute a plan based on plan ID or create and execute a new plan from instruction.
        
        Args:
            plan_id: Optional ID of an existing plan to execute
            instruction: Optional instruction to create a new plan
            
        Returns:
            Dictionary with result information
        """
        try:
            if not plan_id and not instruction:
                return {
                    "success": False,
                    "message": "Either plan_id or instruction must be provided"
                }
                
            # Get or create plan
            plan = None
            if plan_id:
                # TODO: Implement plan retrieval by ID when the feature is available
                return {
                    "success": False,
                    "message": "Plan retrieval by ID not implemented yet"
                }
            elif instruction:
                plan = await self.planning.create_plan_from_instruction(instruction)
                
            if not plan:
                return {
                    "success": False,
                    "message": "Failed to get or create plan"
                }
                
            # Store as current plan
            self.current_plan = plan
            
            # Start execution
            plan.mark_started()
            
            # Return initial status
            return {
                "success": True,
                "plan_id": id(plan),  # Use object ID as plan ID for now
                "instruction": plan.instruction,
                "step_count": len(plan.steps),
                "status": plan.status.name
            }
                
        except Exception as e:
            logger.error(f"Error executing plan: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _update_action_context(self) -> None:
        """Update the current action context with latest perception data."""
        try:
            # Get latest screenshot
            screenshot = await self.perception.capture_screen()
            
            # Update UI elements if needed
            if self.config.enable_ui_interaction and (
                not self.detected_ui_elements or 
                time.time() - self.last_ui_detection_time > self.config.ui_detection_interval * self.config.tick_rate
            ):
                await self._detect_ui_elements()
            
            # Update context
            self.current_action_context = ActionContext(
                screenshot=screenshot,
                ui_elements=self.detected_ui_elements,
                current_instruction=self.current_plan.instruction if self.current_plan else None,
                previous_actions=self.action_history[-5:] if self.action_history else [],
                environment_state={
                    "timestamp": time.time(),
                    "tick_count": self.tick_count
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating action context: {str(e)}")
    
    async def _execute_action_with_context(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action with the current context.
        
        Args:
            action_type: Type of action to execute
            params: Action parameters
            
        Returns:
            Action result dictionary
        """
        start_time = time.time()
        
        try:
            # Check if we have an action handler
            if action_type not in self.planning.action_handlers:
                return {
                    "success": False,
                    "message": f"No handler for action type: {action_type}"
                }
            
            handler = self.planning.action_handlers[action_type]
            
            # Enhance params with context if enabled
            if self.config.context_aware_actions:
                context_enhanced_params = self._enhance_params_with_context(action_type, params)
            else:
                context_enhanced_params = params.copy()
            
            # Execute the action with retry for critical actions
            if action_type in self.critical_actions:
                # Get max retries from config or use default
                max_retries = getattr(self.config, "action_max_retries", 3)
                logger.debug(f"Using retry mechanism for critical action: {action_type}")
                result = await self._execute_action_with_retry(action_type, context_enhanced_params, max_retries)
            else:
                # Execute the action directly for non-critical actions
                result = await handler(**context_enhanced_params)
            
            # Record in history
            action_record = {
                "type": action_type,
                "params": params,
                "enhanced_params": context_enhanced_params,
                "result": result,
                "timestamp": time.time(),
                "duration": time.time() - start_time,
                "is_critical": action_type in self.critical_actions
            }
            self.action_history.append(action_record)
            self.last_action_result = result
            
            # Update performance stats
            self.perf_stats['action_time'].append(time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action_type}: {str(e)}")
            self.last_action_result = {
                "success": False, 
                "error": str(e),
                "error_type": type(e).__name__,
                "duration": time.time() - start_time
            }
            return self.last_action_result
    
    def _enhance_params_with_context(self, action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance action parameters with context information.
        
        Args:
            action_type: Type of action
            params: Original parameters
            
        Returns:
            Enhanced parameters
        """
        enhanced_params = params.copy()
        
        # Add context based on action type
        if action_type in ['click', 'click_at', 'move_mouse', 'drag']:
            # For mouse actions, add information about nearby UI elements
            if 'x' in enhanced_params and 'y' in enhanced_params:
                x, y = enhanced_params['x'], enhanced_params['y']
                nearby_elements = self._find_nearby_ui_elements(x, y, max_distance=50)
                if nearby_elements:
                    enhanced_params['nearby_elements'] = nearby_elements
        
        elif action_type in ['type_text', 'press_key']:
            # For keyboard actions, add information about focused text fields
            text_fields = [elem for elem in self.detected_ui_elements 
                          if elem.element_type in ['text_field', 'input', 'textarea']]
            if text_fields:
                enhanced_params['potential_targets'] = text_fields
        
        elif action_type == 'find_ui_element':
            # For UI element search, add broader context
            if not enhanced_params.get('screenshot') and self.current_action_context.screenshot is not None:
                enhanced_params['screenshot'] = self.current_action_context.screenshot
        
        # Add general context
        enhanced_params['_context'] = {
            'timestamp': time.time(),
            'instruction': self.current_action_context.current_instruction
        }
        
        return enhanced_params
    
    def _find_nearby_ui_elements(self, x: int, y: int, max_distance: int = 50) -> List[Dict[str, Any]]:
        """
        Find UI elements near the specified coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            max_distance: Maximum distance to consider (in pixels)
            
        Returns:
            List of nearby UI elements
        """
        nearby = []
        
        for element in self.detected_ui_elements:
            if element.confidence < self.config.ui_confidence_threshold:
                continue
                
            # Calculate distance to element center
            cx, cy = element.center
            distance = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            
            if distance <= max_distance:
                nearby.append({
                    "id": element.element_id,
                    "type": element.element_type,
                    "text": element.text,
                    "distance": distance,
                    "center": (cx, cy)
                })
                
        # Sort by distance
        nearby.sort(key=lambda e: e["distance"])
        return nearby
    
    async def _handle_plan_step(self, step: PlanStep) -> Dict[str, Any]:
        """
        Handle execution of a single plan step with robust error handling.
        
        Args:
            step: The plan step to execute
            
        Returns:
            Step execution result with detailed error information if applicable
        """
        step_start_time = time.time()
        step_id = id(step)
        
        logger.info(f"Executing plan step [{step_id}]: {step.description} ({step.action_type})")
        
        try:
            # Mark step as in progress
            step.mark_in_progress()
            
            # Update the action context
            await self._update_action_context()
            
            # Execute the action with context
            result = await self._execute_action_with_context(step.action_type, step.params)
            
            # Store the result in the step
            step.result = result
            
            # Update the step status based on result
            if result.get("success", False):
                step.mark_completed()
                logger.info(f"Step [{step_id}] completed successfully: {step.description}")
            else:
                error_message = result.get("message", result.get("error", "Unknown error"))
                step.mark_failed(error_message)
                logger.warning(f"Step [{step_id}] failed: {step.description} - {error_message}")
                
                # Handle failure with dynamic replanning if enabled
                if self.config.dynamic_replanning and self.current_plan:
                    await self._handle_step_failure(step, result)
            
            return result
            
        except asyncio.CancelledError:
            # Handle cancellation explicitly
            cancel_message = "Step execution was cancelled"
            logger.warning(f"Step [{step_id}] cancelled: {step.description}")
            
            step.mark_cancelled(cancel_message)
            result = {
                "success": False,
                "cancelled": True,
                "message": cancel_message,
                "time_elapsed": time.time() - step_start_time
            }
            step.result = result
            
            # Re-raise to properly propagate cancellation
            raise
            
        except asyncio.TimeoutError:
            # Handle timeout explicitly
            timeout_message = "Step execution timed out"
            logger.error(f"Step [{step_id}] timed out: {step.description}")
            
            step.mark_failed(timeout_message)
            result = {
                "success": False,
                "timeout": True,
                "message": timeout_message,
                "time_elapsed": time.time() - step_start_time
            }
            step.result = result
            return result
            
        except Exception as e:
            # Handle general exceptions
            error_message = f"Error executing step: {str(e)}"
            logger.error(f"Step [{step_id}] failed with exception: {step.description} - {error_message}")
            logger.debug(traceback.format_exc())
            
            step.mark_failed(error_message)
            result = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": error_message,
                "time_elapsed": time.time() - step_start_time
            }
            step.result = result
            return result
            
        finally:
            # Always log step completion time
            execution_time = time.time() - step_start_time
            logger.debug(f"Step [{step_id}] execution took {execution_time:.3f} seconds")
            
            # Update plan status if we have a current plan
            if self.current_plan:
                self.current_plan.update_status()
    
    async def _handle_step_failure(self, failed_step: PlanStep, failure_result: Dict[str, Any]) -> None:
        """
        Handle a failed plan step, potentially with dynamic replanning.
        
        Args:
            failed_step: The failed plan step
            failure_result: The failure result
        """
        if not self.config.dynamic_replanning or not self.current_plan:
            return
            
        logger.info(f"Attempting to recover from step failure: {failed_step.description}")
        
        # Create a recovery instruction
        recovery_instruction = (
            f"The action '{failed_step.description}' failed with error: "
            f"{failure_result.get('message', 'Unknown error')}. "
            f"Please create a recovery plan to achieve the original goal: "
            f"{self.current_plan.instruction}"
        )
        
        try:
            # Create a recovery plan
            recovery_plan = await self.planning.create_plan_from_instruction(recovery_instruction)
            
            if recovery_plan and recovery_plan.steps:
                logger.info(f"Created recovery plan with {len(recovery_plan.steps)} steps")
                
                # Replace the current plan with the recovery plan
                self.current_plan = recovery_plan
                recovery_plan.mark_started()
                
                # Log the recovery
                self.memory.log_action(AgentAction(
                    action_type="dynamic_replanning",
                    params={
                        "original_instruction": self.current_plan.instruction,
                        "failed_step": failed_step.description,
                        "recovery_instruction": recovery_instruction
                    },
                    success=True
                ))
            else:
                logger.warning("Failed to create recovery plan")
                
        except Exception as e:
            logger.error(f"Error in dynamic replanning: {str(e)}")
    
    async def execute_planned_actions(self, plan: Optional[Plan] = None, instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a plan from an instruction and execute it with feedback.
        
        Args:
            plan: Optional existing plan to execute
            instruction: Optional instruction to create a plan if none provided
            
        Returns:
            Execution results
        """
        try:
            # Create or use provided plan
            if not plan and not instruction:
                return {
                    "success": False,
                    "message": "Either plan or instruction must be provided"
                }
                
            # Use instruction as is or extract from plan
            actual_instruction = instruction if instruction else plan.instruction
            
            # Start tracking execution using FeedbackManager
            execution_id = self.feedback.start_execution(actual_instruction)
            logger.info(f"Started execution tracking [{execution_id}] for: {actual_instruction}")
            
            if not plan:
                # Create plan
                logger.info(f"Creating plan for instruction: {actual_instruction}")
                create_plan_step_id = self.feedback.add_step("Create plan from instruction", "plan_creation")
                
                plan = await self.planning.create_plan_from_instruction(actual_instruction)
                
                if not plan:
                    # Update feedback and return error
                    self.feedback.update_step(create_plan_step_id, "FAILED", {
                        "success": False, 
                        "error": "Failed to create plan"
                    })
                    self.feedback.complete_execution(False, {
                        "success": False,
                        "message": "Failed to create plan"
                    })
                    return {
                        "success": False,
                        "message": "Failed to create plan",
                        "execution_id": execution_id
                    }
                else:
                    self.feedback.update_step(create_plan_step_id, "COMPLETED", {
                        "success": True,
                        "plan_steps": len(plan.steps)
                    })
            
            # Store plan and start it
            self.current_plan = plan
            plan.mark_started()
            
            init_step_id = self.feedback.add_step("Initialize plan execution", "plan_initialization")
            
            # Create a unique task name for the execution
            task_prefix = "plan_exec"
            timestamp = int(time.time())
            plan_task_name = f"{task_prefix}_{timestamp}"
            
            # Track overall results
            results = []
            step_tasks = {}
            step_ids = {}  # Map step to feedback step ID
            
            # Create tasks for each step
            for i, step in enumerate(plan.steps):
                if step.status == PlanStatus.PENDING:
                    # Add step to feedback manager
                    feedback_step_id = self.feedback.add_step(
                        step.description, 
                        step.action_type
                    )
                    
                    # Mark as in progress in feedback
                    self.feedback.update_step(feedback_step_id, "IN_PROGRESS")
                    
                    # Create the task
                    step_task_name = f"{plan_task_name}_step_{i}"
                    step_task = await self.task_manager.create_task(
                        step_task_name, 
                        self._handle_plan_step(step)
                    )
                    
                    # Store mappings
                    step_tasks[step_task_name] = {"step": step, "task": step_task}
                    step_ids[step_task_name] = feedback_step_id
            
            self.feedback.update_step(init_step_id, "COMPLETED", {
                "success": True,
                "tasks_created": len(step_tasks)
            })
            
            # Execute tasks and collect results
            cancelled_tasks = []
            for step_task_name, task_info in step_tasks.items():
                step = task_info["step"]
                feedback_step_id = step_ids[step_task_name]
                
                try:
                    # Wait for step completion with timeout (if configured)
                    timeout = getattr(self.config, "step_timeout", 60)  # Default 60 seconds if not specified
                    result = await self.task_manager.wait_for_task(step_task_name, timeout)
                    
                    # Update feedback status based on result
                    if result.get("success", False):
                        self.feedback.update_step(feedback_step_id, "COMPLETED", result)
                    else:
                        self.feedback.update_step(feedback_step_id, "FAILED", result)
                    
                    # Add to results
                    results.append({
                        "step": step.description,
                        "action_type": step.action_type,
                        "success": result.get("success", False),
                        "result": result
                    })
                    
                    # Break on failure if dynamic replanning is disabled
                    if not result.get("success", False) and not self.config.dynamic_replanning:
                        # Cancel remaining tasks
                        for remaining_task_name in [name for name in step_tasks.keys() 
                                                   if name not in cancelled_tasks and 
                                                   name != step_task_name]:
                            # Update feedback for cancelled steps
                            self.feedback.update_step(step_ids[remaining_task_name], "CANCELLED", {
                                "success": False,
                                "cancelled": True,
                                "message": "Cancelled due to prior step failure"
                            })
                            
                            # Cancel task
                            self.task_manager.cancel_task(remaining_task_name)
                            cancelled_tasks.append(remaining_task_name)
                        break
                        
                except asyncio.TimeoutError:
                    # Handle timeout
                    timeout_result = {
                        "success": False, 
                        "timeout": True, 
                        "error": f"Execution timed out after {timeout} seconds"
                    }
                    
                    # Update feedback
                    self.feedback.update_step(feedback_step_id, "TIMEOUT", timeout_result)
                    
                    logger.warning(f"Timeout while executing step: {step.description}")
                    step.mark_failed(f"Execution timed out after {timeout} seconds")
                    
                    results.append({
                        "step": step.description,
                        "action_type": step.action_type,
                        "success": False,
                        "result": timeout_result
                    })
                    
                    # Cancel the task
                    self.task_manager.cancel_task(step_task_name)
                    cancelled_tasks.append(step_task_name)
                    
                    # Break if dynamic replanning disabled
                    if not self.config.dynamic_replanning:
                        break
                        
                except Exception as e:
                    # Handle other exceptions
                    error_result = {
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    
                    # Update feedback
                    self.feedback.update_step(feedback_step_id, "FAILED", error_result)
                    
                    logger.error(f"Error executing step {step.description}: {str(e)}")
                    step.mark_failed(f"Execution error: {str(e)}")
                    
                    results.append({
                        "step": step.description,
                        "action_type": step.action_type,
                        "success": False,
                        "result": error_result
                    })
                    
                    # Cancel the task
                    if step_task_name not in cancelled_tasks:
                        try:
                            self.task_manager.cancel_task(step_task_name)
                            cancelled_tasks.append(step_task_name)
                        except Exception:
                            pass
                    
                    # Break if dynamic replanning disabled
                    if not self.config.dynamic_replanning:
                        break
            
            # Update plan status
            plan.update_status()
            
            # Determine overall success
            success = plan.status == PlanStatus.COMPLETED
            
            # Create execution summary
            execution_summary = {
                "success": success,
                "execution_id": execution_id,
                "plan_status": plan.status.name,
                "instruction": plan.instruction,
                "steps_total": len(plan.steps),
                "steps_completed": len([s for s in plan.steps if s.status == PlanStatus.COMPLETED]),
                "steps_failed": len([s for s in plan.steps if s.status == PlanStatus.FAILED]),
                "cancelled_tasks": len(cancelled_tasks),
                "results": results
            }
            
            # Complete execution in feedback manager
            self.feedback.complete_execution(success, execution_summary)
            
            # Return execution summary
            return execution_summary
                
        except Exception as e:
            # If we have started feedback tracking, complete it with error
            if 'execution_id' in locals():
                error_result = {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                self.feedback.complete_execution(False, error_result)
            
            logger.error(f"Error executing planned actions: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                "success": False, 
                "error": str(e),
                "execution_id": locals().get('execution_id')
            }
    
    async def _run_single_tick(self) -> None:
        """Run a single tick of the agent loop."""
        tick_start = time.time()
        
        try:
            # Skip processing if paused
            if self.paused:
                await asyncio.sleep(self.config.tick_rate)
                return
                
            # === PERCEPTION PHASE ===
            perception_start = time.time()
            
            try:
                # Capture the screen
                screenshot = await self.perception.capture_screen()
                
                # Perform UI element detection periodically
                if self.config.enable_ui_interaction and self.tick_count % self.config.ui_detection_interval == 0:
                    await self._detect_ui_elements()
            except Exception as e:
                logger.error(f"Error in perception phase: {str(e)}")
                logger.debug(traceback.format_exc())
                # Non-fatal error, continue with other phases
                
            perception_time = time.time() - perception_start
            
            # === INTELLIGENCE-ACTION CONNECTION PHASE ===
            intelligence_action_start = time.time()
            
            try:
                # Update action context with latest perception data
                await self._update_action_context()
                
                # Get the current plan
                current_plan = self.current_plan
                
                # If we have a plan that's not completed, failed, or cancelled, continue executing it
                if current_plan and current_plan.status == PlanStatus.IN_PROGRESS:
                    # Get the current step
                    current_step = current_plan.get_current_step()
                    
                    # If we have a current step that's pending, execute it with context-aware handler
                    if current_step and current_step.status == PlanStatus.PENDING:
                        logger.info(f"Executing step: {current_step.description} (Action: {current_step.action_type})")
                        await self._handle_plan_step(current_step)
                    
                    # Update plan status
                    current_plan.update_status()
                    
                    # If plan is completed or failed, log and clear
                    if current_plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED):
                        result_status = "succeeded" if current_plan.status == PlanStatus.COMPLETED else "failed"
                        logger.info(f"Plan for '{current_plan.instruction}' {result_status}")
                        
                        # Log more detailed diagnostics for failed plans
                        if current_plan.status == PlanStatus.FAILED:
                            failed_steps = [s for s in current_plan.steps if s.status == PlanStatus.FAILED]
                            for step in failed_steps:
                                logger.error(f"Step failed: {step.description}, Error: {step.error}")
                        
                        # Log plan completion to memory with feedback
                        if self.config.action_feedback_enabled:
                            feedback = {
                                "instruction": current_plan.instruction,
                                "status": current_plan.status.name,
                                "steps_count": len(current_plan.steps),
                                "completed_steps": len([s for s in current_plan.steps if s.status == PlanStatus.COMPLETED]),
                                "failed_steps": len([s for s in current_plan.steps if s.status == PlanStatus.FAILED])
                            }
                            self.memory.store_observation("plan_feedback", feedback)
                        
                        self.current_plan = None
                
                # If we don't have an active plan but have a pending one, start it
                elif current_plan and current_plan.status == PlanStatus.PENDING:
                    logger.info(f"Starting execution of plan: {current_plan.instruction}")
                    current_plan.mark_started()
            except Exception as e:
                logger.error(f"Error in intelligence-action phase: {str(e)}")
                logger.debug(traceback.format_exc())
                # If we hit an error during plan execution, mark the current step as failed
                if current_plan and current_plan.status == PlanStatus.IN_PROGRESS:
                    current_step = current_plan.get_current_step()
                    if current_step and current_step.status == PlanStatus.IN_PROGRESS:
                        current_step.mark_failed(f"Execution error: {str(e)}")
            
            intelligence_action_time = time.time() - intelligence_action_start
            
            # === MEMORY PHASE ===
            memory_start = time.time()
            
            try:
                # Store screenshot in memory periodically
                if self.tick_count % 10 == 0:
                    screenshot_key = f"latest_screenshot"
                    self.memory.store_observation(screenshot_key, {"timestamp": time.time()})
                    
                    # Also store UI element data if available
                    if self.detected_ui_elements:
                        ui_data = [
                            {
                                "id": elem.element_id,
                                "type": elem.element_type,
                                "bounds": elem.bounds,
                                "text": elem.text,
                                "confidence": elem.confidence
                            } 
                            for elem in self.detected_ui_elements
                            if elem.confidence >= self.config.ui_confidence_threshold
                        ]
                        self.memory.store_observation("latest_ui_elements", {
                            "timestamp": time.time(),
                            "elements": ui_data,
                            "count": len(ui_data)
                        })
                        
                    # Store action history
                    if self.action_history:
                        self.memory.store_observation("action_history", {
                            "timestamp": time.time(),
                            "recent_actions": self.action_history[-10:],
                            "total_actions": len(self.action_history)
                        })
                
                # Save memory state to disk periodically
                if self.tick_count % 50 == 0 and self.config.memory_dir:
                    self.memory.save_to_disk()
            except Exception as e:
                logger.error(f"Error in memory phase: {str(e)}")
                logger.debug(traceback.format_exc())
            
            memory_time = time.time() - memory_start
            
            # Update performance stats
            loop_time = time.time() - tick_start
            
            if self.config.monitor_performance:
                self.perf_stats['loop_time'].append(loop_time)
                self.perf_stats['perception_time'].append(perception_time)
                if 'action_time' not in self.perf_stats:
                    self.perf_stats['action_time'] = []
                self.perf_stats['memory_time'].append(memory_time)
                
                # Log performance stats periodically
                if self.tick_count % self.config.perf_sample_rate == 0:
                    self._log_performance_stats()
            
            # Sleep if needed to maintain tick rate
            elapsed = time.time() - tick_start
            if elapsed < self.config.tick_rate:
                await asyncio.sleep(self.config.tick_rate - elapsed)
            
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.info("Tick execution cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Error in agent loop tick: {str(e)}")
            logger.debug(traceback.format_exc())
            
        finally:
            self.tick_count += 1
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        if not self.perf_stats['loop_time']:
            return
            
        # Calculate averages, avoiding division by zero
        avg_loop_time = sum(self.perf_stats['loop_time']) / max(len(self.perf_stats['loop_time']), 1)
        
        # Check if perception_time has data
        if self.perf_stats['perception_time'] and len(self.perf_stats['perception_time']) > 0:
            avg_perception_time = sum(self.perf_stats['perception_time']) / len(self.perf_stats['perception_time'])
        else:
            avg_perception_time = 0
        
        # Check if planning_time has data
        if 'planning_time' in self.perf_stats and self.perf_stats['planning_time'] and len(self.perf_stats['planning_time']) > 0:
            avg_planning_time = sum(self.perf_stats['planning_time']) / len(self.perf_stats['planning_time'])
        else:
            avg_planning_time = 0
        
        # Check if memory_time has data
        if 'memory_time' in self.perf_stats and self.perf_stats['memory_time'] and len(self.perf_stats['memory_time']) > 0:
            avg_memory_time = sum(self.perf_stats['memory_time']) / len(self.perf_stats['memory_time'])
        else:
            avg_memory_time = 0
        
        # UI detection stats if available
        ui_stats = ""
        if 'ui_detection_time' in self.perf_stats and self.perf_stats['ui_detection_time'] and len(self.perf_stats['ui_detection_time']) > 0:
            avg_ui_time = sum(self.perf_stats['ui_detection_time']) / len(self.perf_stats['ui_detection_time'])
            ui_stats = f", UI Detection: {avg_ui_time*1000:.2f}"
        
        # Log the stats
        logger.info(f"Performance stats (ms) - Loop: {avg_loop_time*1000:.2f}, " +
                    f"Perception: {avg_perception_time*1000:.2f}, " +
                    f"Planning: {avg_planning_time*1000:.2f}, " +
                    f"Memory: {avg_memory_time*1000:.2f}" + ui_stats)
        
        # Average tick rate
        actual_tick_rate = avg_loop_time
        target_tick_rate = self.config.tick_rate
        if actual_tick_rate > target_tick_rate:
            logger.warning(f"Agent loop is running slower than configured rate " +
                           f"(actual: {actual_tick_rate:.3f}s, target: {target_tick_rate:.3f}s)")
        
        # Reset the stats (keep last few for stability in measurements)
        sample_size = 10
        for key in self.perf_stats:
            if len(self.perf_stats[key]) > sample_size:
                self.perf_stats[key] = self.perf_stats[key][-sample_size:]
    
    async def run(self) -> None:
        """Run the agent loop until stopped."""
        self._setup_signal_handlers()
        
        logger.info("Starting agent loop")
        self.running = True
        
        try:
            while self.running and not self._shutdown_requested:
                await self._run_single_tick()
                
        except KeyboardInterrupt:
            logger.info("Agent loop interrupted by keyboard")
        
        except Exception as e:
            logger.error(f"Unhandled exception in agent loop: {str(e)}")
            logger.debug(traceback.format_exc())
            
        finally:
            # Clean up and save final state
            logger.info("Agent loop stopped, cleaning up...")
            
            # Cancel any running plan
            if self.current_plan and self.current_plan.status == PlanStatus.IN_PROGRESS:
                self.planning.cancel_current_plan()
            
            # Save final memory state
            if self.config.memory_dir:
                self.memory.save_to_disk("final_state.json")
            
            self.running = False
            logger.info("Agent loop shutdown complete")
    
    async def start_and_wait(self, timeout: Optional[float] = None) -> bool:
        """
        Start the agent loop and wait for it to complete or timeout.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if completed successfully, False if timed out or error occurred
            
        Raises:
            RuntimeError: If the agent is already running
        """
        if self.running:
            raise RuntimeError("Agent loop is already running")
            
        try:
            # Start the task
            logger.info(f"Starting agent loop{f' with {timeout}s timeout' if timeout else ''}")
            
            self.running = True
            self._shutdown_requested = False
            
            if timeout is not None:
                # Start with timeout
                await asyncio.wait_for(self.run(), timeout)
            else:
                # Start without timeout
                await self.run()
                
            logger.info("Agent loop completed successfully")
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Agent loop timed out after {timeout}s")
            # Ensure the agent stops
            self.stop()
            return False
            
        except asyncio.CancelledError:
            logger.info("Agent loop was cancelled")
            self.stop()
            raise
            
        except Exception as e:
            logger.error(f"Error in agent loop: {str(e)}")
            logger.debug(traceback.format_exc())
            # Ensure the agent stops
            self.stop()
            return False
            
        finally:
            # Update running status
            self.running = False
    
    def start(self) -> asyncio.Task:
        """
        Start the agent loop in the current event loop.
        
        Returns:
            The created asyncio task
            
        Raises:
            RuntimeError: If the agent is already running
        """
        if self.running:
            raise RuntimeError("Agent loop is already running")
            
        # Create task and set the running flag
        self.running = True
        self._shutdown_requested = False
        task = asyncio.create_task(self.run(), name="MacAgent_Loop")
        
        # Add done callback to update running status when complete
        task.add_done_callback(lambda _: setattr(self, 'running', False))
        
        logger.info("Agent loop started in background task")
        return task
    
    def stop(self) -> None:
        """Request the agent loop to stop."""
        logger.info("Stop requested for agent loop")
        self._shutdown_requested = True
    
    def pause(self) -> None:
        """Pause the agent loop processing."""
        if not self.paused:
            logger.info("Pausing agent loop")
            self.paused = True
    
    def resume(self) -> None:
        """Resume the agent loop processing."""
        if self.paused:
            logger.info("Resuming agent loop")
            self.paused = False
    
    def cancel_current_plan(self) -> bool:
        """
        Cancel the current plan if one exists.
        
        Returns:
            True if a plan was cancelled, False otherwise
        """
        if self.current_plan:
            logger.info(f"Cancelling current plan: {self.current_plan.instruction}")
            self.planning.cancel_current_plan()
            self.current_plan = None
            return True
        return False

    async def find_and_click_element(self, element_type: str = None, text: str = None) -> Dict[str, Any]:
        """
        Find and click a UI element based on type and/or text content.
        
        Args:
            element_type: Type of element to find (button, text_field, etc)
            text: Text content to match
            
        Returns:
            Dictionary with result information
        """
        # First find the element
        find_result = await self._handle_find_ui_element(element_type=element_type, text=text)
        if not find_result["success"]:
            return find_result
            
        # Then click it
        return await self._handle_click_ui_element(element_id=find_result["element_id"])
    
    async def get_visible_ui_structure(self) -> Dict[str, Any]:
        """
        Get the structure of visible UI elements.
        
        Returns:
            Dictionary with UI structure information
        """
        # Make sure UI detection is up to date
        await self._detect_ui_elements()
        
        # Organize elements by type
        elements_by_type = {}
        for element in self.detected_ui_elements:
            if element.confidence < self.config.ui_confidence_threshold:
                continue
                
            if element.element_type not in elements_by_type:
                elements_by_type[element.element_type] = []
                
            elements_by_type[element.element_type].append({
                "id": element.element_id,
                "bounds": element.bounds,
                "text": element.text,
                "confidence": element.confidence
            })
            
        return {
            "success": True,
            "timestamp": time.time(),
            "ui_structure": elements_by_type,
            "element_count": len(self.detected_ui_elements)
        }

    async def _handle_wait(self, seconds: float = 1.0, **kwargs) -> Dict[str, Any]:
        """
        Handle a wait action by pausing execution for the specified time.
        
        Args:
            seconds: Number of seconds to wait
            
        Returns:
            Dictionary with result information
        """
        try:
            logger.info(f"Waiting for {seconds} seconds")
            await asyncio.sleep(seconds)
            return {"success": True, "waited_seconds": seconds}
        except Exception as e:
            logger.error(f"Error during wait: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _execute_action_with_retry(self, action_type: str, params: Dict[str, Any], max_retries: int = 3) -> Dict[str, Any]:
        """
        Execute an action with retry mechanism.
        
        Args:
            action_type: Type of action to execute
            params: Action parameters
            max_retries: Maximum number of retry attempts
            
        Returns:
            Action result dictionary
        """
        retry_count = 0
        last_error = None
        base_delay = 1.0  # Base delay in seconds
        
        logger.info(f"Executing action '{action_type}' with up to {max_retries} retries")
        
        while retry_count <= max_retries:
            try:
                # On first attempt, don't call it a retry
                attempt_desc = "initial attempt" if retry_count == 0 else f"retry attempt {retry_count}"
                logger.debug(f"Action '{action_type}' {attempt_desc}")
                
                # Execute the action with context
                result = await self._execute_action_with_context(action_type, params)
                
                # If successful, return immediately
                if result.get("success", False):
                    if retry_count > 0:
                        logger.info(f"Action '{action_type}' succeeded after {retry_count} retries")
                    return result
                
                # If this is the last attempt, return the failed result
                if retry_count >= max_retries:
                    logger.warning(f"Action '{action_type}' failed after {retry_count} retries. Last error: {result.get('message', 'Unknown error')}")
                    # Add retry information to result
                    result["retries_attempted"] = retry_count
                    result["retries_exhausted"] = True
                    return result
                
                # Otherwise, prepare for retry
                error_message = result.get("message", result.get("error", "Unknown error"))
                logger.warning(f"Action '{action_type}' failed (attempt {retry_count+1}/{max_retries+1}): {error_message}")
                last_error = error_message
                
                # Increment retry counter
                retry_count += 1
                
                # Exponential backoff with jitter
                delay = base_delay * (1.5 ** retry_count) + (random.random() * 0.5)
                logger.debug(f"Waiting {delay:.2f}s before retry...")
                await asyncio.sleep(delay)
                
            except asyncio.CancelledError:
                logger.warning(f"Action '{action_type}' cancelled during {attempt_desc}")
                raise
                
            except Exception as e:
                # If this is the last attempt, raise the exception
                if retry_count >= max_retries:
                    logger.error(f"Action '{action_type}' failed with exception after {retry_count} retries: {str(e)}")
                    raise
                
                # Otherwise, log and prepare for retry
                logger.error(f"Action '{action_type}' failed with exception (attempt {retry_count+1}/{max_retries+1}): {str(e)}")
                last_error = str(e)
                
                # Increment retry counter
                retry_count += 1
                
                # Exponential backoff with jitter
                delay = base_delay * (1.5 ** retry_count) + (random.random() * 0.5)
                logger.debug(f"Waiting {delay:.2f}s before retry...")
                await asyncio.sleep(delay)
        
        # This code should not be reached but is here for safety
        return {
            "success": False,
            "message": f"Action failed after {max_retries} retries. Last error: {last_error}",
            "retries_attempted": max_retries,
            "retries_exhausted": True,
            "last_error": last_error
        }

    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the agent to complete its current tasks.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            True if completed successfully, False if timed out
        """
        # If agent is not running, nothing to wait for
        if not self.running:
            logger.info("Agent is not running, nothing to wait for")
            return True
        
        # If no current plan, nothing to wait for
        if not self.current_plan:
            logger.info("No current plan, nothing to wait for")
            return True
            
        start_time = time.time()
        check_interval = 0.1  # Check status every 100ms
        
        try:
            logger.info(f"Waiting for current plan to complete: {self.current_plan.instruction}")
            
            while True:
                # Check if we've exceeded the timeout
                if timeout is not None and (time.time() - start_time) > timeout:
                    logger.warning(f"Timeout waiting for plan completion after {timeout}s")
                    return False
                
                # Check if the plan has reached a terminal state
                if not self.current_plan:
                    logger.info("Plan completed (no current plan)")
                    return True
                    
                if self.current_plan.status in [PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.CANCELLED]:
                    success = self.current_plan.status == PlanStatus.COMPLETED
                    logger.info(f"Plan {self.current_plan.status.name.lower()} with status: {success}")
                    return success
                
                # Check if the agent is still running
                if not self.running or self._shutdown_requested:
                    logger.warning("Agent is no longer running while waiting for plan completion")
                    return False
                
                # Wait a bit before checking again
                await asyncio.sleep(check_interval)
                
        except asyncio.CancelledError:
            logger.warning("Wait for completion was cancelled")
            raise
            
        except Exception as e:
            logger.error(f"Error while waiting for plan completion: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
