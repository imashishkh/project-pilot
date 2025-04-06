"""
System Integrator Module for MacAgent.

This module provides a comprehensive integration framework that connects 
vision, interaction, and intelligence components of the MacAgent system.
It manages the data flow between components and provides a unified interface
for the entire system.
"""

import logging
import time
import os
import asyncio
from typing import Dict, List, Optional, Any, Callable, Type, Set, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from MacAgent.src.core.perception import PerceptionModule
from MacAgent.src.core.action import ActionModule
from MacAgent.src.core.planning import PlanningModule
from MacAgent.src.core.memory import MemorySystem
from MacAgent.src.core.pipeline_manager import PipelineManager

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Base configuration for a component."""
    enabled: bool = True
    init_params: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    """Configuration for the entire system."""
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    
    # System-wide settings
    thread_pool_size: int = 4
    max_retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Monitoring and diagnostics
    enable_diagnostics: bool = True
    diagnostic_interval: float = 5.0  # seconds
    
    @classmethod
    def default_config(cls) -> 'SystemConfig':
        """Create a default configuration with standard components."""
        return cls(
            components={
                "perception": ComponentConfig(
                    init_params={"screenshot_interval": 0.2}
                ),
                "action": ComponentConfig(
                    init_params={"move_speed": 0.3, "click_delay": 0.1}
                ),
                "planning": ComponentConfig(),
                "memory": ComponentConfig(
                    init_params={"storage_dir": "memory", "max_actions": 100}
                ),
            },
            pipeline_config={
                "default_timeout": 10.0,
                "enable_parallelism": True,
                "execution_mode": "async"
            }
        )


class SystemIntegrator:
    """
    SystemIntegrator connects and manages all components of the MacAgent system.
    
    This class is responsible for:
    1. Component initialization, dependency resolution and lifecycle management
    2. Connecting components and managing data flow between them
    3. Providing a unified interface for the entire system
    4. Error handling and recovery
    5. System-wide diagnostics and monitoring
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the system integrator.
        
        Args:
            config: Configuration for the system integrator
        """
        self.config = config or SystemConfig.default_config()
        self.components: Dict[str, Any] = {}
        self.initialized = False
        self.running = False
        
        # Set up thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        
        # Create pipeline manager
        self.pipeline_manager = PipelineManager(**self.config.pipeline_config)
        
        # Event hooks for system events
        self.event_handlers: Dict[str, List[Callable]] = {
            "system_start": [],
            "system_stop": [],
            "system_error": [],
            "component_error": []
        }
        
        # Dependency graph for components
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Initialize diagnostics
        self.diagnostic_data = {
            "component_status": {},
            "error_counts": {},
            "last_errors": {},
            "performance_metrics": {}
        }
        
        logger.info("SystemIntegrator created")
    
    async def initialize(self) -> bool:
        """
        Initialize all components in the correct order based on dependencies.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self.initialized:
            logger.warning("System already initialized")
            return True
            
        logger.info("Initializing system components...")
        
        # Build dependency graph
        self._build_dependency_graph()
        
        # Initialize components in the correct order
        initialization_order = self._get_initialization_order()
        
        for component_name in initialization_order:
            if not await self._initialize_component(component_name):
                logger.error(f"Failed to initialize component: {component_name}")
                return False
        
        # Connect components
        self._connect_components()
        
        # Register pipeline stages
        self._setup_pipeline()
        
        self.initialized = True
        logger.info("System initialization complete")
        return True
    
    def _build_dependency_graph(self) -> None:
        """Build the dependency graph for components."""
        self.dependency_graph = {}
        
        for component_name, component_config in self.config.components.items():
            if component_config.enabled:
                self.dependency_graph[component_name] = set(component_config.dependencies)
    
    def _get_initialization_order(self) -> List[str]:
        """
        Determine the order of component initialization based on dependencies.
        
        Returns:
            List of component names in initialization order
        """
        # Implementation of topological sort for dependency resolution
        visited: Set[str] = set()
        initialization_order: List[str] = []
        
        def visit(component: str) -> None:
            if component in visited:
                return
                
            visited.add(component)
            
            for dependency in self.dependency_graph.get(component, set()):
                if dependency not in visited:
                    visit(dependency)
                    
            initialization_order.append(component)
            
        for component in self.dependency_graph:
            if component not in visited:
                visit(component)
                
        return initialization_order
    
    async def _initialize_component(self, component_name: str) -> bool:
        """
        Initialize a single component with retry logic.
        
        Args:
            component_name: Name of the component to initialize
            
        Returns:
            True if initialization succeeded, False otherwise
        """
        component_config = self.config.components.get(component_name)
        if not component_config or not component_config.enabled:
            logger.warning(f"Component {component_name} not found or disabled")
            return False
            
        logger.info(f"Initializing component: {component_name}")
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                if component_name == "perception":
                    self.components[component_name] = PerceptionModule(
                        **component_config.init_params
                    )
                elif component_name == "action":
                    self.components[component_name] = ActionModule(
                        **component_config.init_params
                    )
                elif component_name == "planning":
                    self.components[component_name] = PlanningModule(
                        **component_config.init_params
                    )
                elif component_name == "memory":
                    storage_dir = component_config.init_params.get("storage_dir")
                    if storage_dir and not os.path.exists(storage_dir):
                        os.makedirs(storage_dir, exist_ok=True)
                    
                    self.components[component_name] = MemorySystem(
                        **component_config.init_params
                    )
                else:
                    logger.warning(f"Unknown component type: {component_name}")
                    return False
                
                self.diagnostic_data["component_status"][component_name] = "initialized"
                return True
                
            except Exception as e:
                logger.error(f"Error initializing {component_name} (attempt {attempt+1}/{self.config.max_retry_attempts}): {str(e)}")
                self.diagnostic_data["error_counts"][component_name] = self.diagnostic_data["error_counts"].get(component_name, 0) + 1
                self.diagnostic_data["last_errors"][component_name] = str(e)
                
                if attempt < self.config.max_retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay)
        
        self.diagnostic_data["component_status"][component_name] = "failed"
        return False
    
    def _connect_components(self) -> None:
        """Connect components by setting up references between them."""
        # Connect perception and planning
        if "perception" in self.components and "planning" in self.components:
            planning = self.components["planning"]
            perception = self.components["perception"]
            # Set up any references or callbacks needed
        
        # Connect action and planning
        if "action" in self.components and "planning" in self.components:
            planning = self.components["planning"]
            action = self.components["action"]
            
            # Register action handlers
            planning.register_action_handler('move_mouse', action.move_to)
            planning.register_action_handler('click', action.click)
            planning.register_action_handler('click_at', action.click_at)
            planning.register_action_handler('type_text', action.type_text)
            planning.register_action_handler('press_key', action.press_key)
            planning.register_action_handler('hotkey', action.perform_hotkey)
            planning.register_action_handler('drag', action.drag_to)
            planning.register_action_handler('scroll', action.scroll)
        
        # Connect memory and other components
        if "memory" in self.components:
            memory = self.components["memory"]
            # Set up memory connections
    
    def _setup_pipeline(self) -> None:
        """Set up the main processing pipeline stages."""
        if not self.pipeline_manager:
            logger.error("Pipeline manager not initialized")
            return
        
        # Register stages with the pipeline manager
        if "perception" in self.components:
            self.pipeline_manager.register_stage(
                "perception", 
                self._perception_stage,
                dependencies=[]
            )
        
        if "planning" in self.components:
            self.pipeline_manager.register_stage(
                "planning", 
                self._planning_stage,
                dependencies=["perception"]
            )
        
        if "action" in self.components:
            self.pipeline_manager.register_stage(
                "action", 
                self._action_stage,
                dependencies=["planning"]
            )
        
        if "memory" in self.components:
            self.pipeline_manager.register_stage(
                "memory", 
                self._memory_stage,
                dependencies=["perception", "action"]
            )
    
    async def _perception_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the perception stage of the pipeline.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            Updated pipeline context
        """
        perception = self.components.get("perception")
        if not perception:
            return context
        
        try:
            start_time = time.time()
            
            # Capture screen
            screenshot = await perception.capture_screen()
            
            # Add results to context
            context["screenshot"] = screenshot
            context["perception_time"] = time.time() - start_time
            
            return context
        except Exception as e:
            logger.error(f"Error in perception stage: {str(e)}")
            context["errors"] = context.get("errors", []) + [{"stage": "perception", "error": str(e)}]
            self._handle_component_error("perception", e)
            return context
    
    async def _planning_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the planning stage of the pipeline.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            Updated pipeline context
        """
        planning = self.components.get("planning")
        if not planning:
            return context
        
        # Skip if no instruction
        if "instruction" not in context:
            return context
        
        try:
            start_time = time.time()
            
            # Process instruction and create plan
            plan = await planning.create_plan(context["instruction"], context.get("screenshot"))
            
            # Add results to context
            context["plan"] = plan
            context["planning_time"] = time.time() - start_time
            
            return context
        except Exception as e:
            logger.error(f"Error in planning stage: {str(e)}")
            context["errors"] = context.get("errors", []) + [{"stage": "planning", "error": str(e)}]
            self._handle_component_error("planning", e)
            return context
    
    async def _action_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action stage of the pipeline.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            Updated pipeline context
        """
        # Skip if no plan
        if "plan" not in context:
            return context
        
        try:
            start_time = time.time()
            
            # Execute the plan
            plan = context["plan"]
            result = await plan.execute()
            
            # Add results to context
            context["action_result"] = result
            context["action_time"] = time.time() - start_time
            
            return context
        except Exception as e:
            logger.error(f"Error in action stage: {str(e)}")
            context["errors"] = context.get("errors", []) + [{"stage": "action", "error": str(e)}]
            self._handle_component_error("action", e)
            return context
    
    async def _memory_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the memory stage of the pipeline.
        
        Args:
            context: Pipeline execution context
            
        Returns:
            Updated pipeline context
        """
        memory = self.components.get("memory")
        if not memory:
            return context
        
        try:
            start_time = time.time()
            
            # Store action if available
            if "action_result" in context and context["action_result"]:
                memory.store_action({
                    "timestamp": time.time(),
                    "action": context.get("instruction", ""),
                    "result": context["action_result"]
                })
            
            # Store observation if available
            if "screenshot" in context:
                memory.store_observation({
                    "timestamp": time.time(),
                    "screenshot": True
                })
            
            # Add results to context
            context["memory_time"] = time.time() - start_time
            
            return context
        except Exception as e:
            logger.error(f"Error in memory stage: {str(e)}")
            context["errors"] = context.get("errors", []) + [{"stage": "memory", "error": str(e)}]
            self._handle_component_error("memory", e)
            return context
    
    def _handle_component_error(self, component: str, error: Exception) -> None:
        """
        Handle errors in components.
        
        Args:
            component: The component that had an error
            error: The exception that occurred
        """
        self.diagnostic_data["error_counts"][component] = self.diagnostic_data["error_counts"].get(component, 0) + 1
        self.diagnostic_data["last_errors"][component] = str(error)
        
        # Notify error handlers
        for handler in self.event_handlers.get("component_error", []):
            try:
                handler(component, error)
            except Exception as e:
                logger.error(f"Error in component_error handler: {str(e)}")
    
    async def process_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Process a user instruction through the perception-planning-action pipeline.
        
        Args:
            instruction: The instruction to process
            
        Returns:
            Dictionary with processing results
        """
        if not self.initialized:
            logger.error("System not initialized")
            return {"success": False, "error": "System not initialized"}
        
        try:
            # Create initial pipeline context
            context = {
                "instruction": instruction,
                "timestamp": time.time()
            }
            
            # Execute the pipeline
            result = await self.pipeline_manager.execute_pipeline(context)
            
            # Extract relevant information for return value
            return {
                "success": "errors" not in result or len(result.get("errors", [])) == 0,
                "context": result
            }
        except Exception as e:
            logger.error(f"Error processing instruction: {str(e)}")
            for handler in self.event_handlers.get("system_error", []):
                try:
                    handler(e)
                except Exception as handler_error:
                    logger.error(f"Error in system_error handler: {str(handler_error)}")
            return {"success": False, "error": str(e)}
    
    async def start(self) -> None:
        """Start the system and all its components."""
        if self.running:
            logger.warning("System already running")
            return
            
        if not self.initialized:
            success = await self.initialize()
            if not success:
                logger.error("Failed to initialize system")
                return
        
        logger.info("Starting system...")
        
        # Start components
        for component_name, component in self.components.items():
            if hasattr(component, "start"):
                try:
                    if asyncio.iscoroutinefunction(component.start):
                        await component.start()
                    else:
                        component.start()
                    self.diagnostic_data["component_status"][component_name] = "running"
                except Exception as e:
                    logger.error(f"Error starting component {component_name}: {str(e)}")
                    self._handle_component_error(component_name, e)
        
        # Start pipeline manager
        self.pipeline_manager.start()
        
        # Start diagnostics if enabled
        if self.config.enable_diagnostics:
            self._start_diagnostics()
        
        self.running = True
        
        # Notify handlers of system start
        for handler in self.event_handlers.get("system_start", []):
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in system_start handler: {str(e)}")
        
        logger.info("System started successfully")
    
    async def stop(self) -> None:
        """Stop the system and all its components."""
        if not self.running:
            logger.warning("System not running")
            return
            
        logger.info("Stopping system...")
        
        # Stop diagnostics if enabled
        if self.config.enable_diagnostics:
            self._stop_diagnostics()
        
        # Stop pipeline manager
        self.pipeline_manager.stop()
        
        # Stop components in reverse initialization order
        for component_name in reversed(list(self.components.keys())):
            component = self.components[component_name]
            if hasattr(component, "stop"):
                try:
                    if asyncio.iscoroutinefunction(component.stop):
                        await component.stop()
                    else:
                        component.stop()
                    self.diagnostic_data["component_status"][component_name] = "stopped"
                except Exception as e:
                    logger.error(f"Error stopping component {component_name}: {str(e)}")
                    self._handle_component_error(component_name, e)
        
        self.running = False
        
        # Notify handlers of system stop
        for handler in self.event_handlers.get("system_stop", []):
            try:
                handler()
            except Exception as e:
                logger.error(f"Error in system_stop handler: {str(e)}")
        
        logger.info("System stopped successfully")
    
    def register_event_handler(self, event: str, handler: Callable) -> None:
        """
        Register a handler for a system event.
        
        Args:
            event: The event to handle
            handler: The handler function
        """
        if event not in self.event_handlers:
            self.event_handlers[event] = []
            
        self.event_handlers[event].append(handler)
    
    def unregister_event_handler(self, event: str, handler: Callable) -> bool:
        """
        Unregister a handler for a system event.
        
        Args:
            event: The event
            handler: The handler function
            
        Returns:
            True if the handler was removed, False otherwise
        """
        if event not in self.event_handlers:
            return False
            
        try:
            self.event_handlers[event].remove(handler)
            return True
        except ValueError:
            return False
    
    async def _collect_diagnostics(self) -> None:
        """Collect diagnostic information from all components."""
        while self.running and self.config.enable_diagnostics:
            for component_name, component in self.components.items():
                if hasattr(component, "get_diagnostics"):
                    try:
                        diagnostics = component.get_diagnostics()
                        self.diagnostic_data["performance_metrics"][component_name] = diagnostics
                    except Exception as e:
                        logger.error(f"Error collecting diagnostics from {component_name}: {str(e)}")
            
            # Collect pipeline metrics
            try:
                pipeline_metrics = self.pipeline_manager.get_performance_metrics()
                self.diagnostic_data["performance_metrics"]["pipeline"] = pipeline_metrics
            except Exception as e:
                logger.error(f"Error collecting pipeline metrics: {str(e)}")
                
            await asyncio.sleep(self.config.diagnostic_interval)
    
    def _start_diagnostics(self) -> None:
        """Start the diagnostics collection task."""
        asyncio.create_task(self._collect_diagnostics())
    
    def _stop_diagnostics(self) -> None:
        """Stop the diagnostics collection task."""
        # This will be handled by the loop condition in _collect_diagnostics
        pass
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get current diagnostic information.
        
        Returns:
            Dictionary with diagnostic information
        """
        return self.diagnostic_data
    
    def inject_component(self, name: str, component: Any) -> None:
        """
        Inject a component into the system (for testing or customization).
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        self.diagnostic_data["component_status"][name] = "injected"
        
        # Add to dependency graph if needed
        if name not in self.dependency_graph and self.initialized:
            self.dependency_graph[name] = set()
            self._connect_components()  # Reconnect components
            self._setup_pipeline()  # Refresh pipeline if needed
