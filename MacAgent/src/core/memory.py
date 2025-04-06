"""
Memory Module for MacAgent.

This module provides memory and state management for the agent.
It stores and retrieves information about UI state, past interactions,
and other data needed to maintain context during agent operation.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
import os
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class UIState:
    """Represents the state of a UI element or window."""
    element_id: str
    element_type: str
    position: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # x, y, width, height
    text: str = ""
    visible: bool = True
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the UIState to a dictionary."""
        return asdict(self)


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    action_type: str  # click, type, move, etc.
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the AgentAction to a dictionary."""
        return asdict(self)


class MemorySystem:
    """
    Manages memory and state for the agent.
    
    This system is responsible for storing and retrieving information about
    UI state, past actions, and other contextual data.
    """
    
    def __init__(self, storage_dir: Optional[str] = None, max_actions: int = 100):
        """
        Initialize the memory system.
        
        Args:
            storage_dir: Optional directory path for persistent storage
            max_actions: Maximum number of actions to keep in memory
        """
        self.ui_states: Dict[str, UIState] = {}
        self.recent_actions: List[AgentAction] = []
        self.observations: Dict[str, Any] = {}
        self.max_actions = max_actions
        
        # Set up storage directory if provided
        self.storage_dir = None
        if storage_dir:
            self.storage_dir = Path(storage_dir)
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Memory system using persistent storage at {self.storage_dir}")
        
        logger.info("MemorySystem initialized")
    
    def update_ui_state(self, state: UIState) -> None:
        """
        Update or add a UI element state.
        
        Args:
            state: The UIState to add or update
        """
        self.ui_states[state.element_id] = state
        logger.debug(f"Updated UI state for element {state.element_id}")
    
    def get_ui_state(self, element_id: str) -> Optional[UIState]:
        """
        Retrieve a UI element state by ID.
        
        Args:
            element_id: The ID of the UI element
            
        Returns:
            The UIState if found, None otherwise
        """
        state = self.ui_states.get(element_id)
        if state:
            logger.debug(f"Retrieved UI state for element {element_id}")
            return state
        
        logger.debug(f"UI state not found for element {element_id}")
        return None
    
    def log_action(self, action: AgentAction) -> None:
        """
        Log an action taken by the agent.
        
        Args:
            action: The AgentAction to log
        """
        self.recent_actions.append(action)
        
        # Trim action history if it exceeds maximum size
        if len(self.recent_actions) > self.max_actions:
            self.recent_actions = self.recent_actions[-self.max_actions:]
        
        logger.debug(f"Logged action {action.action_type}")
    
    def get_recent_actions(self, count: int = 10, action_type: Optional[str] = None) -> List[AgentAction]:
        """
        Get recent actions, optionally filtered by type.
        
        Args:
            count: Maximum number of actions to return
            action_type: Optional filter for action type
            
        Returns:
            List of recent AgentAction objects
        """
        if action_type:
            filtered_actions = [a for a in self.recent_actions if a.action_type == action_type]
            return filtered_actions[-count:]
        
        return self.recent_actions[-count:]
    
    def store_observation(self, key: str, data: Any) -> None:
        """
        Store arbitrary observation data.
        
        Args:
            key: Identifier for the data
            data: The data to store
        """
        self.observations[key] = data
        logger.debug(f"Stored observation with key {key}")
    
    def get_observation(self, key: str, default: Any = None) -> Any:
        """
        Retrieve observation data by key.
        
        Args:
            key: Identifier for the data
            default: Value to return if key not found
            
        Returns:
            The stored data or default value
        """
        data = self.observations.get(key, default)
        logger.debug(f"Retrieved observation with key {key}")
        return data
    
    def save_to_disk(self, filename: Optional[str] = None) -> bool:
        """
        Save the current memory state to disk.
        
        Args:
            filename: Optional specific filename to use
            
        Returns:
            True if successful, False otherwise
        """
        if not self.storage_dir:
            logger.warning("No storage directory configured for memory persistence")
            return False
        
        try:
            # Generate filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"memory_state_{timestamp}.json"
            
            # Prepare data for serialization
            memory_data = {
                "ui_states": {k: v.to_dict() for k, v in self.ui_states.items()},
                "recent_actions": [a.to_dict() for a in self.recent_actions],
                "observations": {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict)) else v 
                                 for k, v in self.observations.items()}
            }
            
            # Save to file
            filepath = self.storage_dir / filename
            with open(filepath, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            logger.info(f"Memory state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory state: {str(e)}")
            return False
    
    def load_from_disk(self, filename: str) -> bool:
        """
        Load memory state from disk.
        
        Args:
            filename: Filename to load from
            
        Returns:
            True if successful, False otherwise
        """
        if not self.storage_dir:
            logger.warning("No storage directory configured for memory persistence")
            return False
        
        try:
            filepath = self.storage_dir / filename
            if not filepath.exists():
                logger.error(f"Memory file not found: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                memory_data = json.load(f)
            
            # Restore UI states
            self.ui_states = {}
            for element_id, state_dict in memory_data.get("ui_states", {}).items():
                position = tuple(state_dict.get("position", (0, 0, 0, 0)))
                state = UIState(
                    element_id=element_id,
                    element_type=state_dict.get("element_type", ""),
                    position=position,
                    text=state_dict.get("text", ""),
                    visible=state_dict.get("visible", True),
                    attributes=state_dict.get("attributes", {}),
                    timestamp=state_dict.get("timestamp", time.time())
                )
                self.ui_states[element_id] = state
            
            # Restore recent actions
            self.recent_actions = []
            for action_dict in memory_data.get("recent_actions", []):
                action = AgentAction(
                    action_type=action_dict.get("action_type", ""),
                    params=action_dict.get("params", {}),
                    timestamp=action_dict.get("timestamp", time.time()),
                    success=action_dict.get("success", True)
                )
                self.recent_actions.append(action)
            
            # Restore observations
            self.observations = memory_data.get("observations", {})
            
            logger.info(f"Memory state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory state: {str(e)}")
            return False
    
    def clear(self, clear_ui_states: bool = True, 
             clear_actions: bool = True,
             clear_observations: bool = True) -> None:
        """
        Clear memory components.
        
        Args:
            clear_ui_states: Whether to clear UI states
            clear_actions: Whether to clear action history
            clear_observations: Whether to clear observations
        """
        if clear_ui_states:
            self.ui_states = {}
            logger.debug("Cleared UI states")
        
        if clear_actions:
            self.recent_actions = []
            logger.debug("Cleared action history")
        
        if clear_observations:
            self.observations = {}
            logger.debug("Cleared observations")
        
        logger.info("Memory system cleared")
