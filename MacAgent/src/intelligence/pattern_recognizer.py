"""
Pattern Recognizer Module

This module provides tools for recognizing patterns in user behavior and UI interactions,
and optimizing agent behavior based on these recognized patterns.
"""

import os
import json
import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# Import LLM connector for pattern analysis
from .llm_connector import LLMConnector, LLMProvider, ModelConfig

# Configure logging
logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of recognizable patterns."""
    TIME_BASED = "time_based"        # Patterns related to time of day/week
    SEQUENCE = "sequence"            # Sequence of actions
    UI_INTERACTION = "ui_interaction"  # UI interaction patterns
    COMMAND_USAGE = "command_usage"  # Command usage patterns
    APP_PREFERENCE = "app_preference"  # Application preferences
    SEARCH_PATTERN = "search_pattern"  # Search behavior patterns
    WORKFLOW = "workflow"            # Multi-step workflow patterns


@dataclass
class UserInteraction:
    """Record of a user interaction."""
    interaction_id: str              # Unique identifier for the interaction
    timestamp: float                 # When the interaction occurred
    interaction_type: str            # Type of interaction (click, command, etc.)
    target: str                      # Target of interaction (app, UI element, etc.)
    context: Dict[str, Any] = field(default_factory=dict)  # Context of interaction
    success: bool = True             # Whether interaction was successful
    duration: Optional[float] = None  # Duration of interaction (if applicable)


@dataclass
class RecognizedPattern:
    """A recognized pattern in user behavior."""
    pattern_id: str                  # Unique identifier for the pattern
    pattern_type: PatternType        # Type of pattern
    description: str                 # Human-readable description
    confidence: float                # Confidence in the pattern (0-1)
    frequency: int                   # Number of occurrences
    last_seen: float                 # When the pattern was last observed
    first_seen: float                # When the pattern was first observed
    elements: List[Any]              # Elements that make up the pattern
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


class PatternRecognizer:
    """
    Recognizes patterns in user behavior and UI interactions.
    
    Features:
    - Tracks user interactions
    - Identifies common patterns in behavior
    - Suggests optimizations based on patterns
    - Adapts agent behavior to user preferences
    - Provides insights on usage patterns
    """
    
    def __init__(self,
                llm_connector: LLMConnector,
                default_model: str = "gpt-4",
                default_provider: LLMProvider = LLMProvider.OPENAI,
                data_dir: str = "data/patterns",
                patterns_file: str = "recognized_patterns.json",
                min_confidence: float = 0.6,
                max_history_days: int = 30):
        """
        Initialize the pattern recognizer.
        
        Args:
            llm_connector: LLM connector for pattern analysis
            default_model: Default model for analysis
            default_provider: Default provider for analysis
            data_dir: Directory to store pattern data
            patterns_file: File to store recognized patterns
            min_confidence: Minimum confidence for pattern recognition
            max_history_days: Maximum days of history to maintain
        """
        self.llm_connector = llm_connector
        self.default_model = default_model
        self.default_provider = default_provider
        self.data_dir = data_dir
        self.patterns_file = os.path.join(data_dir, patterns_file)
        self.min_confidence = min_confidence
        self.max_history_days = max_history_days
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data structures
        self.interactions: List[UserInteraction] = []
        self.recognized_patterns: Dict[str, RecognizedPattern] = {}
        self.sequence_buffer: List[UserInteraction] = []
        self.daily_stats: Dict[str, Dict[str, Any]] = {}
        
        # Load existing patterns and interaction history
        self._load_patterns()
        self._load_interactions()
        
        logger.info(f"PatternRecognizer initialized with {len(self.recognized_patterns)} patterns")
    
    def _load_patterns(self) -> None:
        """Load recognized patterns from file."""
        if not os.path.exists(self.patterns_file):
            return
            
        try:
            with open(self.patterns_file, "r") as f:
                patterns_data = json.load(f)
                
                for pattern_dict in patterns_data:
                    # Convert pattern_type from string to enum
                    pattern_type_str = pattern_dict.pop("pattern_type", "")
                    try:
                        pattern_type = PatternType(pattern_type_str)
                    except ValueError:
                        pattern_type = PatternType.SEQUENCE  # Default
                    
                    # Create RecognizedPattern object
                    pattern = RecognizedPattern(
                        pattern_id=pattern_dict.get("pattern_id", ""),
                        pattern_type=pattern_type,
                        description=pattern_dict.get("description", ""),
                        confidence=pattern_dict.get("confidence", 0.0),
                        frequency=pattern_dict.get("frequency", 0),
                        last_seen=pattern_dict.get("last_seen", 0.0),
                        first_seen=pattern_dict.get("first_seen", 0.0),
                        elements=pattern_dict.get("elements", []),
                        metadata=pattern_dict.get("metadata", {})
                    )
                    
                    self.recognized_patterns[pattern.pattern_id] = pattern
                    
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def _save_patterns(self) -> None:
        """Save recognized patterns to file."""
        try:
            patterns_data = []
            
            for pattern in self.recognized_patterns.values():
                pattern_dict = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "frequency": pattern.frequency,
                    "last_seen": pattern.last_seen,
                    "first_seen": pattern.first_seen,
                    "elements": pattern.elements,
                    "metadata": pattern.metadata
                }
                patterns_data.append(pattern_dict)
                
            with open(self.patterns_file, "w") as f:
                json.dump(patterns_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _load_interactions(self) -> None:
        """Load interaction history from files."""
        try:
            # Get all interaction files in data directory
            for filename in os.listdir(self.data_dir):
                if filename.startswith("interactions_") and filename.endswith(".json"):
                    file_path = os.path.join(self.data_dir, filename)
                    
                    try:
                        with open(file_path, "r") as f:
                            interactions_data = json.load(f)
                            
                            for interaction_dict in interactions_data:
                                interaction = UserInteraction(
                                    interaction_id=interaction_dict.get("interaction_id", ""),
                                    timestamp=interaction_dict.get("timestamp", 0.0),
                                    interaction_type=interaction_dict.get("interaction_type", ""),
                                    target=interaction_dict.get("target", ""),
                                    context=interaction_dict.get("context", {}),
                                    success=interaction_dict.get("success", True),
                                    duration=interaction_dict.get("duration")
                                )
                                
                                # Only keep recent interactions based on max_history_days
                                if time.time() - interaction.timestamp < self.max_history_days * 86400:
                                    self.interactions.append(interaction)
                                    
                    except Exception as e:
                        logger.warning(f"Error loading interactions file {filename}: {e}")
                        
            # Sort interactions by timestamp
            self.interactions.sort(key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Error loading interactions: {e}")
    
    def _save_interaction(self, interaction: UserInteraction) -> None:
        """Save an interaction to file."""
        try:
            # Determine file based on date
            date = datetime.fromtimestamp(interaction.timestamp)
            filename = f"interactions_{date.year}_{date.month}.json"
            file_path = os.path.join(self.data_dir, filename)
            
            # Convert interaction to dict
            interaction_dict = {
                "interaction_id": interaction.interaction_id,
                "timestamp": interaction.timestamp,
                "interaction_type": interaction.interaction_type,
                "target": interaction.target,
                "context": interaction.context,
                "success": interaction.success,
                "duration": interaction.duration
            }
            
            # Load existing data if file exists
            interactions_data = []
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    interactions_data = json.load(f)
                    
            # Add new interaction
            interactions_data.append(interaction_dict)
            
            # Save to file
            with open(file_path, "w") as f:
                json.dump(interactions_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
    
    def record_interaction(self, 
                         interaction_type: str,
                         target: str,
                         context: Dict[str, Any] = None,
                         success: bool = True,
                         duration: Optional[float] = None) -> str:
        """
        Record a user interaction.
        
        Args:
            interaction_type: Type of interaction (click, command, etc.)
            target: Target of interaction (app, UI element, etc.)
            context: Context of interaction
            success: Whether interaction was successful
            duration: Duration of interaction (if applicable)
            
        Returns:
            Interaction ID
        """
        # Generate interaction ID
        interaction_id = f"int_{int(time.time())}_{hash(interaction_type + target) % 10000}"
        
        # Create interaction object
        interaction = UserInteraction(
            interaction_id=interaction_id,
            timestamp=time.time(),
            interaction_type=interaction_type,
            target=target,
            context=context or {},
            success=success,
            duration=duration
        )
        
        # Add to interactions list
        self.interactions.append(interaction)
        
        # Add to sequence buffer
        self.sequence_buffer.append(interaction)
        if len(self.sequence_buffer) > 20:  # Limit buffer size
            self.sequence_buffer.pop(0)
            
        # Save interaction
        self._save_interaction(interaction)
        
        # Update daily stats
        self._update_daily_stats(interaction)
        
        # Analyze for patterns (periodically)
        if len(self.interactions) % 10 == 0:
            self.analyze_patterns()
            
        return interaction_id
    
    def _update_daily_stats(self, interaction: UserInteraction) -> None:
        """Update daily statistics for interactions."""
        date_key = datetime.fromtimestamp(interaction.timestamp).strftime("%Y-%m-%d")
        
        if date_key not in self.daily_stats:
            self.daily_stats[date_key] = {
                "interaction_count": 0,
                "interaction_types": Counter(),
                "targets": Counter(),
                "success_rate": 0,
                "success_count": 0,
                "hourly_activity": [0] * 24
            }
            
        stats = self.daily_stats[date_key]
        stats["interaction_count"] += 1
        stats["interaction_types"][interaction.interaction_type] += 1
        stats["targets"][interaction.target] += 1
        
        if interaction.success:
            stats["success_count"] += 1
        
        stats["success_rate"] = stats["success_count"] / stats["interaction_count"]
        
        # Update hourly activity
        hour = datetime.fromtimestamp(interaction.timestamp).hour
        stats["hourly_activity"][hour] += 1
    
    def analyze_patterns(self) -> None:
        """Analyze interactions for patterns."""
        if not self.interactions:
            return
            
        # Analyze different types of patterns
        self._analyze_time_patterns()
        self._analyze_sequence_patterns()
        self._analyze_app_preferences()
        self._analyze_ui_interaction_patterns()
        
        # Save updated patterns
        self._save_patterns()
        
        logger.info(f"Analyzed patterns - now tracking {len(self.recognized_patterns)} patterns")
    
    def _analyze_time_patterns(self) -> None:
        """Analyze time-based patterns in user interactions."""
        # Skip if not enough data
        if len(self.daily_stats) < 3:
            return
            
        # Analyze hourly activity patterns
        hourly_totals = [0] * 24
        for date_stats in self.daily_stats.values():
            for hour, count in enumerate(date_stats["hourly_activity"]):
                hourly_totals[hour] += count
                
        # Find peak activity hours
        total_interactions = sum(hourly_totals)
        if total_interactions == 0:
            return
            
        hourly_percentages = [count / total_interactions for count in hourly_totals]
        
        # Identify peak hours (hours with above-average activity)
        avg_hourly = total_interactions / 24
        peak_hours = []
        
        for hour, count in enumerate(hourly_totals):
            if count > avg_hourly * 1.5:  # 50% above average
                peak_hours.append(hour)
                
        if peak_hours:
            # Create or update pattern
            pattern_id = f"time_peak_hours"
            
            if pattern_id in self.recognized_patterns:
                pattern = self.recognized_patterns[pattern_id]
                pattern.elements = peak_hours
                pattern.last_seen = time.time()
                pattern.frequency += 1
                pattern.confidence = min(0.95, pattern.confidence + 0.05)
            else:
                pattern = RecognizedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.TIME_BASED,
                    description=f"User has peak activity during hours: {', '.join(str(h) for h in peak_hours)}",
                    confidence=0.7,
                    frequency=1,
                    last_seen=time.time(),
                    first_seen=time.time(),
                    elements=peak_hours,
                    metadata={"hourly_percentages": hourly_percentages}
                )
                
            self.recognized_patterns[pattern_id] = pattern
    
    def _analyze_app_preferences(self) -> None:
        """Analyze app usage preferences."""
        # Focus on application launch/focus interactions
        app_interactions = [i for i in self.interactions 
                          if i.interaction_type in ["app_launch", "app_focus", "app_use"]]
        
        if len(app_interactions) < 5:
            return
            
        # Count app usage
        app_counts = Counter([i.target for i in app_interactions])
        total_apps = sum(app_counts.values())
        
        # Find frequently used apps (used more than 15% of the time)
        frequent_apps = [(app, count) for app, count in app_counts.most_common(5)
                        if count / total_apps >= 0.15]
        
        if frequent_apps:
            # Create or update pattern
            pattern_id = "app_preferences"
            
            if pattern_id in self.recognized_patterns:
                pattern = self.recognized_patterns[pattern_id]
                pattern.elements = frequent_apps
                pattern.last_seen = time.time()
                pattern.frequency += 1
                pattern.confidence = min(0.95, pattern.confidence + 0.05)
                pattern.description = f"User frequently uses: {', '.join(app for app, _ in frequent_apps)}"
            else:
                pattern = RecognizedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.APP_PREFERENCE,
                    description=f"User frequently uses: {', '.join(app for app, _ in frequent_apps)}",
                    confidence=0.7,
                    frequency=1,
                    last_seen=time.time(),
                    first_seen=time.time(),
                    elements=frequent_apps,
                    metadata={"app_counts": dict(app_counts)}
                )
                
            self.recognized_patterns[pattern_id] = pattern
    
    def _analyze_sequence_patterns(self) -> None:
        """Analyze sequential patterns in interactions."""
        # Need at least 10 interactions to analyze sequences
        if len(self.interactions) < 10:
            return
            
        # Look for common sequences of length 3-5
        for seq_length in range(3, 6):
            if len(self.interactions) < seq_length * 2:
                continue
                
            sequences = []
            for i in range(len(self.interactions) - seq_length + 1):
                seq = [(inter.interaction_type, inter.target) for inter in self.interactions[i:i+seq_length]]
                sequences.append(seq)
                
            # Count sequence occurrences
            seq_counter = Counter([str(seq) for seq in sequences])
            
            # Find frequent sequences (occurring at least 3 times)
            for seq_str, count in seq_counter.items():
                if count >= 3:
                    # This is a potential pattern
                    pattern_id = f"seq_{hash(seq_str) % 10000}"
                    
                    if pattern_id in self.recognized_patterns:
                        pattern = self.recognized_patterns[pattern_id]
                        pattern.last_seen = time.time()
                        pattern.frequency += 1
                        pattern.confidence = min(0.95, pattern.confidence + 0.05)
                    else:
                        # Parse back the sequence from string representation
                        # (simplified here - would need proper deserialization)
                        seq_elements = eval(seq_str)
                        
                        # Create a human-readable description
                        desc_parts = []
                        for inter_type, target in seq_elements:
                            desc_parts.append(f"{inter_type} {target}")
                            
                        description = f"User often performs sequence: {' → '.join(desc_parts)}"
                        
                        pattern = RecognizedPattern(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.SEQUENCE,
                            description=description,
                            confidence=0.7,
                            frequency=count,
                            last_seen=time.time(),
                            first_seen=time.time(),
                            elements=seq_elements,
                            metadata={"sequence_length": seq_length}
                        )
                        
                    self.recognized_patterns[pattern_id] = pattern
    
    def _analyze_ui_interaction_patterns(self) -> None:
        """Analyze UI interaction patterns."""
        # Focus on UI interactions
        ui_interactions = [i for i in self.interactions 
                         if i.interaction_type in ["click", "drag", "menu_select", "keyboard_shortcut"]]
        
        if len(ui_interactions) < 10:
            return
            
        # Analyze common UI elements interacted with
        ui_elements = Counter([i.target for i in ui_interactions])
        
        # Find elements with multiple interactions (used at least 5 times)
        frequent_elements = [(element, count) for element, count in ui_elements.most_common(10)
                           if count >= 5]
        
        if frequent_elements:
            # Create or update pattern
            pattern_id = "ui_frequent_elements"
            
            if pattern_id in self.recognized_patterns:
                pattern = self.recognized_patterns[pattern_id]
                pattern.elements = frequent_elements
                pattern.last_seen = time.time()
                pattern.frequency += 1
                pattern.confidence = min(0.95, pattern.confidence + 0.05)
                pattern.description = f"User frequently interacts with: {', '.join(el for el, _ in frequent_elements[:3])}"
            else:
                pattern = RecognizedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.UI_INTERACTION,
                    description=f"User frequently interacts with: {', '.join(el for el, _ in frequent_elements[:3])}",
                    confidence=0.7,
                    frequency=1,
                    last_seen=time.time(),
                    first_seen=time.time(),
                    elements=frequent_elements,
                    metadata={"ui_element_counts": dict(ui_elements)}
                )
                
            self.recognized_patterns[pattern_id] = pattern
            
        # Analyze shortcut usage
        shortcuts = [i for i in ui_interactions if i.interaction_type == "keyboard_shortcut"]
        if len(shortcuts) > 5:
            shortcut_usage = len(shortcuts) / len(ui_interactions)
            
            pattern_id = "ui_shortcut_preference"
            
            if pattern_id in self.recognized_patterns:
                pattern = self.recognized_patterns[pattern_id]
                pattern.metadata["shortcut_usage"] = shortcut_usage
                pattern.last_seen = time.time()
                pattern.frequency += 1
            else:
                pattern = RecognizedPattern(
                    pattern_id=pattern_id,
                    pattern_type=PatternType.UI_INTERACTION,
                    description=f"User prefers keyboard shortcuts ({shortcut_usage:.0%} of UI interactions)",
                    confidence=0.7,
                    frequency=1,
                    last_seen=time.time(),
                    first_seen=time.time(),
                    elements=[],
                    metadata={"shortcut_usage": shortcut_usage}
                )
                
            self.recognized_patterns[pattern_id] = pattern
    
    async def optimize_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest behavior optimizations based on recognized patterns.
        
        Args:
            context: Current context (app, task, etc.)
            
        Returns:
            Dictionary of suggested optimizations
        """
        optimizations = {
            "preferred_apps": [],
            "ui_preferences": {},
            "timing_suggestions": {},
            "workflow_optimizations": []
        }
        
        # Consider all patterns with sufficient confidence
        relevant_patterns = [p for p in self.recognized_patterns.values() 
                           if p.confidence >= self.min_confidence]
        
        # Preferred apps
        app_patterns = [p for p in relevant_patterns if p.pattern_type == PatternType.APP_PREFERENCE]
        if app_patterns:
            for pattern in app_patterns:
                for app, _ in pattern.elements:
                    if app not in optimizations["preferred_apps"]:
                        optimizations["preferred_apps"].append(app)
        
        # UI preferences
        ui_patterns = [p for p in relevant_patterns if p.pattern_type == PatternType.UI_INTERACTION]
        for pattern in ui_patterns:
            if pattern.pattern_id == "ui_shortcut_preference":
                shortcut_usage = pattern.metadata.get("shortcut_usage", 0)
                if shortcut_usage > 0.3:  # 30% or more
                    optimizations["ui_preferences"]["prefer_shortcuts"] = True
            
            if pattern.pattern_id == "ui_frequent_elements":
                optimizations["ui_preferences"]["frequent_elements"] = [el for el, _ in pattern.elements[:5]]
        
        # Timing suggestions
        time_patterns = [p for p in relevant_patterns if p.pattern_type == PatternType.TIME_BASED]
        for pattern in time_patterns:
            if pattern.pattern_id == "time_peak_hours":
                optimizations["timing_suggestions"]["peak_hours"] = pattern.elements
                
                # Check if current time is in peak hours
                current_hour = datetime.now().hour
                if current_hour in pattern.elements:
                    optimizations["timing_suggestions"]["is_peak_time"] = True
                else:
                    optimizations["timing_suggestions"]["is_peak_time"] = False
        
        # Workflow optimizations from sequence patterns
        seq_patterns = [p for p in relevant_patterns if p.pattern_type == PatternType.SEQUENCE]
        for pattern in seq_patterns:
            # Check if the current context matches the start of a sequence
            if context.get("last_interaction_type") and context.get("last_target"):
                seq_start = pattern.elements[0]
                if seq_start[0] == context["last_interaction_type"] and seq_start[1] == context["last_target"]:
                    # Suggest the next steps in the sequence
                    next_steps = []
                    for i in range(1, len(pattern.elements)):
                        step_type, step_target = pattern.elements[i]
                        next_steps.append({
                            "type": step_type,
                            "target": step_target,
                            "confidence": pattern.confidence
                        })
                    
                    if next_steps:
                        optimizations["workflow_optimizations"].append({
                            "pattern_id": pattern.pattern_id,
                            "description": pattern.description,
                            "next_steps": next_steps
                        })
        
        # If few patterns recognized, use LLM to analyze and suggest
        if len(relevant_patterns) < 3 and len(self.interactions) > 20:
            llm_suggestions = await self._get_llm_optimization_suggestions(context)
            if llm_suggestions:
                optimizations["llm_suggestions"] = llm_suggestions
        
        return optimizations
    
    async def _get_llm_optimization_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimization suggestions from LLM based on interaction history."""
        # Prepare recent interactions for analysis
        recent_interactions = self.interactions[-30:]  # Last 30 interactions
        
        interactions_data = []
        for interaction in recent_interactions:
            interactions_data.append({
                "type": interaction.interaction_type,
                "target": interaction.target,
                "timestamp": interaction.timestamp,
                "success": interaction.success
            })
        
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=500,
            temperature=0.2
        )
        
        # Build messages for analysis
        messages = [
            {
                "role": "system",
                "content": """You analyze user interaction patterns and suggest optimizations.
Based on the recent interaction history, identify:
1. Common workflows that could be optimized
2. UI preferences that could be accommodated
3. App usage patterns that could be leveraged

Return a JSON array with optimization suggestions.
"""
            },
            {
                "role": "user",
                "content": f"""Analyze these recent user interactions and suggest optimizations:
{json.dumps(interactions_data, indent=2)}

Current context:
{json.dumps(context, indent=2)}
"""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if isinstance(result, list):
                return result
            elif isinstance(result, dict) and "suggestions" in result:
                return result["suggestions"]
                
        except Exception as e:
            logger.error(f"Error getting LLM optimization suggestions: {e}")
            
        return []
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text."""
        import re
        import json
        
        # Try to find JSON in the text
        json_match = re.search(r'```json\n(.*?)\n```|```(.*?)```|\{.*\}|\[.*\]', text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1) or json_match.group(2) or json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Error parsing JSON: {json_str}")
                
        # Try with a more aggressive approach - assume the entire text is JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not extract JSON from text")
            return None
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[RecognizedPattern]:
        """
        Get a pattern by ID.
        
        Args:
            pattern_id: Pattern ID
            
        Returns:
            RecognizedPattern or None if not found
        """
        return self.recognized_patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[RecognizedPattern]:
        """
        Get patterns by type.
        
        Args:
            pattern_type: Type of patterns to retrieve
            
        Returns:
            List of patterns of the specified type
        """
        return [p for p in self.recognized_patterns.values() if p.pattern_type == pattern_type]
    
    def clear_outdated_patterns(self, max_age_days: int = 30) -> int:
        """
        Clear outdated patterns.
        
        Args:
            max_age_days: Maximum age in days for patterns
            
        Returns:
            Number of patterns cleared
        """
        cutoff = time.time() - (max_age_days * 86400)
        
        outdated_patterns = []
        for pattern_id, pattern in self.recognized_patterns.items():
            if pattern.last_seen < cutoff:
                outdated_patterns.append(pattern_id)
                
        # Remove outdated patterns
        for pattern_id in outdated_patterns:
            del self.recognized_patterns[pattern_id]
            
        # Save updated patterns
        if outdated_patterns:
            self._save_patterns()
            
        return len(outdated_patterns)
    
    def get_usage_insights(self) -> Dict[str, Any]:
        """
        Get usage insights based on interactions and patterns.
        
        Returns:
            Dictionary of usage insights
        """
        insights = {
            "total_interactions": len(self.interactions),
            "active_days": len(self.daily_stats),
            "recognized_patterns": len(self.recognized_patterns),
            "top_apps": [],
            "top_interaction_types": [],
            "usage_by_hour": [0] * 24,
            "success_rate": 0.0
        }
        
        if not self.interactions:
            return insights
            
        # Calculate overall success rate
        success_count = sum(1 for i in self.interactions if i.success)
        insights["success_rate"] = success_count / len(self.interactions)
        
        # Calculate app usage
        app_targets = [i.target for i in self.interactions 
                    if i.interaction_type in ["app_launch", "app_focus", "app_use"]]
        app_counter = Counter(app_targets)
        insights["top_apps"] = app_counter.most_common(5)
        
        # Calculate interaction types
        interaction_types = Counter([i.interaction_type for i in self.interactions])
        insights["top_interaction_types"] = interaction_types.most_common(5)
        
        # Calculate usage by hour
        for interaction in self.interactions:
            hour = datetime.fromtimestamp(interaction.timestamp).hour
            insights["usage_by_hour"][hour] += 1
            
        return insights
