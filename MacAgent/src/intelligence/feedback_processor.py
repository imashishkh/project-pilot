"""
Feedback Processor Module

This module provides tools for processing user feedback, tracking success/failure metrics,
and adapting agent behavior based on past experiences.
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

# Import LLM connector for analysis
from .llm_connector import LLMConnector, LLMProvider, ModelConfig, LLMResponse

# Configure logging
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback for actions."""
    SUCCESS = "success"        # Action completed successfully
    PARTIAL = "partial"        # Action partially succeeded
    FAILURE = "failure"        # Action failed completely
    USER_CORRECTION = "user_correction"  # User had to correct the action
    USER_OVERRIDE = "user_override"      # User overrode the action
    USER_APPROVAL = "user_approval"      # User explicitly approved result


@dataclass
class ActionFeedback:
    """Feedback data for a specific action."""
    action_id: str             # Unique identifier for the action
    action_type: str           # Type of action (navigation, search, etc.)
    feedback_type: FeedbackType  # Success, failure, etc.
    timestamp: float = field(default_factory=time.time)  # When feedback was recorded
    context: Dict[str, Any] = field(default_factory=dict)  # Context of the action
    metrics: Dict[str, float] = field(default_factory=dict)  # Any associated metrics
    user_provided: bool = False  # Whether feedback was explicitly given by user
    notes: str = ""            # Additional notes about the feedback


@dataclass
class FeedbackPatternMatch:
    """A pattern identified across multiple feedback instances."""
    pattern_id: str            # Unique identifier for the pattern
    action_type: str           # Type of action this pattern applies to
    context_pattern: Dict[str, Any]  # Pattern in the context
    success_rate: float        # Rate of success when pattern is present
    confidence: float          # Confidence in this pattern
    feedback_count: int        # Number of feedback instances supporting this pattern
    last_updated: float        # When the pattern was last updated
    recommendation: str = ""   # Generated recommendation based on pattern


class FeedbackProcessor:
    """
    Processes feedback about actions, learns patterns, and adapts behavior.
    
    Features:
    - Stores and retrieves action feedback
    - Analyzes success/failure patterns
    - Generates recommendations for future actions
    - Adapts to user preferences over time
    - Provides metrics on action performance
    """
    
    def __init__(self,
                llm_connector: LLMConnector,
                default_model: str = "gpt-4",
                default_provider: LLMProvider = LLMProvider.OPENAI,
                feedback_dir: str = "data/feedback",
                patterns_file: str = "patterns.json",
                learning_rate: float = 0.1,
                min_confidence_threshold: float = 0.6):
        """
        Initialize the feedback processor.
        
        Args:
            llm_connector: LLM connector for pattern analysis
            default_model: Default model for analysis
            default_provider: Default provider for analysis
            feedback_dir: Directory to store feedback data
            patterns_file: File to store learned patterns
            learning_rate: Rate at which to update patterns (0-1)
            min_confidence_threshold: Minimum confidence for recommendation
        """
        self.llm_connector = llm_connector
        self.default_model = default_model
        self.default_provider = default_provider
        self.feedback_dir = feedback_dir
        self.patterns_file = os.path.join(feedback_dir, patterns_file)
        self.learning_rate = learning_rate
        self.min_confidence_threshold = min_confidence_threshold
        
        # Ensure feedback directory exists
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Load feedback history and patterns
        self.feedback_history: Dict[str, ActionFeedback] = {}  # action_id -> feedback
        self.feedback_by_type: Dict[str, List[str]] = {}  # action_type -> [action_id]
        self.patterns: Dict[str, FeedbackPatternMatch] = {}  # pattern_id -> pattern
        
        self._load_feedback_history()
        self._load_patterns()
        
        logger.info(f"FeedbackProcessor initialized with {len(self.feedback_history)} feedback entries and {len(self.patterns)} patterns")
    
    def _load_feedback_history(self) -> None:
        """Load feedback history from files."""
        try:
            # Load all json files in the feedback directory
            for filename in os.listdir(self.feedback_dir):
                if filename.endswith(".json") and not filename == os.path.basename(self.patterns_file):
                    file_path = os.path.join(self.feedback_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            
                            # Process each feedback entry
                            for entry_data in data:
                                try:
                                    # Convert feedback_type from string to enum
                                    entry_data["feedback_type"] = FeedbackType(entry_data["feedback_type"])
                                    
                                    # Create feedback object
                                    feedback = ActionFeedback(**entry_data)
                                    
                                    # Add to history
                                    self.feedback_history[feedback.action_id] = feedback
                                    
                                    # Add to by_type index
                                    if feedback.action_type not in self.feedback_by_type:
                                        self.feedback_by_type[feedback.action_type] = []
                                    self.feedback_by_type[feedback.action_type].append(feedback.action_id)
                                    
                                except Exception as e:
                                    logger.warning(f"Error processing feedback entry: {e}")
                    except Exception as e:
                        logger.warning(f"Error loading feedback file {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading feedback history: {e}")
    
    def _load_patterns(self) -> None:
        """Load feedback patterns from file."""
        if not os.path.exists(self.patterns_file):
            return
        
        try:
            with open(self.patterns_file, "r") as f:
                data = json.load(f)
                
                for pattern_data in data:
                    pattern_id = pattern_data.pop("pattern_id", None)
                    if pattern_id:
                        self.patterns[pattern_id] = FeedbackPatternMatch(**pattern_data, pattern_id=pattern_id)
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
    
    def _save_patterns(self) -> None:
        """Save feedback patterns to file."""
        try:
            with open(self.patterns_file, "w") as f:
                # Convert patterns to dict
                pattern_list = []
                for pattern_id, pattern in self.patterns.items():
                    pattern_dict = {
                        "pattern_id": pattern.pattern_id,
                        "action_type": pattern.action_type,
                        "context_pattern": pattern.context_pattern,
                        "success_rate": pattern.success_rate,
                        "confidence": pattern.confidence,
                        "feedback_count": pattern.feedback_count,
                        "last_updated": pattern.last_updated,
                        "recommendation": pattern.recommendation
                    }
                    pattern_list.append(pattern_dict)
                
                json.dump(pattern_list, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _save_feedback(self, feedback: ActionFeedback) -> None:
        """Save feedback to file."""
        try:
            # Determine file based on month
            date = datetime.fromtimestamp(feedback.timestamp)
            filename = f"feedback_{date.year}_{date.month}.json"
            file_path = os.path.join(self.feedback_dir, filename)
            
            # Load existing data
            data = []
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
            
            # Convert enum to string for JSON
            feedback_dict = {
                "action_id": feedback.action_id,
                "action_type": feedback.action_type,
                "feedback_type": feedback.feedback_type.value,
                "timestamp": feedback.timestamp,
                "context": feedback.context,
                "metrics": feedback.metrics,
                "user_provided": feedback.user_provided,
                "notes": feedback.notes
            }
            
            # Add new feedback
            data.append(feedback_dict)
            
            # Save
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    async def record_feedback(self, feedback: ActionFeedback) -> None:
        """
        Record feedback for an action.
        
        Args:
            feedback: Feedback to record
        """
        try:
            # Add to history
            self.feedback_history[feedback.action_id] = feedback
            
            # Add to by_type index
            if feedback.action_type not in self.feedback_by_type:
                self.feedback_by_type[feedback.action_type] = []
            self.feedback_by_type[feedback.action_type].append(feedback.action_id)
            
            # Save to file
            self._save_feedback(feedback)
            
            # If sufficient feedback has been collected, update patterns
            if len(self.feedback_history) % 10 == 0:
                await self.update_patterns()
            
            logger.info(f"Recorded {feedback.feedback_type.value} feedback for action {feedback.action_id}")
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    def get_feedback(self, action_id: str) -> Optional[ActionFeedback]:
        """
        Get feedback for a specific action.
        
        Args:
            action_id: ID of the action
            
        Returns:
            Feedback for the action, or None if not found
        """
        return self.feedback_history.get(action_id)
    
    def get_feedback_by_type(self, action_type: str, limit: int = 50) -> List[ActionFeedback]:
        """
        Get feedback for a specific action type.
        
        Args:
            action_type: Type of action
            limit: Maximum number of results
            
        Returns:
            List of feedback for the action type
        """
        feedback_list = []
        
        action_ids = self.feedback_by_type.get(action_type, [])
        for action_id in action_ids[-limit:]:  # Get most recent
            feedback = self.feedback_history.get(action_id)
            if feedback:
                feedback_list.append(feedback)
        
        return feedback_list
    
    def get_success_rate(self, action_type: str, time_window: Optional[timedelta] = None) -> float:
        """
        Get success rate for a specific action type.
        
        Args:
            action_type: Type of action
            time_window: Optional time window to consider
            
        Returns:
            Success rate as a float from 0 to 1
        """
        feedback_list = self.get_feedback_by_type(action_type, limit=100)
        
        if not feedback_list:
            return 0.0
        
        # Filter by time window if specified
        if time_window:
            cutoff = time.time() - time_window.total_seconds()
            feedback_list = [f for f in feedback_list if f.timestamp >= cutoff]
        
        if not feedback_list:
            return 0.0
        
        # Count successes
        success_count = sum(1 for f in feedback_list 
                          if f.feedback_type in [FeedbackType.SUCCESS, FeedbackType.USER_APPROVAL])
        
        # Count partials as 0.5 success
        partial_count = sum(0.5 for f in feedback_list 
                          if f.feedback_type == FeedbackType.PARTIAL)
        
        return (success_count + partial_count) / len(feedback_list)
    
    async def update_patterns(self) -> None:
        """Update pattern recognition based on feedback history."""
        logger.info("Updating feedback patterns")
        
        # Group feedback by action type
        for action_type, action_ids in self.feedback_by_type.items():
            # Only process if we have enough feedback
            if len(action_ids) < 5:
                continue
            
            # Get feedback for this action type
            feedback_list = [self.feedback_history[action_id] for action_id in action_ids 
                            if action_id in self.feedback_history]
            
            # Analyze patterns with LLM
            await self._analyze_patterns(action_type, feedback_list)
        
        # Save updated patterns
        self._save_patterns()
    
    async def _analyze_patterns(self, action_type: str, feedback_list: List[ActionFeedback]) -> None:
        """
        Analyze patterns in feedback for an action type using LLM.
        
        Args:
            action_type: Type of action
            feedback_list: List of feedback for the action type
        """
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=800,
            temperature=0.0
        )
        
        # Prepare feedback data for analysis
        feedback_data = []
        for feedback in feedback_list[-20:]:  # Limit to most recent 20
            feedback_data.append({
                "action_id": feedback.action_id,
                "feedback_type": feedback.feedback_type.value,
                "context": feedback.context,
                "user_provided": feedback.user_provided,
                "timestamp": feedback.timestamp
            })
        
        # Build messages for pattern analysis
        messages = [
            {
                "role": "system",
                "content": """You analyze patterns in user feedback to identify context factors that 
correlate with success or failure of actions. 

Identify 2-3 specific patterns in the context that correlate with different outcomes.
For each pattern, specify:
1. A clear description of the context pattern
2. The estimated success rate when this pattern is present
3. A specific recommendation for handling this context in the future

Return a JSON array of pattern objects.
"""
            },
            {
                "role": "user",
                "content": f"""Analyze these feedback entries for action type '{action_type}' and identify patterns:
{json.dumps(feedback_data, indent=2)}"""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if result and isinstance(result, list):
                # Process each pattern
                for pattern_data in result:
                    # Generate a unique ID for this pattern
                    pattern_id = f"{action_type}_{hash(json.dumps(pattern_data.get('context_pattern', {}), sort_keys=True)) % 10000}"
                    
                    # Check if pattern already exists
                    if pattern_id in self.patterns:
                        # Update existing pattern
                        existing = self.patterns[pattern_id]
                        
                        # Update with learning rate
                        existing.success_rate = (1 - self.learning_rate) * existing.success_rate + \
                                               self.learning_rate * pattern_data.get("success_rate", 0.0)
                        existing.confidence = min(1.0, existing.confidence + 0.1)
                        existing.feedback_count += 1
                        existing.last_updated = time.time()
                        
                        # Update recommendation if confidence is high
                        if pattern_data.get("recommendation") and existing.confidence > 0.7:
                            existing.recommendation = pattern_data.get("recommendation")
                    else:
                        # Create new pattern
                        self.patterns[pattern_id] = FeedbackPatternMatch(
                            pattern_id=pattern_id,
                            action_type=action_type,
                            context_pattern=pattern_data.get("context_pattern", {}),
                            success_rate=pattern_data.get("success_rate", 0.0),
                            confidence=0.5,  # Start with moderate confidence
                            feedback_count=1,
                            last_updated=time.time(),
                            recommendation=pattern_data.get("recommendation", "")
                        )
                    
                logger.info(f"Updated {len(result)} patterns for action type {action_type}")
                        
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
    
    def _extract_json(self, text: str) -> Any:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON or None
        """
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
    
    def get_recommendations(self, action_type: str, context: Dict[str, Any]) -> List[str]:
        """
        Get recommendations for handling a specific action type and context.
        
        Args:
            action_type: Type of action
            context: Context of the action
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Find patterns that match this action type
        matching_patterns = [p for p in self.patterns.values() 
                            if p.action_type == action_type 
                            and p.confidence >= self.min_confidence_threshold]
        
        # Sort by confidence
        matching_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Check context against patterns
        for pattern in matching_patterns:
            # Simple pattern matching logic - can be made more sophisticated
            match = True
            for key, value in pattern.context_pattern.items():
                if key not in context or context[key] != value:
                    match = False
                    break
            
            if match and pattern.recommendation:
                recommendations.append(pattern.recommendation)
        
        return recommendations[:3]  # Limit to top 3
    
    def get_pattern_matching_score(self, action_type: str, context: Dict[str, Any]) -> float:
        """
        Get a score predicting success likelihood based on patterns.
        
        Args:
            action_type: Type of action
            context: Context of the action
            
        Returns:
            Predicted success score from 0 to 1
        """
        matching_scores = []
        
        # Find patterns that match this action type
        matching_patterns = [p for p in self.patterns.values() 
                          if p.action_type == action_type]
        
        # No patterns, return 0.5 (neutral)
        if not matching_patterns:
            return 0.5
        
        # Check context against patterns
        for pattern in matching_patterns:
            # Simple pattern matching logic - can be made more sophisticated
            match_quality = 0.0
            
            total_keys = len(pattern.context_pattern)
            if total_keys == 0:
                continue
                
            matching_keys = 0
            for key, value in pattern.context_pattern.items():
                if key in context and context[key] == value:
                    matching_keys += 1
            
            # Calculate match quality
            match_quality = matching_keys / total_keys
            
            # If we have a partial match, add weighted score
            if match_quality > 0:
                weight = match_quality * pattern.confidence
                matching_scores.append((weight, pattern.success_rate))
        
        # No matching patterns, return overall success rate for this action type
        if not matching_scores:
            return self.get_success_rate(action_type)
        
        # Calculate weighted average
        total_weight = sum(weight for weight, _ in matching_scores)
        if total_weight == 0:
            return 0.5
            
        weighted_score = sum(weight * score for weight, score in matching_scores) / total_weight
        
        return weighted_score
    
    def generate_summary_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Generate a summary report of feedback patterns.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Summary report as a dictionary
        """
        # Calculate timeframe
        cutoff = time.time() - (days * 86400)
        
        # Initialize report
        report = {
            "time_period": f"Last {days} days",
            "total_actions": 0,
            "success_rate": 0.0,
            "action_types": {},
            "top_patterns": [],
            "user_corrections": 0
        }
        
        # Get recent feedback
        recent_feedback = [f for f in self.feedback_history.values() if f.timestamp >= cutoff]
        
        if not recent_feedback:
            return report
        
        # Calculate overall stats
        report["total_actions"] = len(recent_feedback)
        
        success_count = sum(1 for f in recent_feedback 
                          if f.feedback_type in [FeedbackType.SUCCESS, FeedbackType.USER_APPROVAL])
        partial_count = sum(0.5 for f in recent_feedback 
                          if f.feedback_type == FeedbackType.PARTIAL)
        
        report["success_rate"] = (success_count + partial_count) / len(recent_feedback)
        
        # Count user corrections
        report["user_corrections"] = sum(1 for f in recent_feedback 
                                      if f.feedback_type == FeedbackType.USER_CORRECTION)
        
        # Breakdown by action type
        action_types = {}
        for feedback in recent_feedback:
            if feedback.action_type not in action_types:
                action_types[feedback.action_type] = {
                    "count": 0,
                    "success_count": 0,
                    "partial_count": 0
                }
            
            action_types[feedback.action_type]["count"] += 1
            
            if feedback.feedback_type in [FeedbackType.SUCCESS, FeedbackType.USER_APPROVAL]:
                action_types[feedback.action_type]["success_count"] += 1
            elif feedback.feedback_type == FeedbackType.PARTIAL:
                action_types[feedback.action_type]["partial_count"] += 0.5
        
        # Calculate success rates by action type
        for action_type, stats in action_types.items():
            success_rate = 0.0
            if stats["count"] > 0:
                success_rate = (stats["success_count"] + stats["partial_count"]) / stats["count"]
            
            report["action_types"][action_type] = {
                "count": stats["count"],
                "success_rate": success_rate
            }
        
        # Top patterns
        patterns = [p for p in self.patterns.values() 
                  if p.confidence >= self.min_confidence_threshold]
        patterns.sort(key=lambda p: p.confidence * p.feedback_count, reverse=True)
        
        for pattern in patterns[:5]:  # Top 5
            report["top_patterns"].append({
                "action_type": pattern.action_type,
                "context_pattern": pattern.context_pattern,
                "success_rate": pattern.success_rate,
                "confidence": pattern.confidence,
                "recommendation": pattern.recommendation
            })
        
        return report
    
    def clear_old_feedback(self, days: int = 90) -> int:
        """
        Clear feedback older than the specified number of days.
        
        Args:
            days: Age in days to clear
            
        Returns:
            Number of feedback items cleared
        """
        cutoff = time.time() - (days * 86400)
        
        # Find old feedback
        old_feedback_ids = [action_id for action_id, feedback in self.feedback_history.items() 
                          if feedback.timestamp < cutoff]
        
        # Remove from history
        for action_id in old_feedback_ids:
            feedback = self.feedback_history.pop(action_id, None)
            if feedback and feedback.action_type in self.feedback_by_type:
                if action_id in self.feedback_by_type[feedback.action_type]:
                    self.feedback_by_type[feedback.action_type].remove(action_id)
        
        logger.info(f"Cleared {len(old_feedback_ids)} old feedback entries")
        return len(old_feedback_ids)
