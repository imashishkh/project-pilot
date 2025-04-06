"""
Decision Making Module for MacAgent.

This module provides decision-making capabilities for the MacAgent, including:
- Evaluating multiple options based on specified criteria
- Selecting optimal choices for implementation
- Handling uncertain or partial information
- Providing justification for decisions
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


class DecisionCriteria(enum.Enum):
    """Types of criteria used in decision making."""
    EFFICIENCY = "efficiency"  # How efficient the solution is
    RELIABILITY = "reliability"  # How reliable the solution is
    SIMPLICITY = "simplicity"  # How simple the solution is
    FLEXIBILITY = "flexibility"  # How adaptable the solution is
    COST = "cost"  # Resource cost (computational, memory, etc.)
    USER_PREFERENCE = "user_preference"  # Alignment with user preferences
    SAFETY = "safety"  # Safety and security considerations
    SPEED = "speed"  # How fast the solution is
    PRECISION = "precision"  # How accurate or precise the solution is
    MAINTAINABILITY = "maintainability"  # Ease of maintenance and updates


@dataclass
class Option:
    """Represents a single option in a decision."""
    id: str  # Unique identifier for the option
    name: str  # Name of the option
    description: str  # Detailed description
    pros: List[str] = field(default_factory=list)  # Advantages
    cons: List[str] = field(default_factory=list)  # Disadvantages
    criteria_scores: Dict[str, float] = field(default_factory=dict)  # Scores for each criterion (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert option to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Option':
        """Create option from dictionary."""
        return cls(**data)


@dataclass
class Decision:
    """Represents a decision with multiple options."""
    id: str  # Unique identifier for the decision
    question: str  # The decision question
    context: Dict[str, Any]  # Context information for the decision
    options: Dict[str, Option] = field(default_factory=dict)  # Available options
    criteria: List[DecisionCriteria] = field(default_factory=list)  # Criteria for evaluation
    criteria_weights: Dict[str, float] = field(default_factory=dict)  # Weight for each criterion (0-1)
    selected_option_id: Optional[str] = None  # ID of the selected option
    justification: str = ""  # Justification for the selection
    created_at: float = field(default_factory=time.time)  # Creation timestamp
    decided_at: Optional[float] = None  # When the decision was made
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "options": {option_id: option.to_dict() for option_id, option in self.options.items()},
            "criteria": [criterion.value for criterion in self.criteria],
            "criteria_weights": self.criteria_weights,
            "selected_option_id": self.selected_option_id,
            "justification": self.justification,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Decision':
        """Create decision from dictionary."""
        # Convert criteria strings to enum
        criteria = [DecisionCriteria(c) for c in data.get("criteria", [])]
        
        # Convert options dict to Option objects
        options_dict = data.pop("options", {})
        
        # Create the decision
        data["criteria"] = criteria
        decision = cls(**data)
        
        # Add options
        decision.options = {
            option_id: Option.from_dict(option_data) 
            for option_id, option_data in options_dict.items()
        }
        
        return decision
    
    def calculate_option_scores(self) -> Dict[str, float]:
        """
        Calculate weighted scores for each option based on criteria.
        
        Returns:
            Dictionary mapping option IDs to their weighted scores
        """
        scores = {}
        
        for option_id, option in self.options.items():
            weighted_score = 0.0
            total_weight = 0.0
            
            for criterion in self.criteria:
                criterion_name = criterion.value
                if criterion_name in option.criteria_scores:
                    weight = self.criteria_weights.get(criterion_name, 1.0)
                    weighted_score += option.criteria_scores[criterion_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                scores[option_id] = weighted_score / total_weight
            else:
                scores[option_id] = 0.0
        
        return scores
    
    def get_best_option(self) -> Optional[Option]:
        """
        Get the option with the highest score.
        
        Returns:
            The best option, or None if no options
        """
        if not self.options:
            return None
        
        scores = self.calculate_option_scores()
        if not scores:
            return None
        
        best_option_id = max(scores, key=scores.get)
        return self.options.get(best_option_id)


class DecisionMaker:
    """
    Decision Maker for MacAgent system.
    
    Evaluates options and makes decisions based on specified criteria.
    """
    
    def __init__(
        self,
        llm_connector: LLMConnector,
        prompt_manager: PromptManager,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        default_model: str = "gpt-3.5-turbo",
        decisions_dir: str = "data/decisions"
    ):
        """
        Initialize the DecisionMaker.
        
        Args:
            llm_connector: The LLM connector for generating analyses
            prompt_manager: The prompt manager for generating prompts
            default_provider: Default LLM provider to use
            default_model: Default model to use
            decisions_dir: Directory to store decision files
        """
        self.llm_connector = llm_connector
        self.prompt_manager = prompt_manager
        self.default_provider = default_provider
        self.default_model = default_model
        self.decisions_dir = decisions_dir
        
        # Ensure decisions directory exists
        import os
        os.makedirs(self.decisions_dir, exist_ok=True)
        
        # Load existing decisions
        self.decisions: Dict[str, Decision] = self._load_decisions()
        
        logger.info(f"DecisionMaker initialized with {len(self.decisions)} existing decisions")
    
    def _load_decisions(self) -> Dict[str, Decision]:
        """Load existing decisions from disk."""
        import os
        decisions = {}
        decision_files = [f for f in os.listdir(self.decisions_dir) 
                         if f.endswith('.json') and os.path.isfile(os.path.join(self.decisions_dir, f))]
        
        for file_name in decision_files:
            file_path = os.path.join(self.decisions_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    decision_data = json.load(f)
                    decision = Decision.from_dict(decision_data)
                    decisions[decision.id] = decision
            except Exception as e:
                logger.error(f"Error loading decision from {file_path}: {e}")
        
        return decisions
    
    def _save_decision(self, decision: Decision) -> None:
        """Save decision to disk."""
        import os
        file_path = os.path.join(self.decisions_dir, f"{decision.id}.json")
        with open(file_path, 'w') as f:
            json.dump(decision.to_dict(), f, indent=2)
    
    async def analyze_options(
        self,
        question: str,
        options: List[str],
        context: Optional[Dict[str, Any]] = None,
        criteria: Optional[List[DecisionCriteria]] = None,
        criteria_weights: Optional[Dict[str, float]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Decision:
        """
        Analyze multiple options for a decision.
        
        Args:
            question: The decision question
            options: List of option descriptions
            context: Additional context for the decision
            criteria: List of criteria to evaluate against
            criteria_weights: Weights for each criterion
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            A Decision object with analyzed options
        """
        # Use default values if not provided
        context = context or {}
        provider = provider or self.default_provider
        model = model or self.default_model
        
        # Set default criteria if not provided
        if not criteria:
            criteria = [
                DecisionCriteria.EFFICIENCY,
                DecisionCriteria.RELIABILITY,
                DecisionCriteria.SIMPLICITY,
                DecisionCriteria.USER_PREFERENCE
            ]
        
        # Set default weights if not provided
        if not criteria_weights:
            criteria_weights = {criterion.value: 1.0 for criterion in criteria}
        
        # Prepare prompt for analysis
        template_name = "decision_analysis"
        prompt_context = {
            "question": question,
            "options": options,
            "context": json.dumps(context, indent=2),
            "criteria": [criterion.value for criterion in criteria],
            "criteria_weights": criteria_weights
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
            analysis = response.get('json', {})
            
            # Create decision ID
            decision_id = str(uuid.uuid4())
            
            # Create options
            analyzed_options = {}
            for i, option_analysis in enumerate(analysis.get("options", [])):
                option_id = f"{decision_id}_option_{i}"
                
                # Extract criteria scores
                criteria_scores = {}
                for criterion in criteria:
                    criterion_name = criterion.value
                    if criterion_name in option_analysis.get("scores", {}):
                        # Ensure score is between 0 and 1
                        score = option_analysis["scores"][criterion_name]
                        criteria_scores[criterion_name] = max(0.0, min(1.0, float(score)))
                
                # Create option
                option = Option(
                    id=option_id,
                    name=option_analysis.get("name", options[i] if i < len(options) else f"Option {i+1}"),
                    description=option_analysis.get("description", options[i] if i < len(options) else ""),
                    pros=option_analysis.get("pros", []),
                    cons=option_analysis.get("cons", []),
                    criteria_scores=criteria_scores,
                    metadata=option_analysis.get("metadata", {})
                )
                analyzed_options[option_id] = option
            
            # Create decision
            decision = Decision(
                id=decision_id,
                question=question,
                context=context,
                options=analyzed_options,
                criteria=criteria,
                criteria_weights=criteria_weights,
                metadata=analysis.get("metadata", {})
            )
            
            # Save the decision
            self.decisions[decision_id] = decision
            self._save_decision(decision)
            
            logger.info(f"Created decision {decision_id} with {len(analyzed_options)} options")
            return decision
            
        except Exception as e:
            logger.error(f"Error analyzing options: {e}")
            raise ValueError(f"Failed to analyze options: {e}")
    
    async def make_decision(
        self,
        decision_id: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Decision:
        """
        Make a decision by selecting the best option.
        
        Args:
            decision_id: ID of the decision to make
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Updated Decision with selection and justification
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision with ID {decision_id} not found")
        
        decision = self.decisions[decision_id]
        
        # Skip if decision was already made
        if decision.selected_option_id:
            return decision
        
        # First calculate scores and get best option
        scores = decision.calculate_option_scores()
        if not scores:
            raise ValueError(f"No options to evaluate for decision {decision_id}")
        
        best_option_id = max(scores, key=scores.get)
        best_option = decision.options[best_option_id]
        
        # Prepare prompt for justification
        template_name = "decision_justification"
        prompt_context = {
            "question": decision.question,
            "context": json.dumps(decision.context, indent=2),
            "options": {
                option_id: {
                    "name": option.name,
                    "description": option.description,
                    "pros": option.pros,
                    "cons": option.cons,
                    "scores": option.criteria_scores
                } for option_id, option in decision.options.items()
            },
            "criteria": [criterion.value for criterion in decision.criteria],
            "criteria_weights": decision.criteria_weights,
            "scores": scores,
            "best_option_id": best_option_id,
            "best_option_name": best_option.name
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
            justification_data = response.get('json', {})
            
            # Update decision
            decision.selected_option_id = best_option_id
            decision.justification = justification_data.get("justification", "")
            decision.decided_at = time.time()
            
            # Save the decision
            self._save_decision(decision)
            
            logger.info(f"Made decision {decision_id}, selected option: {best_option.name}")
            return decision
            
        except Exception as e:
            logger.error(f"Error generating justification: {e}")
            
            # Still make the decision, but with a simpler justification
            decision.selected_option_id = best_option_id
            decision.justification = f"Selected based on highest score: {scores[best_option_id]:.2f}"
            decision.decided_at = time.time()
            
            # Save the decision
            self._save_decision(decision)
            
            logger.info(f"Made decision {decision_id} with basic justification")
            return decision
    
    def get_decision(self, decision_id: str) -> Decision:
        """
        Get a decision by ID.
        
        Args:
            decision_id: ID of the decision
            
        Returns:
            Decision object
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision with ID {decision_id} not found")
        
        return self.decisions[decision_id]
    
    def get_all_decisions(self) -> List[Decision]:
        """
        Get all decisions.
        
        Returns:
            List of all Decision objects
        """
        return list(self.decisions.values())
    
    def delete_decision(self, decision_id: str) -> None:
        """
        Delete a decision.
        
        Args:
            decision_id: ID of the decision to delete
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision with ID {decision_id} not found")
        
        # Remove from memory
        del self.decisions[decision_id]
        
        # Remove from disk
        import os
        file_path = os.path.join(self.decisions_dir, f"{decision_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Deleted decision {decision_id}")
    
    async def update_decision_context(
        self,
        decision_id: str,
        new_context: Dict[str, Any],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Decision:
        """
        Update a decision with new context and recalculate.
        
        Args:
            decision_id: ID of the decision to update
            new_context: New context information to add
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Updated Decision object
        """
        if decision_id not in self.decisions:
            raise ValueError(f"Decision with ID {decision_id} not found")
        
        decision = self.decisions[decision_id]
        
        # Merge contexts
        updated_context = {**decision.context}
        updated_context.update(new_context)
        
        # Update context
        decision.context = updated_context
        
        # Reset decision
        decision.selected_option_id = None
        decision.justification = ""
        decision.decided_at = None
        
        # Re-analyze options with updated context
        options_list = [option.description for option in decision.options.values()]
        
        # Use existing options as a base for reanalysis
        template_name = "decision_reanalysis"
        prompt_context = {
            "question": decision.question,
            "original_context": json.dumps(decision.context, indent=2),
            "new_context": json.dumps(new_context, indent=2),
            "updated_context": json.dumps(updated_context, indent=2),
            "options": {
                option_id: {
                    "name": option.name,
                    "description": option.description,
                    "pros": option.pros,
                    "cons": option.cons,
                    "scores": option.criteria_scores
                } for option_id, option in decision.options.items()
            },
            "criteria": [criterion.value for criterion in decision.criteria],
            "criteria_weights": decision.criteria_weights
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
            
            # Update options
            for option_id, option in decision.options.items():
                option_analysis = analysis.get("options", {}).get(option_id, {})
                
                if option_analysis:
                    # Update option details
                    if "name" in option_analysis:
                        option.name = option_analysis["name"]
                    
                    if "description" in option_analysis:
                        option.description = option_analysis["description"]
                    
                    if "pros" in option_analysis:
                        option.pros = option_analysis["pros"]
                    
                    if "cons" in option_analysis:
                        option.cons = option_analysis["cons"]
                    
                    # Update scores
                    if "scores" in option_analysis:
                        for criterion in decision.criteria:
                            criterion_name = criterion.value
                            if criterion_name in option_analysis["scores"]:
                                # Ensure score is between 0 and 1
                                score = option_analysis["scores"][criterion_name]
                                option.criteria_scores[criterion_name] = max(0.0, min(1.0, float(score)))
            
            # Save the decision
            self._save_decision(decision)
            
            logger.info(f"Updated context for decision {decision_id}")
            
            # Make the decision again
            return await self.make_decision(decision_id, provider, model)
            
        except Exception as e:
            logger.error(f"Error updating decision: {e}")
            raise ValueError(f"Failed to update decision: {e}")
    
    async def compare_decisions(
        self,
        decision_ids: List[str],
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple decisions.
        
        Args:
            decision_ids: List of decision IDs to compare
            provider: LLM provider to use
            model: Model to use
            
        Returns:
            Comparison analysis
        """
        # Validate all decision IDs
        decisions = []
        for decision_id in decision_ids:
            if decision_id not in self.decisions:
                raise ValueError(f"Decision with ID {decision_id} not found")
            decisions.append(self.decisions[decision_id])
        
        # Prepare prompt for comparison
        template_name = "decision_comparison"
        prompt_context = {
            "decisions": [
                {
                    "id": decision.id,
                    "question": decision.question,
                    "context": decision.context,
                    "options": {
                        option_id: {
                            "name": option.name,
                            "description": option.description,
                            "pros": option.pros,
                            "cons": option.cons,
                            "scores": option.criteria_scores
                        } for option_id, option in decision.options.items()
                    },
                    "criteria": [criterion.value for criterion in decision.criteria],
                    "criteria_weights": decision.criteria_weights,
                    "selected_option_id": decision.selected_option_id,
                    "justification": decision.justification
                }
                for decision in decisions
            ]
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
            comparison = response.get('json', {})
            logger.info(f"Generated comparison for {len(decision_ids)} decisions")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing decisions: {e}")
            raise ValueError(f"Failed to compare decisions: {e}")
