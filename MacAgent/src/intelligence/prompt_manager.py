"""
Prompt Manager Module

This module provides tools for managing prompts, templates, and context windows
for more effective LLM interactions.
"""

import os
import re
import json
import time
import logging
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
import string
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

class PromptStrategy(Enum):
    """Different prompt strategies for LLM interactions."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    REACT = "react"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class PromptTemplate:
    """A template for generating prompts with variable substitution."""
    name: str
    template: str
    strategy: PromptStrategy
    variables: Set[str] = field(default_factory=set)
    description: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        """Extract variables from the template."""
        # Find all variables in the template {variable_name}
        self.variables = set(re.findall(r'\{([a-zA-Z0-9_]+)\}', self.template))
    
    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render the template with the provided variables.
        
        Args:
            variables: Dictionary of variable values
            
        Returns:
            Rendered prompt
        """
        # Check for missing variables
        missing = [var for var in self.variables if var not in variables]
        if missing:
            logger.warning(f"Missing variables in template '{self.name}': {missing}")
        
        # Use string.format with variables
        try:
            # Create a kwargs dict with only the variables in the template
            template_vars = {k: v for k, v in variables.items() if k in self.variables}
            return self.template.format(**template_vars)
        except Exception as e:
            logger.error(f"Error rendering template '{self.name}': {e}")
            raise
    
    def get_required_variables(self) -> Set[str]:
        """Get the set of variables required by this template."""
        return self.variables
    
    def estimate_token_count(self, variables: Dict[str, Any]) -> int:
        """
        Estimate the number of tokens in the rendered prompt.
        
        This is a simplified estimator that assumes ~4 chars per token.
        
        Args:
            variables: Dictionary of variable values
            
        Returns:
            Estimated token count
        """
        rendered = self.render(variables)
        # Estimate tokens (rough approximation: ~4 chars per token)
        return len(rendered) // 4 + 1


class ContextWindowManager:
    """
    Manages context window limitations for LLM interactions.
    
    Ensures that prompts don't exceed token limits and implements
    strategies for handling long contexts.
    """
    
    def __init__(self, default_model_token_limit: int = 4096):
        """
        Initialize context window manager.
        
        Args:
            default_model_token_limit: Default token limit for models
        """
        self.default_model_token_limit = default_model_token_limit
        self.model_token_limits = {
            # OpenAI models
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-1106-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            
            # Anthropic models
            "claude-instant-1": 100000,
            "claude-2": 100000,
            "claude-2.1": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000
        }
        
        # Reserve tokens for responses
        self.response_token_reserve = 1024
    
    def get_model_token_limit(self, model_name: str) -> int:
        """
        Get the token limit for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Token limit for the model
        """
        return self.model_token_limits.get(model_name, self.default_model_token_limit)
    
    def check_token_limit(self, 
                         prompt_tokens: int, 
                         model_name: str,
                         max_response_tokens: int = None) -> bool:
        """
        Check if a prompt fits within a model's token limit.
        
        Args:
            prompt_tokens: Estimated number of tokens in the prompt
            model_name: Name of the model
            max_response_tokens: Maximum tokens for the response
            
        Returns:
            True if within limits, False otherwise
        """
        model_limit = self.get_model_token_limit(model_name)
        
        # Use provided max_response_tokens or default reserve
        reserve = max_response_tokens or self.response_token_reserve
        
        # Check if prompt + reserved response tokens exceeds the limit
        return prompt_tokens + reserve <= model_limit
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.
        
        This is a simplified approach that assumes ~4 chars per token.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        # Rough approximation: ~4 chars per token
        char_limit = max_tokens * 4
        
        if len(text) <= char_limit:
            return text
            
        # Simple truncation - more sophisticated approaches could preserve meaning better
        return text[:char_limit] + "..."
    
    def truncate_conversation(self, 
                             messages: List[Dict[str, str]], 
                             max_tokens: int,
                             preserve_system: bool = True) -> List[Dict[str, str]]:
        """
        Truncate a conversation to fit within token limits.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum total tokens allowed
            preserve_system: Whether to preserve system messages
            
        Returns:
            Truncated conversation
        """
        # Estimate current token count
        estimated_tokens = sum(len(m.get("content", "")) // 4 + 1 for m in messages)
        
        if estimated_tokens <= max_tokens:
            return messages
        
        # Preserve system messages if requested
        if preserve_system:
            system_messages = [m for m in messages if m.get("role") == "system"]
            system_tokens = sum(len(m.get("content", "")) // 4 + 1 for m in system_messages)
            
            # If system messages alone exceed the limit, we need to truncate them too
            if system_tokens > max_tokens:
                logger.warning("System messages alone exceed token limit, truncating them too")
                preserve_system = False
        
        # Create a new list of messages
        result = []
        
        if preserve_system:
            # Add all system messages first
            result.extend(system_messages)
            
            # Calculate remaining tokens
            remaining_tokens = max_tokens - system_tokens
            
            # Get non-system messages, newest first
            other_messages = [m for m in messages if m.get("role") != "system"]
            other_messages.reverse()  # Newest first
            
            # Add as many recent messages as possible
            for message in other_messages:
                message_tokens = len(message.get("content", "")) // 4 + 1
                
                if message_tokens <= remaining_tokens:
                    result.insert(len(system_messages), message)  # Insert after system messages
                    remaining_tokens -= message_tokens
                else:
                    # If this message would exceed the limit, truncate it
                    truncated_content = self.truncate_text(message.get("content", ""), remaining_tokens)
                    truncated_message = dict(message)
                    truncated_message["content"] = truncated_content
                    result.insert(len(system_messages), truncated_message)
                    break
        else:
            # If not preserving system messages, start from the most recent messages
            messages_copy = list(messages)
            messages_copy.reverse()  # Newest first
            
            remaining_tokens = max_tokens
            
            for message in messages_copy:
                message_tokens = len(message.get("content", "")) // 4 + 1
                
                if message_tokens <= remaining_tokens:
                    result.insert(0, message)  # Insert at the beginning
                    remaining_tokens -= message_tokens
                else:
                    # If this message would exceed the limit, truncate it
                    truncated_content = self.truncate_text(message.get("content", ""), remaining_tokens)
                    truncated_message = dict(message)
                    truncated_message["content"] = truncated_content
                    result.insert(0, truncated_message)
                    break
        
        return result


class PromptStrategyHandler(ABC):
    """
    Base class for different prompt strategies.
    
    Provides an interface for implementing different prompt strategies
    like zero-shot, few-shot, chain-of-thought, etc.
    """
    
    @abstractmethod
    def format_prompt(self, task: str, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Format a prompt according to the strategy.
        
        Args:
            task: The task or instruction
            variables: Dictionary of variables
            
        Returns:
            Formatted messages for LLM
        """
        pass


class ZeroShotStrategy(PromptStrategyHandler):
    """Simple zero-shot prompting strategy."""
    
    def format_prompt(self, task: str, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format a zero-shot prompt."""
        system_message = variables.get("system_message", "You are a helpful assistant.")
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": task}
        ]


class FewShotStrategy(PromptStrategyHandler):
    """Few-shot prompting with examples."""
    
    def format_prompt(self, task: str, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format a few-shot prompt with examples."""
        system_message = variables.get("system_message", "You are a helpful assistant.")
        examples = variables.get("examples", [])
        
        messages = [{"role": "system", "content": system_message}]
        
        # Add examples
        for example in examples:
            if "input" in example and "output" in example:
                messages.append({"role": "user", "content": example["input"]})
                messages.append({"role": "assistant", "content": example["output"]})
        
        # Add the actual task
        messages.append({"role": "user", "content": task})
        
        return messages


class ChainOfThoughtStrategy(PromptStrategyHandler):
    """Chain-of-thought prompting strategy."""
    
    def format_prompt(self, task: str, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format a chain-of-thought prompt."""
        system_message = variables.get("system_message", 
                                     "You are a helpful assistant that thinks step-by-step.")
        
        cot_instruction = "Think through this step-by-step:"
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"{task}\n\n{cot_instruction}"}
        ]


class StructuredOutputStrategy(PromptStrategyHandler):
    """Strategy for generating structured outputs like JSON."""
    
    def format_prompt(self, task: str, variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format a prompt for structured output."""
        system_message = variables.get("system_message", 
                                     "You are a helpful assistant that provides structured outputs.")
        
        output_format = variables.get("output_format", "json")
        format_instructions = variables.get("format_instructions", "")
        
        if not format_instructions:
            format_instructions = f"Provide your response in {output_format} format."
        
        return [
            {"role": "system", "content": f"{system_message}\n\n{format_instructions}"},
            {"role": "user", "content": task}
        ]


class PromptManager:
    """
    Manages prompt templates and strategies for LLM interactions.
    
    Features:
    - Maintains a library of optimized prompts for different tasks
    - Implements prompt templating with variable substitution
    - Handles context window management
    - Supports different prompt strategies
    """
    
    def __init__(self, 
                templates_path: Optional[str] = "config/prompt_templates",
                default_model_token_limit: int = 4096):
        """
        Initialize the prompt manager.
        
        Args:
            templates_path: Path to prompt templates directory
            default_model_token_limit: Default token limit for models
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_path = templates_path
        self.context_manager = ContextWindowManager(default_model_token_limit)
        
        # Initialize strategy handlers
        self.strategies = {
            PromptStrategy.ZERO_SHOT: ZeroShotStrategy(),
            PromptStrategy.FEW_SHOT: FewShotStrategy(),
            PromptStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy(),
            PromptStrategy.STRUCTURED_OUTPUT: StructuredOutputStrategy()
        }
        
        # Load templates if path exists
        if templates_path and os.path.exists(templates_path):
            self.load_templates()
    
    def load_templates(self) -> None:
        """Load prompt templates from the templates directory."""
        if not self.templates_path or not os.path.exists(self.templates_path):
            logger.warning(f"Templates path not found: {self.templates_path}")
            return
            
        try:
            # Look for JSON files in the templates directory
            for root, _, files in os.walk(self.templates_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                templates_data = json.load(f)
                                
                                # Process each template in the file
                                for template_data in templates_data:
                                    try:
                                        name = template_data.get('name')
                                        if not name:
                                            logger.warning(f"Skipping template without name in {file_path}")
                                            continue
                                            
                                        template = template_data.get('template')
                                        if not template:
                                            logger.warning(f"Skipping template without content: {name}")
                                            continue
                                            
                                        # Parse strategy
                                        strategy_str = template_data.get('strategy', 'zero_shot')
                                        try:
                                            strategy = PromptStrategy(strategy_str)
                                        except ValueError:
                                            logger.warning(f"Unknown strategy '{strategy_str}' for template '{name}', using ZERO_SHOT")
                                            strategy = PromptStrategy.ZERO_SHOT
                                        
                                        # Create template object
                                        prompt_template = PromptTemplate(
                                            name=name,
                                            template=template,
                                            strategy=strategy,
                                            description=template_data.get('description', ''),
                                            version=template_data.get('version', '1.0'),
                                            tags=template_data.get('tags', []),
                                            examples=template_data.get('examples', []),
                                            max_tokens=template_data.get('max_tokens')
                                        )
                                        
                                        # Add to templates dictionary
                                        self.templates[name] = prompt_template
                                        logger.info(f"Loaded template: {name}")
                                        
                                    except Exception as e:
                                        logger.error(f"Error processing template in {file_path}: {e}")
                        
                        except Exception as e:
                            logger.error(f"Error loading template file {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.templates)} prompt templates")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def create_template(self, 
                       name: str, 
                       template: str, 
                       strategy: PromptStrategy = PromptStrategy.ZERO_SHOT,
                       description: str = "",
                       tags: List[str] = None,
                       examples: List[Dict[str, Any]] = None,
                       max_tokens: Optional[int] = None) -> PromptTemplate:
        """
        Create a new prompt template.
        
        Args:
            name: Template name
            template: Template string with variables in {variable_name} format
            strategy: Prompt strategy
            description: Template description
            tags: List of tags for categorization
            examples: Example usages of the template
            max_tokens: Maximum tokens for this prompt
            
        Returns:
            Created PromptTemplate
        """
        if name in self.templates:
            logger.warning(f"Overwriting existing template: {name}")
            
        prompt_template = PromptTemplate(
            name=name,
            template=template,
            strategy=strategy,
            description=description,
            tags=tags or [],
            examples=examples or [],
            max_tokens=max_tokens
        )
        
        self.templates[name] = prompt_template
        
        # Save to file if templates_path exists
        if self.templates_path:
            self.save_template(prompt_template)
            
        return prompt_template
    
    def save_template(self, template: PromptTemplate) -> bool:
        """
        Save a template to a file.
        
        Args:
            template: The template to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.templates_path:
            logger.warning("No templates path specified, cannot save template")
            return False
            
        # Make sure the directory exists
        os.makedirs(self.templates_path, exist_ok=True)
        
        try:
            # Group templates by first letter or category
            first_char = template.name[0].lower()
            file_path = os.path.join(self.templates_path, f"{first_char}_templates.json")
            
            # Check if file exists and load existing templates
            existing_templates = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        existing_templates = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading existing templates file {file_path}: {e}")
                    # If error, start with empty list
                    existing_templates = []
            
            # Convert template to dictionary
            template_dict = {
                "name": template.name,
                "template": template.template,
                "strategy": template.strategy.value,
                "description": template.description,
                "version": template.version,
                "tags": template.tags,
                "examples": template.examples,
                "max_tokens": template.max_tokens
            }
            
            # Check if template already exists
            found = False
            for i, existing in enumerate(existing_templates):
                if existing.get("name") == template.name:
                    existing_templates[i] = template_dict
                    found = True
                    break
                    
            if not found:
                existing_templates.append(template_dict)
                
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(existing_templates, f, indent=2)
                
            logger.info(f"Saved template '{template.name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template '{template.name}': {e}")
            return False
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get a template by name.
        
        Args:
            name: Template name
            
        Returns:
            Template or None if not found
        """
        return self.templates.get(name)
    
    def delete_template(self, name: str) -> bool:
        """
        Delete a template.
        
        Args:
            name: Template name
            
        Returns:
            True if deleted, False otherwise
        """
        if name not in self.templates:
            logger.warning(f"Template not found: {name}")
            return False
            
        del self.templates[name]
        
        # Also remove from file if templates_path exists
        if self.templates_path:
            # Search for the template in all template files
            for root, _, files in os.walk(self.templates_path):
                for file in files:
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                templates_data = json.load(f)
                                
                            # Check if template exists in this file
                            found = False
                            for i, template_data in enumerate(templates_data):
                                if template_data.get('name') == name:
                                    templates_data.pop(i)
                                    found = True
                                    break
                                    
                            if found:
                                # Write updated templates back to file
                                with open(file_path, 'w') as f:
                                    json.dump(templates_data, f, indent=2)
                                    
                                logger.info(f"Deleted template '{name}' from {file_path}")
                                break
                                
                        except Exception as e:
                            logger.error(f"Error updating template file {file_path}: {e}")
        
        return True
    
    def render_prompt(self, 
                     template_name: str, 
                     variables: Dict[str, Any],
                     model_name: Optional[str] = None,
                     max_tokens: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Render a prompt using a template and format it according to the strategy.
        
        Args:
            template_name: Template name
            variables: Dictionary of variable values
            model_name: Target model name (for token limit checks)
            max_tokens: Override max token limit
            
        Returns:
            Formatted messages for LLM
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
            
        # Render the template
        rendered_text = template.render(variables)
        
        # Check token limit if model_name is provided
        if model_name:
            token_limit = max_tokens or template.max_tokens
            estimated_tokens = template.estimate_token_count(variables)
            
            if token_limit and estimated_tokens > token_limit:
                logger.warning(f"Rendered prompt exceeds token limit ({estimated_tokens} > {token_limit})")
            
            if not self.context_manager.check_token_limit(estimated_tokens, model_name):
                logger.warning(f"Prompt may exceed model's token limit ({model_name})")
        
        # Format according to strategy
        strategy = self.strategies.get(template.strategy)
        if not strategy:
            logger.warning(f"Strategy not implemented: {template.strategy}, falling back to zero-shot")
            strategy = self.strategies[PromptStrategy.ZERO_SHOT]
            
        return strategy.format_prompt(rendered_text, variables)
    
    def get_templates_by_tag(self, tag: str) -> List[PromptTemplate]:
        """
        Get all templates with a specific tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of matching templates
        """
        return [template for template in self.templates.values() if tag in template.tags]
    
    def render_dynamic_prompt(self, 
                             text: str, 
                             variables: Dict[str, Any],
                             strategy: PromptStrategy = PromptStrategy.ZERO_SHOT) -> List[Dict[str, str]]:
        """
        Render a dynamic prompt without a pre-defined template.
        
        Args:
            text: Prompt text with variables in {variable_name} format
            variables: Dictionary of variable values
            strategy: Prompt strategy
            
        Returns:
            Formatted messages for LLM
        """
        # Create a temporary template
        temp_template = PromptTemplate(
            name="dynamic_template",
            template=text,
            strategy=strategy
        )
        
        # Render the template
        rendered_text = temp_template.render(variables)
        
        # Format according to strategy
        strategy_handler = self.strategies.get(strategy)
        if not strategy_handler:
            logger.warning(f"Strategy not implemented: {strategy}, falling back to zero-shot")
            strategy_handler = self.strategies[PromptStrategy.ZERO_SHOT]
            
        return strategy_handler.format_prompt(rendered_text, variables)
    
    def optimize_prompt(self, 
                       prompt: List[Dict[str, str]], 
                       model_name: str,
                       max_response_tokens: int = None) -> List[Dict[str, str]]:
        """
        Optimize a prompt to fit within model's token limit.
        
        Args:
            prompt: List of conversation messages
            model_name: Target model name
            max_response_tokens: Maximum tokens for the response
            
        Returns:
            Optimized prompt
        """
        # Use context window manager to truncate if needed
        model_limit = self.context_manager.get_model_token_limit(model_name)
        
        # Reserve tokens for response
        reserve = max_response_tokens or self.context_manager.response_token_reserve
        max_prompt_tokens = model_limit - reserve
        
        # Truncate conversation if needed
        return self.context_manager.truncate_conversation(prompt, max_prompt_tokens)
    
    def register_strategy(self, strategy_type: PromptStrategy, handler: PromptStrategyHandler) -> None:
        """
        Register a custom strategy handler.
        
        Args:
            strategy_type: Strategy type enum
            handler: Strategy handler implementation
        """
        self.strategies[strategy_type] = handler
        logger.info(f"Registered custom handler for strategy: {strategy_type.value}") 