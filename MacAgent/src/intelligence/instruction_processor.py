"""
Instruction Processor Module

This module provides tools for parsing natural language instructions,
extracting intents and parameters, and breaking down complex tasks.
"""

import re
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

# Import LLM connector for intent resolution
from .llm_connector import LLMConnector, LLMProvider, ModelConfig, LLMResponse

# Configure logging
logger = logging.getLogger(__name__)


class InstructionIntentType(Enum):
    """Types of instruction intents."""
    NAVIGATION = "navigation"  # Navigate to a location or website
    SEARCH = "search"          # Search for information
    CREATE = "create"          # Create a file, document, etc.
    EDIT = "edit"              # Edit a file, document, etc.
    DELETE = "delete"          # Delete a file, document, etc.
    EXECUTE = "execute"        # Execute a command or action
    QUERY = "query"            # Ask a question or get information
    CONFIRM = "confirm"        # Confirm an action
    CANCEL = "cancel"          # Cancel an action
    UNKNOWN = "unknown"        # Unknown intent


@dataclass
class Parameter:
    """Parameter extracted from an instruction."""
    name: str
    value: Any
    required: bool = False
    data_type: str = "string"
    confidence: float = 1.0
    alternatives: List[Any] = field(default_factory=list)


@dataclass
class Instruction:
    """Parsed instruction with intent and parameters."""
    raw_text: str
    intent: InstructionIntentType
    parameters: Dict[str, Parameter] = field(default_factory=dict)
    confidence: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    ambiguous: bool = False
    disambiguation_options: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskStep:
    """A single step in a complex task."""
    instruction: Instruction
    description: str
    dependencies: List[int] = field(default_factory=list)  # Indexes of steps this depends on
    completed: bool = False
    
    def __post_init__(self):
        """Initialize with an empty list if dependencies is None."""
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class TaskBreakdown:
    """A breakdown of a complex task into simpler steps."""
    original_instruction: Instruction
    steps: List[TaskStep] = field(default_factory=list)
    description: str = ""
    valid: bool = True
    validation_issues: List[str] = field(default_factory=list)


class InstructionProcessor:
    """
    Processes natural language instructions.
    
    Features:
    - Parses natural language instructions
    - Extracts intents and parameters
    - Disambiguates unclear instructions
    - Breaks down complex instructions into simpler tasks
    - Validates instructions for feasibility
    """
    
    def __init__(self, 
                llm_connector: LLMConnector,
                default_model: str = "gpt-4",
                default_provider: LLMProvider = LLMProvider.OPENAI,
                instruction_cache_size: int = 100):
        """
        Initialize the instruction processor.
        
        Args:
            llm_connector: LLM connector for intent resolution
            default_model: Default model for intent resolution
            default_provider: Default provider for intent resolution
            instruction_cache_size: Size of the instruction cache
        """
        self.llm_connector = llm_connector
        self.default_model = default_model
        self.default_provider = default_provider
        
        # Cache to avoid redundant processing
        self.instruction_cache = {}
        self.instruction_cache_size = instruction_cache_size
        self.instruction_cache_keys = []  # Track order for LRU
        
        # Intent classification patterns
        self.intent_patterns = {
            InstructionIntentType.NAVIGATION: [
                r"(navigate|go|open|browse) to",
                r"visit (website|page|site|url)",
                r"open (the|a)? (website|app|application|program|file)",
            ],
            InstructionIntentType.SEARCH: [
                r"(search|look|find) for",
                r"query (for|about)",
                r"locate"
            ],
            InstructionIntentType.CREATE: [
                r"create (a|an|the)?",
                r"make (a|an|the)?",
                r"generate (a|an|the)?",
                r"new (file|document|folder)"
            ],
            InstructionIntentType.EDIT: [
                r"edit (the|a|an)?",
                r"modify (the|a|an)?",
                r"change (the|a|an)?",
                r"update (the|a|an)?"
            ],
            InstructionIntentType.DELETE: [
                r"delete (the|a|an)?",
                r"remove (the|a|an)?",
                r"trash"
            ],
            InstructionIntentType.EXECUTE: [
                r"run (the|a|an)?",
                r"execute (the|a|an)?",
                r"launch (the|a|an)?",
                r"perform (the|a|an)?"
            ],
            InstructionIntentType.QUERY: [
                r"(what|who|when|where|why|how)",
                r"tell me",
                r"explain",
                r"show me",
                r"give me information"
            ],
            InstructionIntentType.CONFIRM: [
                r"(yes|confirm|proceed|continue|go ahead)",
                r"approve",
                r"accept"
            ],
            InstructionIntentType.CANCEL: [
                r"(no|cancel|stop|abort|halt)",
                r"reject",
                r"decline"
            ]
        }
    
    def _simple_intent_classification(self, text: str) -> Tuple[InstructionIntentType, float]:
        """
        Perform simple rule-based intent classification.
        
        Args:
            text: Instruction text
            
        Returns:
            Tuple of intent type and confidence score
        """
        text_lower = text.lower()
        
        best_match = InstructionIntentType.UNKNOWN
        highest_score = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Count matches and weigh by pattern length
                    matches = re.findall(pattern, text_lower)
                    score = sum(len(match) for match in matches) / len(text)
                    
                    if score > highest_score:
                        highest_score = score
                        best_match = intent
        
        # If no matches found, default to UNKNOWN with low confidence
        if highest_score == 0:
            return InstructionIntentType.UNKNOWN, 0.1
        
        return best_match, min(highest_score + 0.3, 1.0)  # Base + match quality, max 1.0
    
    def _get_from_cache(self, text: str) -> Optional[Instruction]:
        """
        Get a parsed instruction from the cache.
        
        Args:
            text: Instruction text
            
        Returns:
            Cached instruction or None
        """
        if text in self.instruction_cache:
            # Update position in LRU cache
            self.instruction_cache_keys.remove(text)
            self.instruction_cache_keys.append(text)
            
            return self.instruction_cache[text]
        
        return None
    
    def _add_to_cache(self, text: str, instruction: Instruction) -> None:
        """
        Add a parsed instruction to the cache.
        
        Args:
            text: Instruction text
            instruction: Parsed instruction
        """
        # If cache is full, remove oldest entry
        if len(self.instruction_cache_keys) >= self.instruction_cache_size:
            oldest = self.instruction_cache_keys.pop(0)
            del self.instruction_cache[oldest]
        
        # Add new entry
        self.instruction_cache[text] = instruction
        self.instruction_cache_keys.append(text)
    
    async def parse_instruction(self, text: str, context: Dict[str, Any] = None) -> Instruction:
        """
        Parse a natural language instruction.
        
        Args:
            text: Instruction text
            context: Optional context information
            
        Returns:
            Parsed instruction
        """
        # Check cache first
        cached = self._get_from_cache(text)
        if cached:
            logger.debug(f"Cache hit for instruction: {text}")
            return cached
        
        # Start with simple intent classification
        intent, confidence = self._simple_intent_classification(text)
        
        # If low confidence, use LLM for better classification
        if confidence < 0.7:
            intent, confidence = await self._llm_intent_classification(text)
        
        # Extract parameters
        parameters = await self._extract_parameters(text, intent)
        
        # If this is an EXECUTE intent that relates to an action, validate the parameters
        if intent == InstructionIntentType.EXECUTE and parameters:
            # Check if this is an action command
            action_type = None
            for param_name in ["action_type", "command"]:
                if param_name in parameters:
                    action_value = parameters[param_name].value.lower() if parameters[param_name].value else ""
                    # Common action types
                    action_types = ["move_to", "click", "click_at", "drag_to", "type_text", 
                                    "press_key", "perform_hotkey", "scroll"]
                    for action in action_types:
                        if action in action_value:
                            action_type = action
                            break
            
            # If we identified an action type, validate its parameters
            if action_type:
                logger.debug(f"Detected action type: {action_type}")
                valid_params = self.validate_action_parameters(action_type, parameters)
                
                # Replace parameters with validated ones
                for name, value in valid_params.items():
                    if name in parameters:
                        parameters[name].value = value
                    else:
                        parameters[name] = Parameter(
                            name=name,
                            value=value,
                            required=True,
                            data_type="string",
                            confidence=1.0
                        )
        
        # Create instruction object
        instruction = Instruction(
            raw_text=text,
            intent=intent,
            parameters=parameters,
            confidence=confidence,
            context=context or {},
            ambiguous=(confidence < 0.5)
        )
        
        # If ambiguous, get disambiguation options
        if instruction.ambiguous:
            instruction.disambiguation_options = await self._generate_disambiguation_options(instruction)
        
        # Add to cache
        self._add_to_cache(text, instruction)
        
        return instruction
    
    async def _llm_intent_classification(self, text: str) -> Tuple[InstructionIntentType, float]:
        """
        Use LLM to classify instruction intent.
        
        Args:
            text: Instruction text
            
        Returns:
            Tuple of intent type and confidence score
        """
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=100,
            temperature=0.0
        )
        
        # Build messages for intent classification
        messages = [
            {
                "role": "system",
                "content": """You are an instruction parser that analyzes user instructions and categorizes them.
Categories: NAVIGATION, SEARCH, CREATE, EDIT, DELETE, EXECUTE, QUERY, CONFIRM, CANCEL, UNKNOWN.
Return a JSON object with "intent" (one of the categories above) and "confidence" (value from 0 to 1).
"""
            },
            {
                "role": "user",
                "content": f"Classify this instruction into the most appropriate category: \"{text}\""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if result and "intent" in result and "confidence" in result:
                try:
                    intent = InstructionIntentType(result["intent"].lower())
                    confidence = float(result["confidence"])
                    return intent, confidence
                except (ValueError, KeyError):
                    logger.warning(f"Invalid intent classification from LLM: {result}")
                    
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {e}")
        
        # Fallback to UNKNOWN with low confidence
        return InstructionIntentType.UNKNOWN, 0.1
    
    async def _extract_parameters(self, text: str, intent: InstructionIntentType) -> Dict[str, Parameter]:
        """
        Extract parameters from instruction text.
        
        Args:
            text: Instruction text
            intent: Instruction intent
            
        Returns:
            Dictionary of parameters
        """
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=500,
            temperature=0.0
        )
        
        # Get parameter schema based on intent
        param_schema = self._get_parameter_schema(intent)
        
        # Enhanced prompt for action parameters
        action_specific_instructions = ""
        if intent == InstructionIntentType.EXECUTE:
            action_specific_instructions = """
For mouse actions:
- Extract "x" and "y" for coordinates
- Use specific parameter names like "start_x", "start_y", "end_x", "end_y" for drag operations
- For clicks, specify "button" as "left", "right", or "middle" and "clicks" as a number
- Specify "duration" for movement speed in seconds

For keyboard actions:
- Extract "key" for single key presses
- Use "modifiers" as an array of modifier keys (ctrl, shift, alt, cmd)
- Extract "text" for typing operations

Avoid adding parameters that aren't explicitly mentioned or implied in the instruction.
If the instruction doesn't specify a parameter value, do not include that parameter.
"""
        
        # Build messages for parameter extraction
        messages = [
            {
                "role": "system",
                "content": f"""You are a parameter extractor that analyzes instructions and extracts structured parameters.
For a {intent.value.upper()} instruction, extract these parameters: {json.dumps(param_schema)}.
Return a JSON object with parameter names as keys and objects with "value", "required", "data_type", and "confidence" fields.
Only extract parameters that are actually present in the instruction.
{action_specific_instructions}
"""
            },
            {
                "role": "user",
                "content": f"Extract parameters from this instruction: \"{text}\""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if result and isinstance(result, dict):
                # Convert to Parameter objects
                parameters = {}
                for name, param_data in result.items():
                    if isinstance(param_data, dict) and "value" in param_data:
                        parameters[name] = Parameter(
                            name=name,
                            value=param_data["value"],
                            required=param_data.get("required", False),
                            data_type=param_data.get("data_type", "string"),
                            confidence=param_data.get("confidence", 1.0),
                            alternatives=param_data.get("alternatives", [])
                        )
                        
                return parameters
                    
        except Exception as e:
            logger.error(f"Error in parameter extraction: {e}")
        
        # Fallback to empty parameters
        return {}
    
    def _get_parameter_schema(self, intent: InstructionIntentType) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter schema for an intent.
        
        Args:
            intent: Instruction intent
            
        Returns:
            Parameter schema
        """
        # Define expected parameters for each intent
        schemas = {
            InstructionIntentType.NAVIGATION: {
                "location": {"required": True, "data_type": "string", "description": "URL or location to navigate to"},
                "new_tab": {"required": False, "data_type": "boolean", "description": "Whether to open in a new tab"}
            },
            InstructionIntentType.SEARCH: {
                "query": {"required": True, "data_type": "string", "description": "Search query"},
                "source": {"required": False, "data_type": "string", "description": "Where to search (web, files, etc.)"}
            },
            InstructionIntentType.CREATE: {
                "type": {"required": True, "data_type": "string", "description": "Type of item to create (file, folder, etc.)"},
                "name": {"required": False, "data_type": "string", "description": "Name of the item"},
                "location": {"required": False, "data_type": "string", "description": "Where to create the item"}
            },
            InstructionIntentType.EDIT: {
                "target": {"required": True, "data_type": "string", "description": "What to edit"},
                "changes": {"required": False, "data_type": "string", "description": "Changes to make"}
            },
            InstructionIntentType.DELETE: {
                "target": {"required": True, "data_type": "string", "description": "What to delete"},
                "confirm": {"required": False, "data_type": "boolean", "description": "Whether confirmation is needed"}
            },
            InstructionIntentType.EXECUTE: {
                "command": {"required": True, "data_type": "string", "description": "Command to execute"},
                "arguments": {"required": False, "data_type": "array", "description": "Command arguments"},
                # Add action-specific parameters that match expected action method parameters
                "action_type": {"required": False, "data_type": "string", "description": "Type of action to execute (mouse, keyboard, etc.)"},
                # Mouse action parameters
                "x": {"required": False, "data_type": "integer", "description": "X-coordinate for mouse action"},
                "y": {"required": False, "data_type": "integer", "description": "Y-coordinate for mouse action"},
                "button": {"required": False, "data_type": "string", "description": "Mouse button to use (left, right, middle)"},
                "clicks": {"required": False, "data_type": "integer", "description": "Number of clicks to perform"},
                "duration": {"required": False, "data_type": "number", "description": "Duration of mouse movement"},
                "interval": {"required": False, "data_type": "number", "description": "Interval between actions"},
                "start_x": {"required": False, "data_type": "integer", "description": "Starting X-coordinate for drag"},
                "start_y": {"required": False, "data_type": "integer", "description": "Starting Y-coordinate for drag"},
                "end_x": {"required": False, "data_type": "integer", "description": "Ending X-coordinate for drag"},
                "end_y": {"required": False, "data_type": "integer", "description": "Ending Y-coordinate for drag"},
                # Keyboard action parameters
                "key": {"required": False, "data_type": "string", "description": "Key to press"},
                "modifiers": {"required": False, "data_type": "array", "description": "Modifier keys to use (ctrl, shift, alt, cmd)"},
                "text": {"required": False, "data_type": "string", "description": "Text to type"}
            },
            InstructionIntentType.QUERY: {
                "question": {"required": True, "data_type": "string", "description": "The question being asked"},
                "topic": {"required": False, "data_type": "string", "description": "The topic of the question"}
            },
            InstructionIntentType.CONFIRM: {
                "action": {"required": False, "data_type": "string", "description": "The action being confirmed"}
            },
            InstructionIntentType.CANCEL: {
                "action": {"required": False, "data_type": "string", "description": "The action being canceled"}
            },
            InstructionIntentType.UNKNOWN: {
                "text": {"required": True, "data_type": "string", "description": "The instruction text"}
            }
        }
        
        return schemas.get(intent, {})
    
    async def _generate_disambiguation_options(self, instruction: Instruction) -> List[Dict[str, Any]]:
        """
        Generate disambiguation options for ambiguous instructions.
        
        Args:
            instruction: The ambiguous instruction
            
        Returns:
            List of disambiguation options
        """
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=500,
            temperature=0.7  # Higher temperature for more diverse options
        )
        
        # Build messages for disambiguation
        messages = [
            {
                "role": "system",
                "content": """You help disambiguate unclear instructions. Generate 2-3 possible interpretations.
For each interpretation, provide:
1. A clear rephrased instruction
2. The likely intent
3. Any parameters that would be needed
Return a JSON array of these interpretations.
"""
            },
            {
                "role": "user",
                "content": f"Generate possible interpretations for this ambiguous instruction: \"{instruction.raw_text}\""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if result and isinstance(result, list) and len(result) > 0:
                return result
                    
        except Exception as e:
            logger.error(f"Error generating disambiguation options: {e}")
        
        # Fallback to a simple disambiguation option
        return [
            {
                "rephrased": instruction.raw_text,
                "intent": instruction.intent.value,
                "parameters": {param.name: param.value for param in instruction.parameters.values()}
            }
        ]
    
    async def disambiguate(self, instruction: Instruction, selected_option: int) -> Instruction:
        """
        Disambiguate an instruction based on a selected option.
        
        Args:
            instruction: The ambiguous instruction
            selected_option: Index of the selected disambiguation option
            
        Returns:
            Disambiguated instruction
        """
        if not instruction.ambiguous or not instruction.disambiguation_options:
            return instruction
            
        if selected_option < 0 or selected_option >= len(instruction.disambiguation_options):
            logger.warning(f"Invalid disambiguation option: {selected_option}")
            return instruction
            
        # Get the selected option
        option = instruction.disambiguation_options[selected_option]
        
        # Create a new instruction based on the selected option
        try:
            intent_str = option.get("intent", instruction.intent.value)
            intent = InstructionIntentType(intent_str.lower())
        except ValueError:
            intent = instruction.intent
            
        # Extract parameters from the option
        parameters = {}
        option_params = option.get("parameters", {})
        for name, value in option_params.items():
            parameters[name] = Parameter(
                name=name,
                value=value,
                required=True,  # Assume required since it was specified
                data_type="string",
                confidence=1.0
            )
            
        # Create disambiguated instruction
        disambiguated = Instruction(
            raw_text=option.get("rephrased", instruction.raw_text),
            intent=intent,
            parameters=parameters,
            confidence=0.9,  # High confidence since it was manually selected
            context=instruction.context,
            ambiguous=False
        )
        
        # Add to cache
        self._add_to_cache(disambiguated.raw_text, disambiguated)
        
        return disambiguated
    
    async def break_down_complex_task(self, instruction: Instruction) -> TaskBreakdown:
        """
        Break down a complex instruction into simpler tasks.
        
        Args:
            instruction: The complex instruction
            
        Returns:
            Task breakdown
        """
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=1000,
            temperature=0.2
        )
        
        # Build messages for task breakdown
        messages = [
            {
                "role": "system",
                "content": """You break down complex tasks into smaller steps. For each step, provide:
1. A clear instruction
2. A brief description
3. Dependencies (indexes of steps this step depends on)

Return a JSON object with:
- description: Brief description of the overall task
- steps: Array of step objects with "instruction", "description", "dependencies" fields
- valid: Whether the task is feasible to execute
- validation_issues: Array of any issues that might make the task invalid
"""
            },
            {
                "role": "user",
                "content": f"Break down this complex task into smaller steps: \"{instruction.raw_text}\""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if result and isinstance(result, dict):
                # Create task steps
                steps = []
                for i, step_data in enumerate(result.get("steps", [])):
                    # Parse the step instruction
                    step_instruction = await self.parse_instruction(step_data.get("instruction", ""))
                    
                    steps.append(TaskStep(
                        instruction=step_instruction,
                        description=step_data.get("description", ""),
                        dependencies=step_data.get("dependencies", [])
                    ))
                
                # Create task breakdown
                return TaskBreakdown(
                    original_instruction=instruction,
                    steps=steps,
                    description=result.get("description", ""),
                    valid=result.get("valid", True),
                    validation_issues=result.get("validation_issues", [])
                )
                    
        except Exception as e:
            logger.error(f"Error breaking down complex task: {e}")
        
        # Fallback to a single step
        return TaskBreakdown(
            original_instruction=instruction,
            steps=[TaskStep(instruction=instruction, description="Complete the task")],
            description=f"Execute the instruction: {instruction.raw_text}",
            valid=True
        )
    
    async def validate_instruction(self, instruction: Instruction, context: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Validate an instruction for feasibility.
        
        Args:
            instruction: The instruction to validate
            context: Additional context for validation
            
        Returns:
            Tuple of (valid, issues)
        """
        # Create model config
        config = ModelConfig(
            provider=self.default_provider,
            model_name=self.default_model,
            max_tokens=300,
            temperature=0.0
        )
        
        # Combine instruction context with additional context
        combined_context = {}
        combined_context.update(instruction.context)
        if context:
            combined_context.update(context)
            
        # Format context as a string
        context_str = "\n".join([f"{k}: {v}" for k, v in combined_context.items()])
        
        # Build messages for validation
        messages = [
            {
                "role": "system",
                "content": """You validate instructions for feasibility. Check if the instruction can be executed given the context.
Return a JSON object with:
- valid: true/false indicating whether the instruction is valid
- issues: array of strings describing any issues found
Only flag real issues that would prevent execution.
"""
            },
            {
                "role": "user",
                "content": f"""Validate this instruction for feasibility:
Instruction: {instruction.raw_text}
Intent: {instruction.intent.value}
Parameters: {json.dumps({name: param.value for name, param in instruction.parameters.items()})}

Context:
{context_str}
"""
            }
        ]
        
        try:
            # Send to LLM
            response = await self.llm_connector.generate(messages, config)
            
            # Parse the JSON response
            result = self._extract_json(response.text)
            
            if result and isinstance(result, dict) and "valid" in result:
                return result.get("valid", False), result.get("issues", [])
                    
        except Exception as e:
            logger.error(f"Error validating instruction: {e}")
        
        # Default to valid if validation fails
        return True, []
    
    def _extract_json(self, text: str) -> Any:
        """
        Extract JSON from text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON or None
        """
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
    
    def clear_cache(self) -> None:
        """Clear the instruction cache."""
        self.instruction_cache.clear()
        self.instruction_cache_keys.clear()
        logger.info("Instruction cache cleared")

    def validate_action_parameters(self, action_type: str, parameters: Dict[str, Parameter]) -> Dict[str, Any]:
        """
        Validate and prepare parameters for action methods.
        
        This method checks if the parameters match the expected signatures of action methods
        and filters out incompatible parameters to prevent parameter mismatches.
        
        Args:
            action_type: Type of action (move_to, click, type_text, etc.)
            parameters: Dictionary of parameters extracted from instruction
            
        Returns:
            Dictionary of validated parameters suitable for the action method
        """
        # Define expected parameters for common action methods
        expected_params = {
            "move_to": {"x", "y", "duration"},
            "click": {"button", "clicks", "interval"},
            "click_at": {"x", "y", "button", "clicks"},
            "drag_to": {"start_x", "start_y", "end_x", "end_y", "duration"},
            "type_text": {"text", "interval"},
            "press_key": {"key", "modifiers", "key_combination"},
            "perform_hotkey": {"keys"},
            "scroll": {"clicks"}
        }
        
        # Get the set of expected parameters for this action type
        valid_params = expected_params.get(action_type, set())
        
        # If we don't recognize the action type, return all parameters
        if not valid_params:
            logger.warning(f"Unknown action type: {action_type}, cannot validate parameters")
            return {name: param.value for name, param in parameters.items()}
        
        # Filter parameters to only include those that are valid for this action type
        filtered_params = {}
        for name, param in parameters.items():
            if name in valid_params:
                filtered_params[name] = param.value
            else:
                logger.debug(f"Filtering out parameter '{name}' for action '{action_type}' as it's not in the expected parameter list")
        
        # Check for required parameters
        if action_type == "move_to" and ("x" not in filtered_params or "y" not in filtered_params):
            logger.warning(f"Missing required parameters for '{action_type}': needs x and y coordinates")
        elif action_type == "click_at" and ("x" not in filtered_params or "y" not in filtered_params):
            logger.warning(f"Missing required parameters for '{action_type}': needs x and y coordinates")
        elif action_type == "drag_to" and not all(p in filtered_params for p in ["start_x", "start_y", "end_x", "end_y"]):
            logger.warning(f"Missing required parameters for '{action_type}': needs start and end coordinates")
        elif action_type == "type_text" and "text" not in filtered_params:
            logger.warning(f"Missing required parameter for '{action_type}': needs text")
        elif action_type == "press_key" and "key" not in filtered_params and "key_combination" not in filtered_params:
            logger.warning(f"Missing required parameter for '{action_type}': needs key or key_combination")
        elif action_type == "scroll" and "clicks" not in filtered_params:
            logger.warning(f"Missing required parameter for '{action_type}': needs clicks")
            
        logger.debug(f"Validated parameters for action '{action_type}': {filtered_params}")
        return filtered_params 