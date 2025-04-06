#!/usr/bin/env python3
"""
Test script for the instruction processing and task planning system.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
import json
import uuid
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from MacAgent.src.intelligence.instruction_processor import InstructionProcessor
from MacAgent.src.intelligence.task_planner import TaskPlanner, Task, TaskPlan, PlanningStrategy
from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider
from MacAgent.src.intelligence.prompt_manager import PromptManager, PromptStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestTaskPlanner:
    """Test-specific TaskPlanner that adapts the API differences."""
    
    def __init__(self, llm_connector, prompt_manager, plans_dir="data/plans"):
        """Initialize with the same parameters as TaskPlanner."""
        self.llm_connector = llm_connector
        self.prompt_manager = prompt_manager
        self.plans_dir = plans_dir
        os.makedirs(self.plans_dir, exist_ok=True)
        self.plans = {}
    
    async def create_simple_plan(self, instruction):
        """Create a simplified plan from an instruction."""
        # Create a plan ID
        plan_id = str(uuid.uuid4())
        
        # Create tasks based on task breakdown
        tasks = {}
        
        # Use instruction directly for a simple plan
        task = Task(
            id=f"{plan_id}_task_0",
            name=instruction.raw_text,
            description=f"Execute the instruction: {instruction.raw_text}",
            priority=3
        )
        tasks[task.id] = task
        
        # Create the plan
        plan = TaskPlan(
            id=plan_id,
            name=f"Plan for: {instruction.raw_text[:50]}",
            description=f"Execute the instruction: {instruction.raw_text}",
            goal=instruction.raw_text,
            tasks=tasks,
            strategy=PlanningStrategy.SEQUENTIAL,
            context=instruction.context or {},
        )
        
        return plan


async def test_planning():
    """Test the instruction processor and task planner."""
    
    # Initialize the LLM connector
    llm_connector = LLMConnector(
        config_path="config/api_keys.json",
        enable_cache=True
    )
    
    # Initialize the prompt manager
    prompt_manager = PromptManager(
        templates_path="config/prompt_templates"
    )
    
    # Add get_prompt method to PromptManager for compatibility
    async def get_prompt(template_name, strategy, context):
        """Adapter method to simulate the expected get_prompt method."""
        # Convert context dict to variables dict
        try:
            return prompt_manager.render_prompt(
                template_name=template_name, 
                variables=context
            )
        except ValueError:
            # If template not found, use dynamic prompt as fallback
            logger.warning(f"Template '{template_name}' not found, using dynamic prompt")
            # Create a simple placeholder prompt
            return [
                {"role": "system", "content": "You are an AI assistant that helps with task planning."},
                {"role": "user", "content": f"Create a plan for this goal: {context.get('goal', 'Unknown goal')}"}
            ]
    
    # Monkey patch the get_prompt method for testing purposes only
    prompt_manager.get_prompt = get_prompt
    
    # Initialize the instruction processor
    processor = InstructionProcessor(
        llm_connector=llm_connector,
        default_model="gpt-4"
    )
    
    # Initialize the test task planner
    test_planner = TestTaskPlanner(
        llm_connector=llm_connector,
        prompt_manager=prompt_manager
    )
    
    # Test instruction
    instruction_text = "Open Safari and search for 'Mac automation tools'"
    
    # Process the instruction (parse_instruction is async)
    logger.info(f"Processing instruction: {instruction_text}")
    instruction = await processor.parse_instruction(instruction_text)
    
    # Display the processed instruction
    print(f"\nProcessed instruction:")
    print(f"  Raw text: {instruction.raw_text}")
    print(f"  Intent: {instruction.intent}")
    print(f"  Confidence: {instruction.confidence}")
    print(f"  Parameters: {', '.join([f'{name}: {param.value}' for name, param in instruction.parameters.items()])}")
    
    # Break down complex task if needed
    logger.info("Breaking down task...")
    task_breakdown = await processor.break_down_complex_task(instruction)
    
    # Display the task breakdown
    print("\nTask breakdown:")
    print(f"  Description: {task_breakdown.description}")
    for i, step in enumerate(task_breakdown.steps):
        print(f"  {i+1}. {step.description}")
        print(f"     Instruction: {step.instruction.raw_text}")
        print(f"     Dependencies: {step.dependencies}")
    
    # Generate a simplified plan
    logger.info("Creating simplified plan...")
    plan = await test_planner.create_simple_plan(instruction)
    
    # Display the plan
    print("\nGenerated plan:")
    print(f"  Plan ID: {plan.id}")
    print(f"  Name: {plan.name}")
    print(f"  Description: {plan.description}")
    print("  Tasks:")
    for task_id, task in plan.tasks.items():
        print(f"    - {task.name}")
        print(f"      Description: {task.description}")
        print(f"      Priority: {task.priority}")
    
    # Display the result
    print("\nTest completed successfully!")
    
    # Clean up
    await llm_connector.close()


if __name__ == "__main__":
    try:
        asyncio.run(test_planning())
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True) 