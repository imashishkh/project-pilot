#!/usr/bin/env python3
"""
Planning System Example for MacAgent.

This script demonstrates how to use the TaskPlanner, DecisionMaker, and ExecutionMonitor
classes within a planning system for a Mac AI agent.

The example showcases:
1. Creating a task plan
2. Making implementation decisions
3. Monitoring task execution
4. Updating the plan based on new context
"""

import os
import json
import time
import random
import asyncio
import logging
from typing import Dict, List, Any, Optional

from MacAgent.src.intelligence.llm_connector import LLMConnector, LLMProvider
from MacAgent.src.intelligence.prompt_manager import PromptManager, PromptStrategy
from MacAgent.src.intelligence.task_planner import TaskPlanner, PlanningStrategy
from MacAgent.src.intelligence.decision_maker import DecisionMaker, DecisionCriteria
from MacAgent.src.intelligence.execution_monitor import ExecutionMonitor, DeviationType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directories exist
os.makedirs("data/plans", exist_ok=True)
os.makedirs("data/decisions", exist_ok=True)
os.makedirs("data/execution_results", exist_ok=True)
os.makedirs("data/prompts", exist_ok=True)

# Mock prompt templates
PROMPT_TEMPLATES = {
    "task_planning": "Create a plan for: {goal}\nContext: {context}\nStrategy: {strategy}",
    "task_planning_update": "Update plan for: {goal}\nExisting plan: {existing_plan}\nNew context: {new_context}",
    "task_action_plan": "Create action plan for task: {task_name}\nDescription: {task_description}",
    "task_reprioritization": "Reprioritize tasks in plan: {plan}",
    "decision_analysis": "Analyze options for: {question}\nOptions: {options}\nCriteria: {criteria}",
    "decision_justification": "Justify decision for: {question}\nSelected option: {best_option_name}",
    "decision_reanalysis": "Reanalyze decision with new context: {new_context}",
    "decision_comparison": "Compare decisions: {decisions}",
    "expected_outcome_generation": "Generate expected outcome for task: {task_id}\nDescription: {task_description}",
    "outcome_verification": "Verify outcome for task: {task_id}\nExpected: {expected_outcome}\nActual: {execution_result}",
    "failure_analysis": "Analyze failure for task: {task_id}\nResult: {execution_result}",
    "progress_report": "Generate progress report for tasks: {task_results}"
}


class MockLLMConnector:
    """Mock LLM connector for simulation purposes."""
    
    async def generate(self, prompt, provider=None, model=None, json_response=False, temperature=0.0):
        """Simulate LLM response generation."""
        logger.info(f"Generating response with {provider} {model} (temp={temperature})")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        if json_response:
            if "task_planning" in prompt or "task_planning_update" in prompt:
                return self._generate_task_plan_response()
            elif "decision_analysis" in prompt:
                return self._generate_decision_analysis_response()
            elif "decision_justification" in prompt:
                return self._generate_decision_justification_response()
            elif "expected_outcome_generation" in prompt:
                return self._generate_expected_outcome_response()
            elif "outcome_verification" in prompt:
                return self._generate_outcome_verification_response()
            elif "failure_analysis" in prompt:
                return self._generate_failure_analysis_response()
            elif "progress_report" in prompt:
                return self._generate_progress_report_response()
            else:
                return {"json": {"message": "Generic JSON response"}}
        else:
            return "Mock text response"
    
    def _generate_task_plan_response(self):
        """Generate a mock task plan response."""
        return {
            "json": {
                "name": "Image Organization Plan",
                "description": "A plan to organize and categorize image files",
                "tasks": [
                    {
                        "name": "Scan image directories",
                        "description": "Scan directories to identify all image files",
                        "estimated_duration": 5,
                        "dependencies": [],
                        "priority": 5
                    },
                    {
                        "name": "Analyze image metadata",
                        "description": "Extract and analyze metadata from images",
                        "estimated_duration": 10,
                        "dependencies": ["Scan image directories"],
                        "priority": 4
                    },
                    {
                        "name": "Categorize images",
                        "description": "Categorize images based on content and metadata",
                        "estimated_duration": 15,
                        "dependencies": ["Analyze image metadata"],
                        "priority": 3
                    },
                    {
                        "name": "Create folder structure",
                        "description": "Create a folder structure based on categories",
                        "estimated_duration": 5,
                        "dependencies": ["Categorize images"],
                        "priority": 2
                    },
                    {
                        "name": "Organize files",
                        "description": "Move files to appropriate folders",
                        "estimated_duration": 10,
                        "dependencies": ["Create folder structure"],
                        "priority": 1
                    }
                ]
            }
        }
    
    def _generate_decision_analysis_response(self):
        """Generate a mock decision analysis response."""
        return {
            "json": {
                "options": [
                    {
                        "name": "AI-based categorization",
                        "description": "Use ML algorithms to categorize images based on content",
                        "pros": ["High accuracy", "Can identify objects in images", "Fully automated"],
                        "cons": ["Computationally intensive", "May require online API", "Less control over categories"],
                        "scores": {
                            "efficiency": 0.8,
                            "reliability": 0.7,
                            "simplicity": 0.5,
                            "flexibility": 0.6,
                            "user_preference": 0.8
                        }
                    },
                    {
                        "name": "Metadata-based categorization",
                        "description": "Categorize images based on EXIF metadata only",
                        "pros": ["Very fast", "Works offline", "Predictable results"],
                        "cons": ["Limited to available metadata", "Can't categorize by content", "Less accurate"],
                        "scores": {
                            "efficiency": 0.9,
                            "reliability": 0.8,
                            "simplicity": 0.9,
                            "flexibility": 0.4,
                            "user_preference": 0.6
                        }
                    },
                    {
                        "name": "Hybrid approach",
                        "description": "Use metadata as primary method with AI enhancement when needed",
                        "pros": ["Good balance of speed and accuracy", "Works offline with online enhancement", "Flexible"],
                        "cons": ["More complex implementation", "Inconsistent processing time", "More parameters to tune"],
                        "scores": {
                            "efficiency": 0.7,
                            "reliability": 0.8,
                            "simplicity": 0.6,
                            "flexibility": 0.9,
                            "user_preference": 0.9
                        }
                    }
                ]
            }
        }
    
    def _generate_decision_justification_response(self):
        """Generate a mock decision justification response."""
        return {
            "json": {
                "justification": "The Hybrid approach was selected because it provides the best balance between efficiency, reliability, and flexibility. While it scores slightly lower on simplicity, its high score on user preference (0.9) indicates it will deliver the experience users want. The ability to work offline but enhance with AI when needed makes it adaptable to various user scenarios."
            }
        }
    
    def _generate_expected_outcome_response(self):
        """Generate a mock expected outcome response."""
        return {
            "json": {
                "description": "All images should be properly organized into appropriate folders based on their content and metadata.",
                "success_criteria": [
                    "All image files are moved from source directories to categorized folders",
                    "No duplicate images exist in the final structure",
                    "Folder names accurately reflect the content of images within",
                    "Original image metadata is preserved",
                    "A summary report of the organization is generated"
                ],
                "verification_method": "automated",
                "parameters": {
                    "allowed_error_rate": 0.05,
                    "minimum_organization_rate": 0.95
                }
            }
        }
    
    def _generate_outcome_verification_response(self):
        """Generate a mock outcome verification response."""
        # Randomly choose between success and different deviations
        deviation_types = [
            DeviationType.NONE.value,
            DeviationType.ERROR.value,
            DeviationType.PARTIAL_COMPLETION.value,
            DeviationType.UNEXPECTED_RESULT.value
        ]
        
        selected_deviation = random.choice(deviation_types)
        
        if selected_deviation == DeviationType.NONE.value:
            return {
                "json": {
                    "deviation_type": None,
                    "deviation_details": None
                }
            }
        else:
            return {
                "json": {
                    "deviation_type": selected_deviation,
                    "deviation_details": {
                        "description": f"Detected a {selected_deviation} deviation during execution",
                        "severity": "medium",
                        "affected_criteria": ["All image files are moved from source directories"]
                    }
                }
            }
    
    def _generate_failure_analysis_response(self):
        """Generate a mock failure analysis response."""
        return {
            "json": {
                "cause": "Insufficient permissions to access some image directories",
                "impact": "Approximately 20% of images could not be processed",
                "recommended_action": "Request elevated permissions or exclude protected directories",
                "prevention_strategy": "Add permission check at the beginning of the task",
                "severity": "medium"
            }
        }
    
    def _generate_progress_report_response(self):
        """Generate a mock progress report response."""
        return {
            "json": {
                "overall_progress": "60%",
                "completed_tasks": 3,
                "in_progress_tasks": 1,
                "pending_tasks": 1,
                "successful_tasks": 2,
                "failed_tasks": 1,
                "estimated_time_remaining": "15 minutes",
                "recommendations": [
                    "Retry failed task with elevated permissions",
                    "Consider parallel processing for remaining tasks"
                ]
            }
        }


class MockPromptManager:
    """Mock prompt manager for simulation purposes."""
    
    def __init__(self, templates):
        self.templates = templates
    
    async def get_prompt(self, template_name, strategy=None, context=None):
        """Get a prompt from the template and context."""
        if template_name not in self.templates:
            return f"Unknown template: {template_name}"
        
        template = self.templates[template_name]
        if context:
            # Simple format string substitution
            try:
                return template.format(**context)
            except KeyError as e:
                return f"Error formatting template: missing key {e}"
        
        return template


async def simulation_task_execution(monitor, result_id, success_rate=0.8):
    """
    Simulate the execution of a task with random success/failure.
    
    Args:
        monitor: ExecutionMonitor instance
        result_id: ID of the execution result
        success_rate: Probability of successful execution
    
    Returns:
        Updated execution result
    """
    # Simulate task execution time
    execution_time = random.uniform(2, 5)
    logger.info(f"Executing task (result ID: {result_id}), estimated time: {execution_time:.1f}s")
    
    await asyncio.sleep(execution_time)
    
    # Determine success or failure
    success = random.random() < success_rate
    
    if success:
        output = {
            "files_processed": random.randint(50, 200),
            "categories_created": random.randint(5, 15),
            "processing_time": execution_time
        }
        error_message = None
        logger.info(f"Task completed successfully: {json.dumps(output, indent=2)}")
    else:
        output = {
            "files_processed": random.randint(10, 50),
            "error_code": random.randint(1, 5)
        }
        error_message = "Error: " + random.choice([
            "Insufficient permissions",
            "Network connection error",
            "Out of memory",
            "Invalid file format",
            "Timeout during processing"
        ])
        logger.info(f"Task failed: {error_message}")
    
    # Complete the execution in the monitor
    result = monitor.complete_execution(
        result_id=result_id,
        success=success,
        output=output,
        error_message=error_message
    )
    
    return result


async def main():
    """Main function demonstrating the planning system."""
    logger.info("Starting planning system example...")
    
    # Initialize components
    llm_connector = MockLLMConnector()
    prompt_manager = MockPromptManager(PROMPT_TEMPLATES)
    
    task_planner = TaskPlanner(
        llm_connector=llm_connector,
        prompt_manager=prompt_manager,
        default_provider=LLMProvider.OPENAI,
        default_model="gpt-4"
    )
    
    decision_maker = DecisionMaker(
        llm_connector=llm_connector,
        prompt_manager=prompt_manager,
        default_provider=LLMProvider.OPENAI,
        default_model="gpt-4"
    )
    
    execution_monitor = ExecutionMonitor(
        llm_connector=llm_connector,
        prompt_manager=prompt_manager,
        default_provider=LLMProvider.OPENAI,
        default_model="gpt-4"
    )
    
    logger.info("Components initialized successfully")
    
    # Example 1: Creating a Task Plan
    logger.info("\n=== Example 1: Creating a Task Plan ===")
    plan_goal = "Organize image files across multiple directories based on content and metadata"
    plan_context = {
        "image_directories": [
            "~/Pictures",
            "~/Downloads",
            "~/Documents/Photos"
        ],
        "file_types": [".jpg", ".png", ".gif", ".heic"],
        "user_preferences": {
            "prefer_content_based_organization": True,
            "create_date_based_subfolders": True,
            "handle_duplicates": "move_to_duplicates_folder"
        }
    }
    
    plan = await task_planner.create_plan(
        goal=plan_goal,
        context=plan_context,
        strategy=PlanningStrategy.SEQUENTIAL
    )
    
    logger.info(f"Created plan '{plan.name}' with {len(plan.tasks)} tasks")
    for task_id, task in plan.tasks.items():
        logger.info(f"  - Task: {task.name} (Priority: {task.priority})")
    
    # Example 2: Making Implementation Decisions
    logger.info("\n=== Example 2: Making Implementation Decisions ===")
    decision_question = "What approach should be used to categorize images based on content?"
    options = [
        "Use AI-based categorization with image recognition",
        "Use metadata-based categorization (date, camera, location)",
        "Use a hybrid approach combining metadata and selective AI analysis"
    ]
    criteria = [
        DecisionCriteria.EFFICIENCY,
        DecisionCriteria.RELIABILITY,
        DecisionCriteria.SIMPLICITY,
        DecisionCriteria.FLEXIBILITY,
        DecisionCriteria.USER_PREFERENCE
    ]
    criteria_weights = {
        "efficiency": 0.8,
        "reliability": 1.0,
        "simplicity": 0.7,
        "flexibility": 0.9,
        "user_preference": 1.0
    }
    
    decision = await decision_maker.analyze_options(
        question=decision_question,
        options=options,
        context=plan_context,
        criteria=criteria,
        criteria_weights=criteria_weights
    )
    
    logger.info(f"Created decision '{decision.question}' with {len(decision.options)} options")
    for option_id, option in decision.options.items():
        scores = ", ".join([f"{k}: {v:.1f}" for k, v in option.criteria_scores.items()])
        logger.info(f"  - Option: {option.name} (Scores: {scores})")
    
    # Make the decision
    decision = await decision_maker.make_decision(decision.id)
    
    selected_option = decision.options[decision.selected_option_id]
    logger.info(f"Selected option: {selected_option.name}")
    logger.info(f"Justification: {decision.justification}")
    
    # Example 3: Monitoring Task Execution
    logger.info("\n=== Example 3: Monitoring Task Execution ===")
    
    # Get the first task from the plan
    next_tasks = plan.get_next_tasks()
    if not next_tasks:
        logger.error("No tasks available to execute")
        return
    
    task = next_tasks[0]
    
    # Generate expected outcome
    expected_outcome = await execution_monitor.generate_expected_outcome(
        task_id=task.id,
        task_description=task.description,
        context=plan_context
    )
    
    logger.info(f"Generated expected outcome for task '{task.name}':")
    logger.info(f"  Description: {expected_outcome.description}")
    logger.info(f"  Success Criteria: {', '.join(expected_outcome.success_criteria)}")
    
    # Start monitoring execution
    execution_result = execution_monitor.start_execution(
        task_id=task.id,
        expected_outcome_id=expected_outcome.id,
        context={"started_by": "planning_system_example.py"}
    )
    
    logger.info(f"Started monitoring execution of task '{task.name}'")
    
    # Simulate task execution with 80% success rate
    execution_result = await simulation_task_execution(
        monitor=execution_monitor,
        result_id=execution_result.id,
        success_rate=0.8
    )
    
    # Verify the outcome
    execution_result = await execution_monitor.verify_outcome(
        result_id=execution_result.id
    )
    
    logger.info(f"Execution status: {execution_result.status}")
    if execution_result.deviation_type != DeviationType.NONE:
        logger.info(f"Deviation detected: {execution_result.deviation_type.value}")
        
        # Analyze failure if not successful
        if not execution_result.success:
            failure_analysis = await execution_monitor.analyze_failure(
                result_id=execution_result.id
            )
            logger.info(f"Failure analysis: {json.dumps(failure_analysis, indent=2)}")
    
    # Example 4: Updating Plan Based on New Context
    logger.info("\n=== Example 4: Updating Plan Based on New Context ===")
    
    new_context = {
        "user_preferences": {
            "prefer_content_based_organization": False,
            "prefer_date_based_organization": True,
            "create_date_based_subfolders": False,
            "handle_duplicates": "keep_newest_only"
        },
        "system_constraints": {
            "limited_disk_space": True,
            "low_memory_mode": True
        }
    }
    
    updated_plan = await task_planner.update_plan(
        plan_id=plan.id,
        context=new_context
    )
    
    logger.info(f"Updated plan based on new context")
    logger.info(f"Original plan had {len(plan.tasks)} tasks")
    logger.info(f"Updated plan has {len(updated_plan.tasks)} tasks")
    
    # Generate a progress report for all tasks
    task_ids = [task_id for task_id in updated_plan.tasks]
    progress_report = await execution_monitor.generate_progress_report(
        task_ids=task_ids
    )
    
    logger.info(f"Progress report: {json.dumps(progress_report, indent=2)}")
    
    logger.info("\nPlanning system example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 