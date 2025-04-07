#!/usr/bin/env python3
"""
Main script for MacAgent.

This script provides the entry point for the MacAgent system, either with 
a command-line interface or a graphical user interface.
"""

import asyncio
import logging
import argparse
import sys
import os
import traceback
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MacAgent.src.core import AgentLoop, AgentConfig
from MacAgent.src.ui import CommandInterface, FeedbackSystem
from MacAgent.src.ui.main_app import MacAgentApp, main as run_ui

# Configure logger
logger = logging.getLogger(__name__)

async def process_command(agent: AgentLoop, command: str) -> Dict[str, Any]:
    """
    Process a command using the agent.
    
    Args:
        agent: The agent loop instance
        command: The command to process
        
    Returns:
        Dictionary with the execution plan and result
    """
    logger.info(f"Processing command: {command}")
    try:
        # Process instruction to create a plan
        plan = await agent.process_instruction(command)
        
        # Execute the plan
        result = await agent.execute_planned_actions(plan)
        
        return {
            'plan': plan,
            'result': result,
            'success': result.get('success', False)
        }
    except Exception as e:
        logger.error(f"Error processing command: {str(e)}")
        logger.debug(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

async def run_command_line_mode(config: AgentConfig) -> None:
    """
    Run MacAgent in command-line mode.
    
    Args:
        config: Agent configuration
    """
    # Create and initialize agent
    agent = AgentLoop(config)
    await agent.initialize()
    
    print("MacAgent Command Line Interface")
    print("Type 'exit' to quit, 'help' for available commands")
    
    try:
        while True:
            try:
                # Get command from user
                command = input("\nEnter command: ")
                
                # Check for exit command
                if command.lower() in ['exit', 'quit']:
                    break
                
                # Check for help command
                if command.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  help - Show this help message")
                    print("  exit/quit - Exit the application")
                    print("  Any natural language instruction - The agent will attempt to execute it")
                    continue
                
                if not command.strip():
                    continue
                    
                print(f"Processing: {command}")
                
                try:
                    # Process the command
                    task = asyncio.create_task(process_command(agent, command))
                    result = await task
                    
                    # Display results
                    if result.get('success'):
                        print("\n✅ Task completed successfully")
                        if 'result' in result and 'output' in result['result']:
                            print(f"Result: {result['result']['output']}")
                    else:
                        print("\n❌ Task failed")
                        if 'error' in result:
                            print(f"Error: {result['error']}")
                        elif 'result' in result and 'error' in result['result']:
                            print(f"Error: {result['result']['error']}")
                
                except asyncio.CancelledError:
                    print("\n⚠️ Operation was cancelled")
                except Exception as e:
                    logger.error(f"Error executing command: {str(e)}")
                    logger.debug(traceback.format_exc())
                    print(f"\n❌ Error during execution: {str(e)}")
                
            except KeyboardInterrupt:
                print("\nOperation canceled. Press Ctrl+C again to exit or enter a new command.")
                try:
                    # Wait briefly to see if user presses Ctrl+C again
                    await asyncio.sleep(0.5)
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
            except Exception as e:
                logger.error(f"Command line interface error: {str(e)}")
                logger.debug(traceback.format_exc())
                print(f"Error: {str(e)}")
    finally:
        # Clean up agent resources
        await agent.cleanup()
        print("MacAgent terminated.")

async def run_demo_mode(config: AgentConfig) -> None:
    """
    Run MacAgent in demo mode with example tasks.
    
    Args:
        config: Agent configuration
    """
    # Create and initialize agent
    agent = AgentLoop(config)
    await agent.initialize()
    
    try:
        print("MacAgent Demo Mode")
        print("Running example tasks...\n")
        
        # Example of processing a test instruction
        instruction = "Take a screenshot of the desktop"
        print(f"Executing: {instruction}")
        
        plan = await agent.process_instruction(instruction)
        print("\nGenerated Plan:")
        for i, step in enumerate(plan.steps, 1):
            print(f"  Step {i}: {step.description}")
        
        # Execute the plan
        print("\nExecuting plan...")
        result = await agent.execute_planned_actions(plan)
        
        # Display results
        if result and result.get('success'):
            print("\n✅ Demo task completed successfully")
            if 'output' in result:
                print(f"Result: {result['output']}")
        else:
            print("\n❌ Demo task failed")
            if result and 'error' in result:
                print(f"Error: {result['error']}")
    finally:
        # Clean up agent resources
        await agent.cleanup()
        print("\nDemo completed.")

async def main() -> None:
    """Main entry point for MacAgent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MacAgent - AI Agent for Mac")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-file-log", action="store_true", help="Disable logging to file")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode (no GUI)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with example task")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Set up console logging
    logging.basicConfig(level=log_level, format=log_format)
    
    # Set up file logging if enabled
    if not args.no_file_log:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "macagent.log"))
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Configure agent
    config = AgentConfig()
    
    # Load configuration from file if specified
    if args.config:
        try:
            config.load_from_file(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            print(f"Error loading configuration: {str(e)}")
            return
    
    # Apply command line options
    if args.debug:
        config.log_level = logging.DEBUG
    if args.no_file_log:
        config.log_to_file = False
    
    logger.debug("Starting MacAgent")
    
    try:
        # Run in demo mode
        if args.demo:
            await run_demo_mode(config)
        # Run in command-line interface mode
        elif args.cli:
            await run_command_line_mode(config)
        # Run with graphical user interface (default)
        else:
            run_ui()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.debug(traceback.format_exc())
        print(f"Application error: {str(e)}")


if __name__ == "__main__":
    try:
        # Run the async main function with proper asyncio handling
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")
        traceback.print_exc()
    finally:
        # Ensure proper cleanup of resources
        # Close any remaining event loops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        
        # Exit with appropriate status code
        sys.exit(0) 