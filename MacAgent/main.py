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

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MacAgent.src.core import AgentLoop, AgentConfig
from MacAgent.src.ui import CommandInterface, FeedbackSystem
from MacAgent.src.ui.main_app import MacAgentApp, main as run_ui

async def process_command(agent, command):
    """Process a command using the agent."""
    logging.info(f"Processing command: {command}")
    plan = await agent.process_instruction(command)
    return plan

async def run_command_line_mode(config):
    """Run MacAgent in command-line mode."""
    # Create and initialize agent
    agent = AgentLoop(config)
    
    print("MacAgent Command Line Interface")
    print("Type 'exit' to quit")
    
    while True:
        try:
            # Get command from user
            command = input("\nEnter command: ")
            
            # Check for exit command
            if command.lower() in ['exit', 'quit']:
                break
                
            # Process the command
            await process_command(agent, command)
            
            # Run the agent loop for a short time to execute the command
            agent.start()
            await asyncio.sleep(5)  # Give time for the agent to execute
            
        except KeyboardInterrupt:
            print("\nOperation canceled.")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Clean up
    agent.stop()
    print("MacAgent terminated.")

async def main():
    """Main entry point for MacAgent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MacAgent - AI Agent for Mac")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-file-log", action="store_true", help="Disable logging to file")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode (no GUI)")
    parser.add_argument("--demo", action="store_true", help="Run demo mode with example task")
    args = parser.parse_args()
    
    # Configure agent
    config = AgentConfig()
    
    # Apply command line options
    if args.debug:
        config.log_level = logging.DEBUG
    if args.no_file_log:
        config.log_to_file = False
    
    # Run in demo mode (original behavior with example task)
    if args.demo:
        # Create and initialize agent
        agent = AgentLoop(config)
        
        # Example of processing a test instruction
        instruction = "Take a screenshot of the desktop"
        plan = await agent.process_instruction(instruction)
        
        # Run the agent loop
        await agent.run()
    # Run in command-line interface mode
    elif args.cli:
        await run_command_line_mode(config)
    # Run with graphical user interface (default)
    else:
        run_ui()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 