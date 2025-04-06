#!/usr/bin/env python3
"""
Test scenario for application control operations:
- Interacting with applications
- Controlling UI elements
- Performing application-specific tasks
"""

import os
import sys
import logging
import time

from MacAgent.src.core.agent import MacAgent
from MacAgent.src.interaction.app_manager import AppManager
from MacAgent.src.interaction.ui_controller import UIController

class ApplicationControlScenario:
    """Test scenario for application control operations."""
    
    def __init__(self, agent):
        """Initialize with a MacAgent instance."""
        self.agent = agent
        self.app_manager = AppManager()
        self.ui_controller = UIController()
        self.logger = logging.getLogger(__name__)
        self.results = {
            "open_notes": False,
            "create_note": False,
            "type_content": False,
            "save_note": False,
            "close_notes": False
        }
    
    async def setup(self):
        """Prepare the environment for testing."""
        # Ensure Notes is closed before starting
        try:
            await self.app_manager.close_application("Notes")
            time.sleep(1)  # Give the system time to close
        except Exception as e:
            self.logger.info(f"Setup: Notes might not be running: {e}")
    
    async def run_test(self):
        """Execute the test scenario."""
        await self.setup()
        
        # Test 1: Open Notes application
        try:
            self.logger.info("Opening Notes...")
            success = await self.app_manager.open_application("Notes")
            self.results["open_notes"] = success
            time.sleep(2)  # Wait for app to open fully
            
            if not success:
                self.logger.error("Failed to open Notes, aborting remaining tests")
                return self.results
        except Exception as e:
            self.logger.error(f"Error opening Notes: {e}")
            return self.results
        
        # Test 2: Create a new note
        try:
            self.logger.info("Creating a new note...")
            # Command+N to create a new note
            await self.ui_controller.send_keystroke("n", ["command"])
            time.sleep(1)
            self.results["create_note"] = True
        except Exception as e:
            self.logger.error(f"Error creating a new note: {e}")
            self.results["create_note"] = False
        
        # Test 3: Type content into the note
        try:
            self.logger.info("Typing content...")
            test_content = "This is a test note created by MacAgent automated testing.\n"
            test_content += "If you see this note, the test was successful."
            await self.ui_controller.type_text(test_content)
            time.sleep(1)
            self.results["type_content"] = True
        except Exception as e:
            self.logger.error(f"Error typing content: {e}")
            self.results["type_content"] = False
        
        # Test 4: Save the note (Command+S)
        try:
            self.logger.info("Saving the note...")
            await self.ui_controller.send_keystroke("s", ["command"])
            time.sleep(1)
            
            # Type a name for the note and press Enter
            note_name = f"MacAgent Test Note {time.strftime('%Y-%m-%d %H:%M:%S')}"
            await self.ui_controller.type_text(note_name)
            time.sleep(0.5)
            await self.ui_controller.send_keystroke("return")
            time.sleep(1)
            
            self.results["save_note"] = True
        except Exception as e:
            self.logger.error(f"Error saving note: {e}")
            self.results["save_note"] = False
        
        # Test 5: Close Notes application
        try:
            self.logger.info("Closing Notes...")
            success = await self.app_manager.close_application("Notes")
            self.results["close_notes"] = success
        except Exception as e:
            self.logger.error(f"Error closing Notes: {e}")
            self.results["close_notes"] = False
        
        return self.results
    
    async def cleanup(self):
        """Clean up after tests."""
        # Ensure Notes is closed
        try:
            await self.app_manager.close_application("Notes")
        except Exception:
            pass  # Ignore errors during cleanup
    
    def get_results(self):
        """Return the test results."""
        # Calculate success rate
        success_count = sum(1 for result in self.results.values() if result)
        total_count = len(self.results)
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        return {
            "detailed_results": self.results,
            "success_rate": success_rate,
            "overall_success": all(self.results.values())
        }


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import asyncio
    
    async def run_scenario():
        # Initialize the agent
        agent = MacAgent()
        
        # Create and run the test scenario
        scenario = ApplicationControlScenario(agent)
        try:
            await scenario.run_test()
            results = scenario.get_results()
            
            # Print results
            print("\nTest Results:")
            print("=============")
            print(f"Overall Success: {results['overall_success']}")
            print(f"Success Rate: {results['success_rate']:.1f}%")
            
            print("\nDetailed Results:")
            for operation, success in results['detailed_results'].items():
                print(f"  {operation}: {'✅ Success' if success else '❌ Failed'}")
            
        finally:
            await scenario.cleanup()
    
    # Run the scenario
    asyncio.run(run_scenario()) 