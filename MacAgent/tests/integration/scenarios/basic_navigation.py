#!/usr/bin/env python3
"""
Test scenario for basic Mac navigation operations:
- Opening applications
- Switching between applications
- Closing applications
"""

import os
import sys
import logging
import time

from MacAgent.src.core.agent import MacAgent
from MacAgent.src.interaction.app_manager import AppManager

class BasicNavigationScenario:
    """Test scenario for basic Mac navigation operations."""
    
    def __init__(self, agent):
        """Initialize with a MacAgent instance."""
        self.agent = agent
        self.app_manager = AppManager()
        self.logger = logging.getLogger(__name__)
        self.test_apps = ["Safari", "Notes", "Calculator"]
        self.results = {
            "open_application": {},
            "switch_application": {},
            "close_application": {}
        }
    
    async def setup(self):
        """Prepare the environment for testing."""
        # Ensure all test applications are closed before starting
        for app in self.test_apps:
            try:
                await self.app_manager.close_application(app)
                time.sleep(1)  # Give the system time to close apps
            except Exception as e:
                self.logger.info(f"Setup: {app} might not be running: {e}")
    
    async def run_test(self):
        """Execute the test scenario."""
        await self.setup()
        
        # Test 1: Opening applications
        for app in self.test_apps:
            try:
                self.logger.info(f"Opening {app}...")
                success = await self.app_manager.open_application(app)
                self.results["open_application"][app] = success
                time.sleep(2)  # Wait for app to open
            except Exception as e:
                self.logger.error(f"Error opening {app}: {e}")
                self.results["open_application"][app] = False
        
        # Test 2: Switching between applications
        for i in range(len(self.test_apps)):
            current_app = self.test_apps[i]
            next_app = self.test_apps[(i + 1) % len(self.test_apps)]
            try:
                self.logger.info(f"Switching from {current_app} to {next_app}...")
                success = await self.app_manager.switch_to_application(next_app)
                self.results["switch_application"][f"{current_app}->{next_app}"] = success
                time.sleep(1)  # Wait for switch to complete
            except Exception as e:
                self.logger.error(f"Error switching from {current_app} to {next_app}: {e}")
                self.results["switch_application"][f"{current_app}->{next_app}"] = False
        
        # Test 3: Closing applications
        for app in reversed(self.test_apps):  # Close in reverse order
            try:
                self.logger.info(f"Closing {app}...")
                success = await self.app_manager.close_application(app)
                self.results["close_application"][app] = success
                time.sleep(1)  # Wait for app to close
            except Exception as e:
                self.logger.error(f"Error closing {app}: {e}")
                self.results["close_application"][app] = False
        
        return self.results
    
    async def cleanup(self):
        """Clean up after tests."""
        # Ensure all test applications are closed
        for app in self.test_apps:
            try:
                await self.app_manager.close_application(app)
            except Exception:
                pass  # Ignore errors during cleanup
    
    def get_results(self):
        """Return the test results."""
        # Calculate success rates
        success_rates = {}
        for operation, results in self.results.items():
            success_count = sum(1 for r in results.values() if r)
            total_count = len(results)
            success_rates[operation] = (
                (success_count / total_count) * 100 if total_count > 0 else 0
            )
        
        return {
            "detailed_results": self.results,
            "success_rates": success_rates,
            "overall_success": all(
                all(result for result in operation_results.values())
                for operation_results in self.results.values()
            )
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
        scenario = BasicNavigationScenario(agent)
        try:
            await scenario.run_test()
            results = scenario.get_results()
            
            # Print results
            print("\nTest Results:")
            print("=============")
            print(f"Overall Success: {results['overall_success']}")
            
            print("\nSuccess Rates:")
            for operation, rate in results['success_rates'].items():
                print(f"  {operation}: {rate:.1f}%")
            
            print("\nDetailed Results:")
            for operation, op_results in results['detailed_results'].items():
                print(f"\n{operation}:")
                for name, success in op_results.items():
                    print(f"  {name}: {'✅ Success' if success else '❌ Failed'}")
            
        finally:
            await scenario.cleanup()
    
    # Run the scenario
    asyncio.run(run_scenario()) 