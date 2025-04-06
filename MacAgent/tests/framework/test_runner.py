#!/usr/bin/env python3
"""
Test Runner module for executing MacAgent test scenarios.
"""

import os
import sys
import logging
import asyncio
import time
import json
from datetime import datetime
from pathlib import Path
import importlib
import inspect
from typing import List, Dict, Any, Optional, Type, Callable

from MacAgent.src.core.agent import MacAgent


class TestResult:
    """Class to store and manage test results."""
    
    def __init__(self, scenario_name: str, start_time: float):
        """Initialize a new test result."""
        self.scenario_name = scenario_name
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.success: bool = False
        self.detailed_results: Dict[str, Any] = {}
        self.error: Optional[str] = None
    
    def complete(self, success: bool, detailed_results: Dict[str, Any]):
        """Mark the test as completed."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.detailed_results = detailed_results
    
    def fail_with_error(self, error_message: str):
        """Mark the test as failed with an error message."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = False
        self.error = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration, 2) if self.duration is not None else None,
            "success": self.success,
            "detailed_results": self.detailed_results,
            "error": self.error
        }


class TestRunner:
    """Test runner for executing MacAgent test scenarios."""
    
    def __init__(self, output_dir: str = "MacAgent/tests/output"):
        """Initialize the test runner."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.agent = MacAgent()
        self.results: List[TestResult] = []
    
    async def run_scenario(self, scenario_class: Type, *args, **kwargs) -> TestResult:
        """Run a single test scenario."""
        scenario_name = scenario_class.__name__
        self.logger.info(f"Running scenario: {scenario_name}")
        
        # Create a new result object
        result = TestResult(scenario_name, time.time())
        
        # Initialize the scenario
        scenario = scenario_class(self.agent, *args, **kwargs)
        
        try:
            # Run the test
            detailed_results = await scenario.run_test()
            
            # Get the success status
            scenario_results = scenario.get_results()
            overall_success = scenario_results.get("overall_success", False)
            
            # Complete the result
            result.complete(overall_success, scenario_results)
            
        except Exception as e:
            # Log and record the error
            self.logger.error(f"Error running scenario {scenario_name}: {e}", exc_info=True)
            result.fail_with_error(str(e))
        finally:
            # Always run cleanup
            try:
                await scenario.cleanup()
            except Exception as e:
                self.logger.error(f"Error during cleanup of {scenario_name}: {e}", exc_info=True)
        
        # Add to results list
        self.results.append(result)
        
        # Log the result
        status = "PASSED" if result.success else "FAILED"
        self.logger.info(f"Scenario {scenario_name}: {status}")
        
        return result
    
    async def discover_and_run_scenarios(self, scenarios_package: str = "MacAgent.tests.integration.scenarios") -> List[TestResult]:
        """Discover and run all scenarios in the specified package."""
        self.logger.info(f"Discovering scenarios in package: {scenarios_package}")
        
        # Import the package
        package = importlib.import_module(scenarios_package)
        
        # Get the package directory
        package_dir = os.path.dirname(package.__file__)
        
        # Find all scenario modules
        scenario_modules = []
        for file in os.listdir(package_dir):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = f"{scenarios_package}.{file[:-3]}"
                scenario_modules.append(module_name)
        
        self.logger.info(f"Found {len(scenario_modules)} scenario modules")
        
        # Load and run each scenario
        for module_name in scenario_modules:
            module = importlib.import_module(module_name)
            
            # Find scenario classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    name.endswith("Scenario") and 
                    hasattr(obj, "run_test") and 
                    hasattr(obj, "cleanup") and 
                    hasattr(obj, "get_results")):
                    
                    # Run the scenario
                    await self.run_scenario(obj)
        
        return self.results
    
    def generate_report(self, report_format: str = "json") -> str:
        """Generate a report of the test results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_format == "json":
            # Generate JSON report
            report_file = os.path.join(self.output_dir, f"test_report_{timestamp}.json")
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(self.results),
                "passed_scenarios": sum(1 for r in self.results if r.success),
                "failed_scenarios": sum(1 for r in self.results if not r.success),
                "scenarios": [result.to_dict() for result in self.results]
            }
            
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)
            
        else:
            # Generate text report
            report_file = os.path.join(self.output_dir, f"test_report_{timestamp}.txt")
            
            with open(report_file, "w") as f:
                f.write("MacAgent Test Report\n")
                f.write("===================\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                
                # Summary
                total = len(self.results)
                passed = sum(1 for r in self.results if r.success)
                failed = total - passed
                success_rate = (passed / total) * 100 if total > 0 else 0
                
                f.write(f"Total Scenarios: {total}\n")
                f.write(f"Passed: {passed}\n")
                f.write(f"Failed: {failed}\n")
                f.write(f"Success Rate: {success_rate:.1f}%\n\n")
                
                # Detailed results
                f.write("Detailed Results\n")
                f.write("===============\n\n")
                
                for result in self.results:
                    status = "PASSED" if result.success else "FAILED"
                    duration = f"{result.duration:.2f}s" if result.duration is not None else "N/A"
                    
                    f.write(f"Scenario: {result.scenario_name}\n")
                    f.write(f"Status: {status}\n")
                    f.write(f"Duration: {duration}\n")
                    
                    if result.error:
                        f.write(f"Error: {result.error}\n")
                    
                    if result.detailed_results:
                        f.write("Details:\n")
                        for key, value in result.detailed_results.items():
                            f.write(f"  {key}: {value}\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Generated report: {report_file}")
        return report_file


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def run_tests():
        runner = TestRunner()
        await runner.discover_and_run_scenarios()
        runner.generate_report("text")
        runner.generate_report("json")
    
    # Run all tests
    asyncio.run(run_tests()) 