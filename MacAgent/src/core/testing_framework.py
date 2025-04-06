"""
Testing Framework Module for MacAgent.

This module provides a comprehensive testing framework for the MacAgent system,
including automated integration tests, user interaction simulation, 
end-to-end functionality validation, and test reporting.
"""

import logging
import time
import asyncio
import os
import json
import csv
import datetime
from typing import Dict, List, Optional, Any, Callable, Set, Union, Tuple
from dataclasses import dataclass, field
import traceback
from pathlib import Path
import inspect

from MacAgent.src.core.system_integrator import SystemIntegrator, SystemConfig
from MacAgent.src.core.pipeline_manager import PipelineManager

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Test case definition."""
    name: str
    description: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    timeout: float = 30.0
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "timeout": self.timeout,
            "tags": self.tags,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create test case from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            inputs=data["inputs"],
            expected_outputs=data["expected_outputs"],
            timeout=data.get("timeout", 30.0),
            tags=data.get("tags", []),
            enabled=data.get("enabled", True)
        )


@dataclass
class TestResult:
    """Test result information."""
    test_case: TestCase
    success: bool
    start_time: float
    end_time: float
    actual_outputs: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Test duration in seconds."""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test result to dictionary."""
        return {
            "test_case": self.test_case.to_dict(),
            "success": self.success,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "actual_outputs": self.actual_outputs,
            "errors": self.errors
        }


@dataclass
class TestSuite:
    """Collection of related test cases."""
    name: str
    description: str
    test_cases: List[TestCase] = field(default_factory=list)
    setup_function: Optional[Callable] = None
    teardown_function: Optional[Callable] = None
    
    def add_test_case(self, test_case: TestCase) -> None:
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
    
    def remove_test_case(self, name: str) -> bool:
        """Remove a test case from the suite by name."""
        for i, test_case in enumerate(self.test_cases):
            if test_case.name == name:
                self.test_cases.pop(i)
                return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test suite to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "test_cases": [tc.to_dict() for tc in self.test_cases]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestSuite':
        """Create test suite from dictionary."""
        test_cases = [TestCase.from_dict(tc) for tc in data.get("test_cases", [])]
        return cls(
            name=data["name"],
            description=data["description"],
            test_cases=test_cases
        )


@dataclass
class TestReport:
    """Report of test execution results."""
    suite_name: str
    start_time: float
    end_time: float
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Total test duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def success_count(self) -> int:
        """Number of successful tests."""
        return sum(1 for result in self.results if result.success)
    
    @property
    def failure_count(self) -> int:
        """Number of failed tests."""
        return sum(1 for result in self.results if not result.success)
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if not self.results:
            return 0.0
        return (self.success_count / len(self.results)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test report to dictionary."""
        return {
            "suite_name": self.suite_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "results": [result.to_dict() for result in self.results]
        }


class UserSimulator:
    """
    Simulates user interactions for testing.
    
    This class provides methods to simulate various user actions like
    clicking, typing, and other interactions to test the agent.
    """
    
    def __init__(self, action_module=None):
        """
        Initialize the user simulator.
        
        Args:
            action_module: Optional action module to use for simulations
        """
        self.action_module = action_module
        self.queued_actions = []
        self.executing = False
        
        logger.info("UserSimulator initialized")
    
    def queue_action(self, action_type: str, **params) -> None:
        """
        Queue an action to be executed later.
        
        Args:
            action_type: Type of action to queue
            **params: Action parameters
        """
        self.queued_actions.append({
            "type": action_type,
            "params": params,
            "timestamp": time.time()
        })
        
        logger.debug(f"Queued action: {action_type} with params: {params}")
    
    def queue_click(self, x: int, y: int) -> None:
        """Queue a mouse click at the specified coordinates."""
        self.queue_action("click", x=x, y=y)
    
    def queue_type(self, text: str) -> None:
        """Queue typing the specified text."""
        self.queue_action("type", text=text)
    
    def queue_key_press(self, key: str) -> None:
        """Queue pressing the specified key."""
        self.queue_action("key_press", key=key)
    
    def queue_delay(self, seconds: float) -> None:
        """Queue a delay in seconds."""
        self.queue_action("delay", seconds=seconds)
    
    async def execute_queued_actions(self) -> None:
        """Execute all queued actions in order."""
        if self.executing:
            logger.warning("Already executing actions")
            return
            
        if not self.action_module:
            logger.error("No action module provided")
            return
            
        self.executing = True
        logger.info(f"Executing {len(self.queued_actions)} queued actions")
        
        for action in self.queued_actions:
            action_type = action["type"]
            params = action["params"]
            
            try:
                if action_type == "click":
                    await self.action_module.click_at(params["x"], params["y"])
                elif action_type == "type":
                    await self.action_module.type_text(params["text"])
                elif action_type == "key_press":
                    await self.action_module.press_key(params["key"])
                elif action_type == "delay":
                    await asyncio.sleep(params["seconds"])
                else:
                    logger.warning(f"Unknown action type: {action_type}")
            except Exception as e:
                logger.error(f"Error executing action {action_type}: {str(e)}")
        
        self.queued_actions = []
        self.executing = False
        
        logger.info("Finished executing actions")
    
    def clear_actions(self) -> None:
        """Clear all queued actions."""
        self.queued_actions = []
        logger.debug("Cleared all queued actions")


class TestingFramework:
    """
    Testing framework for MacAgent system.
    
    This class provides functionality for:
    1. Running automated integration tests
    2. Simulating user interactions
    3. Validating end-to-end functionality
    4. Generating test reports and visualizations
    5. Supporting continuous integration
    """
    
    def __init__(
        self,
        test_output_dir: str = "test_output",
        system_config: Optional[SystemConfig] = None
    ):
        """
        Initialize the testing framework.
        
        Args:
            test_output_dir: Directory for test output
            system_config: Optional system configuration
        """
        self.test_output_dir = test_output_dir
        self.system_config = system_config or SystemConfig.default_config()
        
        # Create output directory if it doesn't exist
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Initialize test suites and reports
        self.test_suites: Dict[str, TestSuite] = {}
        self.reports: List[TestReport] = []
        
        # Initialize system integrator for testing
        self.system_integrator = None
        
        # Initialize user simulator
        self.user_simulator = UserSimulator()
        
        # Test execution flags
        self.is_running = False
        self.current_suite = None
        
        logger.info(f"TestingFramework initialized with output directory: {test_output_dir}")
    
    def add_test_suite(self, suite: TestSuite) -> None:
        """
        Add a test suite to the framework.
        
        Args:
            suite: Test suite to add
        """
        if suite.name in self.test_suites:
            logger.warning(f"Overwriting existing test suite: {suite.name}")
            
        self.test_suites[suite.name] = suite
        logger.info(f"Added test suite: {suite.name} with {len(suite.test_cases)} test cases")
    
    def load_test_suite_from_file(self, file_path: str) -> bool:
        """
        Load a test suite from a JSON file.
        
        Args:
            file_path: Path to the test suite JSON file
            
        Returns:
            True if the suite was loaded successfully, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            suite = TestSuite.from_dict(data)
            self.add_test_suite(suite)
            return True
            
        except Exception as e:
            logger.error(f"Error loading test suite from {file_path}: {str(e)}")
            return False
    
    def save_test_suite_to_file(self, suite_name: str, file_path: str) -> bool:
        """
        Save a test suite to a JSON file.
        
        Args:
            suite_name: Name of the test suite to save
            file_path: Path to save the test suite
            
        Returns:
            True if the suite was saved successfully, False otherwise
        """
        if suite_name not in self.test_suites:
            logger.error(f"Test suite not found: {suite_name}")
            return False
            
        try:
            suite = self.test_suites[suite_name]
            data = suite.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved test suite {suite_name} to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving test suite to {file_path}: {str(e)}")
            return False
    
    async def setup_test_environment(self) -> bool:
        """
        Set up the test environment.
        
        Returns:
            True if setup was successful, False otherwise
        """
        try:
            logger.info("Setting up test environment")
            
            # Create system integrator for testing
            self.system_integrator = SystemIntegrator(self.system_config)
            
            # Initialize the system
            success = await self.system_integrator.initialize()
            if not success:
                logger.error("Failed to initialize system integrator")
                return False
            
            # Set up user simulator with action module
            if "action" in self.system_integrator.components:
                self.user_simulator.action_module = self.system_integrator.components["action"]
            
            logger.info("Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up test environment: {str(e)}")
            return False
    
    async def teardown_test_environment(self) -> None:
        """Tear down the test environment."""
        logger.info("Tearing down test environment")
        
        if self.system_integrator:
            try:
                await self.system_integrator.stop()
            except Exception as e:
                logger.error(f"Error stopping system integrator: {str(e)}")
                
        self.system_integrator = None
        self.user_simulator.action_module = None
        
        logger.info("Test environment teardown complete")
    
    async def run_test_case(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: Test case to run
            
        Returns:
            Test result
        """
        if not test_case.enabled:
            logger.info(f"Skipping disabled test case: {test_case.name}")
            return TestResult(
                test_case=test_case,
                success=False,
                start_time=time.time(),
                end_time=time.time(),
                actual_outputs={},
                errors=["Test case is disabled"]
            )
            
        if not self.system_integrator:
            logger.error("System integrator not initialized")
            return TestResult(
                test_case=test_case,
                success=False,
                start_time=time.time(),
                end_time=time.time(),
                actual_outputs={},
                errors=["System integrator not initialized"]
            )
            
        logger.info(f"Running test case: {test_case.name}")
        
        start_time = time.time()
        errors = []
        actual_outputs = {}
        
        try:
            # Process test inputs
            instruction = test_case.inputs.get("instruction")
            simulated_actions = test_case.inputs.get("simulated_actions", [])
            
            # Set up simulated actions if any
            if simulated_actions:
                for action in simulated_actions:
                    action_type = action["type"]
                    if action_type == "click":
                        self.user_simulator.queue_click(action["x"], action["y"])
                    elif action_type == "type":
                        self.user_simulator.queue_type(action["text"])
                    elif action_type == "key_press":
                        self.user_simulator.queue_key_press(action["key"])
                    elif action_type == "delay":
                        self.user_simulator.queue_delay(action["seconds"])
            
            # Process instruction if provided
            if instruction:
                # Set timeout for the instruction
                result = await asyncio.wait_for(
                    self.system_integrator.process_instruction(instruction),
                    timeout=test_case.timeout
                )
                actual_outputs["instruction_result"] = result
            
            # Execute any queued actions
            if simulated_actions:
                await self.user_simulator.execute_queued_actions()
            
            # Compare expected outputs with actual outputs
            success = self._validate_test_results(test_case.expected_outputs, actual_outputs, errors)
            
        except asyncio.TimeoutError:
            errors.append(f"Test timed out after {test_case.timeout} seconds")
            success = False
        except Exception as e:
            errors.append(f"Error running test: {str(e)}")
            traceback.print_exc()
            success = False
            
        end_time = time.time()
        
        result = TestResult(
            test_case=test_case,
            success=success,
            start_time=start_time,
            end_time=end_time,
            actual_outputs=actual_outputs,
            errors=errors
        )
        
        logger.info(f"Test case {test_case.name} {'passed' if success else 'failed'} in {result.duration:.3f}s")
        return result
    
    def _validate_test_results(
        self, 
        expected_outputs: Dict[str, Any], 
        actual_outputs: Dict[str, Any],
        errors: List[str]
    ) -> bool:
        """
        Validate test results against expected outputs.
        
        Args:
            expected_outputs: Expected outputs
            actual_outputs: Actual outputs
            errors: List to append errors to
            
        Returns:
            True if all expectations are met, False otherwise
        """
        if not expected_outputs:
            return True
            
        for key, expected_value in expected_outputs.items():
            if key not in actual_outputs:
                errors.append(f"Expected output '{key}' not found in actual outputs")
                return False
                
            actual_value = actual_outputs[key]
            
            # Handle different validation types
            if isinstance(expected_value, dict) and "validation_type" in expected_value:
                validation_type = expected_value["validation_type"]
                
                if validation_type == "contains":
                    if expected_value["value"] not in actual_value:
                        errors.append(f"Expected '{key}' to contain '{expected_value['value']}', got '{actual_value}'")
                        return False
                elif validation_type == "not_contains":
                    if expected_value["value"] in actual_value:
                        errors.append(f"Expected '{key}' to not contain '{expected_value['value']}', got '{actual_value}'")
                        return False
                elif validation_type == "greater_than":
                    if not actual_value > expected_value["value"]:
                        errors.append(f"Expected '{key}' to be greater than {expected_value['value']}, got {actual_value}")
                        return False
                elif validation_type == "less_than":
                    if not actual_value < expected_value["value"]:
                        errors.append(f"Expected '{key}' to be less than {expected_value['value']}, got {actual_value}")
                        return False
            else:
                # Direct equality check
                if actual_value != expected_value:
                    errors.append(f"Expected '{key}' to be '{expected_value}', got '{actual_value}'")
                    return False
                    
        return True
    
    async def run_test_suite(self, suite_name: str) -> TestReport:
        """
        Run all test cases in a test suite.
        
        Args:
            suite_name: Name of the test suite to run
            
        Returns:
            Test report
        """
        if suite_name not in self.test_suites:
            logger.error(f"Test suite not found: {suite_name}")
            return TestReport(
                suite_name=suite_name,
                start_time=time.time(),
                end_time=time.time(),
                results=[]
            )
            
        suite = self.test_suites[suite_name]
        logger.info(f"Running test suite: {suite_name} with {len(suite.test_cases)} test cases")
        
        if self.is_running:
            logger.warning("Tests already running")
            return TestReport(
                suite_name=suite_name,
                start_time=time.time(),
                end_time=time.time(),
                results=[]
            )
            
        self.is_running = True
        self.current_suite = suite_name
        
        start_time = time.time()
        results = []
        
        try:
            # Set up test environment
            success = await self.setup_test_environment()
            if not success:
                logger.error("Failed to set up test environment")
                return TestReport(
                    suite_name=suite_name,
                    start_time=start_time,
                    end_time=time.time(),
                    results=[]
                )
            
            # Run suite setup if provided
            if suite.setup_function:
                try:
                    if asyncio.iscoroutinefunction(suite.setup_function):
                        await suite.setup_function()
                    else:
                        suite.setup_function()
                except Exception as e:
                    logger.error(f"Error in suite setup: {str(e)}")
            
            # Run each test case
            for test_case in suite.test_cases:
                result = await self.run_test_case(test_case)
                results.append(result)
            
            # Run suite teardown if provided
            if suite.teardown_function:
                try:
                    if asyncio.iscoroutinefunction(suite.teardown_function):
                        await suite.teardown_function()
                    else:
                        suite.teardown_function()
                except Exception as e:
                    logger.error(f"Error in suite teardown: {str(e)}")
            
        finally:
            # Tear down test environment
            await self.teardown_test_environment()
            
            self.is_running = False
            self.current_suite = None
            
        end_time = time.time()
        
        # Create test report
        report = TestReport(
            suite_name=suite_name,
            start_time=start_time,
            end_time=end_time,
            results=results
        )
        
        # Save report
        self.reports.append(report)
        self._save_report(report)
        
        logger.info(f"Test suite {suite_name} completed: {report.success_count}/{len(results)} tests passed ({report.success_rate:.1f}%)")
        return report
    
    def _save_report(self, report: TestReport) -> None:
        """
        Save a test report to the output directory.
        
        Args:
            report: Test report to save
        """
        timestamp = datetime.datetime.fromtimestamp(report.start_time).strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.test_output_dir, f"{report.suite_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Save JSON report
        report_path = os.path.join(report_dir, "report.json")
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
            
        # Save CSV summary
        csv_path = os.path.join(report_dir, "summary.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Test Case", "Success", "Duration (s)", "Errors"])
            
            for result in report.results:
                writer.writerow([
                    result.test_case.name,
                    result.success,
                    f"{result.duration:.3f}",
                    "; ".join(result.errors) if result.errors else ""
                ])
                
        logger.info(f"Saved test report to {report_dir}")
    
    def get_last_report(self) -> Optional[TestReport]:
        """
        Get the most recent test report.
        
        Returns:
            Most recent test report or None if no reports exist
        """
        if not self.reports:
            return None
        return self.reports[-1]
    
    def get_all_reports(self) -> List[TestReport]:
        """
        Get all test reports.
        
        Returns:
            List of all test reports
        """
        return self.reports
    
    def create_default_test_suite(self) -> TestSuite:
        """
        Create a default test suite with basic test cases.
        
        Returns:
            Default test suite
        """
        suite = TestSuite(
            name="default_suite",
            description="Default test suite with basic tests"
        )
        
        # Add a simple test case
        suite.add_test_case(TestCase(
            name="basic_perception_test",
            description="Tests basic perception functionality",
            inputs={
                "instruction": "Take a screenshot"
            },
            expected_outputs={
                "instruction_result": {
                    "validation_type": "contains",
                    "value": "success"
                }
            }
        ))
        
        # Add another test case with simulated actions
        suite.add_test_case(TestCase(
            name="basic_action_test",
            description="Tests basic action functionality",
            inputs={
                "simulated_actions": [
                    {"type": "click", "x": 100, "y": 100},
                    {"type": "delay", "seconds": 0.5},
                    {"type": "type", "text": "Hello, world!"}
                ]
            },
            expected_outputs={}  # No specific expectations for this test
        ))
        
        self.add_test_suite(suite)
        return suite
    
    async def generate_test_report_html(self, report: TestReport) -> str:
        """
        Generate an HTML report from a test report.
        
        Args:
            report: Test report to generate HTML for
            
        Returns:
            Path to the generated HTML file
        """
        timestamp = datetime.datetime.fromtimestamp(report.start_time).strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.test_output_dir, f"{report.suite_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        html_path = os.path.join(report_dir, "report.html")
        
        # Generate HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - {report.suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .summary {{ margin-bottom: 20px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Test Report - {report.suite_name}</h1>
            
            <div class="summary">
                <p>
                    <strong>Date:</strong> {datetime.datetime.fromtimestamp(report.start_time).strftime("%Y-%m-%d %H:%M:%S")}
                </p>
                <p>
                    <strong>Duration:</strong> {report.duration:.3f} seconds
                </p>
                <p>
                    <strong>Results:</strong> 
                    <span class="{'success' if report.success_rate == 100 else 'failure'}">
                        {report.success_count}/{len(report.results)} tests passed ({report.success_rate:.1f}%)
                    </span>
                </p>
            </div>
            
            <h2>Test Cases</h2>
            <table>
                <tr>
                    <th>Test Case</th>
                    <th>Description</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Errors</th>
                </tr>
        """
        
        for result in report.results:
            test_case = result.test_case
            status_class = "success" if result.success else "failure"
            status_text = "Passed" if result.success else "Failed"
            
            html += f"""
                <tr>
                    <td>{test_case.name}</td>
                    <td>{test_case.description}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result.duration:.3f}</td>
                    <td>{"<br>".join(result.errors) if result.errors else ""}</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(html_path, 'w') as f:
            f.write(html)
            
        logger.info(f"Generated HTML report at {html_path}")
        return html_path
    
    def register_custom_validation(self, name: str, validation_func: Callable) -> None:
        """
        Register a custom validation function for test results.
        
        Args:
            name: Name of the validation function
            validation_func: Validation function that takes (expected, actual) and returns (success, error_message)
        """
        if not callable(validation_func):
            logger.error("Validation function must be callable")
            return
            
        # Store the validation function
        if not hasattr(self, "_custom_validations"):
            self._custom_validations = {}
            
        self._custom_validations[name] = validation_func
        logger.info(f"Registered custom validation function: {name}")


# Create helper functions for CI integration

async def run_ci_tests(test_suite_path: str, output_dir: str = "test_output") -> bool:
    """
    Run tests in a CI environment.
    
    Args:
        test_suite_path: Path to the test suite JSON file
        output_dir: Output directory for test results
        
    Returns:
        True if all tests passed, False otherwise
    """
    framework = TestingFramework(test_output_dir=output_dir)
    
    if not framework.load_test_suite_from_file(test_suite_path):
        logger.error(f"Failed to load test suite from {test_suite_path}")
        return False
        
    suite_name = next(iter(framework.test_suites.keys()))
    report = await framework.run_test_suite(suite_name)
    
    # Generate HTML report
    await framework.generate_test_report_html(report)
    
    # Return success/failure
    return report.success_rate == 100.0 