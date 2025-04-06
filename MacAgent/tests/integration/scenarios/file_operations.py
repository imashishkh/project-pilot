#!/usr/bin/env python3
"""
Test scenario for file operations:
- Creating files
- Reading files
- Writing to files
- Deleting files
"""

import os
import sys
import logging
import tempfile
import shutil
import time

from MacAgent.src.core.agent import MacAgent
from MacAgent.src.interaction.file_manager import FileManager

class FileOperationsScenario:
    """Test scenario for basic file operations."""
    
    def __init__(self, agent):
        """Initialize with a MacAgent instance."""
        self.agent = agent
        self.file_manager = FileManager()
        self.logger = logging.getLogger(__name__)
        self.test_dir = None
        self.test_files = []
        self.results = {
            "create_file": {},
            "write_file": {},
            "read_file": {},
            "delete_file": {}
        }
    
    async def setup(self):
        """Prepare the environment for testing."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp(prefix="macagent_test_")
        self.logger.info(f"Created test directory: {self.test_dir}")
        
        # Define test file paths
        self.test_files = [
            os.path.join(self.test_dir, "test_file1.txt"),
            os.path.join(self.test_dir, "test_file2.txt"),
            os.path.join(self.test_dir, "test_file3.txt")
        ]
        
        # Create subdirectory for testing
        self.sub_dir = os.path.join(self.test_dir, "subdir")
        os.makedirs(self.sub_dir, exist_ok=True)
        self.test_files.append(os.path.join(self.sub_dir, "nested_file.txt"))
    
    async def run_test(self):
        """Execute the test scenario."""
        await self.setup()
        
        # Test 1: Creating files
        for file_path in self.test_files:
            try:
                self.logger.info(f"Creating file: {file_path}")
                success = await self.file_manager.create_file(file_path)
                self.results["create_file"][file_path] = success
            except Exception as e:
                self.logger.error(f"Error creating file {file_path}: {e}")
                self.results["create_file"][file_path] = False
        
        # Test 2: Writing to files
        for i, file_path in enumerate(self.test_files):
            content = f"This is test content for file {i+1}.\nLine 2 of file {i+1}.\nLine 3 of file {i+1}."
            try:
                self.logger.info(f"Writing to file: {file_path}")
                success = await self.file_manager.write_to_file(file_path, content)
                self.results["write_file"][file_path] = success
            except Exception as e:
                self.logger.error(f"Error writing to file {file_path}: {e}")
                self.results["write_file"][file_path] = False
        
        # Test 3: Reading files
        for file_path in self.test_files:
            try:
                self.logger.info(f"Reading file: {file_path}")
                content = await self.file_manager.read_file(file_path)
                # Check if content is not empty
                self.results["read_file"][file_path] = content is not None and len(content) > 0
            except Exception as e:
                self.logger.error(f"Error reading file {file_path}: {e}")
                self.results["read_file"][file_path] = False
        
        # Test 4: Deleting files
        for file_path in self.test_files:
            try:
                self.logger.info(f"Deleting file: {file_path}")
                success = await self.file_manager.delete_file(file_path)
                self.results["delete_file"][file_path] = success
            except Exception as e:
                self.logger.error(f"Error deleting file {file_path}: {e}")
                self.results["delete_file"][file_path] = False
        
        return self.results
    
    async def cleanup(self):
        """Clean up after tests."""
        if self.test_dir and os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
                self.logger.info(f"Removed test directory: {self.test_dir}")
            except Exception as e:
                self.logger.error(f"Error removing test directory: {e}")
    
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
        scenario = FileOperationsScenario(agent)
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
                    base_name = os.path.basename(name)
                    print(f"  {base_name}: {'✅ Success' if success else '❌ Failed'}")
            
        finally:
            await scenario.cleanup()
    
    # Run the scenario
    asyncio.run(run_scenario()) 