#!/usr/bin/env python3
"""
Main test runner script for MacAgent.
Runs all test scenarios and generates reports.
"""

import os
import sys
import logging
import asyncio
import argparse
from pathlib import Path

from MacAgent.tests.framework.test_runner import TestRunner
from MacAgent.tests.framework.result_analyzer import ResultAnalyzer


async def run_tests(output_dir: str, scenarios_package: str, generate_analysis: bool = True):
    """Run all tests and optionally generate analysis."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "test_run.log"))
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting test run")
    
    # Create test runner
    runner = TestRunner(output_dir=output_dir)
    
    try:
        # Run all tests
        logger.info(f"Running tests from package: {scenarios_package}")
        results = await runner.discover_and_run_scenarios(scenarios_package=scenarios_package)
        
        # Generate reports
        logger.info("Generating reports")
        text_report = runner.generate_report("text")
        json_report = runner.generate_report("json")
        
        # Print summary
        total = len(results)
        passed = sum(1 for r in results if r.success)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        logger.info(f"Test run completed: {passed}/{total} passed ({success_rate:.1f}%)")
        logger.info(f"Text report: {text_report}")
        logger.info(f"JSON report: {json_report}")
        
        # Generate analysis if requested
        if generate_analysis:
            logger.info("Generating test analysis")
            analyzer = ResultAnalyzer(reports_dir=output_dir)
            analysis = analyzer.generate_analysis_report(max_age_days=30)
            analysis_path = analyzer.save_analysis_report(analysis)
            
            logger.info(f"Analysis report: {analysis_path}")
            
            # Print recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                logger.info("Test Analysis Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    logger.info(f"  {i}. {rec}")
        
        return success_rate
    
    except Exception as e:
        logger.error(f"Error during test run: {e}", exc_info=True)
        return 0.0


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MacAgent tests")
    parser.add_argument(
        "--output-dir", 
        default="MacAgent/tests/output",
        help="Directory for test reports and logs"
    )
    parser.add_argument(
        "--package", 
        default="MacAgent.tests.integration.scenarios",
        help="Package containing test scenarios"
    )
    parser.add_argument(
        "--no-analysis", 
        action="store_true",
        help="Skip generating analysis report"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    success_rate = asyncio.run(
        run_tests(
            output_dir=args.output_dir,
            scenarios_package=args.package,
            generate_analysis=not args.no_analysis
        )
    )
    
    # Exit with appropriate code (0 for success, 1 for failure)
    sys.exit(0 if success_rate >= 80.0 else 1)


if __name__ == "__main__":
    main() 