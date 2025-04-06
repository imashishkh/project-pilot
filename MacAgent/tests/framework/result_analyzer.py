#!/usr/bin/env python3
"""
Result Analyzer for MacAgent test framework.
Analyzes test results to identify patterns and provide recommendations.
"""

import os
import sys
import json
import logging
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta


class ResultAnalyzer:
    """Analyzer for test results, identifying patterns and providing recommendations."""
    
    def __init__(self, reports_dir: str = "MacAgent/tests/output"):
        """Initialize the result analyzer."""
        self.logger = logging.getLogger(__name__)
        self.reports_dir = reports_dir
        
        # Ensure reports directory exists
        if not os.path.exists(reports_dir):
            self.logger.warning(f"Reports directory {reports_dir} does not exist.")
            os.makedirs(reports_dir, exist_ok=True)
    
    def load_reports(self, pattern: str = "test_report_*.json", max_age_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load test reports matching the pattern and within the age limit."""
        reports = []
        report_files = glob.glob(os.path.join(self.reports_dir, pattern))
        
        cutoff_date = None
        if max_age_days is not None:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for report_file in report_files:
            file_stat = os.stat(report_file)
            file_date = datetime.fromtimestamp(file_stat.st_mtime)
            
            # Skip files older than the cutoff date
            if cutoff_date and file_date < cutoff_date:
                continue
            
            try:
                with open(report_file, "r") as f:
                    report_data = json.load(f)
                    reports.append(report_data)
            except Exception as e:
                self.logger.error(f"Error loading report {report_file}: {e}")
        
        reports.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        self.logger.info(f"Loaded {len(reports)} reports")
        return reports
    
    def analyze_success_trend(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the trend of success rates over time."""
        if not reports:
            return {"trends": [], "overall_direction": "unknown"}
        
        # Extract success rates by timestamp
        success_rates = []
        for report in reports:
            timestamp = report.get("timestamp")
            try:
                timestamp_dt = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                continue
            
            total = report.get("total_scenarios", 0)
            passed = report.get("passed_scenarios", 0)
            success_rate = (passed / total) * 100 if total > 0 else 0
            
            success_rates.append({
                "timestamp": timestamp,
                "success_rate": success_rate
            })
        
        # Sort by timestamp
        success_rates.sort(key=lambda x: x["timestamp"])
        
        # Calculate trend direction
        trend_direction = "stable"
        if len(success_rates) >= 2:
            first_rate = success_rates[0]["success_rate"]
            last_rate = success_rates[-1]["success_rate"]
            diff = last_rate - first_rate
            
            if diff > 5:
                trend_direction = "improving"
            elif diff < -5:
                trend_direction = "declining"
        
        return {
            "trends": success_rates,
            "overall_direction": trend_direction
        }
    
    def identify_common_failures(self, reports: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Identify scenarios that commonly fail."""
        if not reports:
            return []
        
        # Count failures by scenario
        scenario_failures = defaultdict(int)
        scenario_runs = defaultdict(int)
        
        for report in reports:
            scenarios = report.get("scenarios", [])
            for scenario in scenarios:
                scenario_name = scenario.get("scenario_name")
                if not scenario_name:
                    continue
                
                scenario_runs[scenario_name] += 1
                if not scenario.get("success", True):
                    scenario_failures[scenario_name] += 1
        
        # Calculate failure rates
        common_failures = []
        for scenario_name, runs in scenario_runs.items():
            failures = scenario_failures[scenario_name]
            failure_rate = failures / runs if runs > 0 else 0
            
            if failure_rate >= threshold:
                common_failures.append({
                    "scenario_name": scenario_name,
                    "runs": runs,
                    "failures": failures,
                    "failure_rate": failure_rate
                })
        
        # Sort by failure rate (highest first)
        common_failures.sort(key=lambda x: x["failure_rate"], reverse=True)
        return common_failures
    
    def identify_flaky_tests(self, reports: List[Dict[str, Any]], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Identify flaky tests (tests that sometimes pass, sometimes fail)."""
        if not reports:
            return []
        
        # Count passes and failures by scenario
        scenario_passes = defaultdict(int)
        scenario_failures = defaultdict(int)
        
        for report in reports:
            scenarios = report.get("scenarios", [])
            for scenario in scenarios:
                scenario_name = scenario.get("scenario_name")
                if not scenario_name:
                    continue
                
                if scenario.get("success", True):
                    scenario_passes[scenario_name] += 1
                else:
                    scenario_failures[scenario_name] += 1
        
        # Find flaky tests
        flaky_tests = []
        for scenario_name in set(list(scenario_passes.keys()) + list(scenario_failures.keys())):
            passes = scenario_passes[scenario_name]
            failures = scenario_failures[scenario_name]
            total = passes + failures
            
            # Skip tests with too few runs
            if total < 3:
                continue
            
            pass_rate = passes / total if total > 0 else 0
            
            # Tests that are neither consistently passing nor consistently failing
            if threshold <= pass_rate <= (1 - threshold):
                flaky_tests.append({
                    "scenario_name": scenario_name,
                    "total_runs": total,
                    "passes": passes,
                    "failures": failures,
                    "pass_rate": pass_rate
                })
        
        # Sort by "flakiness" (closest to 50% pass rate)
        flaky_tests.sort(key=lambda x: abs(x["pass_rate"] - 0.5))
        return flaky_tests
    
    def analyze_runtime_performance(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze runtime performance of test scenarios."""
        if not reports:
            return {"average_durations": [], "slowest_scenarios": []}
        
        # Collect durations by scenario
        scenario_durations = defaultdict(list)
        
        for report in reports:
            scenarios = report.get("scenarios", [])
            for scenario in scenarios:
                scenario_name = scenario.get("scenario_name")
                duration = scenario.get("duration_seconds")
                
                if scenario_name and duration is not None:
                    scenario_durations[scenario_name].append(duration)
        
        # Calculate average durations
        average_durations = []
        for scenario_name, durations in scenario_durations.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                average_durations.append({
                    "scenario_name": scenario_name,
                    "average_duration": avg_duration,
                    "runs": len(durations)
                })
        
        # Sort by average duration (descending)
        average_durations.sort(key=lambda x: x["average_duration"], reverse=True)
        
        # Get top 5 slowest scenarios
        slowest_scenarios = average_durations[:5] if len(average_durations) >= 5 else average_durations
        
        return {
            "average_durations": average_durations,
            "slowest_scenarios": slowest_scenarios
        }
    
    def generate_recommendations(self, reports: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        # Identify common failures
        common_failures = self.identify_common_failures(reports)
        if common_failures:
            failures_str = ", ".join([f.get("scenario_name", "Unknown") for f in common_failures[:3]])
            recommendations.append(f"Focus on fixing consistently failing scenarios: {failures_str}")
        
        # Identify flaky tests
        flaky_tests = self.identify_flaky_tests(reports)
        if flaky_tests:
            flaky_str = ", ".join([f.get("scenario_name", "Unknown") for f in flaky_tests[:3]])
            recommendations.append(f"Improve reliability of flaky tests: {flaky_str}")
        
        # Analyze performance
        performance = self.analyze_runtime_performance(reports)
        slow_tests = performance.get("slowest_scenarios", [])
        if slow_tests:
            slow_str = ", ".join([
                f"{t.get('scenario_name', 'Unknown')} ({t.get('average_duration', 0):.2f}s)"
                for t in slow_tests[:3]
            ])
            recommendations.append(f"Optimize slow-running tests: {slow_str}")
        
        # Analyze success trend
        trend = self.analyze_success_trend(reports)
        if trend.get("overall_direction") == "declining":
            recommendations.append("Overall test success rate is declining. Review recent changes to the codebase.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests are performing well. Continue monitoring.")
        
        return recommendations
    
    def generate_analysis_report(self, max_age_days: Optional[int] = 30) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        # Load reports
        reports = self.load_reports(max_age_days=max_age_days)
        
        if not reports:
            self.logger.warning("No reports found for analysis.")
            return {
                "timestamp": datetime.now().isoformat(),
                "message": "No reports found for analysis.",
                "recommendations": ["Set up test runs to generate reports for analysis."]
            }
        
        # Run various analyses
        success_trend = self.analyze_success_trend(reports)
        common_failures = self.identify_common_failures(reports)
        flaky_tests = self.identify_flaky_tests(reports)
        performance = self.analyze_runtime_performance(reports)
        recommendations = self.generate_recommendations(reports)
        
        # Compile results
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "reports_analyzed": len(reports),
            "date_range": {
                "start": reports[-1].get("timestamp") if reports else None,
                "end": reports[0].get("timestamp") if reports else None
            },
            "success_trend": success_trend,
            "common_failures": common_failures,
            "flaky_tests": flaky_tests,
            "performance": performance,
            "recommendations": recommendations
        }
        
        return analysis
    
    def save_analysis_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save the analysis report to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_analysis_{timestamp}.json"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Analysis report saved to: {filepath}")
        return filepath


# For standalone testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create analyzer and generate report
    analyzer = ResultAnalyzer()
    analysis = analyzer.generate_analysis_report()
    
    # Save report
    report_path = analyzer.save_analysis_report(analysis)
    
    # Print recommendations
    print("\nTest Analysis Recommendations:")
    print("=============================")
    for i, rec in enumerate(analysis.get("recommendations", []), 1):
        print(f"{i}. {rec}")
    
    print(f"\nFull analysis saved to: {report_path}") 