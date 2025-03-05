#!/usr/bin/env python3
"""
Benchmark runner for agentic frameworks.

This script runs benchmarks for different frameworks and use cases,
collects metrics, and generates reports.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any, Optional
import importlib
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.metrics import BenchmarkAnalyzer
from src.benchmark.use_cases.base_use_case import BaseUseCase

# Load environment variables
load_dotenv()

# Available frameworks
FRAMEWORKS = ["autogen", "semantic_kernel", "langchain", "crewai", "all"]

# Available use cases
USE_CASES = {
    "article_writing": "src.benchmark.use_cases.article_writing.ArticleWritingUseCase",
    # Add more use cases as they are implemented
}

def load_use_case(use_case_name: str) -> Optional[BaseUseCase]:
    """
    Load a use case class by name.
    
    Args:
        use_case_name: Name of the use case to load
        
    Returns:
        Instance of the use case class, or None if not found
    """
    if use_case_name not in USE_CASES:
        print(f"Error: Use case '{use_case_name}' not found.")
        return None
    
    try:
        # Get the module path and class name
        module_path = USE_CASES[use_case_name]
        module_name, class_name = module_path.rsplit(".", 1)
        
        # Import the module and get the class
        module = importlib.import_module(module_name)
        use_case_class = getattr(module, class_name)
        
        # Create an instance of the use case
        return use_case_class()
    except (ImportError, AttributeError) as e:
        print(f"Error loading use case '{use_case_name}': {e}")
        return None

def run_benchmark(
    framework: str,
    use_case_name: str,
    runs: int = 1,
    save_results: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Run a benchmark for a specific framework and use case.
    
    Args:
        framework: Name of the framework to benchmark
        use_case_name: Name of the use case to run
        runs: Number of times to run the benchmark
        save_results: Whether to save the results to files
        **kwargs: Additional arguments to pass to the use case
        
    Returns:
        List of benchmark results
    """
    use_case = load_use_case(use_case_name)
    if not use_case:
        return []
    
    results = []
    
    # Set inputs from kwargs
    use_case.set_inputs(kwargs)
    
    # Run the benchmark multiple times if requested
    for i in range(runs):
        print(f"Running {framework} on {use_case_name} (run {i+1}/{runs})...")
        start_time = time.time()
        
        try:
            result = use_case.run(framework, save_results=save_results)
            results.append(result)
            
            # Print summary of this run
            print(f"  Execution time: {result.get('execution_time', 0):.2f} seconds")
            print(f"  Token usage: {result.get('token_usage', {}).get('total_tokens', 0)} tokens")
            print(f"  Cost: ${result.get('cost', 0):.6f}")
        except Exception as e:
            print(f"Error running benchmark: {e}")
        
        print(f"  Total time: {time.time() - start_time:.2f} seconds")
        print()
    
    return results

def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run benchmarks for agentic frameworks.")
    parser.add_argument("--framework", choices=FRAMEWORKS, default="all",
                        help="Framework to benchmark (default: all)")
    parser.add_argument("--use-case", choices=list(USE_CASES.keys()), required=True,
                        help="Use case to run")
    parser.add_argument("--runs", type=int, default=int(os.getenv("BENCHMARK_RUNS", "1")),
                        help="Number of runs per benchmark (default: from BENCHMARK_RUNS env var or 1)")
    parser.add_argument("--save-results", action="store_true", default=True,
                        help="Save results to files (default: True)")
    parser.add_argument("--no-save-results", action="store_false", dest="save_results",
                        help="Don't save results to files")
    parser.add_argument("--topic", type=str,
                        help="Topic for article writing use case")
    parser.add_argument("--llm-provider", choices=["openai", "groq"], default="openai",
                        help="LLM provider to use (default: openai)")
    parser.add_argument("--model", type=str, default=os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"),
                        help="Model to use (default: from DEFAULT_MODEL env var or gpt-3.5-turbo)")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate a report after running benchmarks")
    
    args = parser.parse_args()
    
    # Determine which frameworks to run
    frameworks_to_run = FRAMEWORKS[:-1] if args.framework == "all" else [args.framework]
    
    # Collect all results
    all_results = []
    
    # Run benchmarks for each framework
    for framework in frameworks_to_run:
        results = run_benchmark(
            framework=framework,
            use_case_name=args.use_case,
            runs=args.runs,
            save_results=args.save_results,
            topic=args.topic,
            llm_provider=args.llm_provider,
            model=args.model
        )
        all_results.extend(results)
    
    # Generate report if requested
    if args.generate_report and all_results:
        analyzer = BenchmarkAnalyzer()
        report_path = analyzer.generate_report()
        print(f"Report generated: {report_path}")
        
        # Generate plots
        for metric in ["execution_time", "total_tokens", "cost"]:
            analyzer.plot_comparison(metric=metric, use_case=args.use_case)

if __name__ == "__main__":
    main() 