#!/usr/bin/env python3
"""
Multi-Agent Assessment Framework

Main entry point for the framework. This script provides a command-line interface
to run benchmarks, analyze results, and generate reports.
"""

import os
import sys
import argparse
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark.run_benchmark import FRAMEWORKS, USE_CASES, run_benchmark
from src.utils.metrics import BenchmarkAnalyzer

# Load environment variables
load_dotenv()

def run_benchmarks(args) -> None:
    """
    Run benchmarks based on command-line arguments.
    
    Args:
        args: Command-line arguments
    """
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

def analyze_results(args) -> None:
    """
    Analyze benchmark results and generate reports.
    
    Args:
        args: Command-line arguments
    """
    analyzer = BenchmarkAnalyzer(results_dir=args.results_dir)
    
    if args.list:
        # List available results
        df = analyzer.get_dataframe()
        if df.empty:
            print("No benchmark results found.")
            return
        
        print("\nAvailable benchmark results:")
        print(f"Total results: {len(df)}")
        print("\nFrameworks:")
        for framework in df["framework"].unique():
            count = len(df[df["framework"] == framework])
            print(f"  - {framework} ({count} results)")
        
        print("\nUse cases:")
        for use_case in df["use_case"].unique():
            count = len(df[df["use_case"] == use_case])
            print(f"  - {use_case} ({count} results)")
        
        print("\nModels:")
        for model in df["model"].unique():
            count = len(df[df["model"] == model])
            print(f"  - {model} ({count} results)")
        
        return
    
    # Generate report
    report_path = analyzer.generate_report(output_file=args.output)
    print(f"Report generated: {report_path}")
    
    # Generate plots if requested
    if args.plots:
        for metric in ["execution_time", "total_tokens", "cost"]:
            analyzer.plot_comparison(metric=metric, use_case=args.use_case)
            print(f"Plot generated for {metric}")

def main():
    """Main entry point for the framework."""
    parser = argparse.ArgumentParser(description="Multi-Agent Assessment Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--framework", choices=FRAMEWORKS, default="all",
                                 help="Framework to benchmark (default: all)")
    benchmark_parser.add_argument("--use-case", choices=list(USE_CASES.keys()), required=True,
                                 help="Use case to run")
    benchmark_parser.add_argument("--runs", type=int, default=int(os.getenv("BENCHMARK_RUNS", "1")),
                                 help="Number of runs per benchmark (default: from BENCHMARK_RUNS env var or 1)")
    benchmark_parser.add_argument("--save-results", action="store_true", default=True,
                                 help="Save results to files (default: True)")
    benchmark_parser.add_argument("--no-save-results", action="store_false", dest="save_results",
                                 help="Don't save results to files")
    benchmark_parser.add_argument("--topic", type=str,
                                 help="Topic for article writing use case")
    benchmark_parser.add_argument("--llm-provider", choices=["openai", "groq"], default="openai",
                                 help="LLM provider to use (default: openai)")
    benchmark_parser.add_argument("--model", type=str, default=os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"),
                                 help="Model to use (default: from DEFAULT_MODEL env var or gpt-3.5-turbo)")
    benchmark_parser.add_argument("--generate-report", action="store_true",
                                 help="Generate a report after running benchmarks")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze benchmark results")
    analyze_parser.add_argument("--results-dir", default="results",
                               help="Directory containing benchmark results (default: results)")
    analyze_parser.add_argument("--output", default="results/benchmark_report.md",
                               help="Output file for the report (default: results/benchmark_report.md)")
    analyze_parser.add_argument("--use-case", type=str,
                               help="Filter results by use case")
    analyze_parser.add_argument("--plots", action="store_true",
                               help="Generate plots")
    analyze_parser.add_argument("--list", action="store_true",
                               help="List available benchmark results")
    
    args = parser.parse_args()
    
    if args.command == "benchmark":
        run_benchmarks(args)
    elif args.command == "analyze":
        analyze_results(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 