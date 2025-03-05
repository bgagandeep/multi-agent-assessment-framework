#!/usr/bin/env python3
"""
Example script demonstrating how to use the Multi-Agent Assessment Framework
for the collaborative article writing use case.

This example shows how to run the article writing use case with different frameworks
and compare their performance.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmark.use_cases.article_writing import ArticleWritingUseCase
from src.utils.metrics import PerformanceTracker

# Load environment variables
load_dotenv()

def run_example():
    """Run the article writing example with different frameworks."""
    # Define the topic for the article
    topic = "The Impact of Artificial Intelligence on Modern Healthcare"
    
    # Create the use case instance
    use_case = ArticleWritingUseCase()
    
    # Set up inputs
    inputs = {"topic": topic}
    use_case.set_inputs(inputs)
    
    # List of frameworks to test
    frameworks = ["autogen", "semantic_kernel", "langchain", "crewai"]
    
    results = {}
    
    # Run the use case with each framework
    for framework in frameworks:
        print(f"\n{'=' * 80}")
        print(f"Running article writing use case with {framework.upper()}")
        print(f"{'=' * 80}\n")
        
        # Create a performance tracker
        tracker = PerformanceTracker(framework_name=framework, use_case=use_case.name)
        
        try:
            # Start tracking performance
            tracker.start()
            
            # Run the use case
            result = use_case.run(framework=framework)
            
            # Stop tracking performance
            execution_time = tracker.stop()
            
            # Print the result
            print(f"\nArticle generated with {framework}:")
            print(f"{'=' * 40}")
            print(result)
            print(f"{'=' * 40}")
            
            # Print performance metrics
            print(f"\nPerformance metrics for {framework}:")
            print(f"Execution time: {execution_time:.2f} seconds")
            
            # Get token usage from result if available
            if isinstance(result, dict) and 'token_usage' in result:
                token_usage = result['token_usage']
                print(f"Prompt tokens: {token_usage.get('prompt_tokens', 0)}")
                print(f"Completion tokens: {token_usage.get('completion_tokens', 0)}")
                print(f"Total tokens: {token_usage.get('total_tokens', 0)}")
                print(f"Cost: ${result.get('cost', 0):.4f}")
            else:
                print(f"Prompt tokens: {tracker.token_usage['prompt_tokens']}")
                print(f"Completion tokens: {tracker.token_usage['completion_tokens']}")
                print(f"Total tokens: {tracker.token_usage['total_tokens']}")
                print(f"Cost: ${tracker.cost:.4f}")
            
            # Store results
            results[framework] = {
                "execution_time": execution_time,
                "prompt_tokens": token_usage.get('prompt_tokens', 0) if isinstance(result, dict) and 'token_usage' in result else tracker.token_usage['prompt_tokens'],
                "completion_tokens": token_usage.get('completion_tokens', 0) if isinstance(result, dict) and 'token_usage' in result else tracker.token_usage['completion_tokens'],
                "total_tokens": token_usage.get('total_tokens', 0) if isinstance(result, dict) and 'token_usage' in result else tracker.token_usage['total_tokens'],
                "cost": result.get('cost', 0) if isinstance(result, dict) and 'cost' in result else tracker.cost
            }
            
        except Exception as e:
            print(f"Error running {framework}: {str(e)}")
    
    # Compare results
    if results:
        print("\n\nComparison of frameworks:")
        print(f"{'Framework':<20} {'Time (s)':<10} {'Tokens':<10} {'Cost ($)':<10}")
        print("-" * 50)
        
        for framework, metrics in results.items():
            print(f"{framework:<20} {metrics['execution_time']:<10.2f} {metrics['total_tokens']:<10} {metrics['cost']:<10.4f}")

if __name__ == "__main__":
    run_example() 