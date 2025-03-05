"""
Metrics utilities for measuring performance of agentic frameworks.
"""

import time
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import glob
import seaborn as sns

# Load environment variables
load_dotenv()

class PerformanceTracker:
    """Track performance metrics for framework benchmarks."""
    
    def __init__(self, framework_name: str, use_case: str):
        """
        Initialize a performance tracker.
        
        Args:
            framework_name: Name of the framework being benchmarked
            use_case: Name of the use case being tested
        """
        self.framework_name = framework_name
        self.use_case = use_case
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.cost = 0.0
        self.model_name = None
        self.responses = []
        self.metadata = {}
    
    def start(self) -> None:
        """Start the performance timer."""
        self.start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop the performance timer.
        
        Returns:
            The execution time in seconds
        """
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        return self.execution_time
    
    def add_response(self, response: Any, is_final: bool = False) -> None:
        """
        Add a response from the framework.
        
        Args:
            response: The response object or text
            is_final: Whether this is the final response
        """
        self.responses.append({
            "content": response if isinstance(response, str) else str(response),
            "is_final": is_final,
            "timestamp": time.time()
        })
    
    def update_token_usage(self, 
                          prompt_tokens: int = 0, 
                          completion_tokens: int = 0,
                          total_tokens: Optional[int] = None) -> None:
        """
        Update token usage metrics.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (if provided directly)
        """
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        
        if total_tokens is not None:
            self.token_usage["total_tokens"] = total_tokens
        else:
            self.token_usage["total_tokens"] = self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"]
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The estimated number of tokens
        """
        try:
            import tiktoken
            # Use cl100k_base encoding (used by gpt-3.5-turbo and gpt-4)
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: estimate tokens as words / 0.75
            return int(len(text.split()) / 0.75)
    
    def calculate_cost(self, model_name: str) -> float:
        """
        Calculate the cost based on token usage and model.
        
        Args:
            model_name: Name of the model used
            
        Returns:
            The estimated cost in USD
        """
        self.model_name = model_name
        
        # Get cost rates from environment variables
        if "gpt-4" in model_name.lower():
            input_cost = float(os.getenv("GPT4_INPUT_COST", "0.03"))
            output_cost = float(os.getenv("GPT4_OUTPUT_COST", "0.06"))
        elif "gpt-3.5" in model_name.lower() or "gpt35" in model_name.lower():
            input_cost = float(os.getenv("GPT35_INPUT_COST", "0.0015"))
            output_cost = float(os.getenv("GPT35_OUTPUT_COST", "0.002"))
        elif "groq" in model_name.lower() or "llama" in model_name.lower():
            # Assuming flat rate for Groq
            flat_rate = float(os.getenv("GROQ_COST", "0.0007"))
            self.cost = (self.token_usage["total_tokens"] / 1000) * flat_rate
            return self.cost
        else:
            # Default rates if model not recognized
            input_cost = 0.002
            output_cost = 0.002
        
        # Calculate cost
        prompt_cost = (self.token_usage["prompt_tokens"] / 1000) * input_cost
        completion_cost = (self.token_usage["completion_tokens"] / 1000) * output_cost
        self.cost = prompt_cost + completion_cost
        
        return self.cost
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set additional metadata for the benchmark.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get the complete results of the benchmark.
        
        Returns:
            Dictionary with all benchmark results
        """
        return {
            "framework": self.framework_name,
            "use_case": self.use_case,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "cost": self.cost,
            "model": self.model_name,
            "responses": self.responses,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, output_dir: str = "results") -> str:
        """
        Save the benchmark results to a file.
        
        Args:
            output_dir: Directory to save results in
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.framework_name}_{self.use_case}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save results as JSON
        with open(filepath, 'w') as f:
            json.dump(self.get_results(), f, indent=2)
        
        return filepath


class BenchmarkAnalyzer:
    """Analyze and visualize benchmark results."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize a benchmark analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = results_dir
        self.results = []
        self.load_results()
    
    def load_results(self) -> None:
        """Load all benchmark results from the results directory."""
        if not os.path.exists(self.results_dir):
            return
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        result = json.load(f)
                        self.results.append(result)
                    except json.JSONDecodeError:
                        print(f"Error loading {filepath}")
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.
        
        Returns:
            DataFrame with benchmark results
        """
        # Extract key metrics from results
        data = []
        for result in self.results:
            row = {
                "framework": result.get("framework"),
                "use_case": result.get("use_case"),
                "execution_time": result.get("execution_time"),
                "total_tokens": result.get("token_usage", {}).get("total_tokens"),
                "prompt_tokens": result.get("token_usage", {}).get("prompt_tokens"),
                "completion_tokens": result.get("token_usage", {}).get("completion_tokens"),
                "cost": result.get("cost"),
                "model": result.get("model"),
                "timestamp": result.get("timestamp")
            }
            
            # Add metadata fields
            for key, value in result.get("metadata", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    row[key] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def compare_frameworks(self, use_case: Optional[str] = None) -> pd.DataFrame:
        """
        Compare frameworks for a specific use case.
        
        Args:
            use_case: Filter by use case (or None for all)
            
        Returns:
            DataFrame with comparison results
        """
        df = self.get_dataframe()
        
        if use_case:
            df = df[df["use_case"] == use_case]
        
        # Group by framework and aggregate
        comparison = df.groupby("framework").agg({
            "execution_time": ["mean", "min", "max"],
            "total_tokens": ["mean", "min", "max"],
            "cost": ["mean", "min", "max"],
            "framework": "count"  # Count of runs
        })
        
        # Rename columns
        comparison.columns = [
            "avg_time", "min_time", "max_time",
            "avg_tokens", "min_tokens", "max_tokens",
            "avg_cost", "min_cost", "max_cost",
            "runs"
        ]
        
        return comparison
    
    def plot_comparison(self, metric: str = "execution_time", use_case: Optional[str] = None) -> None:
        """
        Plot a comparison of frameworks by the specified metric.
        
        Args:
            metric: The metric to compare ('execution_time', 'total_tokens', or 'cost')
            use_case: Filter by use case (or None for all)
        """
        df = self.get_dataframe()
        
        if use_case:
            df = df[df["use_case"] == use_case]
            title_suffix = f" for {use_case}"
        else:
            title_suffix = " across all use cases"
        
        # Map metric to readable name
        metric_names = {
            "execution_time": "Execution Time (s)",
            "total_tokens": "Total Tokens",
            "cost": "Cost (USD)"
        }
        
        metric_name = metric_names.get(metric, metric)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Bar chart
        ax = df.groupby("framework")[metric].mean().plot(kind="bar")
        
        # Add value labels on top of bars
        for i, v in enumerate(df.groupby("framework")[metric].mean()):
            ax.text(i, v * 1.05, f"{v:.4f}", ha="center")
        
        plt.title(f"Average {metric_name} by Framework{title_suffix}")
        plt.ylabel(metric_name)
        plt.xlabel("Framework")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs("results/plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"results/plots/comparison_{metric}_{timestamp}.png")
        plt.show()
    
    def generate_report(self, output_file: str = "results/benchmark_report.md") -> str:
        """
        Generate a markdown report of benchmark results.
        
        Args:
            output_file: Path to save the report
            
        Returns:
            Path to the saved report
        """
        df = self.get_dataframe()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("# Agentic Framework Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Overall Summary\n\n")
            f.write(f"- Total benchmarks run: {len(self.results)}\n")
            f.write(f"- Frameworks tested: {', '.join(df['framework'].unique())}\n")
            f.write(f"- Use cases tested: {', '.join(df['use_case'].unique())}\n\n")
            
            # Framework comparison
            f.write("## Framework Comparison\n\n")
            comparison = self.compare_frameworks()
            f.write(comparison.to_markdown())
            f.write("\n\n")
            
            # Per use case comparison
            f.write("## Results by Use Case\n\n")
            for use_case in df['use_case'].unique():
                f.write(f"### {use_case}\n\n")
                case_comparison = self.compare_frameworks(use_case)
                f.write(case_comparison.to_markdown())
                f.write("\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### {result['framework']} - {result['use_case']}\n\n")
                f.write(f"- Execution time: {result['execution_time']:.4f} seconds\n")
                f.write(f"- Token usage: {result['token_usage']['total_tokens']} tokens\n")
                f.write(f"- Cost: ${result['cost']:.6f}\n")
                f.write(f"- Model: {result['model']}\n")
                
                # Add metadata
                if result['metadata']:
                    f.write("- Additional metrics:\n")
                    for key, value in result['metadata'].items():
                        f.write(f"  - {key}: {value}\n")
                
                f.write("\n")
        
        return output_file

    def _ensure_results_dir(self) -> None:
        """Ensure that the results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _load_results(self) -> List[Dict[str, Any]]:
        """
        Load benchmark results from files.
        
        Returns:
            List[Dict[str, Any]]: List of benchmark results.
        """
        results = []
        result_files = glob.glob(os.path.join(self.results_dir, "*.json"))
        
        for file_path in result_files:
            try:
                with open(file_path, "r") as f:
                    result = json.load(f)
                    results.append(result)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading {file_path}: {e}")
        
        return results
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate a report from benchmark results.
        
        Args:
            output_file (str, optional): Output file path. If None, a default path
                will be used. Defaults to None.
        
        Returns:
            str: Path to the generated report.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.results_dir, f"benchmark_report_{timestamp}.md")
        
        df = self.get_dataframe()
        
        if df.empty:
            print("No benchmark results found.")
            return None
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write("# Multi-Agent Assessment Framework Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"Total benchmarks: {len(df)}\n")
            f.write(f"Frameworks: {', '.join(df['framework'].unique())}\n")
            f.write(f"Use cases: {', '.join(df['use_case'].unique())}\n")
            f.write(f"Models: {', '.join(df['model'].unique())}\n\n")
            
            # Overall performance metrics
            f.write("## Overall Performance Metrics\n\n")
            
            # Execution time
            f.write("### Execution Time (seconds)\n\n")
            f.write("| Framework | Mean | Median | Min | Max |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            
            for framework in df['framework'].unique():
                framework_df = df[df['framework'] == framework]
                f.write(f"| {framework} | {framework_df['execution_time'].mean():.2f} | "
                        f"{framework_df['execution_time'].median():.2f} | "
                        f"{framework_df['execution_time'].min():.2f} | "
                        f"{framework_df['execution_time'].max():.2f} |\n")
            
            f.write("\n")
            
            # Token usage
            f.write("### Token Usage\n\n")
            f.write("| Framework | Mean Prompt | Mean Completion | Mean Total |\n")
            f.write("| --- | --- | --- | --- |\n")
            
            for framework in df['framework'].unique():
                framework_df = df[df['framework'] == framework]
                f.write(f"| {framework} | {framework_df['prompt_tokens'].mean():.0f} | "
                        f"{framework_df['completion_tokens'].mean():.0f} | "
                        f"{framework_df['total_tokens'].mean():.0f} |\n")
            
            f.write("\n")
            
            # Cost
            f.write("### Cost (USD)\n\n")
            f.write("| Framework | Mean | Median | Min | Max | Total |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            
            for framework in df['framework'].unique():
                framework_df = df[df['framework'] == framework]
                f.write(f"| {framework} | {framework_df['cost'].mean():.4f} | "
                        f"{framework_df['cost'].median():.4f} | "
                        f"{framework_df['cost'].min():.4f} | "
                        f"{framework_df['cost'].max():.4f} | "
                        f"{framework_df['cost'].sum():.4f} |\n")
            
            f.write("\n")
            
            # Per use case analysis
            f.write("## Per Use Case Analysis\n\n")
            
            for use_case in df['use_case'].unique():
                use_case_df = df[df['use_case'] == use_case]
                
                f.write(f"### {use_case}\n\n")
                
                # Execution time
                f.write("#### Execution Time (seconds)\n\n")
                f.write("| Framework | Mean | Median | Min | Max |\n")
                f.write("| --- | --- | --- | --- | --- | --- |\n")
                
                for framework in use_case_df['framework'].unique():
                    framework_df = use_case_df[use_case_df['framework'] == framework]
                    f.write(f"| {framework} | {framework_df['execution_time'].mean():.2f} | "
                            f"{framework_df['execution_time'].median():.2f} | "
                            f"{framework_df['execution_time'].min():.2f} | "
                            f"{framework_df['execution_time'].max():.2f} |\n")
                
                f.write("\n")
                
                # Token usage
                f.write("#### Token Usage\n\n")
                f.write("| Framework | Mean Prompt | Mean Completion | Mean Total |\n")
                f.write("| --- | --- | --- | --- |\n")
                
                for framework in use_case_df['framework'].unique():
                    framework_df = use_case_df[use_case_df['framework'] == framework]
                    f.write(f"| {framework} | {framework_df['prompt_tokens'].mean():.0f} | "
                            f"{framework_df['completion_tokens'].mean():.0f} | "
                            f"{framework_df['total_tokens'].mean():.0f} |\n")
                
                f.write("\n")
                
                # Cost
                f.write("#### Cost (USD)\n\n")
                f.write("| Framework | Mean | Median | Min | Max | Total |\n")
                f.write("| --- | --- | --- | --- | --- | --- |\n")
                
                for framework in use_case_df['framework'].unique():
                    framework_df = use_case_df[use_case_df['framework'] == framework]
                    f.write(f"| {framework} | {framework_df['cost'].mean():.4f} | "
                            f"{framework_df['cost'].median():.4f} | "
                            f"{framework_df['cost'].min():.4f} | "
                            f"{framework_df['cost'].max():.4f} | "
                            f"{framework_df['cost'].sum():.4f} |\n")
                
                f.write("\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("This report provides a comparison of different multi-agent frameworks ")
            f.write("based on execution time, token usage, and cost. ")
            f.write("For more detailed analysis, refer to the plots generated alongside this report.\n")
        
        print(f"Report generated: {output_file}")
        return output_file
    
    def plot_comparison(self, metric: str, use_case: Optional[str] = None, 
                        output_dir: Optional[str] = None) -> str:
        """
        Generate a plot comparing frameworks based on a specific metric.
        
        Args:
            metric (str): Metric to compare (e.g., "execution_time", "total_tokens", "cost").
            use_case (Optional[str], optional): Filter by use case. Defaults to None.
            output_dir (Optional[str], optional): Output directory. Defaults to None.
        
        Returns:
            str: Path to the generated plot.
        """
        df = self.get_dataframe()
        
        if df.empty:
            print("No benchmark results found.")
            return None
        
        if use_case:
            df = df[df['use_case'] == use_case]
            if df.empty:
                print(f"No results found for use case: {use_case}")
                return None
        
        if output_dir is None:
            output_dir = self.results_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plot
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Create the plot
        if metric in df.columns:
            ax = sns.boxplot(x='framework', y=metric, data=df)
            ax.set_title(f"Comparison of {metric.replace('_', ' ').title()} by Framework")
            ax.set_xlabel("Framework")
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Add use case to title if specified
            if use_case:
                ax.set_title(f"Comparison of {metric.replace('_', ' ').title()} by Framework for {use_case}")
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            use_case_suffix = f"_{use_case}" if use_case else ""
            output_file = os.path.join(output_dir, f"{metric}_comparison{use_case_suffix}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            
            print(f"Plot saved to: {output_file}")
            return output_file
        else:
            print(f"Metric '{metric}' not found in benchmark results.")
            return None 