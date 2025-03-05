# Multi-Agent Assessment Framework

A comprehensive framework for evaluating and comparing different multi-agent frameworks including AutoGen, Microsoft Semantic Kernel, LangChain/LangGraph, and CrewAI.

## Overview

This project provides a structured approach to assess various multi-agent frameworks based on:

- Execution time
- Token usage
- Cost
- Scalability
- Ease of use
- Implementation complexity

The framework includes a standardized benchmark suite with common use cases implemented across all frameworks, allowing for fair and consistent comparisons.

## Features

- **Standardized Benchmarks**: Common use cases implemented across all frameworks
- **Performance Tracking**: Detailed metrics on execution time, token usage, and cost
- **Automated Reporting**: Generate comprehensive reports and visualizations
- **Extensible Design**: Easily add new frameworks or use cases
- **Configurable**: Support for different LLM providers and models

## Supported Frameworks

- [AutoGen](https://github.com/microsoft/autogen): Microsoft's framework for building agent-based applications
- [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel): A lightweight SDK for integrating AI services
- [LangChain/LangGraph](https://github.com/langchain-ai/langchain): Framework for developing applications powered by language models
- [CrewAI](https://github.com/joaomdmoura/crewAI): Framework for orchestrating role-playing autonomous AI agents

## Use Cases

The framework currently includes the following use cases:

1. **Collaborative Article Writing**: Multiple agents (Editor, Researcher, Coder, Writer) collaborate to produce an article on a specified topic.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-assessment-framework.git
   cd multi-agent-assessment-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Usage

### Running Benchmarks

To run benchmarks for all frameworks with the article writing use case:

```bash
python src/main.py benchmark --use-case article_writing --topic "Artificial Intelligence in Healthcare"
```

To run benchmarks for a specific framework:

```bash
python src/main.py benchmark --framework autogen --use-case article_writing --topic "Climate Change Solutions"
```

Additional options:
- `--runs`: Number of runs per benchmark (default: 1)
- `--llm-provider`: LLM provider to use (default: openai)
- `--model`: Model to use (default: gpt-3.5-turbo)
- `--generate-report`: Generate a report after running benchmarks

### Analyzing Results

To analyze benchmark results and generate a report:

```bash
python src/main.py analyze --plots
```

To list available benchmark results:

```bash
python src/main.py analyze --list
```

## Project Structure

```
multi-agent-assessment-framework/
├── src/
│   ├── frameworks/
│   │   ├── autogen/
│   │   ├── semantic_kernel/
│   │   ├── langchain/
│   │   └── crewai/
│   ├── benchmark/
│   │   ├── use_cases/
│   │   │   └── article_writing.py
│   │   └── run_benchmark.py
│   ├── utils/
│   │   ├── metrics.py
│   │   └── llm.py
│   └── main.py
├── results/
├── requirements.txt
├── .env.example
└── README.md
```

## Adding New Use Cases

To add a new use case:

1. Create a new file in `src/benchmark/use_cases/`
2. Implement the use case for all frameworks
3. Update the `USE_CASES` dictionary in `src/benchmark/run_benchmark.py`

## Adding New Frameworks

To add a new framework:

1. Create a new directory in `src/frameworks/`
2. Implement the framework-specific code
3. Update the `FRAMEWORKS` list in `src/benchmark/run_benchmark.py`
4. Implement the use case for the new framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [OpenAI](https://openai.com/) for providing the API used in this project
- The developers of AutoGen, Semantic Kernel, LangChain, and CrewAI for their excellent frameworks 