# The Edge AI Benchmark for Large-Language Models (LLMs) on Edge Device
A tool for benchmarking Large Language Models running on Ollama in edge computing environments.

## Features

- Benchmark multiple Ollama models
- Measure inference time and memory usage
- Section-by-section prompt evaluation
- Memory optimization options (swap configuration, memory mapping)
- Parallel or sequential model execution
- Detailed reporting and results export
- ROUGE-1 score evaluation for output quality assessment

## Requirements

- C++17 compatible compiler
- libcurl
- nlohmann-json
- Ollama server running locally

## Project Structure

- `src/`: C++ source files
- `include/`: Header files
- `tools/`: Utility programs source code
- `prompts/`: Sample prompt files
- `data/`: Reference data for evaluation

```bash

edge_ai_benchmark/
│
├── include/
│   ├── api_client.h          # OllamaAPI class declaration
│   ├── memory_monitor.h      # MemoryMonitor class declaration
│   ├── system_utils.h        # System utilities declarations
│   ├── llm_benchmark.h       # LLMBenchmark class declaration
│   └── rouge_evaluator.h     # RougeEvaluator class declaration
│
├── src/
│   ├── api_client.cpp        # OllamaAPI implementation
│   ├── memory_monitor.cpp    # MemoryMonitor implementation
│   ├── system_utils.cpp      # System utilities implementation
│   ├── llm_benchmark.cpp     # LLMBenchmark implementation
│   ├── main.cpp              # Main application entry point
│   └── rouge_evaluator.cpp   # RougeEvaluator implementation
│
├── tools/
│   └── rouge_evaluator.cpp   # ROUGE-1 evaluator tool main function
│
├── prompts/                  # Sample prompts for benchmarking
│   └── standard_prompt.txt   # Standard evaluation prompt
│
├── data/                     # Data directory for reference answers
│   └── reference_answers.json # Reference answers for evaluation
│
├── Makefile                  # Build configuration
│
└── prompt.txt                # Input queries

```


## Building

```bash

# Enter the folder
cd LLM_benchmark

# Install dependencies
make install-deps

# Build the application
make
```

## Usage

### Benchmark Tool

```bash
# Basic usage
./edge_ai_benchmark --prompt prompt.txt --model tinyllama:latest --output results.json

# With specific models
./edge_ai_benchmark --prompt prompt.txt --model tinyllama:latest --verbose --output results.json

# With memory optimization for low-RAM devices
./edge_ai_benchmark --prompt prompt.txt  --model tinyllama:latest --verbose --swap 4096 --swappiness 10 --mmap --output results.json

# For all options
./edge_ai_benchmark --help
```

### ROUGE Evaluator
```bash
# Basic usage (uses predefined model outputs)
./rouge_evaluator

# With custom inputs/outputs
./rouge_evaluator --input results.json --ref data/reference_answers.json --output evaluation_results.json

# Save evaluation results
./rouge_evaluator --output evaluation_results.json

# For all options
./rouge_evaluator --help
```

## Command Line Options

### Benchmark Tool

- `--verbose`, `-v`: Enable verbose output with answers
- `--parallel`, `-p`: Run models in parallel (caution on low-RAM devices)
- `--no-memory`, `-nm`: Disable memory tracking
- `--mmap`, `-mm`: Enable memory-mapped model loading (45% faster initial load)
- `--swap SIZE`, `-s SIZE`: Configure swap file of SIZE MB (e.g., 4096 for 4GB)
- `--swappiness VAL`, `-sw VAL`: Set VM swappiness (0-100, default 10)
- `--prompt`, `-i FILE`: Specify prompt file (default: prompt.txt)
- `--output`, `-o FILE`: Save detailed results to file
- `--model`, `-m MODEL`: Specify a model to test (can be used multiple times)
- `--help`, `-h`: Show help message

### ROUGE Evaluator

- `--input`, `-i FILE`: Read model outputs from JSON file
- `--ref`, `-r FILE`: Read reference answers from JSON file
- `--output`, `-o FILE`: Write results to JSON file
- `--detailed`, `-d`: Show detailed output
- `--help`, `-h`: Show help message


## ROUGE-1 Evaluation

ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) is a metric used to evaluate the quality of model-generated text compared to reference answers. The evaluator implemented in this project:

1. Tokenizes both generated outputs and reference answers into unigrams
2. Calculates precision, recall, and F1 scores
3. Provides both category-specific and overall evaluation scores
4. Offers task-specific accuracy metrics for different question types

The evaluation process helps quantify model performance beyond just speed and memory usage.
