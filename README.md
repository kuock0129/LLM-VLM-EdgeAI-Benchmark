A tool for benchmarking Large Language Models running on Ollama in edge computing environments.

## Features

- Benchmark multiple Ollama models
- Measure inference time and memory usage
- Section-by-section prompt evaluation
- Memory optimization options (swap configuration, memory mapping)
- Parallel or sequential model execution
- Detailed reporting and results export

## Requirements

- C++17 compatible compiler
- libcurl
- nlohmann-json
- Ollama server running locally

## Building

```bash
# Install dependencies
make install-deps

# Build the application
make
```

## Usage

```bash
# Basic usage
./edge_ai_benchmark

# With specific models
./edge_ai_benchmark --model mistral:7b --model phi:latest

# With memory optimization for low-RAM devices
./edge_ai_benchmark --swap 4096 --swappiness 10 --mmap

# For all options
./edge_ai_benchmark --help
```

## Command Line Options

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
