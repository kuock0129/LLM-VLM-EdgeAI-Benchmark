#!/bin/bash

# Ollama Direct Benchmarking Script
# This script uses Ollama CLI directly for benchmarking with verbose metrics

# Create results directory
RESULTS_DIR="ollama_benchmarks_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Function to run benchmark
run_benchmark() {
    local model=$1
    local prompt_file=$2
    
    echo "----------------------------------------"
    echo "Running benchmark on model: $model"
    echo "----------------------------------------"
    
    local output_file="${RESULTS_DIR}/${model}_results.txt"
    
    # Get prompt from file
    if [ ! -f "$prompt_file" ]; then
        echo "Error: Prompt file not found: $prompt_file"
        return 1
    fi
    
    local prompt=$(cat "$prompt_file")
    
    # Run benchmark with Ollama CLI and capture both response and metrics
    {
        echo "BENCHMARK RESULTS FOR MODEL: $model"
        echo "----------------------------------------"
        echo "DATE: $(date)"
        echo "MODEL: $model"
        echo "PROMPT FILE: $prompt_file"
        echo "----------------------------------------"
        echo "PROMPT:"
        echo "$prompt"
        echo "----------------------------------------"
        echo "RESPONSE:"
    } > "$output_file"
    
    # Run Ollama with the prompt and capture output
    {
        # This captures the complete output including metrics
        ollama run "$model" --verbose < "$prompt_file" 2>&1
    } | tee -a "$output_file"
    
    echo "----------------------------------------" >> "$output_file"
    echo "Benchmark results saved to $output_file"
    echo "----------------------------------------"
}

# Parse command line arguments
MODEL=""
PROMPT_FILE="prompt.txt"

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model|-m)
            MODEL="$2"
            shift
            shift
            ;;
        --prompt|-p)
            PROMPT_FILE="$2"
            shift
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--model MODEL] [--prompt PROMPT_FILE]"
            echo "  --model, -m MODEL        Specify model to benchmark (default: runs multiple models)"
            echo "  --prompt, -p FILE        Specify prompt file (default: prompt.txt)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Ollama Direct Benchmark Suite"
echo "Results will be saved to $RESULTS_DIR"
echo "----------------------------------------"

# Run benchmark for specific model or multiple models
if [ -n "$MODEL" ]; then
    run_benchmark "$MODEL" "$PROMPT_FILE"
else
    # List of models to benchmark - adjust as needed
    MODELS=("mistral:7b" "phi:latest" "tinyllama:latest")
    
    for model in "${MODELS[@]}"; do
        run_benchmark "$model" "$PROMPT_FILE"
    done
fi

# Create a summary CSV
echo "Creating summary CSV..."
echo "Model,Total Duration,Prompt Tokens,Generation Tokens,Generation Rate" > "${RESULTS_DIR}/summary.csv"

for file in "${RESULTS_DIR}"/*_results.txt; do
    model=$(basename "$file" | cut -d'_' -f1)
    
    # Parse metrics from the output file - adjust grep patterns as needed
    total_duration=$(grep "total duration:" "$file" | awk '{print $3}')
    eval_count=$(grep "eval count:" "$file" | awk '{print $3}')
    eval_rate=$(grep "eval rate:" "$file" | awk '{print $3}')
    
    echo "$model,$total_duration,$eval_count,$eval_rate" >> "${RESULTS_DIR}/summary.csv"
done

echo "Benchmarking complete!"
echo "Results available in $RESULTS_DIR directory"
echo "Summary: ${RESULTS_DIR}/summary.csv"