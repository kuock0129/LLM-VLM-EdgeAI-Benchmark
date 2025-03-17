#include "llm_benchmark.h"
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Display help message
 * @param argv Program name
 */
void display_help(const char* argv) {
    std::cout << "Ollama Edge AI LLM Benchmark Tool" << std::endl;
    std::cout << "Usage: " << argv << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --verbose, -v          Enable verbose output with answers" << std::endl;
    std::cout << "  --parallel, -p         Run models in parallel (caution on Raspberry Pi)" << std::endl;
    std::cout << "  --no-memory, -nm       Disable memory tracking" << std::endl;
    std::cout << "  --mmap, -mm            Enable memory-mapped model loading (45% faster initial load)" << std::endl;
    std::cout << "  --swap SIZE, -s SIZE   Configure swap file of SIZE MB (e.g. 4096 for 4GB)" << std::endl;
    std::cout << "  --swappiness VAL, -sw VAL  Set VM swappiness (0-100, default 10)" << std::endl;
    std::cout << "  --prompt, -i FILE      Specify prompt file (default: prompt.txt)" << std::endl;
    std::cout << "  --output, -o FILE      Save detailed results to file" << std::endl;
    std::cout << "  --model, -m MODEL      Specify a model to test (can be used multiple times)" << std::endl;
    std::cout << "  --help, -h             Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Memory Optimization:" << std::endl;
    std::cout << "  For models exceeding 4GB RAM, use --swap 4096 --swappiness 10 --mmap" << std::endl;
    std::cout << "  This creates a 4GB swap file with optimal swappiness and enables memory mapping" << std::endl;
    std::cout << "  Memory-mapped loading reduces initial load times by up to 45%" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values
    std::string prompt_file = "prompt.txt";
    std::string output_file = "";
    bool verbose = false;
    bool parallel = false;
    bool track_memory = true;         // Enable memory tracking by default
    bool use_mmap = false;            // Memory-mapped loading disabled by default
    unsigned long swap_size = 0;      // Swap size in MB (0 = don't configure)
    int swappiness = 10;              // Default swappiness value
    std::vector<std::string> specific_models;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--parallel" || arg == "-p") {
            parallel = true;
        } else if (arg == "--no-memory" || arg == "-nm") {
            track_memory = false;
        } else if (arg == "--mmap" || arg == "-mm") {
            use_mmap = true;
        } else if (arg == "--prompt" || arg == "-i") {
            if (i + 1 < argc) {
                prompt_file = argv[++i];
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        } else if (arg == "--model" || arg == "-m") {
            if (i + 1 < argc) {
                specific_models.push_back(argv[++i]);
            }
        } else if (arg == "--swap" || arg == "-s") {
            if (i + 1 < argc) {
                swap_size = std::stoul(argv[++i]);
            }
        } else if (arg == "--swappiness" || arg == "-sw") {
            if (i + 1 < argc) {
                swappiness = std::stoi(argv[++i]);
                if (swappiness < 0) swappiness = 0;
                if (swappiness > 100) swappiness = 100;
            }
        } else if (arg == "--help" || arg == "-h") {
            display_help(argv[0]);
            return 0;
        }
    }
    
    try {
        // Create benchmark instance with memory optimization
        LLMBenchmark benchmark(prompt_file, output_file, verbose, parallel, track_memory, 
                              use_mmap, swap_size, swappiness);
        
        // Add specified models or default models
        if (specific_models.empty()) {
            // Use default models
            benchmark.add_model("mistral:7b");
            benchmark.add_model("tinyllama:latest");
            benchmark.add_model("phi:latest");
        } else {
            for (const auto& model : specific_models) {
                benchmark.add_model(model);
            }
        }
        
        // Run the benchmark
        benchmark.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}