#include "rouge_evaluator.h"
#include <iostream>
#include <string>
#include <vector>

void display_help(const char* program_name) {
    std::cout << "ROUGE-1 Evaluator for LLM Benchmark" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help, -h             Show this help message" << std::endl;
    std::cout << "  --input, -i FILE       Read model outputs from JSON file" << std::endl;
    std::cout << "  --ref, -r FILE         Read reference answers from JSON file" << std::endl;
    std::cout << "  --output, -o FILE      Write results to JSON file" << std::endl;
    std::cout << "  --detailed, -d         Show detailed output" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " -i benchmark_results.json -r reference_answers.json -o rouge_scores.json" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default values
    std::string input_file = "";
    std::string ref_file = "";
    std::string output_file = "";
    bool detailed_output = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            display_help(argv[0]);
            return 0;
        } else if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) {
                input_file = argv[++i];
            }
        } else if (arg == "--ref" || arg == "-r") {
            if (i + 1 < argc) {
                ref_file = argv[++i];
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        } else if (arg == "--detailed" || arg == "-d") {
            detailed_output = true;
        }
    }
    
    // Initialize evaluator
    RougeEvaluator evaluator;
    
    // Set up example model outputs if no input file is provided
    if (input_file.empty()) {
        std::cout << "No input file specified. Using predefined model outputs." << std::endl;
    } else {
        // Load model outputs from file
        if (!evaluator.loadModelOutputs(input_file)) {
            std::cerr << "Failed to load model outputs from " << input_file << std::endl;
            return 1;
        }
    }
    
    // Load reference answers if provided
    if (!ref_file.empty()) {
        if (!evaluator.loadReferenceAnswers(ref_file)) {
            std::cerr << "Failed to load reference answers from " << ref_file << std::endl;
            return 1;
        }
    }
    
    // Calculate scores
    evaluator.calculateScores();
    
    // Print results
    evaluator.printResults(detailed_output);
    
    // Save results if output file is specified
    if (!output_file.empty()) {
        if (evaluator.saveResults(output_file)) {
            std::cout << "\nResults saved to " << output_file << std::endl;
        } else {
            std::cerr << "Failed to save results to " << output_file << std::endl;
            return 1;
        }
    }
    
    return 0;
}