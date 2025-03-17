#ifndef LLM_BENCHMARK_H
#define LLM_BENCHMARK_H

#include "api_client.h"
#include "memory_monitor.h"
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <map>
#include <future>

/**
 * @brief Class to benchmark LLM models
 */
class LLMBenchmark {
private:
    std::vector<std::string> models;
    std::string prompt_file;
    std::string output_file;
    bool verbose;
    bool parallel;
    bool track_memory;
    bool use_mmap;          // Use memory-mapped model loading
    unsigned long swap_size; // Swap size in MB
    int swappiness;         // VM swappiness setting
    OllamaAPI api;
    std::mutex output_mutex;
    
    /**
     * @brief Result structure with memory metrics
     */
    struct Result {
        std::string model_name;
        std::string response;
        std::chrono::milliseconds duration;
        double tokens_per_second;
        unsigned long peak_memory;
        unsigned long baseline_memory;
        std::map<std::string, std::string> section_responses; // For verbose output
        std::map<std::string, std::pair<std::chrono::milliseconds, unsigned long>> section_metrics; // Duration and memory by section
    };
    
    /**
     * @brief Read prompt from file
     * @return Prompt string
     */
    std::string read_prompt();
    
    /**
     * @brief Format time duration
     * @param ms Duration in milliseconds
     * @return Formatted string (e.g., "1m 23.456s")
     */
    std::string format_duration(std::chrono::milliseconds ms);
    
    /**
     * @brief Get current timestamp
     * @return Current time as string
     */
    std::string get_timestamp();
    
    /**
     * @brief Estimate token count (approximate)
     * @param text Input text
     * @return Estimated token count
     */
    int estimate_tokens(const std::string& text);
    
    /**
     * @brief Parse prompt sections for better output
     * @param prompt Full prompt text
     * @return Vector of section name and content pairs
     */
    std::vector<std::pair<std::string, std::string>> parse_prompt_sections(const std::string& prompt);
    
public:
    /**
     * @brief Constructor
     * @param prompt_path Path to the prompt file
     * @param output_path Path for the output file (empty for no output)
     * @param verbose_output Whether to show verbose output
     * @param run_parallel Whether to run models in parallel
     * @param memory_tracking Whether to track memory usage
     * @param use_memory_mapping Whether to use memory-mapped model loading
     * @param swap_mb Size of swap file in MB (0 to disable)
     * @param swap_priority VM swappiness priority (0-100)
     */
    LLMBenchmark(
        const std::string& prompt_path, 
        const std::string& output_path = "", 
        bool verbose_output = false, 
        bool run_parallel = false, 
        bool memory_tracking = true,
        bool use_memory_mapping = false, 
        unsigned long swap_mb = 0, 
        int swap_priority = 10
    );
    
    /**
     * @brief Destructor
     */
    ~LLMBenchmark();
    
    /**
     * @brief Add a specific model to benchmark
     * @param model_name Name of the model
     */
    void add_model(const std::string& model_name);
    
    /**
     * @brief Add all available models
     */
    void add_all_models();
    
    /**
     * @brief Run the benchmark
     */
    void run();
};

#endif // LLM_BENCHMARK_H