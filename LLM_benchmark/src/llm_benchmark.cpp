#include "llm_benchmark.h"
#include "system_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

LLMBenchmark::LLMBenchmark(
    const std::string& prompt_path, 
    const std::string& output_path, 
    bool verbose_output, 
    bool run_parallel, 
    bool memory_tracking,
    bool use_memory_mapping, 
    unsigned long swap_mb, 
    int swap_priority
) : prompt_file(prompt_path), 
    output_file(output_path), 
    verbose(verbose_output), 
    parallel(run_parallel), 
    track_memory(memory_tracking),
    use_mmap(use_memory_mapping), 
    swap_size(swap_mb), 
    swappiness(swap_priority),
    api(OllamaAPI("http://localhost:11434", use_memory_mapping)) {
    
    // Initialize cURL
    OllamaAPI::initialize();
    
    // Configure swap if needed
    if (swap_size > 0) {
        // Get current system memory
        auto mem_info = get_system_memory();
        unsigned long total_mem = mem_info.first;
        unsigned long available_mem = mem_info.second;
        
        std::cout << "System memory: " << total_mem << "MB total, " 
                  << available_mem << "MB available" << std::endl;
        
        if (configure_swap(swap_size, swappiness)) {
            std::cout << "Successfully configured swap memory" << std::endl;
        } else {
            std::cerr << "Failed to configure swap memory. Continuing without swap optimization." << std::endl;
        }
    }
}

LLMBenchmark::~LLMBenchmark() {
    OllamaAPI::cleanup();
}

void LLMBenchmark::add_model(const std::string& model_name) {
    models.push_back(model_name);
}

void LLMBenchmark::add_all_models() {
    models = api.list_models();
    if (verbose) {
        std::cout << "Found " << models.size() << " models:" << std::endl;
        for (const auto& model : models) {
            std::cout << "  - " << model << std::endl;
        }
    }
}

std::string LLMBenchmark::read_prompt() {
    std::ifstream file(prompt_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open prompt file " << prompt_file << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    return buffer.str();
}

std::string LLMBenchmark::format_duration(std::chrono::milliseconds ms) {
    auto total_seconds = ms.count() / 1000;
    auto minutes = total_seconds / 60;
    auto seconds = total_seconds % 60;
    auto remaining_ms = ms.count() % 1000;
    
    std::stringstream ss;
    if (minutes > 0) {
        ss << minutes << "m ";
    }
    ss << seconds << "." << std::setfill('0') << std::setw(3) << remaining_ms << "s";
    return ss.str();
}

std::string LLMBenchmark::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%H:%M:%S");
    return ss.str();
}

int LLMBenchmark::estimate_tokens(const std::string& text) {
    // Simple approximation: ~4 chars per token
    return text.length() / 4;
}

std::vector<std::pair<std::string, std::string>> LLMBenchmark::parse_prompt_sections(const std::string& prompt) {
    std::vector<std::pair<std::string, std::string>> sections;
    std::istringstream stream(prompt);
    std::string line;
    std::string current_section;
    std::string current_content;
    
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        
        // Check if line starts with ## (section header)
        if (line.substr(0, 2) == "##") {
            // Save previous section if exists
            if (!current_section.empty()) {
                sections.push_back({current_section, current_content});
                current_content = "";
            }
            // Extract new section name
            current_section = line.substr(2);
            // Trim leading spaces
            size_t firstChar = current_section.find_first_not_of(" \t");
            if (firstChar != std::string::npos) {
                current_section = current_section.substr(firstChar);
            }
        } else {
            // Add line to current content
            if (!current_content.empty()) {
                current_content += "\n";
            }
            current_content += line;
        }
    }
    
    // Add the last section
    if (!current_section.empty()) {
        sections.push_back({current_section, current_content});
    }
    
    return sections;
}

void LLMBenchmark::run() {
    if (models.empty()) {
        std::cerr << "Error: No models specified for benchmark" << std::endl;
        return;
    }
    
    std::string prompt = read_prompt();
    if (prompt.empty()) {
        std::cerr << "Error: Empty prompt or failed to read prompt file" << std::endl;
        return;
    }
    
    int estimated_tokens = estimate_tokens(prompt);
    auto prompt_sections = parse_prompt_sections(prompt);
    
    std::cout << "========== EDGE AI LLM BENCHMARK ==========" << std::endl;
    std::cout << "Prompt file: " << prompt_file << std::endl;
    std::cout << "Models to test: " << models.size() << std::endl;
    std::cout << "Number of prompt sections: " << prompt_sections.size() << std::endl;
    std::cout << "Estimated tokens in prompt: " << estimated_tokens << std::endl;
    std::cout << "Verbose mode: " << (verbose ? "ON" : "OFF") << std::endl;
    std::cout << "Parallel execution: " << (parallel ? "ON" : "OFF") << std::endl;
    std::cout << "Memory tracking: " << (track_memory ? "ON" : "OFF") << std::endl;
    std::cout << "Memory-mapped loading: " << (use_mmap ? "ON" : "OFF") << std::endl;
    
    if (swap_size > 0) {
        std::cout << "Swap configuration: " << swap_size << "MB with swappiness " << swappiness << std::endl;
    } else {
        std::cout << "Swap configuration: Using system defaults" << std::endl;
    }
    
    std::cout << "===================================" << std::endl;
    
    std::vector<Result> results;
    
    // Get baseline memory before starting
    unsigned long baseline_memory = track_memory ? get_ollama_memory_usage() : 0;
    
    if (track_memory) {
        std::cout << "Baseline Ollama memory usage: " << format_memory(baseline_memory) << std::endl;
    }
    
    auto benchmark_start = std::chrono::high_resolution_clock::now();

    if (parallel) {
        // Run models in parallel
        std::vector<std::future<Result>> futures;
        
        for (const auto& model : models) {
            futures.push_back(std::async(std::launch::async, [this, &prompt, &prompt_sections, model, baseline_memory]() {
                Result result;
                result.model_name = model;
                result.baseline_memory = baseline_memory;
                
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "\n[" << get_timestamp() << "] Starting inference on model " << model << std::endl;
                }
                
                // Setup memory monitoring if enabled
                MemoryMonitor memory_monitor("ollama");
                if (track_memory) {
                    memory_monitor.start();
                }
                
                // Full model evaluation
                auto full_start = std::chrono::high_resolution_clock::now();
                result.response = api.generate(model, prompt, false, verbose);
                auto full_end = std::chrono::high_resolution_clock::now();
                
                // Capture peak memory
                if (track_memory) {
                    memory_monitor.stop();
                    result.peak_memory = memory_monitor.get_peak_memory();
                    
                    // Alternatively, use direct Ollama process monitoring
                    unsigned long ollama_memory = get_ollama_memory_usage();
                    result.peak_memory = std::max(result.peak_memory, ollama_memory);
                }
                
                result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(full_end - full_start);
                
                // Calculate tokens per second (very approximate)
                int output_tokens = estimate_tokens(result.response);
                result.tokens_per_second = 1000.0 * output_tokens / result.duration.count();
                
                {
                    std::lock_guard<std::mutex> lock(output_mutex);
                    std::cout << "[" << get_timestamp() << "] Completed full inference on model " << model 
                            << " in " << format_duration(result.duration) << std::endl;
                    std::cout << "[" << get_timestamp() << "] Response tokens: ~" << output_tokens 
                            << " (" << result.tokens_per_second << " tokens/sec)" << std::endl;
                    
                    if (track_memory) {
                        std::cout << "[" << get_timestamp() << "] Peak memory: " 
                                << format_memory(result.peak_memory) 
                                << " (+" << format_memory(result.peak_memory - result.baseline_memory) 
                                << " from baseline)" << std::endl;
                    }
                }
                
                // Section-by-section evaluation if verbose
                if (verbose) {
                    for (const auto& section : prompt_sections) {
                        {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            std::cout << "[" << get_timestamp() << "] Testing section: " << section.first << std::endl;
                        }
                        
                        std::string section_prompt = "## " + section.first + "\n" + section.second;
                        
                        // Setup memory monitoring for section
                        MemoryMonitor section_memory_monitor("ollama");
                        if (track_memory) {
                            section_memory_monitor.start();
                        }
                        
                        auto section_start = std::chrono::high_resolution_clock::now();
                        std::string section_response = api.generate(model, section_prompt, false, false);
                        auto section_end = std::chrono::high_resolution_clock::now();
                        
                        // Capture section memory
                        unsigned long section_memory = 0;
                        if (track_memory) {
                            section_memory_monitor.stop();
                            section_memory = section_memory_monitor.get_peak_memory();
                            
                            // Use direct Ollama process monitoring if available
                            unsigned long ollama_section_memory = get_ollama_memory_usage();
                            section_memory = std::max(section_memory, ollama_section_memory);
                        }
                        
                        std::chrono::milliseconds section_duration = 
                            std::chrono::duration_cast<std::chrono::milliseconds>(section_end - section_start);
                        
                        {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            std::cout << "[" << get_timestamp() << "] Completed section: " << section.first 
                                    << " in " << format_duration(section_duration) << std::endl;
                                    
                            if (track_memory) {
                                std::cout << "[" << get_timestamp() << "] Section memory: " 
                                        << format_memory(section_memory) << std::endl;
                            }
                        }
                        
                        result.section_responses[section.first] = section_response;
                        result.section_metrics[section.first] = {section_duration, section_memory};
                    }
                }
                
                return result;
            }));
        }
        
        // Collect results
        for (auto& future : futures) {
            results.push_back(future.get());
        }
    } else {
        // Run models sequentially
        for (const auto& model : models) {
            Result result;
            result.model_name = model;
            result.baseline_memory = baseline_memory;
            
            std::cout << "\n[" << get_timestamp() << "] Starting inference on model " << model << std::endl;
            
            // Setup memory monitoring if enabled
            MemoryMonitor memory_monitor("ollama");
            if (track_memory) {
                memory_monitor.start();
            }
            
            // Full model evaluation
            auto full_start = std::chrono::high_resolution_clock::now();
            result.response = api.generate(model, prompt, false, verbose);
            auto full_end = std::chrono::high_resolution_clock::now();
            
            // Capture peak memory
            if (track_memory) {
                memory_monitor.stop();
                result.peak_memory = memory_monitor.get_peak_memory();
                
                // Alternatively, use direct Ollama process monitoring
                unsigned long ollama_memory = get_ollama_memory_usage();
                result.peak_memory = std::max(result.peak_memory, ollama_memory);
            }
            
            result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(full_end - full_start);
            
            // Calculate tokens per second (very approximate)
            int output_tokens = estimate_tokens(result.response);
            result.tokens_per_second = 1000.0 * output_tokens / result.duration.count();
            
            std::cout << "[" << get_timestamp() << "] Completed full inference on model " << model 
                    << " in " << format_duration(result.duration) << std::endl;
            std::cout << "[" << get_timestamp() << "] Response tokens: ~" << output_tokens 
                    << " (" << result.tokens_per_second << " tokens/sec)" << std::endl;
            
            if (track_memory) {
                std::cout << "[" << get_timestamp() << "] Peak memory: " 
                        << format_memory(result.peak_memory) 
                        << " (+" << format_memory(result.peak_memory - result.baseline_memory) 
                        << " from baseline)" << std::endl;
            }
            
            // Section-by-section evaluation if verbose
            if (verbose) {
                for (const auto& section : prompt_sections) {
                    std::cout << "[" << get_timestamp() << "] Testing section: " << section.first << std::endl;
                    
                    std::string section_prompt = "## " + section.first + "\n" + section.second;
                    
                    // Setup memory monitoring for section
                    MemoryMonitor section_memory_monitor("ollama");
                    if (track_memory) {
                        section_memory_monitor.start();
                    }
                    
                    auto section_start = std::chrono::high_resolution_clock::now();
                    std::string section_response = api.generate(model, section_prompt, false, false);
                    auto section_end = std::chrono::high_resolution_clock::now();
                    
                    // Capture section memory
                    unsigned long section_memory = 0;
                    if (track_memory) {
                        section_memory_monitor.stop();
                        section_memory = section_memory_monitor.get_peak_memory();
                        
                        // Use direct Ollama process monitoring if available
                        unsigned long ollama_section_memory = get_ollama_memory_usage();
                        section_memory = std::max(section_memory, ollama_section_memory);
                    }
                    
                    std::chrono::milliseconds section_duration = 
                        std::chrono::duration_cast<std::chrono::milliseconds>(section_end - section_start);
                    
                    std::cout << "[" << get_timestamp() << "] Completed section: " << section.first 
                            << " in " << format_duration(section_duration) << std::endl;
                            
                    if (track_memory) {
                        std::cout << "[" << get_timestamp() << "] Section memory: " 
                                << format_memory(section_memory) << std::endl;
                    }
                    
                    result.section_responses[section.first] = section_response;
                    result.section_metrics[section.first] = {section_duration, section_memory};
                }
            }
            
            results.push_back(result);
        }
    }
    
    auto benchmark_end = std::chrono::high_resolution_clock::now();
    std::chrono::milliseconds total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        benchmark_end - benchmark_start);
    
    // Sort results by duration
    std::sort(results.begin(), results.end(), 
            [](const Result& a, const Result& b) { return a.duration < b.duration; });
    
    // Summary report
    std::cout << "\n========== BENCHMARK RESULTS ==========" << std::endl;
    std::cout << "Total benchmark time: " << format_duration(total_duration) << std::endl;
    std::cout << "\nModels ranked by inference speed:" << std::endl;
    
    // Expanded table with memory usage
    if (track_memory) {
        std::cout << std::left << std::setw(20) << "Model" 
                << std::setw(15) << "Time" 
                << std::setw(15) << "Tokens/sec" 
                << std::setw(15) << "Memory" 
                << std::setw(15) << "Mem increase" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(20) << result.model_name 
                    << std::setw(15) << format_duration(result.duration) 
                    << std::setw(15) << std::fixed << std::setprecision(2) << result.tokens_per_second
                    << std::setw(15) << format_memory(result.peak_memory)
                    << std::setw(15) << format_memory(result.peak_memory - result.baseline_memory) 
                    << std::endl;
        }
    } else {
        std::cout << std::left << std::setw(20) << "Model" 
                << std::setw(15) << "Time" 
                << std::setw(15) << "Tokens/sec" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(20) << result.model_name 
                    << std::setw(15) << format_duration(result.duration) 
                    << std::setw(15) << std::fixed << std::setprecision(2) << result.tokens_per_second << std::endl;
        }
    }
    
    // Save detailed results to file if specified
    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << "========== EDGE AI LLM BENCHMARK DETAILED RESULTS ==========" << std::endl;
            out << "Prompt file: " << prompt_file << std::endl;
            out << "Total benchmark time: " << format_duration(total_duration) << std::endl;
            
            if (track_memory) {
                out << "Baseline Ollama memory usage: " << format_memory(baseline_memory) << std::endl;
            }
            
            out << std::endl;
            
            for (const auto& result : results) {
                out << "MODEL: " << result.model_name << std::endl;
                out << "Time: " << format_duration(result.duration) << std::endl;
                out << "Tokens/sec: " << std::fixed << std::setprecision(2) << result.tokens_per_second << std::endl;
                
                if (track_memory) {
                    out << "Peak memory: " << format_memory(result.peak_memory) << std::endl;
                    out << "Memory increase: " << format_memory(result.peak_memory - result.baseline_memory) << std::endl;
                }
                
                if (!result.section_responses.empty()) {
                    out << "\nSECTION-BY-SECTION METRICS:" << std::endl;
                    
                    for (const auto& section : prompt_sections) {
                        out << "\n=== SECTION: " << section.first << " ===" << std::endl;
                        out << "QUESTION:" << std::endl;
                        out << section.second << std::endl;
                        
                        auto metrics_it = result.section_metrics.find(section.first);
                        if (metrics_it != result.section_metrics.end()) {
                            out << "Time: " << format_duration(metrics_it->second.first) << std::endl;
                            
                            if (track_memory) {
                                out << "Memory: " << format_memory(metrics_it->second.second) << std::endl;
                            }
                        }
                        
                        out << "\nRESPONSE:" << std::endl;
                        
                        auto it = result.section_responses.find(section.first);
                        if (it != result.section_responses.end()) {
                            out << it->second << std::endl;
                        } else {
                            out << "[No response for this section]" << std::endl;
                        }
                        
                        out << "----------------------------------------" << std::endl;
                    }
                } else {
                    out << "\nFULL RESPONSE:" << std::endl;
                    out << "----------------------------------------" << std::endl;
                    out << result.response << std::endl;
                }
                
                out << "========================================" << std::endl;
                out << std::endl;
            }
            
            out.close();
            std::cout << "\nDetailed results saved to " << output_file << std::endl;
        } else {
            std::cerr << "Error: Could not open output file " << output_file << std::endl;
        }
    }
    
    if (verbose) {
        std::cout << "\n===== DETAILED ANSWERS BY MODEL =====" << std::endl;
        for (const auto& result : results) {
            std::cout << "\n======== " << result.model_name << " ========" << std::endl;
            std::cout << "Time: " << format_duration(result.duration) << " | ";
            std::cout << "Tokens/sec: " << std::fixed << std::setprecision(2) << result.tokens_per_second;
            
            if (track_memory) {
                std::cout << " | Memory: " << format_memory(result.peak_memory) 
                        << " (+" << format_memory(result.peak_memory - result.baseline_memory) 
                        << " from baseline)";
            }
            
            std::cout << std::endl << std::endl;
            
            if (!result.section_responses.empty()) {
                std::cout << "SECTION-BY-SECTION RESPONSES:" << std::endl;
                
                for (const auto& section : prompt_sections) {
                    std::cout << "\n--- " << section.first << " ---" << std::endl;
                    
                    auto metrics_it = result.section_metrics.find(section.first);
                    if (metrics_it != result.section_metrics.end()) {
                        std::cout << "Time: " << format_duration(metrics_it->second.first);
                        
                        if (track_memory) {
                            std::cout << " | Memory: " << format_memory(metrics_it->second.second);
                        }
                        
                        std::cout << std::endl;
                    }
                    
                    std::cout << "Q: " << section.second << std::endl;
                    std::cout << "\nA: ";
                    
                    auto it = result.section_responses.find(section.first);
                    if (it != result.section_responses.end()) {
                        std::cout << it->second << std::endl;
                    } else {
                        std::cout << "[No response]" << std::endl;
                    }
                }
            } else {
                std::cout << "FULL RESPONSE:" << std::endl;
                std::cout << result.response << std::endl;
            }
            
            std::cout << "----------------------------------------" << std::endl;
        }
    }
    
    std::cout << "\n=======================================" << std::endl;
    
    // Optionally generate memory usage comparison chart
    if (track_memory && results.size() > 1) {
        std::cout << "\nMEMORY USAGE COMPARISON:" << std::endl;
        std::cout << "Memory baseline: " << format_memory(baseline_memory) << std::endl;
        
        // Find the model with the highest memory usage for scaling
        unsigned long max_memory_increase = 0;
        for (const auto& result : results) {
            max_memory_increase = std::max(max_memory_increase, result.peak_memory - result.baseline_memory);
        }
        
        // Simple ASCII bar chart
        const int chart_width = 50; // characters
        
        for (const auto& result : results) {
            unsigned long memory_increase = result.peak_memory - result.baseline_memory;
            int bar_length = (max_memory_increase > 0) 
                           ? static_cast<int>((memory_increase * chart_width) / max_memory_increase) 
                           : 0;
            
            std::cout << std::left << std::setw(20) << result.model_name << " ";
            std::cout << "[" << std::string(bar_length, '#') << std::string(chart_width - bar_length, ' ') << "] ";
            std::cout << format_memory(memory_increase) << std::endl;
        }
    }
}