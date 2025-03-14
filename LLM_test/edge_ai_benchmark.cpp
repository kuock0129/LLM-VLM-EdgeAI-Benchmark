#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <mutex>
#include <future>
#include <ctime>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Callback function for cURL to write response data
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append((char*)contents, total_size);
    return total_size;
}

// Class to handle Ollama API interactions
class OllamaAPI {
private:
    std::string base_url;
    
public:
    OllamaAPI(const std::string& url = "http://localhost:11434") : base_url(url) {}
    
    // Initialize curl once at the beginning
    static bool initialize() {
        return curl_global_init(CURL_GLOBAL_ALL) == CURLE_OK;
    }
    
    // Clean up curl at the end
    static void cleanup() {
        curl_global_cleanup();
    }
    
    // Get list of available models
    std::vector<std::string> list_models() {
        std::vector<std::string> models;
        std::string response;
        
        CURL* curl = curl_easy_init();
        if (curl) {
            std::string url = base_url + "/api/tags";
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            
            CURLcode res = curl_easy_perform(curl);
            if (res == CURLE_OK) {
                try {
                    json j = json::parse(response);
                    if (j.contains("models") && j["models"].is_array()) {
                        for (const auto& model : j["models"]) {
                            if (model.contains("name")) {
                                models.push_back(model["name"]);
                            }
                        }
                    }
                } catch (json::parse_error& e) {
                    std::cerr << "JSON parse error: " << e.what() << std::endl;
                }
            } else {
                std::cerr << "cURL error: " << curl_easy_strerror(res) << std::endl;
            }
            
            curl_easy_cleanup(curl);
        }
        
        return models;
    }
    
    // Modify the OllamaAPI generate method to capture verbose metrics
    std::string generate(const std::string& model, const std::string& prompt, bool stream = false, bool verbose = false) {
        std::string response_text;
        
        CURL* curl = curl_easy_init();
        if (curl) {
            std::string url = base_url + "/api/generate";
            struct curl_slist* headers = NULL;
            headers = curl_slist_append(headers, "Content-Type: application/json");
            
            // Create JSON request body with verbose flag
            json request_body = {
                {"model", model},
                {"prompt", prompt},
                {"stream", stream},
                {"options", {
                    {"num_gpu", 1},   // Use GPU if available
                    {"temperature", 0.7}
                }}
            };
            
            std::string request_str = request_body.dump();
            
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request_str.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_text);
            
            if (verbose) {
                std::cout << "[DEBUG] Requesting completion from " << model << std::endl;
                std::cout << "[DEBUG] Request body: " << request_str << std::endl;
            }
            
            auto start_time = std::chrono::high_resolution_clock::now();
            CURLcode res = curl_easy_perform(curl);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> elapsed = end_time - start_time;
            
            if (res != CURLE_OK) {
                std::cerr << "cURL error: " << curl_easy_strerror(res) << std::endl;
                response_text = "Error: Failed to connect to Ollama API";
            } else if (verbose) {
                std::cout << "[DEBUG] Raw response received with length: " << response_text.length() << " bytes" << std::endl;
                std::cout << "[DEBUG] API request took: " << elapsed.count() << "s" << std::endl;
            }
            
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            
            // Parse the response for metrics
            if (!response_text.empty()) {
                try {
                    json j = json::parse(response_text);
                    
                    // Extract and display metrics similar to Ollama CLI
                    if (verbose && j.contains("eval_count") && j.contains("eval_duration")) {
                        double total_duration = elapsed.count();
                        int eval_count = j["eval_count"];
                        double eval_duration = j["eval_duration"].get<double>() / 1000.0; // Convert ms to s
                        double token_rate = (eval_count > 0 && eval_duration > 0) ? 
                                            eval_count / eval_duration : 0;
                        
                        // Display metrics in similar format to Ollama CLI
                        std::cout << "\nPERFORMANCE METRICS:" << std::endl;
                        std::cout << std::left << std::setw(25) << "total duration:" 
                                << total_duration << "s" << std::endl;
                        
                        if (j.contains("prompt_eval_count")) {
                            int prompt_tokens = j["prompt_eval_count"];
                            double prompt_duration = j["prompt_eval_duration"].get<double>() / 1000.0;
                            double prompt_rate = (prompt_tokens > 0 && prompt_duration > 0) ? 
                                                prompt_tokens / prompt_duration : 0;
                            
                            std::cout << std::left << std::setw(25) << "prompt eval count:" 
                                    << prompt_tokens << " token(s)" << std::endl;
                            std::cout << std::left << std::setw(25) << "prompt eval duration:" 
                                    << prompt_duration << "s" << std::endl;
                            std::cout << std::left << std::setw(25) << "prompt eval rate:" 
                                    << std::fixed << std::setprecision(2) << prompt_rate 
                                    << " tokens/s" << std::endl;
                        }
                        
                        std::cout << std::left << std::setw(25) << "eval count:" 
                                << eval_count << " token(s)" << std::endl;
                        std::cout << std::left << std::setw(25) << "eval duration:" 
                                << eval_duration << "s" << std::endl;
                        std::cout << std::left << std::setw(25) << "eval rate:" 
                                << std::fixed << std::setprecision(2) << token_rate 
                                << " tokens/s" << std::endl;
                    }
                    
                    if (j.contains("response")) {
                        return j["response"];
                    }
                } catch (json::parse_error& e) {
                    std::cerr << "JSON parse error: " << e.what() << std::endl;
                    return "Error: Failed to parse response";
                }
            }
        }
        
        return response_text;
    }
};

// Class to benchmark LLM models
class LLMBenchmark {
private:
    std::vector<std::string> models;
    std::string prompt_file;
    std::string output_file;
    bool verbose;
    bool parallel;
    OllamaAPI api;
    std::mutex output_mutex;
    
    // Read prompt from file
    std::string read_prompt() {
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
    
    // Format time duration
    std::string format_duration(std::chrono::milliseconds ms) {
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
    
    // Get current timestamp
    std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t_now), "%H:%M:%S");
        return ss.str();
    }
    
    // Estimate token count (very rough approximation)
    int estimate_tokens(const std::string& text) {
        // Simple approximation: ~4 chars per token
        return text.length() / 4;
    }
    
    // Parse prompt sections and responses for better output
    std::vector<std::pair<std::string, std::string>> parse_prompt_sections(const std::string& prompt) {
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
    
public:
    LLMBenchmark(const std::string& prompt_path, const std::string& output_path = "", 
                 bool verbose_output = false, bool run_parallel = false)
        : prompt_file(prompt_path), output_file(output_path), 
          verbose(verbose_output), parallel(run_parallel) {
        
        // Initialize cURL
        OllamaAPI::initialize();
    }
    
    ~LLMBenchmark() {
        OllamaAPI::cleanup();
    }
    
    // Add a specific model to benchmark
    void add_model(const std::string& model_name) {
        models.push_back(model_name);
    }
    
    // Add all available models
    void add_all_models() {
        models = api.list_models();
        if (verbose) {
            std::cout << "Found " << models.size() << " models:" << std::endl;
            for (const auto& model : models) {
                std::cout << "  - " << model << std::endl;
            }
        }
    }
    
    // Run the benchmark
    void run() {
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
        std::cout << "===================================" << std::endl;
        
        // Results storage
        struct Result {
            std::string model_name;
            std::string response;
            std::chrono::milliseconds duration;
            double tokens_per_second;
            std::map<std::string, std::string> section_responses; // For verbose output
        };
        
        std::vector<Result> results;
        
        auto benchmark_start = std::chrono::high_resolution_clock::now();
        
        if (parallel) {
            // Run models in parallel
            std::vector<std::future<Result>> futures;
            
            for (const auto& model : models) {
                futures.push_back(std::async(std::launch::async, [this, &prompt, &prompt_sections, model]() {
                    Result result;
                    result.model_name = model;
                    
                    {
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "\n[" << get_timestamp() << "] Starting inference on model " << model << std::endl;
                        if (verbose) {
                            std::cout << "[" << get_timestamp() << "] Testing " << prompt_sections.size() << " sections" << std::endl;
                        }
                    }
                    
                    // Full model evaluation
                    auto full_start = std::chrono::high_resolution_clock::now();
                    result.response = api.generate(model, prompt, false, verbose);
                    auto full_end = std::chrono::high_resolution_clock::now();
                    
                    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(full_end - full_start);
                    
                    // Calculate tokens per second (very approximate)
                    int output_tokens = estimate_tokens(result.response);
                    result.tokens_per_second = 1000.0 * output_tokens / result.duration.count();
                    
                    // Section-by-section evaluation if verbose
                    if (verbose) {
                        for (const auto& section : prompt_sections) {
                            {
                                std::lock_guard<std::mutex> lock(output_mutex);
                                std::cout << "[" << get_timestamp() << "] Testing section: " << section.first << std::endl;
                            }
                            
                            std::string section_prompt = "## " + section.first + "\n" + section.second;
                            auto section_start = std::chrono::high_resolution_clock::now();
                            std::string section_response = api.generate(model, section_prompt, false, false);
                            auto section_end = std::chrono::high_resolution_clock::now();
                            
                            std::chrono::milliseconds section_duration = 
                                std::chrono::duration_cast<std::chrono::milliseconds>(section_end - section_start);
                            
                            {
                                std::lock_guard<std::mutex> lock(output_mutex);
                                std::cout << "[" << get_timestamp() << "] Completed section: " << section.first 
                                        << " in " << format_duration(section_duration) << std::endl;
                            }
                            
                            result.section_responses[section.first] = section_response;
                        }
                    }
                    
                    {
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "[" << get_timestamp() << "] Completed full inference on model " << model 
                                << " in " << format_duration(result.duration) << std::endl;
                        std::cout << "[" << get_timestamp() << "] Response tokens: ~" << output_tokens 
                                << " (" << result.tokens_per_second << " tokens/sec)" << std::endl;
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
                
                std::cout << "\n[" << get_timestamp() << "] Starting inference on model " << model << std::endl;
                if (verbose) {
                    std::cout << "[" << get_timestamp() << "] Testing " << prompt_sections.size() << " sections" << std::endl;
                }
                
                // Full model evaluation
                auto full_start = std::chrono::high_resolution_clock::now();
                result.response = api.generate(model, prompt, false, verbose);
                auto full_end = std::chrono::high_resolution_clock::now();
                
                result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(full_end - full_start);
                
                // Calculate tokens per second (very approximate)
                int output_tokens = estimate_tokens(result.response);
                result.tokens_per_second = 1000.0 * output_tokens / result.duration.count();
                
                std::cout << "[" << get_timestamp() << "] Completed full inference on model " << model 
                        << " in " << format_duration(result.duration) << std::endl;
                std::cout << "[" << get_timestamp() << "] Response tokens: ~" << output_tokens 
                        << " (" << result.tokens_per_second << " tokens/sec)" << std::endl;
                
                // Section-by-section evaluation if verbose
                if (verbose) {
                    for (const auto& section : prompt_sections) {
                        std::cout << "[" << get_timestamp() << "] Testing section: " << section.first << std::endl;
                        
                        std::string section_prompt = "## " + section.first + "\n" + section.second;
                        auto section_start = std::chrono::high_resolution_clock::now();
                        std::string section_response = api.generate(model, section_prompt, false, false);
                        auto section_end = std::chrono::high_resolution_clock::now();
                        
                        std::chrono::milliseconds section_duration = 
                            std::chrono::duration_cast<std::chrono::milliseconds>(section_end - section_start);
                        
                        std::cout << "[" << get_timestamp() << "] Completed section: " << section.first 
                                << " in " << format_duration(section_duration) << std::endl;
                        
                        result.section_responses[section.first] = section_response;
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
        std::cout << std::left << std::setw(20) << "Model" 
                << std::setw(15) << "Time" 
                << std::setw(15) << "Tokens/sec" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::left << std::setw(20) << result.model_name 
                    << std::setw(15) << format_duration(result.duration) 
                    << std::setw(15) << std::fixed << std::setprecision(2) << result.tokens_per_second << std::endl;
        }
        
        // Save detailed results to file if specified
        if (!output_file.empty()) {
            std::ofstream out(output_file);
            if (out.is_open()) {
                out << "========== EDGE AI LLM BENCHMARK DETAILED RESULTS ==========" << std::endl;
                out << "Prompt file: " << prompt_file << std::endl;
                out << "Total benchmark time: " << format_duration(total_duration) << std::endl;
                out << std::endl;
                
                for (const auto& result : results) {
                    out << "MODEL: " << result.model_name << std::endl;
                    out << "Time: " << format_duration(result.duration) << std::endl;
                    out << "Tokens/sec: " << std::fixed << std::setprecision(2) << result.tokens_per_second << std::endl;
                    
                    if (!result.section_responses.empty()) {
                        out << "\nSECTION-BY-SECTION RESPONSES:" << std::endl;
                        
                        for (const auto& section : prompt_sections) {
                            out << "\n=== SECTION: " << section.first << " ===" << std::endl;
                            out << "QUESTION:" << std::endl;
                            out << section.second << std::endl;
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
                
                if (!result.section_responses.empty()) {
                    std::cout << "SECTION-BY-SECTION RESPONSES:" << std::endl;
                    
                    for (const auto& section : prompt_sections) {
                        std::cout << "\n--- " << section.first << " ---" << std::endl;
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
    }
};

int main(int argc, char* argv[]) {
    // Default values
    std::string prompt_file = "prompt.txt";
    std::string output_file = "";
    bool verbose = false;
    bool parallel = false;
    std::vector<std::string> specific_models;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--parallel" || arg == "-p") {
            parallel = true;
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
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Ollama Edge AI LLM Benchmark Tool" << std::endl;
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --verbose, -v          Enable verbose output with answers" << std::endl;
            std::cout << "  --parallel, -p         Run models in parallel (caution on Raspberry Pi)" << std::endl;
            std::cout << "  --prompt, -i FILE      Specify prompt file (default: prompt.txt)" << std::endl;
            std::cout << "  --output, -o FILE      Save detailed results to file" << std::endl;
            std::cout << "  --model, -m MODEL      Specify a model to test (can be used multiple times)" << std::endl;
            std::cout << "  --help, -h             Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create benchmark instance
    LLMBenchmark benchmark(prompt_file, output_file, verbose, parallel);
    
    // Add specified models or default models
    if (specific_models.empty()) {
        // Use default models
        // benchmark.add_model("llava:7b");
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
    
    return 0;
}