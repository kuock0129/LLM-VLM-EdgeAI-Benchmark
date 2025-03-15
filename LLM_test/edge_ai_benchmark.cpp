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
#include <sys/resource.h> // For getrusage and RUSAGE_SELF
#include <sys/time.h>     // For timeval structure
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

// Class to monitor process memory usage
class MemoryMonitor {
    private:
        std::mutex mtx;
        bool should_run;
        std::thread monitor_thread;
        unsigned long peak_memory;
        std::string process_name;
        int sample_interval_ms;
        
        // Get current RSS memory usage in KB
        unsigned long get_memory_usage() {
            // Method 1: Use getrusage
            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            unsigned long self_memory = usage.ru_maxrss;
            
            // Method 2: Read from /proc/self/status (Linux only)
            unsigned long proc_memory = 0;
            std::ifstream status_file("/proc/self/status");
            if (status_file.is_open()) {
                std::string line;
                while (std::getline(status_file, line)) {
                    if (line.substr(0, 6) == "VmRSS:") {
                        std::stringstream ss(line.substr(6));
                        ss >> proc_memory;
                        break;
                    }
                }
                status_file.close();
            }
            
            // Method 3: Read from /proc/PID/smaps (Linux only, more detailed)
            unsigned long detailed_memory = 0;
            if (!process_name.empty()) {
                // Use pgrep to find PID
                std::string cmd = "pgrep -f " + process_name;
                FILE* pipe = popen(cmd.c_str(), "r");
                if (pipe) {
                    char buffer[128];
                    std::string pid;
                    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                        pid = buffer;
                        pid.erase(pid.find_last_not_of(" \n\r\t") + 1);
                    }
                    pclose(pipe);
                    
                    if (!pid.empty()) {
                        std::string smaps_path = "/proc/" + pid + "/smaps";
                        std::ifstream smaps_file(smaps_path);
                        if (smaps_file.is_open()) {
                            std::string smaps_line;
                            while (std::getline(smaps_file, smaps_line)) {
                                if (smaps_line.substr(0, 4) == "Rss:") {
                                    unsigned long rss_value;
                                    std::stringstream ss(smaps_line.substr(4));
                                    ss >> rss_value;
                                    detailed_memory += rss_value;
                                }
                            }
                            smaps_file.close();
                        }
                    }
                }
            }
            
            // Use the highest value among the methods that returned data
            return std::max({self_memory, proc_memory, detailed_memory});
        }
        
        // Monitor thread function
        void monitor_memory() {
            while (should_run) {
                unsigned long current = get_memory_usage();
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    peak_memory = std::max(peak_memory, current);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
            }
        }
        
    public:
        MemoryMonitor(const std::string& process = "", int interval_ms = 100) 
            : should_run(false), peak_memory(0), process_name(process), sample_interval_ms(interval_ms) {}
        
        ~MemoryMonitor() {
            stop();
        }
        
        // Start monitoring
        void start() {
            std::lock_guard<std::mutex> lock(mtx);
            if (!should_run) {
                should_run = true;
                peak_memory = 0;
                monitor_thread = std::thread(&MemoryMonitor::monitor_memory, this);
            }
        }
        
        // Stop monitoring
        void stop() {
            {
                std::lock_guard<std::mutex> lock(mtx);
                should_run = false;
            }
            
            if (monitor_thread.joinable()) {
                monitor_thread.join();
            }
        }
        
        // Get peak memory in KB
        unsigned long get_peak_memory() {
            std::lock_guard<std::mutex> lock(mtx);
            return peak_memory;
        }
        
        // Format memory size for display
        static std::string format_memory(unsigned long memory_kb) {
            if (memory_kb > 1024*1024) {
                return std::to_string(memory_kb / (1024*1024)) + " GB";
            } else if (memory_kb > 1024) {
                return std::to_string(memory_kb / 1024) + " MB";
            } else {
                return std::to_string(memory_kb) + " KB";
            }
        }
    };
    
    // Add a function to get Ollama process memory usage specifically
    unsigned long get_ollama_memory_usage() {
        unsigned long total_memory = 0;
        
        // Use ps command to get memory usage of Ollama process
        FILE* pipe = popen("ps -C ollama -o rss=", "r");
        if (pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                total_memory = std::stoul(buffer);
            }
            pclose(pipe);
        }
        
        return total_memory;
    }



// Class to benchmark LLM models
class LLMBenchmark {
private:
    std::vector<std::string> models;
    std::string prompt_file;
    std::string output_file;
    bool verbose;
    bool parallel;
    bool track_memory;
    OllamaAPI api;
    std::mutex output_mutex;
    
    // Result structure with memory metrics
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
                 bool verbose_output = false, bool run_parallel = false, bool memory_tracking = true)
        : prompt_file(prompt_path), output_file(output_path), 
          verbose(verbose_output), parallel(run_parallel), track_memory(memory_tracking) {
        
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
        std::cout << "Memory tracking: " << (track_memory ? "ON" : "OFF") << std::endl;
        std::cout << "===================================" << std::endl;
        
        std::vector<Result> results;
        
        // Get baseline memory before starting
        unsigned long baseline_memory = track_memory ? get_ollama_memory_usage() : 0;
        
        if (track_memory) {
            std::cout << "Baseline Ollama memory usage: " << MemoryMonitor::format_memory(baseline_memory) << std::endl;
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
                                    << MemoryMonitor::format_memory(result.peak_memory) 
                                    << " (+" << MemoryMonitor::format_memory(result.peak_memory - result.baseline_memory) 
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
                                            << MemoryMonitor::format_memory(section_memory) << std::endl;
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
                            << MemoryMonitor::format_memory(result.peak_memory) 
                            << " (+" << MemoryMonitor::format_memory(result.peak_memory - result.baseline_memory) 
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
                                    << MemoryMonitor::format_memory(section_memory) << std::endl;
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
                        << std::setw(15) << MemoryMonitor::format_memory(result.peak_memory)
                        << std::setw(15) << MemoryMonitor::format_memory(result.peak_memory - result.baseline_memory) 
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
                    out << "Baseline Ollama memory usage: " << MemoryMonitor::format_memory(baseline_memory) << std::endl;
                }
                
                out << std::endl;
                
                for (const auto& result : results) {
                    out << "MODEL: " << result.model_name << std::endl;
                    out << "Time: " << format_duration(result.duration) << std::endl;
                    out << "Tokens/sec: " << std::fixed << std::setprecision(2) << result.tokens_per_second << std::endl;
                    
                    if (track_memory) {
                        out << "Peak memory: " << MemoryMonitor::format_memory(result.peak_memory) << std::endl;
                        out << "Memory increase: " << MemoryMonitor::format_memory(result.peak_memory - result.baseline_memory) << std::endl;
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
                                    out << "Memory: " << MemoryMonitor::format_memory(metrics_it->second.second) << std::endl;
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
                    std::cout << " | Memory: " << MemoryMonitor::format_memory(result.peak_memory) 
                            << " (+" << MemoryMonitor::format_memory(result.peak_memory - result.baseline_memory) 
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
                                std::cout << " | Memory: " << MemoryMonitor::format_memory(metrics_it->second.second);
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
            std::cout << "Memory baseline: " << MemoryMonitor::format_memory(baseline_memory) << std::endl;
            
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
                std::cout << MemoryMonitor::format_memory(memory_increase) << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    // Default values
    std::string prompt_file = "prompt.txt";
    std::string output_file = "";
    bool verbose = false;
    bool parallel = false;
    bool track_memory = true; // Enable memory tracking by default
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
            std::cout << "  --no-memory, -nm       Disable memory tracking" << std::endl;
            std::cout << "  --prompt, -i FILE      Specify prompt file (default: prompt.txt)" << std::endl;
            std::cout << "  --output, -o FILE      Save detailed results to file" << std::endl;
            std::cout << "  --model, -m MODEL      Specify a model to test (can be used multiple times)" << std::endl;
            std::cout << "  --help, -h             Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create benchmark instance with memory tracking
    LLMBenchmark benchmark(prompt_file, output_file, verbose, parallel, track_memory);
    
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
    
    return 0;
}