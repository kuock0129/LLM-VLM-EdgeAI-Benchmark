#include "api_client.h"
#include <iostream>
#include <iomanip>
#include <chrono>

size_t OllamaAPI::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
    size_t total_size = size * nmemb;
    response->append(static_cast<char*>(contents), total_size);
    return total_size;
}

OllamaAPI::OllamaAPI(const std::string& url, bool memory_mapping) 
    : base_url(url), use_mmap(memory_mapping) {}

bool OllamaAPI::initialize() {
    return curl_global_init(CURL_GLOBAL_ALL) == CURLE_OK;
}

void OllamaAPI::cleanup() {
    curl_global_cleanup();
}

std::vector<std::string> OllamaAPI::list_models() {
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

std::string OllamaAPI::generate(
    const std::string& model, 
    const std::string& prompt, 
    bool stream, 
    bool verbose
) {
    std::string response_text;
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        return "Error: Failed to initialize cURL";
    }
    
    std::string url = base_url + "/api/generate";
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    // Create JSON request body with memory options
    json request_body = {
        {"model", model},
        {"prompt", prompt},
        {"stream", stream},
        {"options", {
            {"num_gpu", 1},      // Use GPU if available
            {"temperature", 0.7},
            {"mmap", use_mmap}   // Add memory-mapped option
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
        std::cout << "[DEBUG] Memory mapping: " << (use_mmap ? "enabled" : "disabled") << std::endl;
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
    
    return response_text;
}