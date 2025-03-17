#ifndef API_CLIENT_H
#define API_CLIENT_H

#include <string>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @brief Class to handle Ollama API interactions
 */
class OllamaAPI {
private:
    std::string base_url;
    bool use_mmap;  // Use memory-mapped model loading
    
    // Callback function for cURL to write response data
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response);

public:
    /**
     * @brief Constructor
     * @param url The base URL for the Ollama API
     * @param memory_mapping Whether to use memory-mapped model loading
     */
    explicit OllamaAPI(const std::string& url = "http://localhost:11434", bool memory_mapping = false);
    
    /**
     * @brief Initialize curl once at the beginning
     * @return true if initialization was successful, false otherwise
     */
    static bool initialize();
    
    /**
     * @brief Clean up curl at the end
     */
    static void cleanup();
    
    /**
     * @brief Get list of available models
     * @return Vector of model names
     */
    std::vector<std::string> list_models();
    
    /**
     * @brief Generate text from a model
     * @param model The model name
     * @param prompt The input prompt
     * @param stream Whether to stream the output
     * @param verbose Whether to print verbose information
     * @return The model's response
     */
    std::string generate(
        const std::string& model, 
        const std::string& prompt, 
        bool stream = false, 
        bool verbose = false
    );
};

#endif // API_CLIENT_H