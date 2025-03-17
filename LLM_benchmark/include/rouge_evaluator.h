#ifndef ROUGE_EVALUATOR_H
#define ROUGE_EVALUATOR_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @brief ROUGE-1 Evaluator for Large Language Model outputs
 * 
 * This class calculates ROUGE-1 scores (unigram overlap)
 * between model outputs and reference answers, providing
 * a quantitative measure of output quality.
 */
class RougeEvaluator {
private:
    // Reference answers for different categories
    std::unordered_map<std::string, std::string> reference_answers;
    
    // Model outputs keyed by model name
    std::unordered_map<std::string, std::string> model_outputs;
    
    // Question categories
    std::vector<std::string> categories;
    
    // Models to evaluate
    std::vector<std::string> models;
    
    // Results storage
    std::unordered_map<std::string, std::unordered_map<std::string, std::map<std::string, double>>> results;
    std::unordered_map<std::string, double> average_f1;
    std::unordered_map<std::string, double> task_accuracy;
    
    /**
     * @brief Tokenize a string into unigrams
     * @param text Input text to tokenize
     * @return Vector of tokens
     */
    std::vector<std::string> tokenize(const std::string& text);
    
    /**
     * @brief Calculate ROUGE-1 score between two texts
     * @param candidate Candidate text (model output)
     * @param reference Reference text (gold standard)
     * @return Map with precision, recall, and F1 scores
     */
    std::map<std::string, double> calculateROUGE1(const std::string& candidate, const std::string& reference);
    
    /**
     * @brief Extract answers for different categories from model output
     * @param model_output Full output text from a model
     * @return Map of category to extracted answer
     */
    std::unordered_map<std::string, std::string> extractAnswers(const std::string& model_output);
    
    /**
     * @brief Calculate task-based accuracy for each model
     */
    void evaluateTaskAccuracy();
    
public:
    /**
     * @brief Constructor
     */
    RougeEvaluator();
    
    /**
     * @brief Set reference answers
     * @param answers Map of category to reference answer
     */
    void setReferenceAnswers(const std::unordered_map<std::string, std::string>& answers);
    
    /**
     * @brief Add model output
     * @param model_name Name of the model
     * @param output Output text from the model
     */
    void addModelOutput(const std::string& model_name, const std::string& output);
    
    /**
     * @brief Load reference answers from JSON file
     * @param filename Path to JSON file
     * @return true if successful, false otherwise
     */
    bool loadReferenceAnswers(const std::string& filename);
    
    /**
     * @brief Load model outputs from JSON file
     * @param filename Path to JSON file
     * @return true if successful, false otherwise
     */
    bool loadModelOutputs(const std::string& filename);
    
    /**
     * @brief Calculate ROUGE scores for all models
     */
    void calculateScores();
    
    /**
     * @brief Get results for all models
     * @return Map of model to category to score metrics
     */
    const std::unordered_map<std::string, std::unordered_map<std::string, std::map<std::string, double>>>& getResults() const;
    
    /**
     * @brief Get average F1 scores for all models
     * @return Map of model to average F1 score
     */
    const std::unordered_map<std::string, double>& getAverageF1() const;
    
    /**
     * @brief Get task accuracy for all models
     * @return Map of model to task accuracy
     */
    const std::unordered_map<std::string, double>& getTaskAccuracy() const;
    
    /**
     * @brief Print results to console
     * @param detailed Whether to print detailed results
     */
    void printResults(bool detailed = false);
    
    /**
     * @brief Save results to JSON file
     * @param filename Path to output file
     * @return true if successful, false otherwise
     */
    bool saveResults(const std::string& filename);
};

#endif // ROUGE_EVALUATOR_H