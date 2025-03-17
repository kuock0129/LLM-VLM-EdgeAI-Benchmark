#include "rouge_evaluator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <regex>
#include <iomanip>

RougeEvaluator::RougeEvaluator() {
    // Initialize default categories
    categories = {"generalKnowledge", "reasoning", "mathematics", "coding"};
    
    // Initialize default reference answers
    reference_answers = {
        {"generalKnowledge", "Neil Armstrong was the first person to walk on the moon and it happened in 1969."},
        {"reasoning", "If a ball costs $1.05 and a bat costs $1.00 more than the ball, they cost together $3.10."},
        {"mathematics", "The derivative of f(x) = 3x^4 - 2x^2 + 5x - 7 is 12x^3 - 4x + 5."},
        {"coding", "def is_palindrome(s):\n    return s == s[::-1]"}
    };
}

std::vector<std::string> RougeEvaluator::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string lower_text = text;
    
    // Convert to lowercase
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                  [](unsigned char c) { return std::tolower(c); });
    
    // Use regex to tokenize (split by whitespace and punctuation)
    std::regex word_regex("[\\w']+");
    auto words_begin = std::sregex_iterator(lower_text.begin(), lower_text.end(), word_regex);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        tokens.push_back(match.str());
    }
    
    return tokens;
}

std::map<std::string, double> RougeEvaluator::calculateROUGE1(const std::string& candidate, const std::string& reference) {
    // Tokenize candidate and reference
    std::vector<std::string> candidate_tokens = tokenize(candidate);
    std::vector<std::string> reference_tokens = tokenize(reference);
    
    // Create sets for faster lookup
    std::unordered_set<std::string> reference_set(reference_tokens.begin(), reference_tokens.end());
    
    // Count matching tokens
    size_t matches = 0;
    for (const auto& token : candidate_tokens) {
        if (reference_set.count(token) > 0) {
            matches++;
        }
    }
    
    // Calculate precision, recall, and F1
    double precision = candidate_tokens.empty() ? 0.0 : static_cast<double>(matches) / candidate_tokens.size();
    double recall = reference_tokens.empty() ? 0.0 : static_cast<double>(matches) / reference_tokens.size();
    double f1 = (precision + recall > 0.0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
    
    return {
        {"precision", precision},
        {"recall", recall},
        {"f1", f1}
    };
}

std::unordered_map<std::string, std::string> RougeEvaluator::extractAnswers(const std::string& model_output) {
    std::unordered_map<std::string, std::string> answers;
    
    // General Knowledge - Extract statements about Neil Armstrong and 1969
    if (model_output.find("Neil Armstrong") != std::string::npos) {
        std::regex general_regex("([^.]*Neil Armstrong[^.]*\\d{4}[^.]*)");
        std::smatch general_match;
        if (std::regex_search(model_output, general_match, general_regex)) {
            answers["generalKnowledge"] = general_match[1];
        }
    }
    
    // Reasoning - Extract statements about ball and bat costs
    if (model_output.find("ball costs") != std::string::npos && 
        model_output.find("bat costs") != std::string::npos) {
        std::regex reasoning_regex("([^.]*ball costs[^.]*bat costs[^.]*together[^.]*)");
        std::smatch reasoning_match;
        if (std::regex_search(model_output, reasoning_match, reasoning_regex)) {
            answers["reasoning"] = reasoning_match[1];
        }
    }
    
    // Mathematics - Extract derivatives
    if (model_output.find("derivative") != std::string::npos && 
        model_output.find("3x^4") != std::string::npos) {
        std::regex math_regex1("([^.]*derivative[^.]*3x\\^4[^.]*is[^.]*)");
        std::regex math_regex2("([^.]*f'[^=]*=[^.]*12x\\^3[^.]*)");
        
        std::smatch math_match;
        if (std::regex_search(model_output, math_match, math_regex1)) {
            answers["mathematics"] = math_match[1];
        } else if (std::regex_search(model_output, math_match, math_regex2)) {
            answers["mathematics"] = math_match[1];
        }
    }
    
    // Coding - Extract palindrome function
    if (model_output.find("palindrome") != std::string::npos) {
        std::regex coding_regex("(def is_palindrome[\\s\\S]*?return[\\s\\S]*?\\n)");
        std::smatch coding_match;
        if (std::regex_search(model_output, coding_match, coding_regex)) {
            answers["coding"] = coding_match[1];
        }
    }
    
    return answers;
}

void RougeEvaluator::setReferenceAnswers(const std::unordered_map<std::string, std::string>& answers) {
    reference_answers = answers;
}

void RougeEvaluator::addModelOutput(const std::string& model_name, const std::string& output) {
    model_outputs[model_name] = output;
    if (std::find(models.begin(), models.end(), model_name) == models.end()) {
        models.push_back(model_name);
    }
}

bool RougeEvaluator::loadReferenceAnswers(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        
        json j;
        file >> j;
        
        for (const auto& category : categories) {
            if (j.contains(category)) {
                reference_answers[category] = j[category];
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading reference answers: " << e.what() << std::endl;
        return false;
    }
}

bool RougeEvaluator::loadModelOutputs(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return false;
        }
        
        json j;
        file >> j;
        
        // Check if there's a "model_outputs" key for the nested structure
        if (j.contains("model_outputs")) {
            // Handle nested structure from benchmark output
            for (const auto& [model_name, output] : j["model_outputs"].items()) {
                if (output.is_string()) {
                    addModelOutput(model_name, output);
                } else {
                    std::cerr << "Warning: Output for model " << model_name 
                              << " is not a string. Skipping." << std::endl;
                }
            }
        } else {
            // Handle flat structure (direct model to output mapping)
            for (const auto& [model_name, output] : j.items()) {
                if (output.is_string()) {
                    addModelOutput(model_name, output);
                } else {
                    std::cerr << "Warning: Output for model " << model_name 
                              << " is not a string. Skipping." << std::endl;
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model outputs: " << e.what() << std::endl;
        return false;
    }
}

void RougeEvaluator::calculateScores() {
    results.clear();
    average_f1.clear();
    
    for (const auto& model : models) {
        if (model_outputs.find(model) == model_outputs.end()) {
            continue;
        }
        
        const std::string& output = model_outputs[model];
        std::unordered_map<std::string, std::string> extracted_answers = extractAnswers(output);
        
        results[model] = {};
        double total_f1 = 0.0;
        int question_count = 0;
        
        for (const auto& category : categories) {
            if (extracted_answers.find(category) != extracted_answers.end() && 
                !extracted_answers[category].empty()) {
                
                results[model][category] = calculateROUGE1(
                    extracted_answers[category], 
                    reference_answers[category]
                );
                
                total_f1 += results[model][category]["f1"];
                question_count++;
            } else {
                results[model][category] = {{"precision", 0.0}, {"recall", 0.0}, {"f1", 0.0}};
            }
        }
        
        average_f1[model] = question_count > 0 ? total_f1 / question_count : 0.0;
    }
    
    // Calculate task-based accuracy
    evaluateTaskAccuracy();
}

void RougeEvaluator::evaluateTaskAccuracy() {
    task_accuracy.clear();
    
    for (const auto& model : models) {
        if (model_outputs.find(model) == model_outputs.end()) {
            continue;
        }
        
        const std::string& output = model_outputs[model];
        double general_knowledge = (output.find("Neil Armstrong") != std::string::npos && 
                                  output.find("1969") != std::string::npos) ? 1.0 : 0.0;
        
        double reasoning = output.find("$3.10") != std::string::npos ? 1.0 : 
                          (output.find("$2.05") != std::string::npos ? 0.5 : 0.0);
        
        double mathematics = output.find("12x^3 - 4x + 5") != std::string::npos ? 1.0 : 0.0;
        
        double coding = output.find("return s == s[::-1]") != std::string::npos ? 1.0 : 0.0;
        
        task_accuracy[model] = (general_knowledge + reasoning + mathematics + coding) / 4.0;
    }
}

const std::unordered_map<std::string, std::unordered_map<std::string, std::map<std::string, double>>>& RougeEvaluator::getResults() const {
    return results;
}

const std::unordered_map<std::string, double>& RougeEvaluator::getAverageF1() const {
    return average_f1;
}

const std::unordered_map<std::string, double>& RougeEvaluator::getTaskAccuracy() const {
    return task_accuracy;
}

void RougeEvaluator::printResults(bool detailed) {
    std::cout << "ROUGE-1 F1 Scores by Model and Question:" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    for (const auto& model : models) {
        if (results.find(model) == results.end()) {
            continue;
        }
        
        std::cout << "\n" << model << ":" << std::endl;
        
        for (const auto& category : categories) {
            if (results[model].find(category) != results[model].end()) {
                std::cout << "  " << category << ": " 
                         << std::fixed << std::setprecision(3) << results[model][category]["f1"] << std::endl;
            }
        }
        
        std::cout << "  Average F1: " << std::fixed << std::setprecision(3) << average_f1[model] << std::endl;
    }
    
    std::cout << "\nModel Summary (Average ROUGE-1 F1):" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Model                Avg ROUGE-1 F1" << std::endl;
    std::cout << "------------------------------------" << std::endl;
    
    for (const auto& model : models) {
        if (average_f1.find(model) == average_f1.end()) {
            continue;
        }
        
        std::cout << std::left << std::setw(20) << model << std::fixed << std::setprecision(3) 
                 << average_f1[model] << std::endl;
    }
    
    // std::cout << "\nTask-based Accuracy:" << std::endl;
    // std::cout << "====================" << std::endl;
    // std::cout << "Model                Avg Accuracy" << std::endl;
    // std::cout << "--------------------" << std::endl;
    
    // for (const auto& model : models) {
    //     if (task_accuracy.find(model) == task_accuracy.end()) {
    //         continue;
    //     }
        
    //     std::cout << std::left << std::setw(20) << model << std::fixed << std::setprecision(3) 
    //              << task_accuracy[model] << std::endl;
    // }
    
    if (detailed) {
        for (const auto& model : models) {
            if (model_outputs.find(model) == model_outputs.end()) {
                continue;
            }
            
            std::cout << "\n========== " << model << " Output ==========" << std::endl;
            std::cout << model_outputs[model] << std::endl;
            std::cout << "=======================================" << std::endl;
        }
    }
}

bool RougeEvaluator::saveResults(const std::string& filename) {
    try {
        json j;
        
        // Add ROUGE-1 scores
        for (const auto& model : models) {
            if (results.find(model) == results.end()) {
                continue;
            }
            
            j["rouge"][model] = {};
            
            for (const auto& category : categories) {
                if (results[model].find(category) != results[model].end()) {
                    j["rouge"][model][category] = results[model][category];
                }
            }
            
            j["rouge"][model]["average_f1"] = average_f1[model];
        }
        
        // Add task accuracy
        for (const auto& [model, accuracy] : task_accuracy) {
            j["task_accuracy"][model] = accuracy;
        }
        
        // Add model outputs
        for (const auto& [model, output] : model_outputs) {
            j["model_outputs"][model] = output;
        }
        
        // Add reference answers
        for (const auto& [category, answer] : reference_answers) {
            j["reference_answers"][category] = answer;
        }
        
        // Write to file
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
            return false;
        }
        
        file << std::setw(4) << j << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving results: " << e.what() << std::endl;
        return false;
    }
}