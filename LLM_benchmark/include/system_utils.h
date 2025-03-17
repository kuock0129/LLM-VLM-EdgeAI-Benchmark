#ifndef SYSTEM_UTILS_H
#define SYSTEM_UTILS_H

#include <string>
#include <utility>

/**
 * @brief Configure swap file for better performance with large models
 * 
 * @param swap_size_mb Size of swap file in MB
 * @param swappiness VM swappiness value (0-100)
 * @return true if successful, false otherwise
 */
bool configure_swap(unsigned long swap_size_mb, int swappiness);

/**
 * @brief Get the total and available system memory
 * 
 * @return Pair of (total_memory_mb, available_memory_mb)
 */
std::pair<unsigned long, unsigned long> get_system_memory();

/**
 * @brief Get Ollama process memory usage
 * 
 * @return Memory usage in KB
 */
unsigned long get_ollama_memory_usage();

/**
 * @brief Format memory size for human-readable display
 * 
 * @param memory_kb Memory size in KB
 * @return Formatted string (e.g., "1.2 GB")
 */
std::string format_memory(unsigned long memory_kb);

#endif // SYSTEM_UTILS_H