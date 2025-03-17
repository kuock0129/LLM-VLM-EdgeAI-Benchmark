#ifndef MEMORY_MONITOR_H
#define MEMORY_MONITOR_H

#include <string>
#include <thread>
#include <mutex>
#include <atomic>

/**
 * @brief Class to monitor process memory usage
 */
class MemoryMonitor {
private:
    std::mutex mtx;
    std::atomic<bool> should_run;
    std::thread monitor_thread;
    unsigned long peak_memory;
    std::string process_name;
    int sample_interval_ms;
    
    /**
     * @brief Get current RSS memory usage in KB
     * @return Current memory usage in KB
     */
    unsigned long get_memory_usage();
    
    /**
     * @brief Monitor thread function
     */
    void monitor_memory();
    
public:
    /**
     * @brief Constructor
     * @param process Name of the process to monitor
     * @param interval_ms Sampling interval in milliseconds
     */
    explicit MemoryMonitor(const std::string& process = "", int interval_ms = 100);
    
    /**
     * @brief Destructor
     */
    ~MemoryMonitor();
    
    /**
     * @brief Start monitoring
     */
    void start();
    
    /**
     * @brief Stop monitoring
     */
    void stop();
    
    /**
     * @brief Get peak memory in KB
     * @return Peak memory usage in KB
     */
    unsigned long get_peak_memory();
};

#endif // MEMORY_MONITOR_H