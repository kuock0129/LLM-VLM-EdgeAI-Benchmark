#include "memory_monitor.h"
#include "system_utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/resource.h> // For getrusage and RUSAGE_SELF

MemoryMonitor::MemoryMonitor(const std::string& process, int interval_ms) 
    : should_run(false), peak_memory(0), process_name(process), sample_interval_ms(interval_ms) {}

MemoryMonitor::~MemoryMonitor() {
    stop();
}

unsigned long MemoryMonitor::get_memory_usage() {
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

void MemoryMonitor::monitor_memory() {
    while (should_run) {
        unsigned long current = get_memory_usage();
        {
            std::lock_guard<std::mutex> lock(mtx);
            peak_memory = std::max(peak_memory, current);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
    }
}

void MemoryMonitor::start() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!should_run) {
        should_run = true;
        peak_memory = 0;
        monitor_thread = std::thread(&MemoryMonitor::monitor_memory, this);
    }
}

void MemoryMonitor::stop() {
    should_run = false;
    
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }
}

unsigned long MemoryMonitor::get_peak_memory() {
    std::lock_guard<std::mutex> lock(mtx);
    return peak_memory;
}