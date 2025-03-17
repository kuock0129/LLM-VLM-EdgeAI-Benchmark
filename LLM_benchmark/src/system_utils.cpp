#include "system_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h> // For geteuid()

bool configure_swap(unsigned long swap_size_mb, int swappiness) {
    // Check if we have root privileges (will need sudo otherwise)
    bool is_root = (geteuid() == 0);
    std::string sudo_prefix = is_root ? "" : "sudo ";
    
    std::cout << "Configuring swap file (" << swap_size_mb << "MB) with swappiness " << swappiness << std::endl;
    
    try {
        // Check if swap already exists
        std::string check_cmd = "free -m | grep Swap | awk '{print $2}'";
        FILE* check_pipe = popen(check_cmd.c_str(), "r");
        if (!check_pipe) return false;
        
        char buffer[128];
        std::string existing_swap;
        if (fgets(buffer, sizeof(buffer), check_pipe) != nullptr) {
            existing_swap = buffer;
            existing_swap.erase(existing_swap.find_last_not_of(" \n\r\t") + 1);
        }
        pclose(check_pipe);
        
        unsigned long current_swap = std::stoul(existing_swap);
        
        if (current_swap >= swap_size_mb) {
            std::cout << "Sufficient swap already exists (" << current_swap << "MB)" << std::endl;
        } else {
            // Create or resize swap file
            std::cout << "Creating/modifying swap file..." << std::endl;
            
            // Disable existing swap if any
            if (current_swap > 0) {
                std::string cmd1 = sudo_prefix + "swapoff -a";
                if (system(cmd1.c_str()) != 0) {
                    std::cerr << "Failed to disable existing swap" << std::endl;
                    return false;
                }
            }
            
            // Create swap file
            std::string cmd2 = sudo_prefix + "fallocate -l " + std::to_string(swap_size_mb) + "M /swapfile";
            if (system(cmd2.c_str()) != 0) {
                std::cerr << "Failed to create swap file" << std::endl;
                return false;
            }
            
            // Set permissions
            std::string cmd3 = sudo_prefix + "chmod 600 /swapfile";
            if (system(cmd3.c_str()) != 0) {
                std::cerr << "Failed to set swap file permissions" << std::endl;
                return false;
            }
            
            // Make swap
            std::string cmd4 = sudo_prefix + "mkswap /swapfile";
            if (system(cmd4.c_str()) != 0) {
                std::cerr << "Failed to make swap file" << std::endl;
                return false;
            }
            
            // Enable swap
            std::string cmd5 = sudo_prefix + "swapon /swapfile";
            if (system(cmd5.c_str()) != 0) {
                std::cerr << "Failed to enable swap file" << std::endl;
                return false;
            }
            
            std::cout << "Swap file created and enabled (" << swap_size_mb << "MB)" << std::endl;
        }
        
        // Set swappiness
        std::string cmd6 = sudo_prefix + "sysctl -w vm.swappiness=" + std::to_string(swappiness);
        if (system(cmd6.c_str()) != 0) {
            std::cerr << "Failed to set swappiness" << std::endl;
            return false;
        }
        
        std::cout << "Swappiness set to " << swappiness << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error configuring swap: " << e.what() << std::endl;
        return false;
    }
}

std::pair<unsigned long, unsigned long> get_system_memory() {
    unsigned long total_mem = 0;
    unsigned long available_mem = 0;
    
    // Read from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.substr(0, 9) == "MemTotal:") {
                std::stringstream ss(line.substr(9));
                ss >> total_mem;
            } else if (line.substr(0, 12) == "MemAvailable:") {
                std::stringstream ss(line.substr(12));
                ss >> available_mem;
            }
        }
        meminfo.close();
    }
    
    // Convert from KB to MB
    return {total_mem / 1024, available_mem / 1024};
}

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

std::string format_memory(unsigned long memory_kb) {
    if (memory_kb > 1024*1024) {
        return std::to_string(memory_kb / (1024*1024)) + " GB";
    } else if (memory_kb > 1024) {
        return std::to_string(memory_kb / 1024) + " MB";
    } else {
        return std::to_string(memory_kb) + " KB";
    }
}