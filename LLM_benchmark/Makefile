CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O2
LDFLAGS = -lcurl -lpthread
INCLUDES = -I./include -I/usr/include/nlohmann

BENCHMARK_TARGET = edge_ai_benchmark
ROUGE_TARGET = rouge_evaluator
BUILD_DIR = build
SRC_DIR = src
TOOLS_DIR = tools
INCLUDE_DIR = include

BENCHMARK_SRCS = $(SRC_DIR)/api_client.cpp \
                 $(SRC_DIR)/memory_monitor.cpp \
                 $(SRC_DIR)/system_utils.cpp \
                 $(SRC_DIR)/llm_benchmark.cpp \
                 $(SRC_DIR)/main.cpp

# Fix: Correctly specify source files with their proper paths
ROUGE_SRCS = $(SRC_DIR)/rouge_evaluator.cpp \
             $(TOOLS_DIR)/rouge_evaluator.cpp

# Fix: Properly map source files to their corresponding object files
BENCHMARK_OBJS = $(BENCHMARK_SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
ROUGE_OBJS = $(SRC_DIR)/rouge_evaluator.cpp:$(BUILD_DIR)/rouge_evaluator.o \
             $(TOOLS_DIR)/rouge_evaluator.cpp:$(BUILD_DIR)/$(TOOLS_DIR)/rouge_evaluator.o

DEPS = $(BENCHMARK_OBJS:.o=.d)
DEPS += $(BUILD_DIR)/rouge_evaluator.d $(BUILD_DIR)/$(TOOLS_DIR)/rouge_evaluator.d

# Create build directory and subdirectories if they don't exist
$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BUILD_DIR)/$(TOOLS_DIR))

.PHONY: all clean install-deps

all: $(BENCHMARK_TARGET) $(ROUGE_TARGET)

# Link the benchmark executable
$(BENCHMARK_TARGET): $(BENCHMARK_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Link the ROUGE evaluator executable
$(ROUGE_TARGET): $(BUILD_DIR)/rouge_evaluator.o $(BUILD_DIR)/$(TOOLS_DIR)/rouge_evaluator.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# Compile tools files
$(BUILD_DIR)/$(TOOLS_DIR)/%.o: $(TOOLS_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BENCHMARK_TARGET) $(ROUGE_TARGET)

install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential g++ libcurl4-openssl-dev nlohmann-json3-dev

-include $(DEPS)