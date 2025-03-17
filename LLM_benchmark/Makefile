CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -pedantic -O2
LDFLAGS = -lcurl -lpthread
INCLUDES = -I./include -I/usr/include/nlohmann

TARGET = edge_ai_benchmark
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
DEPS = $(OBJS:.o=.d)

# Create build directory if it doesn't exist
$(shell mkdir -p $(BUILD_DIR))

.PHONY: all clean install-deps

all: $(TARGET)

# Link the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compile each source file and generate dependencies
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential g++ libcurl4-openssl-dev nlohmann-json3-dev

-include $(DEPS)