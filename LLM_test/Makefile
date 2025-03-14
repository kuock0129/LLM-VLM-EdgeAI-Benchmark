CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
LDFLAGS = -lcurl -lpthread
INCLUDES = -I/usr/include/nlohmann

TARGET = edge_ai_benchmark
SRCS = edge_ai_benchmark.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential g++ libcurl4-openssl-dev nlohmann-json3-dev

.PHONY: all clean install-deps