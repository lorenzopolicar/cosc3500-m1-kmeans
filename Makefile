# COSC3500 M1 K-Means Makefile
# Portable build system with anti-vectorization toggle

# Compiler and flags
CXX ?= g++
CXXFLAGS_BASE = -std=c++17 -O2 -Wall -Wextra -Wshadow -Wconversion

# Detect compiler for anti-vectorization flags
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null | head -n1)
ifeq ($(findstring GCC,$(CXX_VERSION)),GCC)
    ANTIVEC_FLAGS = -fno-tree-vectorize
else ifeq ($(findstring clang,$(CXX_VERSION)),clang)
    ANTIVEC_FLAGS = -fno-vectorize -fno-slp-vectorize
else
    ANTIVEC_FLAGS = 
endif

# Build configuration
BUILD_DIR = build
TARGET = $(BUILD_DIR)/kmeans
SOURCES = src/main.cpp src/kmeans.cpp
OBJECTS = $(SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)

# Default target
all: $(TARGET)

# Release build with optional anti-vectorization
release: CXXFLAGS = $(CXXFLAGS_BASE) $(if $(ANTIVEC),$(ANTIVEC_FLAGS))
release: $(TARGET)

# Debug build with sanitizers
debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -Wshadow -Wconversion -fsanitize=address -fsanitize=undefined
debug: LDFLAGS = -fsanitize=address -fsanitize=undefined
debug: $(TARGET)

# Profile build with gprof
profile: CXXFLAGS = $(CXXFLAGS_BASE) -pg
profile: LDFLAGS = -pg
profile: $(TARGET)

# Main target
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

# Object files
$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Run cachegrind on a tiny default command
cachegrind: $(TARGET)
	valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out.$$(date +%s) $(TARGET) 1000 8 16 5

# Print compiler and system info
info:
	@echo "=== Build Configuration ==="
	@echo "Compiler: $(CXX)"
	@echo "Version: $(CXX_VERSION)"
	@echo "Base flags: $(CXXFLAGS_BASE)"
	@echo "Anti-vectorization flags: $(ANTIVEC_FLAGS)"
	@echo "ANTIVEC setting: $(ANTIVEC)"
	@echo ""
	@echo "=== System Info ==="
	@uname -a
	@echo ""
	@echo "=== CPU Info ==="
	@lscpu 2>/dev/null | head -n 10 || echo "lscpu not available"
	@echo ""
	@echo "=== Available Targets ==="
	@echo "  all      - Default build with -O2"
	@echo "  release  - Release build (use ANTIVEC=1 for anti-vectorization)"
	@echo "  debug    - Debug build with AddressSanitizer and UBSan"
	@echo "  profile  - Build with gprof profiling"
	@echo "  clean    - Remove build artifacts"
	@echo "  cachegrind - Run valgrind cachegrind on tiny test"
	@echo "  info     - Show this information"

# Phony targets
.PHONY: all release debug profile clean cachegrind info
