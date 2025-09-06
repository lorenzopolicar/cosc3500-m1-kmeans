# COSC3500 M1 K-Means Makefile
# Portable build system with anti-vectorization toggle

# Compiler and flags
CXX ?= g++
CXXFLAGS_BASE = -std=c++17 -O2 -Wall -Wextra -Wshadow -Wconversion -Iinclude

# Detect compiler for anti-vectorization flags and filesystem library
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null | head -n1)
ifeq ($(findstring GCC,$(CXX_VERSION)),GCC)
    ANTIVEC_FLAGS = -fno-tree-vectorize
    # GCC < 9.0 needs -lstdc++fs for filesystem support
    ifeq ($(shell $(CXX) -dumpversion | cut -d. -f1),8)
        FS_LIB = -lstdc++fs
    else
        FS_LIB = 
    endif
else ifeq ($(findstring clang,$(CXX_VERSION)),clang)
    ANTIVEC_FLAGS = -fno-vectorize -fno-slp-vectorize
    # Clang on macOS has filesystem support built-in
    FS_LIB = 
else
    ANTIVEC_FLAGS = 
    FS_LIB = 
endif

# Build configuration
BUILD_DIR = build
TARGET = $(BUILD_DIR)/kmeans
SOURCES = src/main.cpp src/kmeans.cpp
OBJECTS = $(SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)

# Default target with anti-vectorization
all: CXXFLAGS = $(CXXFLAGS_BASE) $(if $(ANTIVEC),$(ANTIVEC_FLAGS))
all: $(TARGET)

# Release build with optional anti-vectorization and definitions
release: CXXFLAGS = $(CXXFLAGS_BASE) $(if $(ANTIVEC),$(ANTIVEC_FLAGS)) $(DEFS)
release: $(TARGET)

# Debug build with sanitizers and anti-vectorization
debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -Wshadow -Wconversion $(ANTIVEC_FLAGS) -fsanitize=address -fsanitize=undefined
debug: LDFLAGS = -fsanitize=address -fsanitize=undefined
debug: $(TARGET)

# Profile build with gprof, anti-vectorization, and optional definitions
profile: CXXFLAGS = $(CXXFLAGS_BASE) $(ANTIVEC_FLAGS) -pg $(DEFS)
profile: LDFLAGS = -pg
profile: $(TARGET)

# Main target
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) $(LDFLAGS) $(FS_LIB) -o $@

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
	@echo "Build definitions: $(DEFS)"
	@echo "Filesystem library: $(FS_LIB)"
	@echo ""
	@echo "=== System Info ==="
	@uname -a
	@echo ""
	@echo "=== CPU Info ==="
	@lscpu 2>/dev/null | head -n 10 || echo "lscpu not available"
	@echo ""
	@echo "=== Available Targets ==="
	@echo "  all      - Default build with -O2 (respects ANTIVEC setting)"
	@echo "  release  - Release build (respects ANTIVEC setting)"
	@echo "  debug    - Debug build with sanitizers and anti-vectorization"
	@echo "  profile  - Build with gprof and anti-vectorization"
	@echo "  clean    - Remove build artifacts"
	@echo "  cachegrind - Run valgrind cachegrind on tiny test"
	@echo "  info     - Show this information"

# Phony targets
.PHONY: all release debug profile clean cachegrind info