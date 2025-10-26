# COSC3500 Milestone 2 - K-Means Parallel Implementation
# Makefile for CUDA, OpenMP, and Serial builds

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# CUDA paths (adjust for your cluster)
CUDA_INC = /opt/local/stow/cuda-11.1/include
CUDA_LIB = /opt/local/stow/cuda-11.1/lib64

# MPI paths (for future use if needed)
MPI_INC = /usr/include/openmpi-x86_64
MPI_LIB = /usr/lib64/openmpi/lib

# =============================================================================
# COMPILER CONFIGURATION
# =============================================================================

# C++ compiler
CXX ?= g++
MPICXX ?= mpicxx

# CUDA compiler
NVCC ?= nvcc

# Compute capability for target GPU
# sm_80 for A100, sm_70 for V100, sm_60 for P100
GPU_ARCH ?= sm_80

# =============================================================================
# COMPILATION FLAGS
# =============================================================================

# Base C++ flags
CXXFLAGS_BASE = -std=c++17 -Wall -Wextra -Wshadow -Wconversion

# Optimization flags
OPT_FLAGS = -O3 -march=native -mtune=native

# Debug flags
DEBUG_FLAGS = -g -O0 -DDEBUG

# OpenMP flags
OPENMP_FLAGS = -fopenmp

# CUDA compilation flags
NVFLAGS = -O3 --gpu-architecture=$(GPU_ARCH) -std=c++17
NVFLAGS_DEBUG = -g -G -O0 --gpu-architecture=$(GPU_ARCH) -std=c++17

# Include paths
INCLUDES = -I../include -Iinclude -I$(CUDA_INC)

# =============================================================================
# LIBRARY CONFIGURATION
# =============================================================================

# Math libraries (Intel MKL if available, otherwise standard BLAS)
MATH_LIBS = -lm

# CUDA libraries
CUDA_LIBS = -L$(CUDA_LIB) -lcudart -lcublas

# OpenMP library
OPENMP_LIBS = -lgomp

# All libraries for different targets
LIBS_SERIAL = $(MATH_LIBS)
LIBS_OPENMP = $(MATH_LIBS) $(OPENMP_LIBS)
LIBS_CUDA = $(MATH_LIBS) $(CUDA_LIBS)

# =============================================================================
# SOURCE FILES
# =============================================================================

# Common sources (utilities, data loading, etc.)
COMMON_SRCS = src/common/kmeans_common.cpp

# CUDA sources
CUDA_SRCS = src/cuda/kmeans_cuda.cu \
            src/cuda/main_cuda.cu

# Serial sources (if needed later)
# SERIAL_SRCS = src/baseline/kmeans_serial.cpp \
#               src/baseline/main_serial.cpp

# OpenMP sources (if needed later)
# OPENMP_SRCS = src/openmp/kmeans_openmp.cpp \
#               src/openmp/main_openmp.cpp

# =============================================================================
# OBJECT FILES
# =============================================================================

# Object directories
OBJ_DIR = build

# Object files
SERIAL_OBJS = $(SERIAL_SRCS:src/%.cpp=$(OBJ_DIR)/%.o)
OPENMP_OBJS = $(OPENMP_SRCS:src/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJS = $(CUDA_SRCS:src/%.cu=$(OBJ_DIR)/%.o)
COMMON_OBJS = $(COMMON_SRCS:src/%.cpp=$(OBJ_DIR)/%.o)

# =============================================================================
# TARGET EXECUTABLES
# =============================================================================

TARGETS = kmeans_cuda
# Add these later: kmeans_serial kmeans_openmp

# Default target
all: directories $(TARGETS)

# =============================================================================
# BUILD TARGETS
# =============================================================================

# Serial version (baseline)
serial: directories kmeans_serial

kmeans_serial: CXXFLAGS = $(CXXFLAGS_BASE) $(OPT_FLAGS) $(INCLUDES)
kmeans_serial: $(SERIAL_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS_SERIAL)
	@echo "Built serial baseline: $@"

# OpenMP version
openmp: directories kmeans_openmp

kmeans_openmp: CXXFLAGS = $(CXXFLAGS_BASE) $(OPT_FLAGS) $(OPENMP_FLAGS) $(INCLUDES)
kmeans_openmp: $(OPENMP_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS_OPENMP)
	@echo "Built OpenMP version: $@"

# CUDA version
cuda: directories kmeans_cuda

kmeans_cuda: $(CUDA_OBJS) $(COMMON_OBJS)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(LIBS_CUDA)
	@echo "Built CUDA version: $@"

# =============================================================================
# DEBUG TARGETS
# =============================================================================

debug: debug_serial debug_openmp debug_cuda

debug_serial: CXXFLAGS = $(CXXFLAGS_BASE) $(DEBUG_FLAGS) $(INCLUDES)
debug_serial: directories $(SERIAL_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(SERIAL_OBJS) $(COMMON_OBJS) -o kmeans_serial_debug $(LIBS_SERIAL)
	@echo "Built debug serial: kmeans_serial_debug"

debug_openmp: CXXFLAGS = $(CXXFLAGS_BASE) $(DEBUG_FLAGS) $(OPENMP_FLAGS) $(INCLUDES)
debug_openmp: directories $(OPENMP_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(OPENMP_OBJS) $(COMMON_OBJS) -o kmeans_openmp_debug $(LIBS_OPENMP)
	@echo "Built debug OpenMP: kmeans_openmp_debug"

debug_cuda: NVFLAGS = $(NVFLAGS_DEBUG)
debug_cuda: directories $(CUDA_OBJS) $(COMMON_OBJS)
	$(NVCC) $(NVFLAGS) $(CUDA_OBJS) $(COMMON_OBJS) -o kmeans_cuda_debug $(LIBS_CUDA)
	@echo "Built debug CUDA: kmeans_cuda_debug"

# =============================================================================
# PROFILING TARGETS
# =============================================================================

profile: profile_serial profile_openmp profile_cuda

profile_serial: CXXFLAGS = $(CXXFLAGS_BASE) $(OPT_FLAGS) -pg $(INCLUDES)
profile_serial: directories $(SERIAL_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(SERIAL_OBJS) $(COMMON_OBJS) -o kmeans_serial_prof $(LIBS_SERIAL)
	@echo "Built profiling serial: kmeans_serial_prof"

profile_openmp: CXXFLAGS = $(CXXFLAGS_BASE) $(OPT_FLAGS) $(OPENMP_FLAGS) -pg $(INCLUDES)
profile_openmp: directories $(OPENMP_OBJS) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(OPENMP_OBJS) $(COMMON_OBJS) -o kmeans_openmp_prof $(LIBS_OPENMP)
	@echo "Built profiling OpenMP: kmeans_openmp_prof"

profile_cuda: NVFLAGS = $(NVFLAGS) -lineinfo
profile_cuda: directories $(CUDA_OBJS) $(COMMON_OBJS)
	$(NVCC) $(NVFLAGS) $(CUDA_OBJS) $(COMMON_OBJS) -o kmeans_cuda_prof $(LIBS_CUDA)
	@echo "Built profiling CUDA: kmeans_cuda_prof"

# =============================================================================
# COMPILATION RULES
# =============================================================================

# Create directories
directories:
	@mkdir -p $(OBJ_DIR)/{baseline,openmp,cuda,common}

# Compile C++ files
$(OBJ_DIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA files
$(OBJ_DIR)/%.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $< -o $@

# =============================================================================
# TEST TARGETS
# =============================================================================

test: test_serial test_openmp test_cuda

test_serial: serial
	./kmeans_serial -N 10000 -D 16 -K 8 -I 5 -S 42
	@echo "Serial test passed"

test_openmp: openmp
	./kmeans_openmp -N 10000 -D 16 -K 8 -I 5 -S 42 -T 4
	@echo "OpenMP test passed"

test_cuda: cuda
	./kmeans_cuda -N 10000 -D 16 -K 8 -I 5 -S 42
	@echo "CUDA test passed"

# Validation test - compare outputs
validate: all
	@echo "Running validation tests..."
	./kmeans_serial -N 10000 -D 16 -K 8 -I 10 -S 42 -o serial_output.txt
	./kmeans_openmp -N 10000 -D 16 -K 8 -I 10 -S 42 -T 4 -o openmp_output.txt
	./kmeans_cuda -N 10000 -D 16 -K 8 -I 10 -S 42 -o cuda_output.txt
	@echo "Comparing outputs..."
	diff serial_output.txt openmp_output.txt || echo "OpenMP differs from serial!"
	diff serial_output.txt cuda_output.txt || echo "CUDA differs from serial!"
	@echo "Validation complete"

# =============================================================================
# BENCHMARKING TARGETS
# =============================================================================

bench_quick: all
	@echo "Quick benchmark: N=100000, D=64, K=32"
	@echo "Serial:"
	./kmeans_serial -N 100000 -D 64 -K 32 -I 10 -S 42
	@echo "OpenMP (4 threads):"
	OMP_NUM_THREADS=4 ./kmeans_openmp -N 100000 -D 64 -K 32 -I 10 -S 42
	@echo "CUDA:"
	./kmeans_cuda -N 100000 -D 64 -K 32 -I 10 -S 42

bench_full: all
	@echo "Running full benchmark suite..."
	bash ../m2-scripts/benchmark/run_full_benchmark.sh

# =============================================================================
# UTILITY TARGETS
# =============================================================================

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR)
	rm -f $(TARGETS)
	rm -f kmeans_*_debug kmeans_*_prof
	rm -f *.o *.out *.txt
	rm -f gmon.out

# Clean everything including results
distclean: clean
	rm -rf ../m2-bench/*
	rm -rf ../m2-plots/*

# Show configuration
info:
	@echo "=== Build Configuration ==="
	@echo "CXX: $(CXX)"
	@echo "NVCC: $(NVCC)"
	@echo "GPU_ARCH: $(GPU_ARCH)"
	@echo "CUDA_INC: $(CUDA_INC)"
	@echo "CUDA_LIB: $(CUDA_LIB)"
	@echo "Targets: $(TARGETS)"

# Help
help:
	@echo "COSC3500 M2 K-Means Makefile"
	@echo ""
	@echo "Main targets:"
	@echo "  all          - Build all versions (serial, openmp, cuda)"
	@echo "  serial       - Build serial baseline"
	@echo "  openmp       - Build OpenMP parallel version"
	@echo "  cuda         - Build CUDA GPU version"
	@echo ""
	@echo "Debug targets:"
	@echo "  debug        - Build all debug versions"
	@echo "  debug_serial - Build debug serial"
	@echo "  debug_openmp - Build debug OpenMP"
	@echo "  debug_cuda   - Build debug CUDA"
	@echo ""
	@echo "Profiling targets:"
	@echo "  profile      - Build all profiling versions"
	@echo ""
	@echo "Testing targets:"
	@echo "  test         - Run basic tests"
	@echo "  validate     - Compare outputs across implementations"
	@echo "  bench_quick  - Quick benchmark"
	@echo "  bench_full   - Full benchmark suite"
	@echo ""
	@echo "Other targets:"
	@echo "  clean        - Remove build artifacts"
	@echo "  distclean    - Remove all generated files"
	@echo "  info         - Show build configuration"
	@echo "  help         - Show this help"

.PHONY: all serial openmp cuda debug test validate bench_quick bench_full clean distclean info help directories
.PHONY: debug_serial debug_openmp debug_cuda profile profile_serial profile_openmp profile_cuda test_serial test_openmp test_cuda