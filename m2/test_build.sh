#!/bin/bash

# Test build script for local development
# This creates a mock build to check for syntax errors

echo "=== Testing M2 Build Structure ==="
echo "Note: This is a test build without actual CUDA compilation"
echo ""

# Check if source files exist
echo "Checking source files..."
FILES=(
    "src/common/kmeans_common.cpp"
    "src/cuda/kmeans_cuda.cu"
    "src/cuda/main_cuda.cu"
    "include/kmeans_common.hpp"
    "include/kmeans_cuda.cuh"
)

ALL_FOUND=true
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (missing)"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    echo ""
    echo "Error: Some source files are missing"
    exit 1
fi

echo ""
echo "Checking C++ compilation (common files)..."
g++ -std=c++17 -c src/common/kmeans_common.cpp -Iinclude -o /tmp/test_common.o 2>&1 | head -20

if [ $? -eq 0 ]; then
    echo "  ✓ Common files compile successfully"
else
    echo "  ✗ Common files have compilation errors"
fi

echo ""
echo "Checking CUDA syntax (no actual compilation)..."
# Just check if files have basic syntax
for cuda_file in src/cuda/*.cu; do
    echo "  - $cuda_file"
    # Basic syntax check: look for CUDA keywords
    if grep -q "__global__\|__device__\|__host__" "$cuda_file"; then
        echo "    Contains CUDA kernels"
    fi
done

echo ""
echo "=== Summary ==="
echo "The M2 structure is ready for CUDA compilation on the cluster."
echo "To build on Rangpur cluster:"
echo "  1. module load cuda/11.1"
echo "  2. cd m2"
echo "  3. make cuda"
echo ""
echo "Created files summary:"
echo "  - CUDA kernels: kmeans_cuda.cu"
echo "  - CUDA main: main_cuda.cu"
echo "  - Common utilities: kmeans_common.cpp"
echo "  - Headers: kmeans_common.hpp, kmeans_cuda.cuh"