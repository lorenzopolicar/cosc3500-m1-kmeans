#!/bin/bash

# First test script for CUDA K-Means on cluster
# Run this after loading CUDA module

echo "=== M2 CUDA K-Means First Test ==="
echo "Date: $(date)"
echo "Directory: $(pwd)"
echo ""

# Check CUDA availability
echo "Checking CUDA environment..."
which nvcc > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: nvcc not found. Please load CUDA module first:"
    echo "  module load cuda/11.1"
    exit 1
fi

echo "CUDA compiler found: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"
echo ""

# Clean and build
echo "Building CUDA implementation..."
make clean
make cuda

if [ $? -ne 0 ]; then
    echo "Build failed! Check compilation errors above."
    exit 1
fi

echo ""
echo "Build successful! Running test cases..."
echo ""

# Test 1: Tiny test for validation
echo "=== Test 1: Tiny (N=1000, D=8, K=4) ==="
./kmeans_cuda -N 1000 -D 8 -K 4 -I 10 -S 42 --verbose --warmup 1 --bench 1

echo ""
echo "=== Test 2: Small (N=10000, D=16, K=8) ==="
./kmeans_cuda -N 10000 -D 16 -K 8 -I 10 -S 42 --warmup 2 --bench 3

echo ""
echo "=== Test 3: Canonical from M1 (N=200000, D=16, K=8) ==="
./kmeans_cuda -N 200000 -D 16 -K 8 -I 15 -S 42 --warmup 3 --bench 5

echo ""
echo "=== Test 4: Stress from M1 (N=100000, D=64, K=64) ==="
./kmeans_cuda -N 100000 -D 64 -K 64 -I 15 -S 42 --warmup 3 --bench 5

echo ""
echo "=== Test Complete ==="
echo "If all tests passed, you're ready to run full benchmarks!"
echo "Next: sbatch ../m2-scripts/slurm/cuda_baseline.slurm"