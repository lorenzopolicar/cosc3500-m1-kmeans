#!/bin/bash -l
#
#SBATCH --job-name=KMeans_CUDA_Baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres shard:1
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500
#SBATCH --time=0-00:30:00
#SBATCH --output=cuda_baseline_%j.out
#SBATCH --error=cuda_baseline_%j.err

# Module loads for CUDA environment
module load compiler-rt/latest
module load cuda/11.1

# Add CUDA binaries to PATH
export PATH=/opt/local/stow/cuda-11.1/bin:$PATH

# Print system information
echo "=== System Information ==="
hostname
nvidia-smi
nvcc --version

echo ""
echo "=== Git Information ==="
git rev-parse HEAD
git status --short

# Navigate to M2 directory
cd m2 || exit 1

echo ""
echo "=== Building CUDA Implementation ==="
make clean
make cuda

echo ""
echo "=== Running CUDA Baseline Benchmarks ==="

# Create output directory for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../m2-bench/cuda/baseline_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Save build information
echo "CUDA Baseline Build Information" > "${OUTPUT_DIR}/build_info.txt"
echo "Date: $(date)" >> "${OUTPUT_DIR}/build_info.txt"
echo "Compiler: $(nvcc --version | head -n 1)" >> "${OUTPUT_DIR}/build_info.txt"
echo "Git SHA: $(git rev-parse HEAD)" >> "${OUTPUT_DIR}/build_info.txt"
echo "SLURM Job ID: ${SLURM_JOB_ID}" >> "${OUTPUT_DIR}/build_info.txt"

# Test 1: Canonical Configuration (Small Working Set)
echo ""
echo "=== Test 1: Canonical Configuration (N=200000, D=16, K=8) ==="
./kmeans_cuda -N 200000 -D 16 -K 8 -I 20 -S 42 \
    -o "${OUTPUT_DIR}/canonical_N200000_D16_K8.csv" \
    --warmup 3 --bench 5 --verbose

# Test 2: Stress Configuration (Large Working Set)
echo ""
echo "=== Test 2: Stress Configuration (N=100000, D=64, K=64) ==="
./kmeans_cuda -N 100000 -D 64 -K 64 -I 20 -S 42 \
    -o "${OUTPUT_DIR}/stress_N100000_D64_K64.csv" \
    --warmup 3 --bench 5 --verbose

# Test 3: Medium Scale
echo ""
echo "=== Test 3: Medium Scale (N=500000, D=32, K=32) ==="
./kmeans_cuda -N 500000 -D 32 -K 32 -I 15 -S 42 \
    -o "${OUTPUT_DIR}/medium_N500000_D32_K32.csv" \
    --warmup 3 --bench 5 --verbose

# Test 4: Large Scale
echo ""
echo "=== Test 4: Large Scale (N=1000000, D=64, K=64) ==="
./kmeans_cuda -N 1000000 -D 64 -K 64 -I 10 -S 42 \
    -o "${OUTPUT_DIR}/large_N1000000_D64_K64.csv" \
    --warmup 3 --bench 5 --verbose

# Test 5: Very Large Scale (if memory permits)
echo ""
echo "=== Test 5: Very Large Scale (N=5000000, D=128, K=128) ==="
./kmeans_cuda -N 5000000 -D 128 -K 128 -I 10 -S 42 \
    -o "${OUTPUT_DIR}/xlarge_N5000000_D128_K128.csv" \
    --warmup 2 --bench 3 --verbose || echo "Very large scale test failed (likely out of memory)"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: ${OUTPUT_DIR}"
echo "Job completed at: $(date)"