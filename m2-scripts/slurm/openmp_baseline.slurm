#!/bin/bash -l
#
#SBATCH --job-name=KMeans_OpenMP_Baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500
#SBATCH --time=0-00:15:00
#SBATCH --output=openmp_baseline_%j.out
#SBATCH --error=openmp_baseline_%j.err

# Module loads for OpenMP environment
module load compiler-rt/latest

# Print system information
echo "=== System Information ==="
hostname
lscpu | grep -E "Model name|Socket|Core|Thread"
echo "OpenMP Max Threads: $OMP_NUM_THREADS"

echo ""
echo "=== Git Information ==="
git rev-parse HEAD
git status --short

# Navigate to M2 directory
cd m2 || exit 1

echo ""
echo "=== Building OpenMP Implementation ==="
make clean
make openmp

echo ""
echo "=== Running OpenMP Baseline Benchmarks ==="

# Create output directory for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../m2-bench/openmp/baseline_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Save build information
echo "OpenMP Baseline Build Information" > "${OUTPUT_DIR}/build_info.txt"
echo "Date: $(date)" >> "${OUTPUT_DIR}/build_info.txt"
echo "Compiler: $(g++ --version | head -n 1)" >> "${OUTPUT_DIR}/build_info.txt"
echo "Git SHA: $(git rev-parse HEAD)" >> "${OUTPUT_DIR}/build_info.txt"
echo "SLURM Job ID: ${SLURM_JOB_ID}" >> "${OUTPUT_DIR}/build_info.txt"

# Thread configurations to test
THREAD_COUNTS="1 2 4 8 16 32"

# Test different thread counts for scaling analysis
for THREADS in ${THREAD_COUNTS}; do
    echo ""
    echo "=== Testing with ${THREADS} threads ==="
    export OMP_NUM_THREADS=${THREADS}

    # Test 1: Canonical Configuration
    echo "Test 1: Canonical (N=200000, D=16, K=8)"
    ./kmeans_openmp -N 200000 -D 16 -K 8 -I 20 -S 42 -T ${THREADS} \
        -o "${OUTPUT_DIR}/canonical_N200000_D16_K8_T${THREADS}.csv" \
        --warmup 3 --bench 5 --verbose

    # Test 2: Stress Configuration
    echo "Test 2: Stress (N=100000, D=64, K=64)"
    ./kmeans_openmp -N 100000 -D 64 -K 64 -I 20 -S 42 -T ${THREADS} \
        -o "${OUTPUT_DIR}/stress_N100000_D64_K64_T${THREADS}.csv" \
        --warmup 3 --bench 5 --verbose

    # Test 3: Medium Scale
    echo "Test 3: Medium (N=500000, D=32, K=32)"
    ./kmeans_openmp -N 500000 -D 32 -K 32 -I 15 -S 42 -T ${THREADS} \
        -o "${OUTPUT_DIR}/medium_N500000_D32_K32_T${THREADS}.csv" \
        --warmup 3 --bench 5 --verbose
done

# Large scale test with optimal thread count
echo ""
echo "=== Large Scale Test with 32 threads ==="
export OMP_NUM_THREADS=32

# Test 4: Large Scale
echo "Test 4: Large Scale (N=1000000, D=64, K=64)"
./kmeans_openmp -N 1000000 -D 64 -K 64 -I 10 -S 42 -T 32 \
    -o "${OUTPUT_DIR}/large_N1000000_D64_K64_T32.csv" \
    --warmup 3 --bench 5 --verbose

# Test 5: Very Large Scale
echo "Test 5: Very Large Scale (N=5000000, D=128, K=128)"
./kmeans_openmp -N 5000000 -D 128 -K 128 -I 10 -S 42 -T 32 \
    -o "${OUTPUT_DIR}/xlarge_N5000000_D128_K128_T32.csv" \
    --warmup 2 --bench 3 --verbose

echo ""
echo "=== OpenMP Scaling Analysis ==="
echo "Thread counts tested: ${THREAD_COUNTS}"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Job completed at: $(date)"