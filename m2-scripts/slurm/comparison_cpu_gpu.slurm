#!/bin/bash -l
#
#SBATCH --job-name=KMeans_CPU_GPU_Comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres shard:1
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500
#SBATCH --time=0-00:15:00
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err

# Module loads for both CUDA and OpenMP
module load compiler-rt/latest
module load cuda/11.1

# Add CUDA binaries to PATH
export PATH=/opt/local/stow/cuda-11.1/bin:$PATH

# Print system information
echo "=== System Information ==="
hostname
echo ""
echo "CPU Information:"
lscpu | grep -E "Model name|Socket|Core|Thread"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

echo "=== Git Information ==="
git rev-parse HEAD
git status --short

# Navigate to M2 directory
cd m2 || exit 1

echo ""
echo "=== Building All Implementations ==="
make clean
make all

echo ""
echo "=== Running CPU vs GPU Comparison ==="

# Create output directory for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../m2-bench/comparison/run_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

# Save system information
echo "System Configuration" > "${OUTPUT_DIR}/system_info.txt"
echo "Date: $(date)" >> "${OUTPUT_DIR}/system_info.txt"
echo "Node: $(hostname)" >> "${OUTPUT_DIR}/system_info.txt"
lscpu >> "${OUTPUT_DIR}/system_info.txt"
echo "" >> "${OUTPUT_DIR}/system_info.txt"
nvidia-smi >> "${OUTPUT_DIR}/system_info.txt"

# Test configurations - increasing problem sizes
declare -a CONFIGS=(
    "10000 16 8"      # Small
    "50000 32 16"     # Small-Medium
    "100000 64 32"    # Medium
    "200000 16 8"     # Canonical
    "100000 64 64"    # Stress
    "500000 32 32"    # Large-Medium
    "1000000 64 64"   # Large
    "2000000 128 128" # Very Large
)

echo "Configuration,Implementation,Threads,Total_Time(ms),Assign_Time(ms),Update_Time(ms),MLUPS,Speedup" > "${OUTPUT_DIR}/comparison_results.csv"

for config in "${CONFIGS[@]}"; do
    read -r N D K <<< "$config"

    echo ""
    echo "=== Testing Configuration: N=$N, D=$D, K=$K ==="

    # Run serial baseline (reference)
    echo "Running serial baseline..."
    SERIAL_OUTPUT=$(./kmeans_serial -N $N -D $D -K $K -I 10 -S 42 --warmup 2 --bench 3 2>&1)
    SERIAL_TIME=$(echo "$SERIAL_OUTPUT" | grep "Total time" | awk '{print $3}')
    SERIAL_ASSIGN=$(echo "$SERIAL_OUTPUT" | grep "Assign time" | awk '{print $3}')
    SERIAL_UPDATE=$(echo "$SERIAL_OUTPUT" | grep "Update time" | awk '{print $3}')
    SERIAL_MLUPS=$(echo "$SERIAL_OUTPUT" | grep "MLUPS" | awk '{print $2}')

    echo "$N,$D,$K,Serial,1,$SERIAL_TIME,$SERIAL_ASSIGN,$SERIAL_UPDATE,$SERIAL_MLUPS,1.0" >> "${OUTPUT_DIR}/comparison_results.csv"

    # Run OpenMP with different thread counts
    for THREADS in 2 4 8 16; do
        echo "Running OpenMP with $THREADS threads..."
        export OMP_NUM_THREADS=$THREADS

        OMP_OUTPUT=$(./kmeans_openmp -N $N -D $D -K $K -I 10 -S 42 -T $THREADS --warmup 2 --bench 3 2>&1)
        OMP_TIME=$(echo "$OMP_OUTPUT" | grep "Total time" | awk '{print $3}')
        OMP_ASSIGN=$(echo "$OMP_OUTPUT" | grep "Assign time" | awk '{print $3}')
        OMP_UPDATE=$(echo "$OMP_OUTPUT" | grep "Update time" | awk '{print $3}')
        OMP_MLUPS=$(echo "$OMP_OUTPUT" | grep "MLUPS" | awk '{print $2}')
        OMP_SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $OMP_TIME" | bc)

        echo "$N,$D,$K,OpenMP,$THREADS,$OMP_TIME,$OMP_ASSIGN,$OMP_UPDATE,$OMP_MLUPS,$OMP_SPEEDUP" >> "${OUTPUT_DIR}/comparison_results.csv"
    done

    # Run CUDA
    echo "Running CUDA..."
    CUDA_OUTPUT=$(./kmeans_cuda -N $N -D $D -K $K -I 10 -S 42 --warmup 2 --bench 3 2>&1)
    CUDA_TIME=$(echo "$CUDA_OUTPUT" | grep "Total time" | awk '{print $3}')
    CUDA_ASSIGN=$(echo "$CUDA_OUTPUT" | grep "Assign time" | awk '{print $3}')
    CUDA_UPDATE=$(echo "$CUDA_OUTPUT" | grep "Update time" | awk '{print $3}')
    CUDA_MLUPS=$(echo "$CUDA_OUTPUT" | grep "MLUPS" | awk '{print $2}')
    CUDA_SPEEDUP=$(echo "scale=2; $SERIAL_TIME / $CUDA_TIME" | bc)

    echo "$N,$D,$K,CUDA,GPU,$CUDA_TIME,$CUDA_ASSIGN,$CUDA_UPDATE,$CUDA_MLUPS,$CUDA_SPEEDUP" >> "${OUTPUT_DIR}/comparison_results.csv"

    # Print summary for this configuration
    echo ""
    echo "Summary for N=$N, D=$D, K=$K:"
    echo "  Serial: ${SERIAL_TIME}ms"
    echo "  OpenMP (16 threads): ${OMP_TIME}ms (Speedup: ${OMP_SPEEDUP}x)"
    echo "  CUDA: ${CUDA_TIME}ms (Speedup: ${CUDA_SPEEDUP}x)"
done

echo ""
echo "=== Creating Analysis Plots ==="
# Assuming we have a Python analysis script
if [ -f "../m2-scripts/analysis/plot_comparison.py" ]; then
    python ../m2-scripts/analysis/plot_comparison.py "${OUTPUT_DIR}/comparison_results.csv" "${OUTPUT_DIR}"
else
    echo "Analysis script not found, skipping plots"
fi

echo ""
echo "=== Comparison Complete ==="
echo "Results saved to: ${OUTPUT_DIR}"
echo "CSV file: ${OUTPUT_DIR}/comparison_results.csv"
echo "Job completed at: $(date)"