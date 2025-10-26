#!/bin/bash -l
#
#SBATCH --job-name=KMeans_CUDA_Test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=0-00:10:00
#SBATCH --output=cuda_test_%j.out
#SBATCH --error=cuda_test_%j.err

# Quick test job for CUDA K-Means

# Load CUDA module
module load cuda/11.1

# Print system info
echo "=== System Information ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Navigate to M2 directory
cd ../../m2 || exit 1

# Build
echo "=== Building CUDA Implementation ==="
make clean
make cuda

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Run quick test
echo ""
echo "=== Running Quick Test ==="
./kmeans_cuda -N 100000 -D 64 -K 32 -I 10 -S 42 --verbose --warmup 3 --bench 5

echo ""
echo "=== Test Complete ==="
echo "Check output above for performance metrics"