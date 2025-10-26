#!/bin/bash

# CUDA K-Means Benchmark Script
# Usage: ./run_cuda_bench.sh N D K iterations [output_dir]

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 N D K iterations [output_dir]"
    echo "Example: $0 100000 64 32 10 ../m2-bench/cuda"
    exit 1
fi

N=$1
D=$2
K=$3
ITERS=$4
OUTPUT_DIR=${5:-"../m2-bench/cuda"}

# Configuration
WARMUP_ITERS=3
BENCH_ITERS=5
SEED=42

# Create output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/N${N}_D${D}_K${K}_${TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "=== CUDA K-Means Benchmark ==="
echo "Configuration: N=$N, D=$D, K=$K"
echo "Iterations: $ITERS"
echo "Warmup: $WARMUP_ITERS, Benchmark: $BENCH_ITERS"
echo "Output directory: $RUN_DIR"
echo ""

# Build if needed
cd ../../m2 || exit 1
if [ ! -f kmeans_cuda ]; then
    echo "Building CUDA implementation..."
    make cuda || exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be available."
fi

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv 2>/dev/null || echo "Unable to query GPU"
echo ""

# Run benchmark
echo "=== Running Benchmark ==="
OUTPUT_FILE="${RUN_DIR}/results.csv"
LOG_FILE="${RUN_DIR}/output.log"

# Run with timing
{ time ./kmeans_cuda \
    -N $N -D $D -K $K -I $ITERS -S $SEED \
    --warmup $WARMUP_ITERS --bench $BENCH_ITERS \
    --verbose \
    -o "$OUTPUT_FILE"; } 2>&1 | tee "$LOG_FILE"

# Extract key metrics from log
echo ""
echo "=== Extracting Metrics ==="
grep -E "(Total time|Assign time|Update time|MLUPS|Inertia)" "$LOG_FILE" > "${RUN_DIR}/summary.txt"

# Save configuration
cat > "${RUN_DIR}/config.json" << EOF
{
    "N": $N,
    "D": $D,
    "K": $K,
    "iterations": $ITERS,
    "warmup_iterations": $WARMUP_ITERS,
    "benchmark_iterations": $BENCH_ITERS,
    "seed": $SEED,
    "timestamp": "$TIMESTAMP",
    "implementation": "CUDA"
}
EOF

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $RUN_DIR"
echo "  - results.csv: Detailed timing data"
echo "  - output.log: Full program output"
echo "  - summary.txt: Key metrics"
echo "  - config.json: Configuration"

# Parse and display summary
if [ -f "${RUN_DIR}/summary.txt" ]; then
    echo ""
    echo "=== Performance Summary ==="
    cat "${RUN_DIR}/summary.txt"
fi