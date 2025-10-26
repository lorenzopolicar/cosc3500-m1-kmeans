#!/bin/bash

# Full Benchmark Suite for M2 K-Means Implementations
# Runs serial, OpenMP, and CUDA implementations across multiple configurations

# Set script to exit on error
set -e

# Base directory for results
BASE_DIR="../m2-bench"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${BASE_DIR}/full_benchmark_${TIMESTAMP}"

# Create directory structure
mkdir -p "${RUN_DIR}"/{serial,openmp,cuda,comparison,plots}

echo "============================================"
echo "     COSC3500 M2 - Full Benchmark Suite    "
echo "============================================"
echo "Run ID: ${TIMESTAMP}"
echo "Output directory: ${RUN_DIR}"
echo ""

# Save system information
echo "=== Capturing System Information ===" | tee "${RUN_DIR}/system_info.txt"
{
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo ""
    echo "CPU Information:"
    lscpu | grep -E "Model name|Socket|Core|Thread|CPU MHz"
    echo ""
    echo "Memory Information:"
    free -h
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "No GPU detected"
    echo ""
    echo "Compiler Versions:"
    g++ --version | head -n 1
    nvcc --version 2>/dev/null | grep "release" || echo "NVCC not available"
} >> "${RUN_DIR}/system_info.txt"

# Define test configurations
# Format: "name N D K iterations"
CONFIGS=(
    "tiny 1000 8 4 20"
    "small 10000 16 8 20"
    "canonical 200000 16 8 15"
    "stress 100000 64 64 15"
    "medium 500000 32 32 10"
    "large 1000000 64 64 10"
)

# Build all implementations
echo ""
echo "=== Building All Implementations ===" | tee -a "${RUN_DIR}/build.log"
cd ../m2 || exit 1
make clean >> "${RUN_DIR}/build.log" 2>&1
make all >> "${RUN_DIR}/build.log" 2>&1
echo "Build complete"

# Function to run a single benchmark
run_benchmark() {
    local impl=$1
    local name=$2
    local N=$3
    local D=$4
    local K=$5
    local iters=$6
    local threads=${7:-""}

    local output_file="${RUN_DIR}/${impl}/${name}_N${N}_D${D}_K${K}.csv"
    local log_file="${RUN_DIR}/${impl}/${name}_N${N}_D${D}_K${K}.log"

    echo "  Running ${impl}${threads:+ with $threads threads}: $name (N=$N, D=$D, K=$K)"

    case $impl in
        serial)
            ./kmeans_serial -N $N -D $D -K $K -I $iters -S 42 \
                --warmup 3 --bench 5 --verbose \
                -o "$output_file" > "$log_file" 2>&1
            ;;
        openmp)
            OMP_NUM_THREADS=$threads ./kmeans_openmp -N $N -D $D -K $K -I $iters -S 42 \
                -T $threads --warmup 3 --bench 5 --verbose \
                -o "$output_file" > "$log_file" 2>&1
            ;;
        cuda)
            ./kmeans_cuda -N $N -D $D -K $K -I $iters -S 42 \
                --warmup 3 --bench 5 --verbose \
                -o "$output_file" > "$log_file" 2>&1
            ;;
    esac

    # Extract summary metrics
    if [ -f "$log_file" ]; then
        grep -E "(Total time|MLUPS|Final inertia)" "$log_file" | tail -3
    fi
}

# Run Serial Baseline
echo ""
echo "=== Running Serial Baseline ===" | tee -a "${RUN_DIR}/progress.log"
for config in "${CONFIGS[@]}"; do
    read -r name N D K iters <<< "$config"
    run_benchmark serial "$name" $N $D $K $iters
done

# Run OpenMP with different thread counts
echo ""
echo "=== Running OpenMP Benchmarks ===" | tee -a "${RUN_DIR}/progress.log"
THREAD_COUNTS="1 2 4 8 16 32"
for config in "${CONFIGS[@]}"; do
    read -r name N D K iters <<< "$config"
    for threads in $THREAD_COUNTS; do
        # Skip large configs with low thread counts to save time
        if [[ "$name" == "large" && $threads -lt 8 ]]; then
            continue
        fi
        run_benchmark openmp "${name}_T${threads}" $N $D $K $iters $threads
    done
done

# Run CUDA
echo ""
echo "=== Running CUDA Benchmarks ===" | tee -a "${RUN_DIR}/progress.log"
for config in "${CONFIGS[@]}"; do
    read -r name N D K iters <<< "$config"
    run_benchmark cuda "$name" $N $D $K $iters
done

# Generate comparison CSV
echo ""
echo "=== Generating Comparison Data ===" | tee -a "${RUN_DIR}/progress.log"
COMPARISON_FILE="${RUN_DIR}/comparison/all_results.csv"
echo "Implementation,Config,N,D,K,Threads,Total_ms,Assign_ms,Update_ms,MLUPS" > "$COMPARISON_FILE"

# Parse all log files and extract metrics
for log_file in "${RUN_DIR}"/*/*.log; do
    if [ -f "$log_file" ]; then
        impl=$(basename $(dirname "$log_file"))
        config=$(basename "$log_file" .log)

        # Extract values
        total_ms=$(grep "Total time" "$log_file" 2>/dev/null | tail -1 | awk '{print $3}')
        assign_ms=$(grep "Assign time" "$log_file" 2>/dev/null | tail -1 | awk '{print $3}')
        update_ms=$(grep "Update time" "$log_file" 2>/dev/null | tail -1 | awk '{print $3}')
        mlups=$(grep "MLUPS" "$log_file" 2>/dev/null | tail -1 | awk '{print $2}')

        # Parse config name
        IFS='_' read -r cfg_name rest <<< "$config"
        threads="N/A"
        if [[ "$impl" == "openmp" ]]; then
            threads=$(echo "$config" | grep -oP 'T\K[0-9]+' || echo "1")
        fi

        # Extract N, D, K from filename
        N=$(echo "$config" | grep -oP 'N\K[0-9]+')
        D=$(echo "$config" | grep -oP 'D\K[0-9]+')
        K=$(echo "$config" | grep -oP 'K\K[0-9]+')

        if [ ! -z "$total_ms" ]; then
            echo "$impl,$cfg_name,$N,$D,$K,$threads,$total_ms,$assign_ms,$update_ms,$mlups" >> "$COMPARISON_FILE"
        fi
    fi
done

# Calculate speedups
echo ""
echo "=== Calculating Speedups ===" | tee -a "${RUN_DIR}/progress.log"
python3 - << 'EOF' 2>/dev/null || echo "Python analysis skipped"
import csv
import sys
import os

run_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('RUN_DIR', '.')
comparison_file = f"{run_dir}/comparison/all_results.csv"
speedup_file = f"{run_dir}/comparison/speedups.csv"

try:
    # Read results
    results = {}
    with open(comparison_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['Config']}_{row['N']}_{row['D']}_{row['K']}"
            impl_key = f"{row['Implementation']}_{row['Threads']}"
            if key not in results:
                results[key] = {}
            results[key][impl_key] = float(row['Total_ms']) if row['Total_ms'] else None

    # Calculate speedups
    with open(speedup_file, 'w') as f:
        f.write("Configuration,N,D,K,OpenMP_Best,OpenMP_Speedup,CUDA_Speedup\n")
        for key in results:
            if 'serial_N/A' in results[key] and results[key]['serial_N/A']:
                serial_time = results[key]['serial_N/A']

                # Find best OpenMP
                best_omp_speedup = 0
                best_omp_threads = 0
                for impl_key in results[key]:
                    if impl_key.startswith('openmp_') and results[key][impl_key]:
                        speedup = serial_time / results[key][impl_key]
                        if speedup > best_omp_speedup:
                            best_omp_speedup = speedup
                            best_omp_threads = impl_key.split('_')[1]

                # Get CUDA speedup
                cuda_speedup = 0
                if 'cuda_N/A' in results[key] and results[key]['cuda_N/A']:
                    cuda_speedup = serial_time / results[key]['cuda_N/A']

                config_parts = key.split('_')
                f.write(f"{config_parts[0]},{config_parts[1]},{config_parts[2]},{config_parts[3]},{best_omp_threads},{best_omp_speedup:.2f},{cuda_speedup:.2f}\n")

    print("Speedup analysis saved to:", speedup_file)
except Exception as e:
    print(f"Error in speedup calculation: {e}")
EOF

# Generate summary report
echo ""
echo "=== Generating Summary Report ===" | tee "${RUN_DIR}/SUMMARY.md"
{
    echo "# M2 K-Means Benchmark Summary"
    echo "## Run Information"
    echo "- Date: $(date)"
    echo "- Run ID: ${TIMESTAMP}"
    echo ""
    echo "## System Configuration"
    head -n 20 "${RUN_DIR}/system_info.txt" | sed 's/^/    /'
    echo ""
    echo "## Configurations Tested"
    for config in "${CONFIGS[@]}"; do
        echo "- $config"
    done
    echo ""
    echo "## Key Results"

    if [ -f "${RUN_DIR}/comparison/speedups.csv" ]; then
        echo "### Speedup Summary"
        echo '```'
        cat "${RUN_DIR}/comparison/speedups.csv"
        echo '```'
    fi

    echo ""
    echo "## File Locations"
    echo "- Serial results: ${RUN_DIR}/serial/"
    echo "- OpenMP results: ${RUN_DIR}/openmp/"
    echo "- CUDA results: ${RUN_DIR}/cuda/"
    echo "- Comparison data: ${RUN_DIR}/comparison/"
} >> "${RUN_DIR}/SUMMARY.md"

echo ""
echo "============================================"
echo "        Benchmark Suite Complete!          "
echo "============================================"
echo "Results directory: ${RUN_DIR}"
echo "Summary report: ${RUN_DIR}/SUMMARY.md"
echo "Comparison data: ${RUN_DIR}/comparison/all_results.csv"

# Make scripts executable
chmod +x "${RUN_DIR}"/../benchmark/*.sh 2>/dev/null || true