#!/usr/bin/env bash
set -euo pipefail

# Check if experiment number is provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <experiment_number> [build_defs]"
    echo "Example: $0 0                    # for E0 baseline"
    echo "Example: $0 1 '-DTRANSPOSED_C=1' # for E1 with transposed centroids"
    echo "Example: $0 2 '-DUNROLL=4'       # for E2 with loop unrolling"
    exit 1
fi

EXPERIMENT_NUM=$1
BUILD_DEFS="${2:-}"  # Optional build definitions
EXPERIMENT_DIR="e${EXPERIMENT_NUM}"

# Detect OS for profiling guidance
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="Linux"
    PROFILING_TOOLS="gprof, perf, valgrind/cachegrind"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macOS"
    PROFILING_TOOLS="sample, instruments"
else
    OS_TYPE="Unknown"
    PROFILING_TOOLS="gprof (if available)"
fi

echo "=== COSC3500 M1 K-Means Experiment E${EXPERIMENT_NUM} ==="
echo "Running experiments and profiling on ${OS_TYPE}..."
echo "Available profiling tools: ${PROFILING_TOOLS}"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "Makefile" ]]; then
    print_error "Must run from project root directory"
    exit 1
fi

# Build the project for TIMINGS (authoritative)
print_status "Building project for TIMINGS..."
make clean
if [[ -n "$BUILD_DEFS" ]]; then
    print_status "Building with definitions: $BUILD_DEFS"
    ANTIVEC=1 DEFS="$BUILD_DEFS" make release
else
    ANTIVEC=1 make release
fi
print_success "Build complete"

# Create bench directory structure
print_status "Creating benchmark directory structure..."
mkdir -p bench/${EXPERIMENT_DIR}

# Capture system information
print_status "Capturing system information..."
./scripts/print_sysinfo.sh | tee bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt

# Capture Slurm node information for HPC clusters
if [[ -n "${SLURM_NODELIST:-}" ]]; then
    print_status "Capturing Slurm node information..."
    echo "Slurm Node Information:" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_NODELIST: ${SLURM_NODELIST:-}" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_NODEID: ${SLURM_NODEID:-}" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_PARTITION: ${SLURM_PARTITION:-}" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_CPUS_ON_NODE: ${SLURM_CPUS_ON_NODE:-}" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_MEM_PER_NODE: ${SLURM_MEM_PER_NODE:-}" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    print_success "Slurm node info captured"
fi

print_success "System info captured"

# Canonical config: N=200000, D=16, K=8, iters=10, seed=1
print_status "Running canonical config: N=200000, D=16, K=8, iters=10, seed=1"
print_status "Running 3 warm-up + 5 measurement runs for statistical significance..."
for run in {1..8}; do
    if [[ $run -le 3 ]]; then
        print_status "Warm-up run $run/3"
        EXPERIMENT=${EXPERIMENT_DIR} RUN_NUM=$run ANTIVEC=1 ./build/kmeans --n 200000 --d 16 --k 8 --iters 10 --seed 1 > /dev/null 2>&1
    else
        print_status "Measurement run $((run-3))/5"
        EXPERIMENT=${EXPERIMENT_DIR} RUN_NUM=$run ANTIVEC=1 ./build/kmeans --n 200000 --d 16 --k 8 --iters 10 --seed 1
    fi
done
print_success "Canonical config complete (5 measurement runs)"

# Stress config: D=64, K=64 (larger dimensions and clusters)
print_status "Running stress config: N=100000, D=64, K=64, iters=10, seed=1"
print_status "Running 3 warm-up + 5 measurement runs for statistical significance..."
for run in {1..8}; do
    if [[ $run -le 3 ]]; then
        print_status "Warm-up run $run/3"
        EXPERIMENT=${EXPERIMENT_DIR} RUN_NUM=$run ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 10 --seed 1 > /dev/null 2>&1
    else
        print_status "Measurement run $((run-3))/5"
        EXPERIMENT=${EXPERIMENT_DIR} RUN_NUM=$run ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 10 --seed 1
    fi
done
print_success "Stress config complete (5 measurement runs)"

# Profile the stress config
print_status "Running profiling on stress config..."

# Function-level profiling with gprof (Linux/macOS)
if command -v gprof &> /dev/null; then
    print_status "Using gprof for function-level profiling..."
    make clean
    if [[ -n "$BUILD_DEFS" ]]; then
        ANTIVEC=1 DEFS="$BUILD_DEFS" make profile
    else
        ANTIVEC=1 make profile
    fi
    EXPERIMENT=${EXPERIMENT_DIR} ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > /dev/null 2>&1
    if [[ -f "gmon.out" ]]; then
        gprof ./build/kmeans gmon.out > bench/${EXPERIMENT_DIR}/gprof_e${EXPERIMENT_NUM}.txt
        print_success "gprof profile saved to bench/${EXPERIMENT_DIR}/gprof_e${EXPERIMENT_NUM}.txt"
    else
        print_warning "gprof output not generated - program may not have exited cleanly"
    fi
# Use macOS sample if gprof not available
elif command -v sample &> /dev/null; then
    print_status "Using macOS sample for function-level profiling..."
    make clean
    if [[ -n "$BUILD_DEFS" ]]; then
        ANTIVEC=1 DEFS="$BUILD_DEFS" make release
    else
        ANTIVEC=1 make release
    fi
    # Use a larger dataset that runs longer for sampling
    ./build/kmeans --n 500000 --d 32 --k 32 --iters 20 --seed 1 > /dev/null 2>&1 &
    KMEANS_PID=$!
    sleep 2
    sample $KMEANS_PID 5 > bench/${EXPERIMENT_DIR}/sample_stress_config.txt 2>&1
    wait $KMEANS_PID 2>/dev/null || true
    print_success "sample profile saved to bench/${EXPERIMENT_DIR}/sample_stress_config.txt"
else
    print_warning "No function-level profiling tools available (gprof/sample)"
fi

# Cache performance analysis with perf (Linux only)
if command -v perf &> /dev/null; then
    print_status "Using perf for hardware cache analysis..."
    make clean
    if [[ -n "$BUILD_DEFS" ]]; then
        ANTIVEC=1 DEFS="$BUILD_DEFS" make release
    else
        ANTIVEC=1 make release
    fi
    
    # Record cache performance with perf
    print_status "Recording hardware cache performance with perf..."
    perf record -e L1-dcache-load-misses,L1-dcache-loads,L2-dcache-load-misses,L2-dcache-loads \
                -o bench/${EXPERIMENT_DIR}/perf_cache.data \
                EXPERIMENT=${EXPERIMENT_DIR} ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > /dev/null 2>&1
    
    if [[ -f "bench/${EXPERIMENT_DIR}/perf_cache.data" ]]; then
        print_success "Found perf cache data: bench/${EXPERIMENT_DIR}/perf_cache.data"
        
        # Generate cache performance report
        perf report -i bench/${EXPERIMENT_DIR}/perf_cache.data > bench/${EXPERIMENT_DIR}/perf_cache_report.txt 2>&1
        print_success "perf cache report saved to bench/${EXPERIMENT_DIR}/perf_cache_report.txt"
        
        # Get detailed cache statistics
        print_status "Computing detailed cache statistics with perf stat..."
        perf stat -e L1-dcache-load-misses,L1-dcache-loads,L2-dcache-load-misses,L2-dcache-loads \
                  EXPERIMENT=${EXPERIMENT_DIR} ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > bench/${EXPERIMENT_DIR}/perf_cache_stats.txt 2>&1
        
        # Extract and normalize cache statistics
        print_status "Extracting normalized cache statistics..."
        N=100000
        D=64
        K=64
        ITERS=5
        TOTAL_OPERATIONS=$((N * K * ITERS))
        
        # Extract cache miss counts from perf stat output
        L1_MISSES=$(grep "L1-dcache-load-misses" bench/${EXPERIMENT_DIR}/perf_cache_stats.txt | awk '{print $1}' | sed 's/,//g' | head -1)
        L1_LOADS=$(grep "L1-dcache-loads" bench/${EXPERIMENT_DIR}/perf_cache_stats.txt | awk '{print $1}' | sed 's/,//g' | head -1)
        L2_MISSES=$(grep "L2-dcache-load-misses" bench/${EXPERIMENT_DIR}/perf_cache_stats.txt | awk '{print $1}' | sed 's/,//g' | head -1)
        L2_LOADS=$(grep "L2-dcache-loads" bench/${EXPERIMENT_DIR}/perf_cache_stats.txt | awk '{print $1}' | sed 's/,//g' | head -1)
        
        if [[ -n "$L1_MISSES" && -n "$L1_LOADS" && -n "$L2_MISSES" && -n "$L2_LOADS" ]]; then
            L1_MISS_RATE=$(echo "scale=6; $L1_MISSES / $L1_LOADS" | bc -l 2>/dev/null || echo "N/A")
            L2_MISS_RATE=$(echo "scale=6; $L2_MISSES / $L2_LOADS" | bc -l 2>/dev/null || echo "N/A")
            L1_MISSES_PER_OP=$(echo "scale=6; $L1_MISSES / $TOTAL_OPERATIONS" | bc -l 2>/dev/null || echo "N/A")
            L2_MISSES_PER_OP=$(echo "scale=6; $L2_MISSES / $TOTAL_OPERATIONS" | bc -l 2>/dev/null || echo "N/A")
            
            echo "Hardware Cache Performance Analysis:" > bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "Total operations (N×K×iters): $TOTAL_OPERATIONS" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "L1 Data Cache:" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Loads: $L1_LOADS" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Misses: $L1_MISSES" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Miss Rate: $L1_MISS_RATE" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Misses per operation: $L1_MISSES_PER_OP" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "L2 Data Cache:" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Loads: $L2_LOADS" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Misses: $L2_MISSES" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Miss Rate: $L2_MISS_RATE" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "  Misses per operation: $L2_MISSES_PER_OP" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            echo "Build definitions: $BUILD_DEFS" >> bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt
            
            print_success "Hardware cache analysis saved to bench/${EXPERIMENT_DIR}/cache_analysis_e${EXPERIMENT_NUM}.txt"
        else
            print_warning "Could not extract cache statistics from perf output"
        fi
    else
        print_warning "perf cache data not generated"
    fi
else
    print_warning "perf not available, skipping cache analysis"
    print_info "Note: perf provides hardware cache performance counters (recommended for Linux HPC)"
fi


# Additional Linux profiling: CPU info and system details
if command -v lscpu &> /dev/null; then
    print_status "Capturing system information (Linux)..."
    lscpu > bench/${EXPERIMENT_DIR}/system_info.txt 2>/dev/null || true
    print_success "System info saved to bench/${EXPERIMENT_DIR}/system_info.txt"
fi

# Clean up profiling artifacts
make clean
if [[ -n "$BUILD_DEFS" ]]; then
    ANTIVEC=1 DEFS="$BUILD_DEFS" make release
else
    ANTIVEC=1 make release
fi

# Show results summary
echo
echo "=== Experiment E${EXPERIMENT_NUM} Complete ==="
echo "Results saved in bench/${EXPERIMENT_DIR}/"
echo
echo "Files generated:"
ls -la bench/${EXPERIMENT_DIR}/
echo
echo "Key improvements in this run:"
echo "  ✓ 3 warm-up + 5 measurement runs for statistical significance"
echo "  ✓ Normalized cache statistics (misses per operation)"
echo "  ✓ Slurm node pinning information captured"
echo "  ✓ Build definitions: $BUILD_DEFS"
echo
echo "Next steps:"
if [[ ${EXPERIMENT_NUM} -eq 0 ]]; then
    echo "1. Review baseline results in bench/${EXPERIMENT_DIR}/"
    echo "2. Proceed to E1 optimization experiments"
    echo "3. Use same configs for comparison"
else
    echo "1. Review optimization results in bench/${EXPERIMENT_DIR}/"
    echo "2. Compare with E0 baseline results"
    echo "3. Analyze performance improvements"
    echo "4. Proceed to next optimization (E$((EXPERIMENT_NUM + 1)))"
fi
