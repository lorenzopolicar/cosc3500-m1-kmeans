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

# Try gprof first (Linux - preferred for HPC)
if command -v gprof &> /dev/null; then
    print_status "Using gprof for profiling (Linux)..."
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
# Use perf if available (Linux performance monitoring)
elif command -v perf &> /dev/null; then
    print_status "Using perf for profiling (Linux)..."
    make clean
    if [[ -n "$BUILD_DEFS" ]]; then
        ANTIVEC=1 DEFS="$BUILD_DEFS" make release
    else
        ANTIVEC=1 make release
    fi
    perf record -o bench/${EXPERIMENT_DIR}/perf_stress_config.data \
                ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > /dev/null 2>&1
    if [[ -f "bench/${EXPERIMENT_DIR}/perf_stress_config.data" ]]; then
        perf report -i bench/${EXPERIMENT_DIR}/perf_stress_config.data > bench/${EXPERIMENT_DIR}/perf_stress_config.txt
        print_success "perf profile saved to bench/${EXPERIMENT_DIR}/perf_stress_config.txt"
    else
        print_warning "perf output not generated"
    fi
# Use macOS sample if available
elif command -v sample &> /dev/null; then
    print_status "Using macOS sample for profiling..."
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
    print_warning "No profiling tools available (gprof/perf/sample)"
fi

# Profile with cachegrind if available (Linux)
if command -v valgrind &> /dev/null; then
    print_status "Running cachegrind profile on stress config (Linux)..."
    make clean
    if [[ -n "$BUILD_DEFS" ]]; then
        ANTIVEC=1 DEFS="$BUILD_DEFS" make release
    else
        ANTIVEC=1 make release
    fi
    # Try to load valgrind module (common on HPC clusters)
    module load valgrind 2>/dev/null || echo "valgrind module not available, using system valgrind"
    
    valgrind --tool=cachegrind --cache-sim=yes \
             EXPERIMENT=${EXPERIMENT_DIR} ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 3 --seed 1 > /dev/null 2>&1
    
    # Find the cachegrind output file and annotate it
    CACHEGRIND_OUT=$(ls cachegrind.out.* 2>/dev/null | head -1)
    if [[ -n "$CACHEGRIND_OUT" ]]; then
        print_success "Found cachegrind output: $CACHEGRIND_OUT"
        if command -v cg_annotate &> /dev/null; then
            cg_annotate "$CACHEGRIND_OUT" > bench/${EXPERIMENT_DIR}/cachegrind_e${EXPERIMENT_NUM}.txt
            print_success "cachegrind profile saved to bench/${EXPERIMENT_DIR}/cachegrind_e${EXPERIMENT_NUM}.txt"
            
            # Normalize cache stats: compute L1/L2 misses per label-update
            print_status "Computing normalized cache statistics..."
            N=100000
            D=64
            K=64
            ITERS=3
            TOTAL_OPERATIONS=$((N * K * ITERS))
            
            # Extract L1 and L2 miss counts from cachegrind output
            L1_MISSES=$(grep "D1  misses" "$CACHEGRIND_OUT" | awk '{print $4}' | head -1)
            L2_MISSES=$(grep "LL misses" "$CACHEGRIND_OUT" | awk '{print $4}' | head -1)
            
            if [[ -n "$L1_MISSES" && -n "$L2_MISSES" ]]; then
                L1_MISSES_PER_OP=$(echo "scale=6; $L1_MISSES / $TOTAL_OPERATIONS" | bc -l 2>/dev/null || echo "N/A")
                L2_MISSES_PER_OP=$(echo "scale=6; $L2_MISSES / $TOTAL_OPERATIONS" | bc -l 2>/dev/null || echo "N/A")
                
                echo "Normalized cache statistics:" > bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                echo "Total operations (N×K×iters): $TOTAL_OPERATIONS" >> bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                echo "L1 misses: $L1_MISSES" >> bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                echo "L2 misses: $L2_MISSES" >> bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                echo "L1 misses per operation: $L1_MISSES_PER_OP" >> bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                echo "L2 misses per operation: $L2_MISSES_PER_OP" >> bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                echo "Build definitions: $BUILD_DEFS" >> bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt
                
                print_success "Normalized cache stats saved to bench/${EXPERIMENT_DIR}/cache_stats_e${EXPERIMENT_NUM}.txt"
            else
                print_warning "Could not extract cache miss counts for normalization"
            fi
        fi
    else
        print_warning "cachegrind output not generated"
    fi
else
    print_warning "valgrind not available, skipping cachegrind profiling"
    print_status "Note: Install valgrind for cache behavior analysis (recommended for Linux HPC)"
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
