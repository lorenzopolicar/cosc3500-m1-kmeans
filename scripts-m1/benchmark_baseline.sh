#!/usr/bin/env bash
set -euo pipefail

# Check if experiment number is provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <experiment_number>"
    echo "Example: $0 0    # for E0 baseline"
    echo "Example: $0 1    # for E1 optimization"
    exit 1
fi

EXPERIMENT_NUM=$1
EXPERIMENT_DIR="e${EXPERIMENT_NUM}"

echo "=== COSC3500 M1 K-Means Experiment E${EXPERIMENT_NUM} ==="
echo "Running experiments and profiling..."
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

# Build the project
print_status "Building project..."
make clean
make release
print_success "Build complete"

# Create bench directory structure
print_status "Creating benchmark directory structure..."
mkdir -p bench/${EXPERIMENT_DIR}

# Canonical config: N=200000, D=16, K=8, iters=10, seed=1
print_status "Running canonical config: N=200000, D=16, K=8, iters=10, seed=1"
EXPERIMENT=${EXPERIMENT_DIR} ./build/kmeans --n 200000 --d 16 --k 8 --iters 10 --seed 1
print_success "Canonical config complete"

# Stress config: D=64, K=64 (larger dimensions and clusters)
print_status "Running stress config: N=100000, D=64, K=64, iters=10, seed=1"
EXPERIMENT=${EXPERIMENT_DIR} ./build/kmeans --n 100000 --d 64 --k 64 --iters 10 --seed 1
print_success "Stress config complete"

# Profile the stress config with gprof
print_status "Running gprof profile on stress config..."
make clean
make profile
EXPERIMENT=${EXPERIMENT_DIR} ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > /dev/null 2>&1
if [[ -f "gmon.out" ]]; then
    gprof ./build/kmeans gmon.out > bench/${EXPERIMENT_DIR}/gprof_stress_config.txt
    print_success "gprof profile saved to bench/${EXPERIMENT_DIR}/gprof_stress_config.txt"
else
    print_warning "gprof output not generated"
fi

# Profile with cachegrind if available
if command -v valgrind &> /dev/null; then
    print_status "Running cachegrind profile on stress config..."
    make clean
    make release
    valgrind --tool=cachegrind --cachegrind-out-file=bench/${EXPERIMENT_DIR}/cachegrind_stress_config.out \
             --log-file=bench/${EXPERIMENT_DIR}/cachegrind_stress_config.log \
             EXPERIMENT=${EXPERIMENT_DIR} ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > /dev/null 2>&1
    
    if [[ -f "bench/${EXPERIMENT_DIR}/cachegrind_stress_config.out" ]]; then
        print_success "cachegrind profile saved to bench/${EXPERIMENT_DIR}/cachegrind_stress_config.out"
        print_status "cachegrind log saved to bench/${EXPERIMENT_DIR}/cachegrind_stress_config.log"
    else
        print_warning "cachegrind output not generated"
    fi
else
    print_warning "valgrind not available, skipping cachegrind profiling"
fi

# Clean up profiling artifacts
make clean
make release

# Show results summary
echo
echo "=== Experiment E${EXPERIMENT_NUM} Complete ==="
echo "Results saved in bench/${EXPERIMENT_DIR}/"
echo
echo "Files generated:"
ls -la bench/${EXPERIMENT_DIR}/
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
