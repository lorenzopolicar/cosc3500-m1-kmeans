# COSC3500 M1 K-Means Benchmarking System

## Overview

This document describes the comprehensive benchmarking system for the COSC3500 Milestone 1 K-Means clustering optimization project. The system provides statistical rigor, complete reproducibility, and professional-grade benchmarking capabilities for serial optimization experiments.

## System Architecture

### Core Components

1. **Experiment Runner Script**: `scripts/run_experiment.sh`
2. **Build System**: `Makefile` with optimization variants
3. **Data Generation**: Synthetic Gaussian blobs with configurable parameters
4. **Performance Measurement**: Kernel-split timing with statistical significance
5. **Metadata Capture**: Complete build and runtime environment recording
6. **Profiling Integration**: gprof and cachegrind support for Linux HPC

### File Organization

```
bench/e{x}/                           # Experiment directory
├── sysinfo_e{x}.txt                  # System information capture
├── meta_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.txt  # Run metadata
├── times_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.csv # Timing data
├── inertia_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.csv # Convergence data
├── gprof_e{x}.txt                    # Function-level profiling (Linux)
├── cachegrind_e{x}.txt               # Cache performance analysis (Linux)
└── cache_stats_e{x}.txt              # Normalized cache statistics (Linux)
```

## Experiment Runner Script (`scripts/run_experiment.sh`)

### Usage

```bash
./scripts/run_experiment.sh <experiment_number> [build_defs]
```

**Examples:**
```bash
./scripts/run_experiment.sh 0                    # E0 baseline
./scripts/run_experiment.sh 1 '-DTRANSPOSED_C=1' # E1 with transposed centroids
./scripts/run_experiment.sh 2 '-DUNROLL=4'       # E2 with loop unrolling
```

### Script Workflow

#### 1. Build for Timings (Authoritative)

```bash
# Build the project for TIMINGS (authoritative)
print_status "Building project for TIMINGS..."
make clean
if [[ -n "$BUILD_DEFS" ]]; then
    print_status "Building with definitions: $BUILD_DEFS"
    ANTIVEC=1 DEFS="$BUILD_DEFS" make release
else
    ANTIVEC=1 make release
fi
```

**Key Features:**
- **ANTIVEC=1**: Disables vectorization for consistent single-threaded performance
- **Build definitions**: Optional compiler definitions for optimization variants
- **Clean build**: Ensures no profiling artifacts interfere with timing

#### 2. System Information Capture

```bash
# Capture system information
print_status "Capturing system information..."
./scripts/print_sysinfo.sh | tee bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt

# Capture Slurm node information for HPC clusters
if [[ -n "$SLURM_NODELIST" ]]; then
    print_status "Capturing Slurm node information..."
    echo "Slurm Node Information:" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_NODELIST: $SLURM_NODELIST" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_NODEID: $SLURM_NODEID" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_JOB_ID: $SLURM_JOB_ID" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_PARTITION: $SLURM_PARTITION" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
    echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE" >> bench/${EXPERIMENT_DIR}/sysinfo_e${EXPERIMENT_NUM}.txt
fi
```

**Captured Information:**
- **Compiler**: Version and flags from Makefile
- **Build environment**: CXX, CXXFLAGS, ANTIVEC, DEFS
- **Git state**: SHA, branch, modification count
- **System details**: OS, architecture, CPU info
- **HPC integration**: Slurm environment variables for cluster consistency

#### 3. Canonical Configuration (Statistical Rigor)

```bash
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
```

**Statistical Design:**
- **3 warm-up runs**: Eliminate cold-start effects (CPU caches, branch predictors)
- **5 measurement runs**: Provide statistical significance and outlier detection
- **Environment consistency**: ANTIVEC=1 maintained across all runs
- **Output separation**: Each run creates unique files with `_run{run}` suffix

#### 4. Stress Configuration (Memory Effects)

```bash
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
```

**Purpose:**
- **Higher D×K**: Amplifies memory access patterns and cache effects
- **Baseline for optimization**: E1-E8 will reuse these exact configurations
- **Memory pressure**: Reveals layout and access pattern optimizations

#### 5. gprof Profiling (Linux HPC)

```bash
# Try gprof first (Linux - preferred for HPC)
if command -v gprof &> /dev/null; then
    print_status "Using gprof for profiling (Linux)..."
    make clean
    if [[ -n "$BUILD_DEFS" ]]; then
        DEFS="$BUILD_DEFS" make profile
    else
        make profile
    fi
    EXPERIMENT=${EXPERIMENT_DIR} ANTIVEC=1 ./build/kmeans --n 100000 --d 64 --k 64 --iters 5 --seed 1 > /dev/null 2>&1
    if [[ -f "gmon.out" ]]; then
        gprof ./build/kmeans gmon.out > bench/${EXPERIMENT_DIR}/gprof_e${EXPERIMENT_NUM}.txt
        print_success "gprof profile saved to bench/${EXPERIMENT_DIR}/gprof_e${EXPERIMENT_NUM}.txt"
    fi
fi
```

**Features:**
- **Separate build**: Profile build with `-pg` flag, separate from timing builds
- **Build definitions**: Maintains optimization variants in profiling
- **Output**: Function-level performance analysis with call graphs

#### 6. cachegrind Profiling (Linux HPC)

```bash
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
            fi
        fi
    fi
fi
```

**Advanced Features:**
- **HPC integration**: `module load valgrind` for cluster environments
- **Normalized statistics**: L1/L2 misses per operation for cross-problem comparison
- **Build consistency**: Same build flags as timing runs
- **Reduced iterations**: 3 iterations for faster completion

## Build System Integration

### Makefile Targets

```makefile
# Release build with optional anti-vectorization and definitions
release: CXXFLAGS = $(CXXFLAGS_BASE) $(if $(ANTIVEC),$(ANTIVEC_FLAGS)) $(DEFS)
release: $(TARGET)

# Profile build with gprof and optional definitions
profile: CXXFLAGS = $(CXXFLAGS_BASE) -pg $(DEFS)
profile: LDFLAGS = -pg
profile: $(TARGET)
```

**Environment Variables:**
- **ANTIVEC=1**: Enables anti-vectorization flags
- **DEFS**: Compiler definitions for optimization variants
- **CXX**: Compiler selection (defaults to g++)

### Anti-Vectorization Support

```makefile
# Detect compiler for anti-vectorization flags
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null | head -n1)
ifeq ($(findstring GCC,$(CXX_VERSION)),GCC)
    ANTIVEC_FLAGS = -fno-tree-vectorize
else ifeq ($(findstring clang,$(CXX_VERSION)),clang)
    ANTIVEC_FLAGS = -fno-vectorize -fno-slp-vectorize
else
    ANTIVEC_FLAGS = 
endif
```

## Data Generation and Algorithm

### Synthetic Data Parameters

```cpp
// Data generation parameters
float noise_std = 1.5f;           // Gaussian noise standard deviation
float center_range = 3.0f;        // Range for cluster centers [-3.0, 3.0]
std::normal_distribution<float> noise_dist(0.0f, noise_std);
std::uniform_real_distribution<float> center_dist(-center_range, center_range);

// Initialize centroids to RANDOM positions (not the true centers)
std::uniform_real_distribution<float> centroid_dist(-8.0f, 8.0f);
```

**Design Rationale:**
- **Challenging clustering**: Higher noise and random initialization create realistic convergence scenarios
- **Deterministic**: Fixed seeds ensure reproducibility
- **Edge cases**: Random centroids may result in empty clusters (handled gracefully)

### K-Means Algorithm Implementation

```cpp
// Core kernels with separate timing
for (size_t iter = 0; iter < iters; ++iter) {
    // Time assign kernel
    auto assign_start = std::chrono::steady_clock::now();
    assign_labels(data);
    auto assign_end = std::chrono::steady_clock::now();
    assign_ms[iter] = std::chrono::duration<double, std::milli>(assign_end - assign_start).count();
    
    // Time update kernel
    auto update_start = std::chrono::steady_clock::now();
    update_centroids(data);
    auto update_end = std::chrono::steady_clock::now();
    update_ms[iter] = std::chrono::duration<double, std::milli>(update_end - update_start).count();
    
    // Compute inertia (not timed)
    inertia_vals[iter] = inertia(data);
}
```

**Kernel Separation:**
- **assign_labels**: N×K×D complexity, dominates runtime
- **update_centroids**: K×D complexity, minimal runtime
- **inertia**: Convergence monitoring, excluded from timing

## Output File Formats

### Timing Data (`times_*.csv`)

```csv
iter,assign_ms,update_ms,total_ms
0,8.07679,0.984292,9.06108
1,7.28538,1.09996,8.38533
2,7.67971,0.866917,8.54663
...
```

**Columns:**
- **iter**: Iteration number (0-based)
- **assign_ms**: Time for label assignment kernel
- **update_ms**: Time for centroid update kernel
- **total_ms**: Combined time for both kernels

### Convergence Data (`inertia_*.csv`)

```csv
iter,inertia,N,D,K,iters,seed
0,1.32646e+07,200000,16,8,10,1
1,9.53124e+06,200000,16,8,10,1
2,8.91557e+06,200000,16,8,10,1
...
```

**Columns:**
- **iter**: Iteration number (0-based)
- **inertia**: Sum of squared distances to assigned centroids
- **N, D, K**: Problem dimensions
- **iters, seed**: Configuration parameters

### Metadata (`meta_*.txt`)

```txt
Experiment: e0
Timestamp: 1756877803568588
Git SHA: 7905c6c
Compiler: c++
Flags: (default)
ANTIVEC: 1
CLI Args: --n 200000 --d 16 --k 8 --iters 10 --seed 1
Data Params: noise_std=1.5, center_range=3.0, init=random
```

**Information Captured:**
- **Experiment identifier**: e0, e1, e2, etc.
- **Timestamp**: Unix timestamp for reproducibility
- **Git state**: Current commit SHA
- **Build environment**: Compiler, flags, ANTIVEC setting
- **Runtime parameters**: CLI arguments and data generation settings

## Statistical Analysis

### Run Structure

```
Run 1-3: Warm-up runs (output discarded)
├── Eliminates cold-start effects
├── CPU caches warmed up
├── Branch predictors trained
└── System in steady state

Run 4-8: Measurement runs (data collected)
├── Statistical significance (n=5)
├── Outlier detection
├── Median performance calculation
└── Confidence interval estimation
```

### Performance Metrics

**Time Distribution:**
- **Canonical config**: assign ~88.8%, update ~11.2%
- **Stress config**: assign ~98.9%, update ~1.1%

**Convergence Behavior:**
- **Monotonic decrease**: Inertia should never increase
- **Stable final state**: Convergence within 10 iterations
- **Deterministic**: Same seed produces identical results

## HPC Cluster Integration

### Slurm Environment Variables

```bash
# Captured automatically when available
SLURM_NODELIST      # Node allocation list
SLURM_NODEID        # Node identifier
SLURM_JOB_ID        # Job identifier
SLURM_PARTITION     # Queue/partition name
SLURM_CPUS_ON_NODE  # CPU allocation
SLURM_MEM_PER_NODE  # Memory allocation
```

**Purpose:**
- **Node consistency**: Ensures runs on same hardware
- **Resource tracking**: Documents allocation for reproducibility
- **Cluster identification**: Distinguishes between different HPC environments

### Module Loading

```bash
# Try to load valgrind module (common on HPC clusters)
module load valgrind 2>/dev/null || echo "valgrind module not available, using system valgrind"
```

**HPC Optimization:**
- **Module system**: Standard on most HPC clusters
- **Fallback handling**: Graceful degradation if module unavailable
- **Environment consistency**: Ensures same tool versions across runs

## Optimization Workflow

### Baseline Establishment (E0)

```bash
./scripts/run_experiment.sh 0
```

**Output:**
- Complete performance baseline
- Statistical confidence intervals
- Cache behavior characterization
- System configuration documentation

### Optimization Experiments (E1-E8)

```bash
# Example optimization variants
./scripts/run_experiment.sh 1 '-DTRANSPOSED_C=1'      # Memory layout optimization
./scripts/run_experiment.sh 2 '-DUNROLL=4'             # Loop unrolling
./scripts/run_experiment.sh 3 '-DBLOCK_SIZE=32'        # Cache blocking
./scripts/run_experiment.sh 4 '-DALIGNED=1'            # Memory alignment
```

**Comparison Process:**
1. **Same configurations**: Identical N, D, K, iters, seed
2. **Statistical comparison**: Median performance across runs
3. **Cache analysis**: Normalized miss rates
4. **Build transparency**: Clear optimization identification

## Quality Assurance

### Reproducibility Checks

1. **Fixed seeds**: Deterministic data generation
2. **Environment capture**: Complete build and runtime state
3. **Git integration**: Source code version tracking
4. **Hardware consistency**: Slurm node pinning

### Validation Criteria

1. **Convergence**: Inertia decreases monotonically
2. **Performance stability**: Low variance across measurement runs
3. **Build consistency**: ANTIVEC and DEFS properly applied
4. **File integrity**: All expected files generated with correct content

### Error Handling

1. **Graceful degradation**: Profiling tools not required
2. **Warning messages**: Non-critical issues reported
3. **Exit codes**: Script fails on critical errors
4. **Logging**: Comprehensive status reporting

## Usage Examples

### Basic Baseline Run

```bash
# Run E0 baseline experiment
./scripts/run_experiment.sh 0

# Expected output: 55 files in bench/e0/
# - 16 timing files (8 runs × 2 configs)
# - 16 inertia files (8 runs × 2 configs)
# - 16 metadata files (8 runs × 2 configs)
# - 1 system info file
# - 1 profiling file
# - 1 profiling config file
```

### Optimization Experiment

```bash
# Run E1 with transposed centroids
./scripts/run_experiment.sh 1 '-DTRANSPOSED_C=1'

# Expected output: 55 files in bench/e1/
# - Same structure as E0
# - Build definitions clearly identified
# - Performance comparison ready
```

### Custom Configuration

```bash
# Run with custom build definitions
./scripts/run_experiment.sh 5 '-DUNROLL=8 -DALIGNED=1'

# Build system applies:
# - ANTIVEC=1 (anti-vectorization)
# - DEFS="-DUNROLL=8 -DALIGNED=1"
# - All optimizations documented in metadata
```

## Best Practices

### For Reproducibility

1. **Always use fixed seeds**: Ensures deterministic results
2. **Document build environment**: Capture all relevant variables
3. **Version control**: Commit before each experiment
4. **Hardware consistency**: Use same cluster nodes when possible

### For Statistical Rigor

1. **Multiple measurement runs**: Minimum 5 for confidence
2. **Warm-up runs**: Eliminate system effects
3. **Outlier detection**: Identify and investigate anomalies
4. **Median reporting**: More robust than mean

### For Optimization Analysis

1. **Baseline comparison**: Always compare against E0
2. **Same configurations**: Identical problem sizes and parameters
3. **Cache analysis**: Normalized statistics for fair comparison
4. **Build transparency**: Clear identification of optimizations

## Troubleshooting

### Common Issues

1. **ANTIVEC not captured**: Check script environment variable passing
2. **Missing files**: Verify directory permissions and disk space
3. **Profiling failures**: Ensure tools available on target system
4. **Build errors**: Check compiler compatibility and dependencies

### Debug Commands

```bash
# Check script execution
bash -x ./scripts/run_experiment.sh 0

# Verify environment variables
env | grep -E "(EXPERIMENT|ANTIVEC|DEFS)"

# Check file generation
ls -la bench/e0/ | wc -l

# Validate CSV content
head -5 bench/e0/times_N200000_D16_K8_iters10_seed1_run4.csv
```

## Conclusion

This benchmarking system provides:

1. **Statistical rigor**: Multiple runs with warm-up elimination
2. **Complete reproducibility**: Environment and build state capture
3. **Professional quality**: Production-grade benchmarking pipeline
4. **HPC integration**: Cluster-aware profiling and execution
5. **Optimization ready**: Systematic comparison framework

The system is designed for academic research and provides the foundation for systematic serial optimization experiments in K-Means clustering algorithms.
