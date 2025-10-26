# COSC3500 M1 Benchmarking Process

## Overview

This document describes the benchmarking infrastructure for tracking optimization experiments (E0-E8) in the K-means project.

## Directory Structure

```
bench/
├── e0/                    # E0 baseline results
│   ├── meta_*.txt        # Run metadata (git SHA, compiler, flags, etc.)
│   ├── times_*.csv       # Kernel timing data
│   ├── inertia_*.csv     # Convergence data
│   ├── gprof_*.txt       # gprof profiles (if available)
│   └── cachegrind_*.out  # cachegrind profiles (if available)
├── e1/                    # E1 optimization results
├── e2/                    # E2 optimization results
└── ...                    # Additional experiments
```

## Metadata Capture

Each run automatically creates a metadata file containing:

- **Experiment**: Experiment identifier (e0, e1, e2, etc.)
- **Timestamp**: Unix timestamp of run
- **Git SHA**: Current commit hash for reproducibility
- **Compiler**: C++ compiler used
- **Flags**: Compiler flags (CXXFLAGS environment variable)
- **ANTIVEC**: Anti-vectorization setting (ANTIVEC environment variable)
- **CLI Args**: All command line parameters
- **Data Params**: Data generation parameters (noise, center range, initialization)

## Baseline Configurations

### Canonical Config (Primary Benchmark)
- **N=200000, D=16, K=8, iters=10, seed=1**
- **Purpose**: Standard performance measurement
- **Use**: Primary comparison baseline for all optimizations

### Stress Config (Memory Effects)
- **N=100000, D=64, K=64, iters=10, seed=1**
- **Purpose**: Amplify memory effects and cache behavior
- **Use**: Secondary baseline for memory-intensive optimizations

## Running Baseline Experiments

### Automated Baseline Script
```bash
./scripts/benchmark_baseline.sh
```

This script:
1. Builds the project
2. Runs canonical and stress configs
3. Generates gprof profiles
4. Generates cachegrind profiles (if valgrind available)
5. Organizes results in `bench/e0/`

### Manual Baseline Runs
```bash
# Canonical config
./build/kmeans --n 200000 --d 16 --k 8 --iters 10 --seed 1

# Stress config
./build/kmeans --n 100000 --d 64 --k 64 --iters 10 --seed 1
```

## Profiling

### gprof Profiling
```bash
make profile
./build/kmeans [args] > /dev/null 2>&1
gprof ./build/kmeans gmon.out > profile.txt
```

### cachegrind Profiling
```bash
valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out \
         ./build/kmeans [args] > /dev/null 2>&1
```

## Optimization Workflow

### For Each Optimization (E1-E8):

1. **Create experiment directory**: `bench/e{N}/`
2. **Run with same configs**: Use identical N, D, K, iters, seed
3. **Compare results**: 
   - Timing data: `times_*.csv`
   - Convergence: `inertia_*.csv`
   - Metadata: `meta_*.txt`
4. **Profile if needed**: Generate new profiles for analysis

### Example E1 Workflow:
```bash
# 1. Modify code for E1 optimization
# 2. Build and test
make clean && make release

# 3. Run canonical config
./build/kmeans --n 200000 --d 16 --k 8 --iters 10 --seed 1

# 4. Run stress config  
./build/kmeans --n 100000 --d 64 --k 64 --iters 10 --seed 1

# 5. Compare with E0 baseline
diff bench/e0/times_N200000_D16_K8_iters10_seed1.csv \
      bench/e1/times_N200000_D16_K8_iters10_seed1.csv
```

## Data Analysis

### Timing Analysis
- **assign_ms**: Assignment kernel time (should dominate)
- **update_ms**: Update kernel time (should be smaller)
- **total_ms**: Total iteration time
- **Percentage split**: Shows optimization impact on each kernel

### Convergence Analysis
- **inertia**: Clustering quality measure
- **Monotonicity**: Should decrease or stay constant
- **Convergence rate**: How quickly algorithm stabilizes

### Performance Metrics
- **Speedup**: E0_time / E{N}_time
- **Efficiency**: Speedup / optimization effort
- **Cache behavior**: Miss rates from cachegrind
- **Hotspots**: Function profiles from gprof

## Best Practices

1. **Reproducibility**: Always use same seeds and parameters
2. **Consistency**: Run same configs across all experiments
3. **Documentation**: Update metadata for any environment changes
4. **Validation**: Verify correctness before measuring performance
5. **Multiple runs**: Use median times for stability

## Troubleshooting

### Common Issues:
- **gprof not working**: Ensure `-pg` flag is used and program runs to completion
- **valgrind not available**: Install valgrind or skip cachegrind profiling
- **Metadata missing**: Check file permissions and directory creation
- **Results inconsistent**: Verify same compiler, flags, and environment

### Debugging:
- Check metadata files for environment differences
- Verify git SHA matches expected commit
- Compare compiler flags across experiments
- Validate input parameters are identical
