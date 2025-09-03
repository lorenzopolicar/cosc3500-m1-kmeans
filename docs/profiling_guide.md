# Profiling Guide for COSC3500 M1

## Overview

This guide explains the profiling tools available for different operating systems and how to use them effectively for K-means optimization experiments.

## Operating System Detection

The `run_experiment.sh` script automatically detects your OS and uses appropriate profiling tools:

- **Linux**: gprof, perf, valgrind/cachegrind
- **macOS**: sample, instruments
- **Other**: gprof (if available)

## Linux Profiling (Rangpur Cluster)

### 1. gprof (GNU Profiler) - **Recommended for HPC**

**What it provides:**
- Function-level timing analysis
- Call graph analysis
- Flat profile with percentage breakdown

**How it works:**
```bash
make profile  # Builds with -pg flag
./build/kmeans [args]  # Runs and generates gmon.out
gprof ./build/kmeans gmon.out > profile.txt
```

**Output analysis:**
- **Flat profile**: Shows time spent in each function
- **Call graph**: Shows function call relationships
- **Percentage**: Time distribution across functions

**Best for:** Identifying hot functions and bottlenecks

### 2. perf (Linux Performance Counters)

**What it provides:**
- Hardware performance counters
- CPU cycles, cache misses, branch mispredictions
- Low-overhead sampling

**How it works:**
```bash
perf record -o perf.data ./build/kmeans [args]
perf report -i perf.data > profile.txt
```

**Output analysis:**
- **CPU cycles**: Function-level cycle counts
- **Cache behavior**: L1/L2/L3 miss rates
- **Branch prediction**: Misprediction rates

**Best for:** Hardware-level performance analysis

### 3. valgrind/cachegrind - **Essential for Cache Analysis**

**What it provides:**
- Detailed cache behavior analysis
- Memory access patterns
- Cache miss rates by function

**How it works:**
```bash
valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out \
         ./build/kmeans [args]
cg_annotate cachegrind.out > summary.txt
```

**Output analysis:**
- **L1 cache**: Instruction and data cache misses
- **L2 cache**: Unified cache miss rates
- **Memory access**: Patterns and locality

**Best for:** Cache optimization and memory layout analysis

## macOS Profiling

### 1. sample (Command Line Profiler)

**What it provides:**
- Process sampling at regular intervals
- Stack traces and function calls
- Low overhead

**How it works:**
```bash
./build/kmeans [args] &
sample $PID 5 > profile.txt  # Sample for 5 seconds
```

**Output analysis:**
- **Stack traces**: Function call sequences
- **Time distribution**: Where time is spent
- **Call frequency**: Function call counts

**Best for:** Quick performance overview

### 2. Instruments (GUI Profiler)

**What it provides:**
- Comprehensive profiling suite
- Time profiler, allocations, leaks
- Interactive analysis

**How to use:**
```bash
instruments -t Time\ Profiler ./build/kmeans [args]
```

**Best for:** Detailed interactive analysis

## Profiling Strategy for K-means

### E0 Baseline Profiling

1. **gprof**: Identify hot functions (assign vs update)
2. **cachegrind**: Baseline cache behavior
3. **System info**: Document hardware configuration

### E1-E8 Optimization Profiling

1. **Compare gprof**: Function timing changes
2. **Compare cachegrind**: Cache behavior improvements
3. **Performance metrics**: Speedup calculations

## Interpreting Results

### gprof Analysis

Look for:
- **assign_labels()**: Should dominate time (~90%)
- **update_centroids()**: Should be smaller (~10%)
- **inertia()**: Minimal impact on total time

### cachegrind Analysis

Look for:
- **L1 cache misses**: Should decrease with optimizations
- **Memory access patterns**: Contiguous vs scattered
- **Cache line utilization**: Efficiency improvements

### Performance Metrics

Calculate:
- **Speedup**: E0_time / E{N}_time
- **Cache improvement**: Miss rate reduction
- **Efficiency**: Performance per optimization effort

## Troubleshooting

### gprof Issues

**Problem**: No gmon.out generated
**Solution**: Ensure program exits normally (no signals)

**Problem**: Inaccurate timing
**Solution**: Use larger datasets for statistical significance

### cachegrind Issues

**Problem**: Very slow execution
**Solution**: Use smaller datasets for profiling runs

**Problem**: No cg_annotate
**Solution**: Install valgrind-dev package

### macOS Issues

**Problem**: sample permission denied
**Solution**: Grant accessibility permissions to Terminal

**Problem**: No profiling output
**Solution**: Ensure process runs long enough to sample

## Best Practices

1. **Profile representative datasets**: Not too small, not too large
2. **Multiple runs**: Use median values for stability
3. **Baseline comparison**: Always compare against E0
4. **Document environment**: OS, compiler, hardware details
5. **Focus on trends**: Relative improvements matter more than absolute values

## Example Analysis Workflow

```bash
# 1. Run baseline
./scripts/run_experiment.sh 0

# 2. Implement optimization
# ... modify code ...

# 3. Run optimization
./scripts/run_experiment.sh 1

# 4. Compare results
diff bench/e0/times_N200000_D16_K8_iters10_seed1.csv \
      bench/e1/times_N200000_D16_K8_iters10_seed1.csv

# 5. Analyze profiles
# Compare gprof outputs
# Compare cachegrind summaries
# Calculate speedups
```

## Next Steps

After profiling:
1. **Identify bottlenecks**: Which functions consume most time?
2. **Cache analysis**: Where are cache misses occurring?
3. **Optimization targets**: Focus on highest-impact areas
4. **Measure improvements**: Quantify each optimization's effect
