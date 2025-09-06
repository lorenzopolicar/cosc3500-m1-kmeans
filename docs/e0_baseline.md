# E0: Baseline K-Means Implementation

## Overview

E0 is the baseline implementation of K-Means clustering using Lloyd's algorithm. This serves as the reference point for all subsequent optimizations and provides the foundation for performance comparisons.

## Implementation Details

### Core Algorithm
- **Algorithm**: Lloyd's algorithm (standard K-Means)
- **Distance Metric**: Squared L2 distance (no square root for efficiency)
- **Initialization**: Random centroid positions
- **Convergence**: Fixed number of iterations (no early stopping)

### Data Structures
```cpp
struct Data {
    std::vector<float> points;      // N points of dimension D (row-major: [i*D + d])
    std::vector<float> centroids;   // K centroids of dimension D (row-major: [k*D + d])
    std::vector<int> labels;        // N labels (0 to K-1)
    size_t N, D, K;                 // Number of points, dimensions, clusters
};
```

### Key Features
- **Single-threaded**: No parallelization (OpenMP, threads, SIMD)
- **Memory Layout**: Row-major for both points and centroids
- **Precision**: Double accumulators for centroid updates
- **Timing**: Kernel-split timing (assign vs update)
- **Validation**: Inertia monitoring and monotonicity checks

## Performance Results

### Cluster Test Environment
- **Node**: a100-a (AMD EPYC 7542 32-Core Processor)
- **OS**: Linux 4.18.0-513.9.1.el8_9.x86_64
- **Compiler**: GCC 8.5.0 20210514 (Red Hat 8.5.0-20)
- **Build Flags**: `-std=c++17 -O2 -Wall -Wextra -Wshadow -Wconversion -Iinclude -fno-tree-vectorize`
- **Job ID**: 270437

### Performance Metrics

#### Canonical Configuration (N=200K, D=16, K=8)
- **Assign Kernel**: 17.944 ms (median)
- **Update Kernel**: 2.203 ms (median)
- **Total Time**: 20.157 ms (median)
- **Performance**: 89.17 MLUPS
- **Time Distribution**: Assign 89.0%, Update 10.9%

#### Stress Configuration (N=100K, D=64, K=64)
- **Assign Kernel**: 329.314 ms (median)
- **Update Kernel**: 4.807 ms (median)
- **Total Time**: 334.078 ms (median)
- **Performance**: 19.43 MLUPS
- **Time Distribution**: Assign 98.6%, Update 1.4%

### Profiling Analysis (gprof)

#### Function-Level Performance Breakdown
```
Flat profile:
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ns/call  ns/call  name    
 91.02      1.62     1.62                             assign_labels(Data&)
  3.37      1.68     0.06                             generate_data(Data&, unsigned int)
  2.25      1.72     0.04  8255364     4.85     4.85  std::mersenne_twister_engine<...>::operator()()
  1.69      1.75     0.03                             inertia(Data const&)
  1.12      1.77     0.02                             update_centroids(Data&)
```

#### Key Insights
1. **assign_labels dominates**: 91.02% of total execution time
2. **update_centroids is minor**: Only 1.12% of execution time
3. **Random number generation**: 2.25% overhead from data generation
4. **Inertia computation**: 1.69% overhead for convergence monitoring

### Performance Characteristics

#### Bottleneck Analysis
- **Primary Bottleneck**: `assign_labels` function (91% of time)
- **Memory Access Pattern**: Row-major layout causes cache misses
- **Computational Complexity**: O(N×K×D) for assign, O(N×D) for update
- **Cache Behavior**: Poor spatial locality in centroid access

#### Scalability Observations
- **Canonical vs Stress**: 18.3x slower assign time (17.944ms → 329.314ms)
- **D×K Impact**: Stress config has 64× larger D×K product (128 vs 8192)
- **Performance Degradation**: MLUPS drops from 89.17 to 19.43 (4.6x slower)

## Build Configuration

```bash
# Standard build
make release

# With anti-vectorization (used in cluster tests)
ANTIVEC=1 make release
```

## Correctness Validation

### Algorithm Correctness
- **Deterministic Results**: Fixed seed ensures reproducibility
- **Monotonic Convergence**: Inertia decreases monotonically across iterations
- **Numerical Stability**: Double precision prevents accumulation errors
- **Label Assignment**: Proper nearest-centroid assignment

### Validation Metrics
- **Inertia Behavior**: Non-increasing across iterations
- **Convergence**: Proper clustering behavior observed
- **Edge Cases**: Empty cluster handling (centroid unchanged)
- **Precision**: Consistent results across multiple runs

## Benchmarking Methodology

### Experimental Setup
- **Warm-up Runs**: 3 runs (excluded from analysis)
- **Measurement Runs**: 5 runs (median reported)
- **Statistical Analysis**: Median timing for robustness
- **Kernel Isolation**: Separate timing for assign vs update

### Data Generation
- **Synthetic Data**: Gaussian blobs with controlled parameters
- **Noise Level**: 1.5 standard deviation for challenging clustering
- **Center Range**: 3.0 for closer cluster centers
- **Initialization**: Random centroid positions (not true centers)

## Performance Baseline Summary

E0 establishes the performance baseline with:
- **Canonical Performance**: 89.17 MLUPS
- **Stress Performance**: 19.43 MLUPS
- **Assign Dominance**: 89-98% of execution time
- **Clear Bottleneck**: Memory access pattern in assign_labels

This baseline provides the foundation for measuring the effectiveness of subsequent optimizations (E1, E2) and validates the optimization strategy of targeting the assign_labels kernel.

## Next Steps

E0 serves as the baseline for:
- **E1**: Memory layout optimization (transposed centroids)
- **E2**: Micro-optimizations (invariant hoisting, branchless argmin, strided pointers)
- **Future optimizations**: SIMD, threading, algorithmic improvements
- **Performance comparisons**: Quantified improvement measurements
