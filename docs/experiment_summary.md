# K-Means Optimization Experiments: Comprehensive Analysis

## Executive Summary

This document provides a comprehensive analysis of three K-Means clustering optimization experiments conducted on the Rangpur cluster. The experiments demonstrate a systematic approach to serial optimization, achieving **7.7-12.6% total performance improvement** through memory layout optimization and micro-optimizations.

## Experiment Overview

| Experiment | Optimization Type | Key Technique | Build Flags |
|------------|------------------|---------------|-------------|
| **E0** | Baseline | Standard Lloyd's algorithm | `-fno-tree-vectorize` |
| **E1** | Memory Layout | Transposed centroids | `-DTRANSPOSED_C=1` |
| **E2** | Micro-optimization | Invariant hoisting + Branchless argmin + Strided pointers | `-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1` |

## Test Environment

### Hardware Configuration
- **Cluster**: Rangpur HPC
- **Node**: a100-a (consistent across all experiments)
- **CPU**: AMD EPYC 7542 32-Core Processor @ 2894.561 MHz
- **Memory**: 62GB RAM, 5GB Swap
- **Cache**: L1d cache: 32K, L1i cache: 32K
- **OS**: Linux 4.18.0-513.9.1.el8_9.x86_64 (Red Hat Enterprise Linux 8.9)

### Software Configuration
- **Compiler**: GCC 8.5.0 20210514 (Red Hat 8.5.0-20)
- **Build Flags**: `-std=c++17 -O2 -Wall -Wextra -Wshadow -Wconversion -Iinclude -fno-tree-vectorize`
- **Methodology**: 3 warm-up + 5 measurement runs, median values reported
- **Job IDs**: E0 (270437), E1 (270438), E2 (270439)

## Performance Results

### Canonical Configuration (N=200K, D=16, K=8)

| Metric | E0 (Baseline) | E1 (Transposed) | E2 (Micro-opt) | E0→E1 | E1→E2 | E0→E2 |
|--------|---------------|-----------------|----------------|-------|-------|-------|
| **Assign Kernel (ms)** | 17.944 | 17.317 | 16.495 | -3.5% | -4.7% | **-8.1%** |
| **Update Kernel (ms)** | 2.203 | 2.105 | 2.101 | -4.4% | -0.2% | **-4.6%** |
| **Total Time (ms)** | 20.157 | 19.423 | 18.604 | -3.6% | -4.2% | **-7.7%** |
| **Performance (MLUPS)** | 89.17 | 92.39 | 97.00 | +3.6% | +5.0% | **+8.8%** |
| **Assign %** | 89.0% | 89.2% | 88.7% | +0.2% | -0.5% | **-0.3%** |

### Stress Configuration (N=100K, D=64, K=64)

| Metric | E0 (Baseline) | E1 (Transposed) | E2 (Micro-opt) | E0→E1 | E1→E2 | E0→E2 |
|--------|---------------|-----------------|----------------|-------|-------|-------|
| **Assign Kernel (ms)** | 329.314 | 319.483 | 287.299 | -3.0% | -10.1% | **-12.8%** |
| **Update Kernel (ms)** | 4.807 | 4.715 | 4.767 | -1.9% | +1.1% | **-0.8%** |
| **Total Time (ms)** | 334.078 | 324.187 | 292.040 | -3.0% | -9.9% | **-12.6%** |
| **Performance (MLUPS)** | 19.43 | 20.03 | 22.28 | +3.1% | +11.2% | **+14.7%** |
| **Assign %** | 98.6% | 98.5% | 98.4% | -0.1% | -0.1% | **-0.2%** |

## Profiling Analysis

### Function-Level Performance Breakdown

| Experiment | assign_labels % | Total Time (s) | Key Function | Improvement |
|------------|----------------|----------------|--------------|-------------|
| **E0** | 91.02% | 1.78s | `assign_labels(Data&)` | Baseline |
| **E1** | 90.60% | 1.70s | `assign_labels(Data&)` | -4.5% |
| **E2** | 88.13% | 1.60s | `assign_labels_strided(Data&)` | -10.1% |

### Profiling Insights

1. **Bottleneck Identification**: `assign_labels` dominates execution time (88-91%)
2. **Progressive Optimization**: Each experiment reduces assign_labels overhead
3. **Function Specialization**: E2 shows specialized `assign_labels_strided` function
4. **Total Runtime**: Consistent reduction from 1.78s → 1.70s → 1.60s

## Optimization Analysis

### E1: Memory Layout Optimization

**Technique**: Transposed centroids for cache-friendly access
- **Canonical Impact**: 3.5-3.6% improvement
- **Stress Impact**: 3.0% improvement
- **Effectiveness**: Consistent but modest improvement
- **Key Success**: Improved cache locality in assign_labels

**Implementation**:
```cpp
// Original (Cache-Unfriendly)
centroids[k * D + d]  // Row-major access

// Optimized (Cache-Friendly)
centroidsT[d * K + k]  // Column-major access
```

### E2: Micro-optimizations

**Techniques**: Invariant hoisting + Branchless argmin + Strided pointers
- **Canonical Impact**: 4.2-4.7% improvement
- **Stress Impact**: 9.9-10.1% improvement
- **Effectiveness**: Excellent, especially on large datasets
- **Key Success**: Strided pointers eliminate d*K multiplies

**Implementation**:
```cpp
// Invariant hoisting
const size_t N = data.N, D = data.D, K = data.K;
const float* __restrict__ points = data.points.data();

// Branchless argmin
best_d2 = (d2 < best_d2) ? d2 : best_d2;
best_k = (d2 < best_d2) ? static_cast<int>(k) : best_k;

// Strided pointers
const float* __restrict__ ck = &centroidsT[k];
for (size_t d = 0; d < D; ++d) {
    float diff = px[d] - *ck;
    ck += K;  // Stride by K
}
```

## Key Findings

### 1. Optimization Strategy Success
- **Targeted Approach**: Both optimizations focus on the assign_labels bottleneck
- **Cumulative Effect**: E0→E2 shows 7.7-12.6% total improvement
- **Synergy**: E1 + E2 optimizations work well together

### 2. Scalability Characteristics
- **Stress Config Benefits Most**: 12.6% improvement vs 7.7% on canonical
- **Large D×K Impact**: Micro-optimizations particularly effective on stress config
- **Memory Access**: Transposed layout + strided pointers optimal for large datasets

### 3. Algorithm Correctness
- **Identical Results**: All experiments produce identical inertia values (±1e-6)
- **Monotonic Convergence**: Proper clustering behavior maintained
- **Numerical Stability**: Double precision prevents accumulation errors

### 4. Profiling Validation
- **Bottleneck Confirmation**: assign_labels remains dominant across all experiments
- **Optimization Effectiveness**: Progressive reduction in function overhead
- **Hardware Utilization**: Better cache utilization and instruction efficiency

## Technical Insights

### Memory Access Patterns
- **E0**: Poor spatial locality in centroid access
- **E1**: Improved cache locality with transposed layout
- **E2**: Optimal access pattern with strided pointers

### Instruction-Level Optimizations
- **Invariant Hoisting**: Reduces repeated memory access
- **Branchless Argmin**: Eliminates branch misprediction penalties
- **Strided Pointers**: Eliminates d*K multiplication overhead

### Compiler Optimization
- **Anti-vectorization**: `-fno-tree-vectorize` ensures single-threaded compliance
- **Restrict Pointers**: `__restrict__` enables better compiler optimization
- **Function Routing**: Conditional compilation for optimization variants

## Conclusions

### Optimization Success
The systematic optimization approach achieved significant performance improvements:
- **E1**: 3.0-3.6% improvement through memory layout optimization
- **E2**: 4.2-10.1% improvement through micro-optimizations
- **Combined**: 7.7-12.6% total improvement from baseline

### Key Success Factors
1. **Bottleneck Identification**: Profiling correctly identified assign_labels as the bottleneck
2. **Targeted Optimization**: Both optimizations focus on the critical path
3. **Incremental Approach**: Each optimization builds on the previous
4. **Correctness Validation**: All optimizations maintain algorithm correctness

### Scalability Insights
- **Large Datasets**: Stress configuration benefits most from optimizations
- **Memory Access**: Cache-friendly patterns crucial for performance
- **Instruction Efficiency**: Micro-optimizations provide significant gains

### Future Optimization Opportunities
1. **SIMD Vectorization**: Next milestone could explore SIMD optimizations
2. **Algorithmic Improvements**: Early stopping, better initialization
3. **Parallelization**: Multi-threading for larger datasets
4. **Advanced Memory Layouts**: Blocked layouts for very large datasets

## Methodology Validation

### Experimental Rigor
- **Consistent Environment**: Same node, compiler, and build configuration
- **Statistical Analysis**: Median values from 5 measurement runs
- **Reproducibility**: Fixed seeds and deterministic algorithms
- **Comprehensive Profiling**: gprof analysis validates optimization effectiveness

### Performance Measurement
- **Kernel Isolation**: Separate timing for assign vs update kernels
- **Warm-up Elimination**: 3 warm-up runs excluded from analysis
- **Hardware Consistency**: Same CPU frequency and memory configuration
- **Build Reproducibility**: Identical compiler flags and optimization levels

This comprehensive analysis demonstrates the effectiveness of systematic serial optimization in K-Means clustering, providing a solid foundation for future optimization efforts and serving as a reference for performance engineering in scientific computing.
