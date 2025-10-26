# K-Means Optimization Experiments: Comprehensive Analysis

## Executive Summary

This document provides a comprehensive analysis of six K-Means clustering optimization experiments conducted on the Rangpur cluster. The experiments demonstrate a systematic approach to serial optimization, achieving **conditional performance improvements** through memory layout optimization, micro-optimizations, and register blocking techniques.

## Experiment Overview

| Experiment | Optimization Type | Key Technique | Build Flags | Status |
|------------|------------------|---------------|-------------|---------|
| **E0** | Baseline | Standard Lloyd's algorithm | `-fno-tree-vectorize` | ✅ Success |
| **E1** | Memory Layout | Transposed centroids | `-DTRANSPOSED_C=1` | ✅ Success |
| **E2** | Micro-optimization | Invariant hoisting + Branchless argmin + Strided pointers | `-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1` | ✅ Success |
| **E3** | Cache Blocking | K-tiling (centroid blocking) | `-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_K=16` | ❌ Failed |
| **E4** | Cache Blocking | D-tiling (dimension blocking) | `-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_D=8` | ❌ Failed |
| **E5** | Register Blocking | K-register blocking (TK=4) | `-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTK=4` | ⚖️ Conditional Success |

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
- **Job IDs**: E0 (270437), E1 (270438), E2 (270439), E3 (270440), E4 (270441), E5 (270442)

## Performance Results

### Canonical Configuration (N=200K, D=16, K=8)

| Metric | E0 | E1 | E2 | E3 | E4 | E5 | E0→E2 | E2→E5 |
|--------|----|----|----|----|----|----|-------|-------|
| **Assign Kernel (ms)** | 17.944 | 17.317 | 16.495 | 16.612 | 17.234 | 17.478 | **-8.1%** | **+6.0%** |
| **Update Kernel (ms)** | 2.203 | 2.105 | 2.101 | 2.098 | 2.103 | 2.090 | **-4.6%** | **-0.5%** |
| **Total Time (ms)** | 20.157 | 19.423 | 18.604 | 18.710 | 19.337 | 19.556 | **-7.7%** | **+5.1%** |
| **Performance (MLUPS)** | 89.17 | 92.39 | 97.00 | 96.35 | 93.12 | 91.54 | **+8.8%** | **-5.6%** |
| **Assign %** | 89.0% | 89.2% | 88.7% | 88.8% | 89.1% | 89.4% | **-0.3%** | **+0.7%** |

### Stress Configuration (N=100K, D=64, K=64)

| Metric | E0 | E1 | E2 | E3 | E4 | E5 | E0→E2 | E2→E5 |
|--------|----|----|----|----|----|----|-------|-------|
| **Assign Kernel (ms)** | 329.314 | 319.483 | 287.299 | 289.123 | 328.456 | 216.697 | **-12.8%** | **-24.6%** |
| **Update Kernel (ms)** | 4.807 | 4.715 | 4.767 | 4.789 | 4.823 | 4.802 | **-0.8%** | **+0.7%** |
| **Total Time (ms)** | 334.078 | 324.187 | 292.040 | 293.912 | 333.279 | 221.375 | **-12.6%** | **-24.2%** |
| **Performance (MLUPS)** | 19.43 | 20.03 | 22.28 | 22.12 | 19.48 | 29.53 | **+14.7%** | **+32.4%** |
| **Assign %** | 98.6% | 98.5% | 98.4% | 98.4% | 98.5% | 97.9% | **-0.2%** | **-0.5%** |

## Profiling Analysis

### Function-Level Performance Breakdown

| Experiment | assign_labels % | Total Time (s) | Key Function | Improvement vs E0 |
|------------|----------------|----------------|--------------|-------------------|
| **E0** | 91.02% | 1.78s | `assign_labels(Data&)` | Baseline |
| **E1** | 90.60% | 1.70s | `assign_labels(Data&)` | -4.5% |
| **E2** | 88.13% | 1.60s | `assign_labels_strided(Data&)` | -10.1% |
| **E3** | 88.45% | 1.61s | `assign_labels_tiled(Data&)` | -9.6% |
| **E4** | 89.12% | 1.68s | `assign_labels_d_tiled(Data&)` | -5.6% |
| **E5** | 86.89% | 1.22s | `assign_labels(Data&)` | -31.5% |

### Profiling Insights

1. **Bottleneck Identification**: `assign_labels` dominates execution time (86-91%)
2. **Progressive Optimization**: E0→E2 shows consistent improvement, E5 shows dramatic improvement
3. **Function Specialization**: E2, E3, E4 show specialized functions, E5 returns to unified approach
4. **Total Runtime**: E0→E2 gradual improvement, E5 dramatic improvement (1.78s → 1.22s)
5. **Failed Experiments**: E3 and E4 show minimal or negative improvements

## Optimization Analysis

### E1: Memory Layout Optimization ✅

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

### E2: Micro-optimizations ✅

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
bool better = (d2 < best_d2);
best_k = better ? static_cast<int>(k) : best_k;
best_d2 = better ? d2 : best_d2;

// Strided pointers
const float* __restrict__ ck = &centroidsT[k];
for (size_t d = 0; d < D; ++d) {
    float diff = px[d] - *ck;
    ck += K;  // Stride by K
}
```

### E3: K-Tiling (Centroid Blocking) ❌

**Technique**: Process centroids in tiles of TK=16 for cache blocking
- **Canonical Impact**: -0.6% regression
- **Stress Impact**: -0.7% regression
- **Effectiveness**: Failed - no actual cache blocking benefits
- **Root Cause**: Nested loops inside per-point loop = same access pattern + overhead

**Why It Failed**:
- Loop structure: `for (point) { for (tile) { for (centroid) } }`
- Same memory access pattern as E2 with added loop overhead
- No temporal reuse benefits
- True cache blocking would require: `for (tile) { for (point) { for (centroid) } }`

### E4: D-Tiling (Dimension Blocking) ❌

**Technique**: Process dimensions in tiles of TD=8 for cache blocking
- **Canonical Impact**: -3.9% regression
- **Stress Impact**: -14.1% regression
- **Effectiveness**: Failed - wrong temporal reuse target
- **Root Cause**: Dimensions already have optimal spatial locality

**Why It Failed**:
- Spatial locality already optimal over dimensions
- Temporal reuse lives across centroids, not dimensions
- Added loop overhead without cache benefits
- Modern prefetchers already handle contiguous dimension access

### E5: K-Register Blocking ⚖️

**Technique**: Process TK=4 centroids simultaneously using scalar accumulators
- **Canonical Impact**: -5.6% regression (small working set)
- **Stress Impact**: +32.4% improvement (large working set)
- **Effectiveness**: Conditional success - depends on problem size
- **Key Success**: Temporal reuse of px[d] across TK centroids

**Implementation**:
```cpp
// TK=4 specialized path with scalar accumulators
for (size_t k0 = 0; k0 < K; k0 += 4) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    
    for (size_t d = 0; d < D; ++d) {
        const float x = px[d];  // Load once per dimension
        const float* base = &centroidsT[d * K + k0];
        
        // Reuse x across 4 centroids
        float t = x - base[0]; s0 += (double)t * t;
        t = x - base[1]; s1 += (double)t * t;
        t = x - base[2]; s2 += (double)t * t;
        t = x - base[3]; s3 += (double)t * t;
    }
    
    // Branchless argmin across 4 accumulators
    // ... (argmin logic)
}
```

**Configuration Analysis**:
- **Canonical (K×D=128 floats)**: Fits in L1 cache, E5 adds overhead
- **Stress (K×D=4096 floats)**: Approaches L1 limit, E5 provides cache benefits

## Key Findings

### 1. Optimization Strategy Success
- **Targeted Approach**: All optimizations focus on the assign_labels bottleneck
- **Cumulative Effect**: E0→E2 shows 7.7-12.6% total improvement
- **Conditional Success**: E5 shows dramatic improvement on stress config (+32.4%)
- **Synergy**: E1 + E2 + E5 optimizations work well together

### 2. Scalability Characteristics
- **Stress Config Benefits Most**: E2 shows 12.6% improvement, E5 shows 32.4% improvement
- **Large D×K Impact**: Register blocking particularly effective on stress config
- **Memory Access**: Transposed layout + strided pointers + register blocking optimal for large datasets
- **Problem Size Dependency**: E5 effectiveness scales with K×D product

### 3. Algorithm Correctness
- **Identical Results**: All experiments produce identical inertia values (±1e-6)
- **Monotonic Convergence**: Proper clustering behavior maintained
- **Numerical Stability**: Double precision prevents accumulation errors
- **Perfect Validation**: No correctness regressions across any experiment

### 4. Profiling Validation
- **Bottleneck Confirmation**: assign_labels remains dominant across all experiments (86-91%)
- **Optimization Effectiveness**: E0→E2 gradual improvement, E5 dramatic improvement
- **Hardware Utilization**: Better cache utilization and instruction efficiency
- **Function Specialization**: E2, E3, E4 show specialized functions, E5 returns to unified approach

### 5. Failed Optimization Analysis
- **E3 K-Tiling**: No actual cache blocking benefits, just loop overhead
- **E4 D-Tiling**: Wrong temporal reuse target, dimensions already optimal
- **Common Failure Pattern**: Both failed due to misunderstanding of temporal reuse patterns
- **Learning Value**: Failures provide insights into effective optimization strategies

## Technical Insights

### Memory Access Patterns
- **E0**: Poor spatial locality in centroid access
- **E1**: Improved cache locality with transposed layout
- **E2**: Optimal access pattern with strided pointers
- **E3**: Same as E2 with added loop overhead (failed)
- **E4**: Same as E2 with added loop overhead (failed)
- **E5**: Register blocking with temporal reuse across centroids

### Instruction-Level Optimizations
- **Invariant Hoisting**: Reduces repeated memory access
- **Branchless Argmin**: Eliminates branch misprediction penalties
- **Strided Pointers**: Eliminates d*K multiplication overhead
- **Register Blocking**: Temporal reuse of px[d] across TK centroids
- **Scalar Accumulators**: TK=4 uses s0..s3 instead of arrays to avoid register spilling

### Compiler Optimization
- **Anti-vectorization**: `-fno-tree-vectorize` ensures single-threaded compliance
- **Restrict Pointers**: `__restrict__` enables better compiler optimization
- **Function Routing**: Conditional compilation for optimization variants
- **Combined Optimizations**: E5 integrates all E1+E2+E5 benefits in single function

### Cache Behavior Analysis
- **L1 Cache Size**: 32KB on AMD EPYC 7542
- **Working Set Sizes**: Canonical (512 bytes), Stress (16KB)
- **Cache Pressure**: E5 benefits when working set approaches L1 limit
- **Temporal Reuse**: E5 provides 4× reduction in point memory accesses

## Conclusions

### Optimization Success Summary
The systematic optimization approach achieved significant performance improvements:

| Experiment | Canonical Config | Stress Config | Status |
|------------|------------------|---------------|---------|
| **E1** | +3.6% | +3.0% | ✅ Success |
| **E2** | +4.2% | +9.9% | ✅ Success |
| **E3** | -0.6% | -0.7% | ❌ Failed |
| **E4** | -3.9% | -14.1% | ❌ Failed |
| **E5** | -5.6% | +32.4% | ⚖️ Conditional Success |

**Combined E0→E2**: 7.7-12.6% total improvement from baseline
**E5 on Stress Config**: 32.4% improvement over E2 baseline

### Key Success Factors
1. **Bottleneck Identification**: Profiling correctly identified assign_labels as the bottleneck
2. **Targeted Optimization**: Successful optimizations focus on the critical path
3. **Incremental Approach**: Each optimization builds on the previous
4. **Correctness Validation**: All optimizations maintain algorithm correctness
5. **Implementation Quality**: Small bugs can cause major performance regressions
6. **Combined Optimizations**: E5 integrates all E1+E2+E5 benefits effectively

### Scalability Insights
- **Large Datasets**: Stress configuration benefits most from optimizations
- **Memory Access**: Cache-friendly patterns crucial for performance
- **Instruction Efficiency**: Micro-optimizations provide significant gains
- **Problem Size Dependency**: E5 effectiveness scales with K×D product
- **Cache Pressure**: Register blocking benefits when working set approaches L1 limit

### Failed Optimization Lessons
1. **E3 K-Tiling**: No actual cache blocking benefits, just loop overhead
2. **E4 D-Tiling**: Wrong temporal reuse target, dimensions already optimal
3. **Common Pattern**: Both failed due to misunderstanding of temporal reuse patterns
4. **Learning Value**: Failures provide insights into effective optimization strategies

### Future Optimization Opportunities
1. **SIMD Vectorization**: Next milestone could explore SIMD optimizations
2. **Algorithmic Improvements**: Early stopping, better initialization
3. **Parallelization**: Multi-threading for larger datasets
4. **Advanced Memory Layouts**: Blocked layouts for very large datasets
5. **Adaptive Optimization**: Choose E2 vs E5 based on problem size

## Methodology Validation

### Experimental Rigor
- **Consistent Environment**: Same node, compiler, and build configuration across all experiments
- **Statistical Analysis**: Median values from 5 measurement runs for robust results
- **Reproducibility**: Fixed seeds and deterministic algorithms ensure consistent results
- **Comprehensive Profiling**: gprof analysis validates optimization effectiveness
- **Correctness Validation**: All experiments produce identical results (±1e-6)

### Performance Measurement
- **Kernel Isolation**: Separate timing for assign vs update kernels
- **Warm-up Elimination**: 3 warm-up runs excluded from analysis
- **Hardware Consistency**: Same CPU frequency and memory configuration
- **Build Reproducibility**: Identical compiler flags and optimization levels
- **Configuration Testing**: Both canonical and stress configurations tested

### Optimization Validation
- **Incremental Testing**: Each optimization builds on previous successes
- **Failure Analysis**: Failed optimizations (E3, E4) provide valuable insights
- **Conditional Success**: E5 demonstrates problem-size dependent optimization effectiveness
- **Implementation Quality**: Critical bugs in E5 initial implementation caused major regression

## Final Assessment

This comprehensive analysis demonstrates the effectiveness of systematic serial optimization in K-Means clustering, achieving:

- **Universal Success**: E1 and E2 provide consistent improvements across all configurations
- **Conditional Success**: E5 provides dramatic improvements on stress configurations
- **Learning from Failures**: E3 and E4 failures provide insights into effective optimization strategies
- **Implementation Matters**: Small bugs can cause major performance regressions
- **Problem Size Dependency**: Optimization effectiveness depends on working set characteristics

The results provide a solid foundation for future optimization efforts and serve as a reference for performance engineering in scientific computing, demonstrating both the potential and the challenges of serial optimization in K-Means clustering.
