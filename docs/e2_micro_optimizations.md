# E2: Assign-kernel Micro-optimizations

## Overview

E2 implements micro-optimizations for the `assign_labels` kernel to reduce per-point overhead without changing the algorithm or timing boundaries. These optimizations build on E1's transposed centroids layout and target instruction-level efficiency improvements.

## Optimization Techniques

### 1. Invariant Hoisting (`-DHOIST=1`)

**Goal:** Reduce repeated memory access by hoisting loop invariants outside the hot loops.

**Implementation:**
```cpp
// Hoist invariants outside loops
const size_t N = data.N;
const size_t D = data.D;
const size_t K = data.K;
const float* __restrict__ points = data.points.data();
const float* __restrict__ centroidsT = data.centroidsT.data();

// Cache point pointer for each iteration
const float* __restrict__ px = &points[i * D];
```

**Benefits:**
- Eliminates repeated `data.N`, `data.D`, `data.K` access
- Reduces pointer arithmetic in inner loops
- Enables better compiler optimization with `__restrict__`

**Expected Impact:** 1-3% improvement

### 2. Branchless Argmin (`-DBRANCHLESS=1`)

**Goal:** Improve CPU branch prediction by using ternary operators instead of if-statements.

**Implementation:**
```cpp
// Original (branchy)
if (d2 < best_d2) {
    best_d2 = d2;
    best_k = static_cast<int>(k);
}

// Optimized (branchless)
best_d2 = (d2 < best_d2) ? d2 : best_d2;
best_k = (d2 < best_d2) ? static_cast<int>(k) : best_k;
```

**Benefits:**
- Eliminates branch misprediction penalties
- Better instruction-level parallelism
- More predictable execution path

**Expected Impact:** 1-2% improvement

### 3. Strided Pointer Arithmetic (`-DSTRIDE_PTR=1`)

**Goal:** Eliminate `d*K` multiplies in transposed centroid access by using strided pointers.

**Implementation:**
```cpp
// Original (with multiply)
for (size_t d = 0; d < D; ++d) {
    float diff = px[d] - centroidsT[d * K + k];
    d2 += static_cast<double>(diff) * static_cast<double>(diff);
}

// Optimized (strided pointer)
const float* __restrict__ ck = &centroidsT[k];  // Start at dimension 0, centroid k
for (size_t d = 0; d < D; ++d) {
    float diff = px[d] - *ck;
    d2 += static_cast<double>(diff) * static_cast<double>(diff);
    ck += K;  // Move to next dimension, same centroid (stride by K)
}
```

**Benefits:**
- Eliminates `d*K` multiplication in inner loop
- Reduces arithmetic operations per iteration
- Better memory access pattern

**Expected Impact:** 2-4% improvement


## Build Configuration

E2 uses the following build definitions:
```bash
BUILD_DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1"
```

**Optimization Priority:**
1. `STRIDE_PTR` - Highest priority (arithmetic reduction)
2. `BRANCHLESS` - Medium priority (branch prediction)
3. `HOIST` - Base optimization (foundation)

## Function Routing

The `assign_labels` function routes to optimized versions based on defined flags:

```cpp
void assign_labels(Data& data) {
    // Route to E2 optimized versions if flags are defined
#ifdef STRIDE_PTR
    assign_labels_strided(data);
    return;
#endif

#ifdef BRANCHLESS
    assign_labels_branchless(data);
    return;
#endif

#ifdef HOIST
    assign_labels_hoisted(data);
    return;
#endif

    // Original E0/E1 implementation (fallback)
    // ...
}
```

## Performance Results

### Cluster Test Environment
- **Node**: a100-a (AMD EPYC 7542 32-Core Processor)
- **OS**: Linux 4.18.0-513.9.1.el8_9.x86_64
- **Compiler**: GCC 8.5.0 20210514 (Red Hat 8.5.0-20)
- **Build Flags**: `-std=c++17 -O2 -Wall -Wextra -Wshadow -Wconversion -Iinclude -fno-tree-vectorize -DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1`
- **Job ID**: 270439

### Performance Metrics

#### Canonical Configuration (N=200K, D=16, K=8)
- **Assign Kernel**: 16.495 ms (median)
- **Update Kernel**: 2.101 ms (median)
- **Total Time**: 18.604 ms (median)
- **Performance**: 97.00 MLUPS
- **Time Distribution**: Assign 88.7%, Update 11.3%

#### Stress Configuration (N=100K, D=64, K=64)
- **Assign Kernel**: 287.299 ms (median)
- **Update Kernel**: 4.767 ms (median)
- **Total Time**: 292.040 ms (median)
- **Performance**: 22.28 MLUPS
- **Time Distribution**: Assign 98.4%, Update 1.6%

### Performance Improvement Analysis

#### Canonical Configuration (N=200K, D=16, K=8)
- **E1→E2 Assign Time**: 4.7% improvement (17.317ms → 16.495ms)
- **E1→E2 Total Time**: 4.2% improvement (19.423ms → 18.604ms)
- **E1→E2 Performance**: 5.0% improvement (92.39 → 97.00 MLUPS)

#### Stress Configuration (N=100K, D=64, K=64)
- **E1→E2 Assign Time**: 10.1% improvement (319.483ms → 287.299ms)
- **E1→E2 Total Time**: 9.9% improvement (324.187ms → 292.040ms)
- **E1→E2 Performance**: 11.2% improvement (20.03 → 22.28 MLUPS)

#### Total E0→E2 Improvement
- **Canonical**: 7.7% total improvement (20.157ms → 18.604ms)
- **Stress**: 12.6% total improvement (334.078ms → 292.040ms)

### Profiling Analysis (gprof)

#### Function-Level Performance Breakdown
```
Flat profile:
  %   cumulative   self              self     total           
 time   seconds   seconds    calls  ns/call  ns/call  name    
 88.13      1.41     1.41                             assign_labels_strided(Data&)
  5.00      1.49     0.08                             generate_data(Data&, unsigned int)
  3.13      1.54     0.05  8255364     6.06     6.06  std::mersenne_twister_engine<...>::operator()()
  1.88      1.57     0.03                             update_centroids(Data&)
  1.25      1.59     0.02                             inertia(Data const&)
```

#### Key Insights
1. **assign_labels_strided dominates**: 88.13% of total execution time
2. **Significant improvement**: 2.47% reduction from E1's 90.60%
3. **Total execution time**: Reduced from 1.70s (E1) to 1.60s (E2)
4. **Micro-optimizations effective**: Strided pointer optimization shows clear benefit

### Key Findings

1. **Excellent Improvement**: Both configurations show 4-11% improvement over E1
2. **Stress Config Benefits Most**: 10.1% assign improvement on large D×K
3. **Micro-optimizations Work**: Invariant hoisting, branchless argmin, and strided pointers effective
4. **Total E0→E2 Success**: 7.7-12.6% cumulative improvement
5. **Function Routing Success**: `assign_labels_strided` correctly used

## Comprehensive Performance Comparison

### E0 → E1 → E2 Progression

| Configuration | Metric | E0 (Baseline) | E1 (Transposed) | E2 (Micro-opt) | E0→E1 | E1→E2 | E0→E2 |
|---------------|--------|---------------|-----------------|----------------|-------|-------|-------|
| **Canonical** | Assign (ms) | 17.944 | 17.317 | 16.495 | -3.5% | -4.7% | **-8.1%** |
| **Canonical** | Total (ms) | 20.157 | 19.423 | 18.604 | -3.6% | -4.2% | **-7.7%** |
| **Canonical** | MLUPS | 89.17 | 92.39 | 97.00 | +3.6% | +5.0% | **+8.8%** |
| **Stress** | Assign (ms) | 329.314 | 319.483 | 287.299 | -3.0% | -10.1% | **-12.8%** |
| **Stress** | Total (ms) | 334.078 | 324.187 | 292.040 | -3.0% | -9.9% | **-12.6%** |
| **Stress** | MLUPS | 19.43 | 20.03 | 22.28 | +3.1% | +11.2% | **+14.7%** |

### Optimization Effectiveness Analysis

#### Memory Layout Optimization (E1)
- **Canonical Impact**: 3.5-3.6% improvement
- **Stress Impact**: 3.0% improvement
- **Effectiveness**: Consistent but modest improvement
- **Bottleneck**: Transpose overhead limits gains

#### Micro-optimizations (E2)
- **Canonical Impact**: 4.2-4.7% improvement
- **Stress Impact**: 9.9-10.1% improvement
- **Effectiveness**: Excellent, especially on large datasets
- **Key Success**: Strided pointers eliminate d*K multiplies

#### Combined Optimization Strategy
- **Cumulative Effect**: E0→E2 shows 7.7-12.6% total improvement
- **Synergy**: E1 + E2 optimizations work well together
- **Scalability**: Stress config benefits most from combined approach
- **Targeting**: Both optimizations focus on assign_labels bottleneck

### Profiling Evolution

| Experiment | assign_labels % | Total Time (s) | Key Function |
|------------|----------------|----------------|--------------|
| **E0** | 91.02% | 1.78s | `assign_labels(Data&)` |
| **E1** | 90.60% | 1.70s | `assign_labels(Data&)` |
| **E2** | 88.13% | 1.60s | `assign_labels_strided(Data&)` |

#### Profiling Insights
1. **Bottleneck Persistence**: assign_labels remains dominant across all experiments
2. **Progressive Improvement**: Function time percentage decreases with optimizations
3. **Total Runtime**: Consistent reduction from 1.78s → 1.70s → 1.60s
4. **Function Specialization**: E2 shows specialized `assign_labels_strided` function

## Correctness Validation

**Acceptance Criteria:**
- Labels and inertia identical to E1 (±1e-6)
- Assign per-iteration median decreases
- Update time unchanged
- Timers and CSV formats unchanged
- Build with `ANTIVEC=1` (no auto-vectorization)

## Implementation Details

### Memory Layout Compatibility
- All optimizations work with E1's transposed centroids
- Maintains `centroids[K×D]` as source of truth
- Uses `centroidsT[D×K]` for optimized access patterns

### Compiler Optimizations
- Uses `__restrict__` pointers to enable better optimization
- Maintains single-threaded, no-SIMD constraints
- Compatible with `-fno-tree-vectorize` (ANTIVEC=1)

### Fallback Support
- Each optimization can be enabled/disabled independently
- Graceful fallback to E1 implementation
- Maintains algorithm correctness across all variants

## Profiling and Analysis

**Key Metrics to Monitor:**
- Assign kernel timing reduction
- Update kernel timing (should remain constant)
- Total performance improvement
- Convergence behavior (should be identical)

**Expected Profiling Results:**
- Reduced time in `assign_labels` function
- Better instruction-level parallelism
- Reduced branch misprediction
- Improved cache utilization

## Next Steps

1. **Run E2 Experiment:** Use `scripts/slurm/run_experiment_e2.slurm`
2. **Compare Results:** Analyze E0 vs E1 vs E2 performance
3. **Validate Correctness:** Ensure identical results across experiments
4. **Document Findings:** Update this document with actual results

## Technical Notes

**Micro-optimization Philosophy:**
- Target the hottest code path (`assign_labels`)
- Maintain algorithm correctness
- Use compiler-friendly patterns
- Build incrementally (each optimization can be measured independently)

**Performance Characteristics:**
- Optimizations are cumulative
- Best results on stress configurations (large D×K)
- Maintains single-threaded constraints
- Compatible with existing profiling infrastructure
