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

### 4. Loop Unrolling (`-DUNROLL_D=4`)

**Goal:** Improve instruction-level parallelism by unrolling the inner D loop.

**Implementation:**
```cpp
// Unroll inner D loop by UNROLL_D factor
size_t d = 0;
for (; d < D - (D % UNROLL_D); d += UNROLL_D) {
    // Process UNROLL_D dimensions at once
    for (size_t unroll_idx = 0; unroll_idx < UNROLL_D; ++unroll_idx) {
        float diff = px[d + unroll_idx] - ck[0];
        d2 += static_cast<double>(diff) * static_cast<double>(diff);
        ck += K;  // Move to next dimension
    }
}

// Handle remainder dimensions
for (; d < D; ++d) {
    float diff = px[d] - *ck;
    d2 += static_cast<double>(diff) * static_cast<double>(diff);
    ck += K;
}
```

**Benefits:**
- Reduces loop overhead
- Enables better instruction scheduling
- Improves register utilization

**Expected Impact:** 2-5% improvement

## Build Configuration

E2 uses the following build definitions:
```bash
BUILD_DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DUNROLL_D=4"
```

**Optimization Priority:**
1. `UNROLL_D` - Highest priority (most aggressive)
2. `STRIDE_PTR` - High priority (arithmetic reduction)
3. `BRANCHLESS` - Medium priority (branch prediction)
4. `HOIST` - Base optimization (foundation)

## Function Routing

The `assign_labels` function routes to optimized versions based on defined flags:

```cpp
void assign_labels(Data& data) {
    // Route to E2 optimized versions if flags are defined
#ifdef UNROLL_D
    assign_labels_unrolled(data);
    return;
#endif

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

## Expected Performance Impact

**Combined E2 Improvements:**
- **Canonical Config (N=200K, D=16, K=8):** 6-14% improvement over E1
- **Stress Config (N=100K, D=64, K=64):** 8-16% improvement over E1

**Total E0→E1→E2 Improvement:**
- **Canonical:** 10-20% total improvement
- **Stress:** 12-22% total improvement

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
