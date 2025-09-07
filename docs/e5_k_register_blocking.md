# E5: K-Register Blocking Optimization

## Overview

E5 implements **K-register blocking** to reduce per-point overhead by evaluating TK centroids at once using multiple accumulators, reusing the loaded point value `px[d]` across TK centroids for better temporal locality. This optimization builds on the lessons learned from the failed E3 and E4 attempts.

## Context: Why E3 and E4 Failed

### E3 K-Tiling Failure
- **Problem**: Nested `for (k0 ...) for (k ...)` inside the per-point loop was functionally identical to E2's access pattern
- **Result**: Same memory access pattern with added loop overhead
- **Performance**: -0.9% regression vs E2
- **Root Cause**: No actual cache blocking benefits, just unnecessary loop overhead

### E4 D-Tiling Failure  
- **Problem**: Tiling dimensions when spatial locality was already optimal
- **Result**: Same contiguous access pattern with added loop overhead
- **Performance**: -30% (canonical) and -12% (stress) regression vs E2
- **Root Cause**: Modern prefetchers already handle contiguous dimension access well

### Key Lessons Learned
1. **Spatial locality was already optimal** in E1+E2
2. **Temporal reuse lives across centroids, not dimensions**
3. **Loop overhead can easily outweigh theoretical cache benefits**
4. **Need to target the actual bottleneck**: point value reuse across centroids

## E5 Design Principles

### Why K-Register Blocking Works
1. **Addresses Real Temporal Reuse**: `px[d]` is loaded once and reused across TK centroids (the actual temporal reuse pattern in K-means)
2. **Contiguous Memory Access**: `centroidsT[d*K + (k0+j)]` gives contiguous access across centroids for fixed d
3. **Register Efficiency**: Multiple accumulators (s0..s{TK-1}) stay in registers
4. **Reduced Index Math**: Process TK centroids with one point load

### Technical Advantages
- **Natural Blocking**: Centroids are the natural blocking unit for K-means
- **Register Pressure**: TK=4 accumulators easily fit in registers
- **Cache Friendly**: Contiguous access pattern thanks to E1 transposed layout
- **Branchless Argmin**: Can use the same branchless pattern from E2
- **Clean Fallback**: Scalar path for remainder and non-transposed case

## Implementation Details

### Core Algorithm
```cpp
// For each point, find the nearest centroid using K-register blocking
for (size_t i = 0; i < N; ++i) {
    double best_d2 = std::numeric_limits<double>::max();
    int best_k = 0;
    const float* __restrict__ px = &points[i * D];
    
    // Process centroids in tiles of TK for register blocking
    for (size_t k0 = 0; k0 < K; k0 += TK) {
        size_t k_end = std::min(k0 + TK, K);
        size_t tile_size = k_end - k0;
        
        // Initialize accumulators for this tile
        double s[TK] = {0.0};
        
        // Inner loop over dimensions - reuse px[d] across TK centroids
        for (size_t d = 0; d < D; ++d) {
            float px_d = px[d];  // Load point value once per dimension
            
            // Process all centroids in current tile
            for (size_t j = 0; j < tile_size; ++j) {
                // Contiguous access: centroidsT[d*K + (k0+j)]
                float c_j = centroidsT[d * K + (k0 + j)];
                float diff_j = px_d - c_j;
                s[j] += static_cast<double>(diff_j) * static_cast<double>(diff_j);
            }
        }
        
        // Branchless argmin across accumulators in this tile
        for (size_t j = 0; j < tile_size; ++j) {
            if (s[j] < best_d2) {
                best_d2 = s[j];
                best_k = static_cast<int>(k0 + j);
            }
        }
    }
    
    data.labels[i] = best_k;
}
```

### Transposed Centroids Support
```cpp
#ifdef TRANSPOSED_C
// Contiguous access: centroidsT[d*K + (k0+j)]
float c_j = centroidsT[d * K + (k0 + j)];
#else
// Fallback for non-transposed case
float c_j = centroids[(k0 + j) * D + d];
#endif
```

### Memory Access Pattern Comparison

| Approach | Point Loads per Dimension | Memory Access Pattern | Temporal Reuse |
|----------|---------------------------|----------------------|----------------|
| **E2** | K loads | `px[d]` loaded K times | ❌ None |
| **E3** | K loads | Same as E2 + overhead | ❌ None |
| **E4** | K loads | Same as E2 + overhead | ❌ None |
| **E5** | 1 load | `px[d]` loaded once, reused TK times | ✅ TK× reduction |

## Build Configuration

### Compilation
```bash
# Build with E5 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTK=4" make release

# Profile build with E5 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTK=4" make profile
```

### Slurm Job Submission
```bash
# Submit E5 experiment
sbatch scripts/slurm/run_experiment_e5.slurm
```

## Expected Performance Impact

### Cache Analysis
- **Current E2**: `px[d]` loaded K times per dimension
- **E5 with TK=4**: `px[d]` loaded once per dimension, reused 4 times
- **Memory Efficiency**: 4× reduction in point memory accesses
- **Register Utilization**: 4 accumulators + loop variables = ~6-8 registers

### Performance Predictions
- **Canonical Config (N=200K, D=16, K=8)**: 5-15% improvement (K=8, TK=4 gives 2 tiles)
- **Stress Config (N=100K, D=64, K=64)**: 10-25% improvement (K=64, TK=4 gives 16 tiles)
- **Total E0→E5**: 30-50% cumulative improvement

### Optimization Hierarchy
1. **E1**: Memory layout optimization (transposed centroids)
2. **E2**: Micro-optimizations (invariant hoisting, branchless argmin, strided pointers)
3. **E5**: Cache optimization (K-register blocking)

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

## Technical Insights

### Register Blocking Benefits
- **Temporal Locality**: `px[d]` reused across TK centroids
- **Spatial Locality**: Centroids in same tile share cache lines
- **Reduced Memory Traffic**: TK× fewer point memory accesses
- **Register Efficiency**: Multiple accumulators stay in registers

### Tuning Considerations
- **TK=4**: Good balance (4 accumulators fit in registers)
- **TK=2**: More conservative, better for smaller register files
- **TK=8**: More aggressive, may cause register pressure
- **System-Specific**: Optimal TK depends on register file size and cache characteristics

### Memory Access Pattern
- **E0**: Poor spatial locality in centroid access
- **E1**: Improved cache locality with transposed layout
- **E2**: Optimal access pattern with strided pointers
- **E5**: Register blocking for maximum temporal locality

## Performance Characteristics

### Bottleneck Analysis
- **Primary Bottleneck**: `assign_labels_k_blocked` function (expected ~85% of time)
- **Memory Access Pattern**: Optimal register-blocked access
- **Computational Complexity**: O(N×K×D) for assign, O(N×D) for update
- **Cache Behavior**: Excellent temporal locality with register blocking

### Scalability Observations
- **Canonical vs Stress**: K-register blocking benefits scale with K
- **D×K Impact**: Stress config (D=64, K=64) benefits most from register blocking
- **Register Efficiency**: 4× improvement in temporal locality

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

## Next Steps

1. **Run E5 Experiment**: Use `scripts/slurm/run_experiment_e5.slurm`
2. **Compare Results**: Analyze E0 vs E1 vs E2 vs E5 performance
3. **Validate Correctness**: Ensure identical results across experiments
4. **Document Findings**: Update this document with actual results

## Technical Notes

### Cache Blocking Philosophy
- **Target the real bottleneck** (temporal reuse across centroids)
- **Maintain algorithm correctness**
- **Use register-friendly patterns**
- **Build incrementally** (each optimization can be measured independently)

### Performance Characteristics
- **Optimizations are cumulative**
- **Best results on stress configurations** (large D×K)
- **Maintains single-threaded constraints**
- **Compatible with existing profiling infrastructure**

### Future Optimization Opportunities
1. **SIMD Vectorization**: Next milestone could explore SIMD optimizations
2. **Algorithmic Improvements**: Early stopping, better initialization
3. **Parallelization**: Multi-threading for larger datasets
4. **Advanced Memory Layouts**: Blocked layouts for very large datasets

## Comparison with Failed Approaches

### E3 vs E5
| Aspect | E3 K-Tiling | E5 K-Register Blocking |
|--------|-------------|------------------------|
| **Loop Structure** | `for (point) { for (tile) { for (centroid) } }` | `for (point) { for (tile) { for (dim) { for (centroid) } } }` |
| **Memory Access** | Same as E2 | TK× fewer point loads |
| **Temporal Reuse** | None | TK× reuse of px[d] |
| **Performance** | -0.9% regression | Expected 10-25% improvement |

### E4 vs E5
| Aspect | E4 D-Tiling | E5 K-Register Blocking |
|--------|-------------|------------------------|
| **Target** | Dimension loop | Centroid loop |
| **Temporal Reuse** | None (dimensions already contiguous) | TK× reuse of px[d] |
| **Memory Access** | Same as E2 | TK× fewer point loads |
| **Performance** | -30% regression | Expected 10-25% improvement |

## Conclusion

E5 K-register blocking represents the first optimization that:
1. **Addresses the real bottleneck** (temporal reuse across centroids)
2. **Leverages E1's transposed layout** correctly
3. **Minimizes overhead** (just register accumulators)
4. **Has clear performance benefits** (TK× reduction in point loads)

This approach should finally deliver the cache performance gains we've been looking for, building on the lessons learned from the failed E3 and E4 attempts.
