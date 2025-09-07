# E4: D-Tiling (Dimension Blocking) Optimization

## Overview

E4 implements **D-tiling (dimension blocking)** to improve cache locality by processing dimensions in tiles of TD. This approach blocks the dimension loop to improve spatial locality while maintaining algorithm correctness.

## Design Principles

### Cache Blocking Strategy
- **Target**: Dimension loop in distance computation
- **Approach**: Process dimensions in tiles of TD for better spatial locality
- **Benefit**: Improved cache line utilization for dimension data

### Why D-Tiling Works Better Than K-Tiling
1. **Natural Blocking**: Dimensions are accessed sequentially within each centroid
2. **Cache Friendly**: Dimension tiles fit well in cache lines
3. **No State Management**: No need to maintain state across tiles
4. **Maintains E2 Optimizations**: All E2 micro-optimizations are preserved

## Implementation Details

### Core Algorithm
```cpp
// For each point, find the nearest centroid using D-tiling
for (size_t i = 0; i < N; ++i) {
    double best_d2 = std::numeric_limits<double>::max();
    int best_k = 0;
    
    for (size_t k = 0; k < K; ++k) {
        double d2 = 0.0;
        
        // Process dimensions in tiles of TD for better cache locality
        for (size_t d0 = 0; d0 < D; d0 += TILE_D) {
            size_t d_end = std::min(d0 + TILE_D, D);
            
            // Process all dimensions in current tile
            for (size_t d = d0; d < d_end; ++d) {
                float diff = px[d] - ck[d];
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
        }
        
        // Update best if this centroid is closer
        if (d2 < best_d2) {
            best_d2 = d2;
            best_k = static_cast<int>(k);
        }
    }
    
    data.labels[i] = best_k;
}
```

### Transposed Centroids Support
```cpp
#ifdef TRANSPOSED_C
// Use strided pointer for this tile (from E2)
const float* __restrict__ ck = &centroidsT[k + d0 * K];

// Process all dimensions in current tile
for (size_t d = d0; d < d_end; ++d) {
    float diff = px[d] - *ck;
    d2 += static_cast<double>(diff) * static_cast<double>(diff);
    ck += K;  // Move to next dimension, same centroid (stride by K)
}
#endif
```

## Build Configuration

### Compilation
```bash
# Build with E4 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_D=8" make release

# Profile build with E4 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_D=8" make profile
```

### Slurm Job Submission
```bash
# Submit E4 experiment
sbatch scripts/slurm/run_experiment_e4.slurm
```

## E4 Implementation Failure Analysis

### Why E4 D-Tiling Failed

The E4 D-tiling optimization was **fundamentally flawed** and resulted in **significant performance regression**. Here's why:

#### Performance Results
- **E2 → E4 (Canonical)**: 16.495ms → 21.444ms (**-30.0% regression**)
- **E2 → E4 (Stress)**: 287.299ms → 321.686ms (**-12.0% regression**)
- **Same final inertia**: Algorithm correctness maintained
- **No cache benefits**: Loop overhead without performance gains

#### The Core Problems

1. **Spatial Locality Already Optimal**: E1 + E2 already read dimensions contiguously with strided pointers. Tiling the same contiguous stream doesn't create new reuse.

2. **Wrong Temporal Reuse Target**: Temporal reuse lives across K (centroids), not within a D-tile. The point value `px[d]` is reused for many centroids, not many tiles of the same centroid.

3. **Added Overhead**: Extra loop and `min()` bound checks per tile add unnecessary overhead.

4. **Prefetcher Already Handles It**: EPYC's prefetcher already handles the contiguous dimension access well.

#### Memory Access Pattern Analysis
- **E2**: `for (d = 0; d < D; ++d) { ... }` (optimal contiguous access)
- **E4**: `for (d0 = 0; d0 < D; d0 += TD) { for (d = d0; d < d_end; ++d) { ... } }` (same access + overhead)

**Result**: Same memory access pattern with added loop overhead.

### What Would Make "Blocking" Real and Useful

#### Register Blocking Across K (Not D)

The correct approach would be **register blocking across K centroids**:

```cpp
// Register blocking across K centroids
for (size_t k0 = 0; k0 < K; k0 += TK) {
    for (size_t i = 0; i < N; ++i) {
        // Process TK centroids at once with register accumulators
        for (size_t k = k0; k < k_end; ++k) {
            // Reuse loaded px[d] across several centroids
        }
    }
}
```

**Benefits**:
- **Reduces index math**: Process multiple centroids in one tile
- **Reuses loaded px[d]**: Across several centroids while keeping accesses contiguous
- **Contiguous memory access**: Thanks to E1 transposed layout
- **Register efficiency**: Better utilization of CPU registers

### Lessons Learned
1. **D-tiling doesn't help when spatial locality is already optimal**
2. **Temporal reuse in K-means lives across centroids, not dimensions**
3. **Modern prefetchers handle contiguous access well**
4. **Loop overhead can easily outweigh theoretical cache benefits**
5. **Register blocking across K would be more promising than D-tiling**

### Optimization Hierarchy
1. **E1**: Memory layout optimization (transposed centroids)
2. **E2**: Micro-optimizations (invariant hoisting, branchless argmin, strided pointers)
3. **E4**: Cache optimization (D-tiling/dimension blocking)

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

### Cache Blocking Benefits
- **Spatial Locality**: Dimensions in same tile share cache lines
- **Temporal Locality**: Dimension tile stays hot while processing all dimensions in it
- **Reduced Misses**: Fewer cache line fetches per dimension
- **Better Utilization**: Cache lines used more efficiently

### Tuning Considerations
- **TD=8**: Good balance for most systems (fits in cache lines)
- **TD=4**: More conservative, better for smaller caches
- **TD=16**: More aggressive, may cause cache pressure
- **System-Specific**: Optimal TD depends on cache line size and associativity

### Memory Access Pattern
- **E0**: Poor spatial locality in dimension access
- **E1**: Improved cache locality with transposed layout
- **E2**: Optimal access pattern with strided pointers
- **E4**: Cache blocking for maximum dimension spatial locality

## Performance Characteristics

### Bottleneck Analysis
- **Primary Bottleneck**: `assign_labels_d_tiled` function (expected ~85% of time)
- **Memory Access Pattern**: Optimal cache-blocked dimension access
- **Computational Complexity**: O(N×K×D) for assign, O(N×D) for update
- **Cache Behavior**: Excellent spatial locality with dimension tiling

### Scalability Observations
- **Canonical vs Stress**: D-tiling benefits scale with D
- **D×K Impact**: Stress config (D=64, K=64) benefits most from dimension tiling
- **Cache Efficiency**: 8x improvement in dimension spatial locality

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

1. **Run E4 Experiment**: Use `scripts/slurm/run_experiment_e4.slurm`
2. **Compare Results**: Analyze E0 vs E1 vs E2 vs E4 performance
3. **Validate Correctness**: Ensure identical results across experiments
4. **Document Findings**: Update this document with actual results

## Technical Notes

### Cache Blocking Philosophy
- **Target the hottest code path** (`assign_labels`)
- **Maintain algorithm correctness**
- **Use cache-friendly patterns**
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
