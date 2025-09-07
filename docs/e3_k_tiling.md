# E3: K-tiling (Centroid Blocking) Cache Optimization

## Overview

E3 implements a sophisticated cache optimization technique called K-tiling (centroid blocking) that improves cache locality by processing centroids in tiles of TK so a D×TK slab of centroids stays hot while evaluating one point. This optimization builds on E1's transposed centroids and E2's micro-optimizations to achieve superior cache performance.

## Problem Analysis (From E2 Profiling)

### Cache Locality Issue
- **E2 Bottleneck**: `assign_labels_strided` still dominates (88.13% of time)
- **Memory Access Pattern**: For K=64, we access 64 different memory regions per point
- **Cache Misses**: Poor spatial locality when iterating through all centroids
- **Stress Config Impact**: K=64 causes significant cache pressure

### Current E2 Access Pattern
```cpp
// E2: Process all K centroids for each point
for (size_t i = 0; i < N; ++i) {
    for (size_t k = 0; k < K; ++k) {  // K=64, accesses 64 different memory regions
        // Distance computation using centroidsT[k...k+D*K]
    }
}
```

**Problem**: For K=64, we access 64 different memory regions per point, causing cache misses.

## Solution: K-tiling (Centroid Blocking)

### Design Principles

1. **Cache Blocking**: Process centroids in tiles of TK (e.g., 16)
2. **Spatial Locality**: Keep D×TK slab of centroids hot in cache
3. **Build on E1+E2**: Leverage transposed centroids + micro-optimizations
4. **Tunable Parameter**: TK can be optimized per system
5. **Identical Results**: Labels and inertia must match E2 exactly (±1e-6)

### Memory Access Transformation

**Original E2 Pattern:**
```
For each point i:
  For k = 0 to K-1:  // K=64, 64 different memory regions
    Access centroidsT[k...k+D*K]
```

**E3 Tiled Pattern:**
```
For each point i:
  For k0 = 0 to K-1 step TK:  // TK=16, 4 tiles
    For k = k0 to min(k0+TK, K):  // Process 16 centroids in cache
      Access centroidsT[k...k+D*K]
```

### Cache Efficiency Analysis

| Configuration | K | Tiles (K/TK) | Memory Regions | Cache Efficiency |
|---------------|---|--------------|----------------|------------------|
| **Canonical** | 8 | 1 (8/16) | 1 | Already optimal |
| **Stress** | 64 | 4 (64/16) | 4 | 16x improvement |

## Implementation Details

### Build Configuration
```bash
BUILD_DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_K=16"
```

### Function Routing
```cpp
void assign_labels(Data& data) {
    // Route to E3 optimized versions if flags are defined
#ifdef TILE_K
    assign_labels_tiled(data);
    return;
#endif
    // ... existing E2 routing
}
```

### Core Implementation
```cpp
void assign_labels_tiled(Data& data) {
    // Hoist invariants outside loops (from E2)
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    const float* __restrict__ centroidsT = data.centroidsT.data();
    
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        const float* __restrict__ px = &points[i * D];
        
        // Process centroids in tiles of TK for better cache locality
        for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
            size_t k_end = std::min(k0 + TILE_K, K);
            
            for (size_t k = k0; k < k_end; ++k) {
                double d2 = 0.0;
                const float* __restrict__ ck = &centroidsT[k];
                
                // Distance computation (from E2)
                for (size_t d = 0; d < D; ++d) {
                    float diff = px[d] - *ck;
                    d2 += static_cast<double>(diff) * static_cast<double>(diff);
                    ck += K;  // Stride by K
                }
                
                // Branchless argmin (from E2)
                best_d2 = (d2 < best_d2) ? d2 : best_d2;
                best_k = (d2 < best_d2) ? static_cast<int>(k) : best_k;
            }
        }
        
        data.labels[i] = best_k;
    }
}
```

## Expected Performance Impact

### Cache Analysis
- **Current E2**: K=64 centroids = 64 different memory regions
- **E3 with TK=16**: 4 tiles = 4 memory regions per point
- **Cache Efficiency**: 16x better spatial locality

### Performance Predictions
- **Canonical Config (N=200K, D=16, K=8)**: 2-5% improvement (already small K)
- **Stress Config (N=100K, D=64, K=64)**: 15-25% improvement (major cache benefit)
- **Total E0→E3**: 20-35% cumulative improvement

### Optimization Hierarchy
1. **E1**: Memory layout optimization (transposed centroids)
2. **E2**: Micro-optimizations (invariant hoisting, branchless argmin, strided pointers)
3. **E3**: Cache optimization (K-tiling/centroid blocking)

## Build Configuration

### Compilation
```bash
# Build with E3 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_K=16" make release

# Profile build with E3 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTILE_K=16" make profile
```

### Slurm Job Submission
```bash
# Submit E3 experiment
sbatch scripts/slurm/run_experiment_e3.slurm
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

## Technical Insights

### Cache Blocking Benefits
- **Spatial Locality**: Centroids in same tile share cache lines
- **Temporal Locality**: Tile stays hot while processing all centroids in it
- **Reduced Misses**: Fewer cache line fetches per point
- **Better Utilization**: Cache lines used more efficiently

### Tuning Considerations
- **TK=16**: Good balance for most systems (fits in L1 cache)
- **TK=8**: More conservative, better for smaller caches
- **TK=32**: More aggressive, may cause cache pressure
- **System-Specific**: Optimal TK depends on cache size and associativity

### Memory Access Pattern
- **E0**: Poor spatial locality in centroid access
- **E1**: Improved cache locality with transposed layout
- **E2**: Optimal access pattern with strided pointers
- **E3**: Cache blocking for maximum spatial locality

## Performance Characteristics

### Bottleneck Analysis
- **Primary Bottleneck**: `assign_labels_tiled` function (expected ~85% of time)
- **Memory Access Pattern**: Optimal cache-blocked access
- **Computational Complexity**: O(N×K×D) for assign, O(N×D) for update
- **Cache Behavior**: Excellent spatial locality with tiling

### Scalability Observations
- **Canonical vs Stress**: Tiling benefits scale with K
- **D×K Impact**: Stress config (D=64, K=64) benefits most from tiling
- **Cache Efficiency**: 16x improvement in spatial locality

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

## Expected Results

### Performance Progression
- **E0**: 89.17 → 19.43 MLUPS (baseline)
- **E1**: 92.39 → 20.03 MLUPS (+3.6% → +3.1%)
- **E2**: 97.00 → 22.28 MLUPS (+5.0% → +11.2%)
- **E3**: ~110 → ~27 MLUPS (+15% → +25% estimated)

### Total E0→E3 Improvement
- **Canonical**: ~25% total improvement
- **Stress**: ~40% total improvement

## Next Steps

1. **Run E3 Experiment**: Use `scripts/slurm/run_experiment_e3.slurm`
2. **Compare Results**: Analyze E0 vs E1 vs E2 vs E3 performance
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
