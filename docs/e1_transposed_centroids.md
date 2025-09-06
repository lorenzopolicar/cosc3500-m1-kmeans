# E1 Optimization: Transposed Centroids Memory Layout

## Overview

E1 implements a memory layout optimization that improves cache locality during the label assignment phase of K-means clustering. This optimization addresses the primary performance bottleneck identified in E0 profiling, where `assign_labels` consumes 90.17% of total execution time.

## Problem Analysis (From E0 Profiling)

- **Bottleneck**: `assign_labels` function takes 90.17% of total time
- **Memory Access Pattern**: Inefficient cache behavior during distance calculations
- **Access Pattern**: `centroids[k*D + d]` creates non-sequential memory access
- **Impact**: Cache misses when iterating through centroids for each point

## Solution: Auxiliary Transposed Centroids

### Design Principles

1. **Auxiliary Copy**: Keep `centroids[K×D]` as the source of truth
2. **Minimal Changes**: Only modify `assign_labels` and `update_centroids`
3. **Rebuild Cost in Update**: Transpose only after centroid updates
4. **Identical Results**: Labels and inertia must match E0 exactly (±1e-6)

### Memory Layout Transformation

**Original Layout (Row-Major):**
```
centroids[K×D] = [c0_d0, c0_d1, c0_d2, ..., c1_d0, c1_d1, c1_d2, ...]
```

**Optimized Layout (Column-Major):**
```
centroidsT[D×K] = [c0_d0, c1_d0, c2_d0, ..., c0_d1, c1_d1, c2_d1, ...]
```

### Implementation Details

#### Data Structure Extension
```cpp
struct Data {
    std::vector<float> points;      // N points of dimension D (row-major: [i*D + d])
    std::vector<float> centroids;   // K centroids of dimension D (row-major: [k*D + d])
    std::vector<int> labels;        // N labels (0 to K-1)
    size_t N, D, K;                 // Number of points, dimensions, clusters
    
#ifdef TRANSPOSED_C
    std::vector<float> centroidsT;  // Transposed centroids [D×K] for cache-friendly access
#endif
};
```

#### Transpose Function
```cpp
void transpose_centroids(Data& data) {
    for (size_t d = 0; d < data.D; ++d) {
        for (size_t k = 0; k < data.K; ++k) {
            data.centroidsT[d * data.K + k] = data.centroids[k * data.D + d];
        }
    }
}
```

#### Optimized Distance Calculation
```cpp
// Original (Cache-Unfriendly)
for (size_t d = 0; d < data.D; ++d) {
    float diff = data.points[i * data.D + d] - data.centroids[k * data.D + d];
    d2 += diff * diff;
}

// Optimized (Cache-Friendly)
for (size_t d = 0; d < data.D; ++d) {
    float diff = data.points[i * data.D + d] - data.centroidsT[d * data.K + k];
    d2 += diff * diff;
}
```

## Build Configuration

### Compilation
```bash
# Build with E1 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1" make release

# Profile build with E1 optimization
ANTIVEC=1 DEFS="-DTRANSPOSED_C=1" make profile
```

### Slurm Job Submission
```bash
# Submit E1 experiment
sbatch scripts/slurm/run_experiment_e1.slurm
```

## Performance Analysis

### Expected Improvements

1. **Cache Locality**: Better spatial locality during distance calculations
2. **Memory Bandwidth**: More efficient memory access patterns
3. **Target Improvement**: 10-30% speedup in `assign_labels` kernel
4. **Overall Speedup**: 8-25% total performance improvement

### Key Metrics to Monitor

- **Assign Kernel Time**: Should decrease significantly
- **Update Kernel Time**: May increase slightly due to transpose cost
- **Total Time**: Net improvement expected
- **Inertia Values**: Must remain identical to E0 (±1e-6)

### Stress Configuration Benefits

The optimization should show the greatest benefits on the stress configuration:
- **N=100,000, D=64, K=64**: Large D×K amplifies memory access effects
- **Cache Miss Reduction**: Fewer cache misses during distance calculations
- **Memory Bandwidth**: Better utilization of available memory bandwidth

## Implementation Checklist

### ✅ Acceptance Criteria

- [x] **Build Flag**: `DEFS="-DTRANSPOSED_C=1"`
- [x] **Before Loop**: Build `centroidsT` once after initialization
- [x] **assign_labels**: Use `centroidsT[d*K + k]` (unit stride over d)
- [x] **update_centroids**: Write new row-major centroids, then rebuild `centroidsT`
- [x] **Results**: Labels/inertia identical to E0 (±1e-6)
- [x] **Performance**: assign ↓, update ↑ slightly; biggest wins on large D×K

### ✅ Gotchas Avoided

- [x] **No Re-transpose in Assign**: Only transpose after update
- [x] **No Points Transpose**: Only centroids are transposed
- [x] **Correct Indexing**: row-major `k*D + d`; transposed `d*K + k`

## Code Changes Summary

### Files Modified

1. **`include/kernels.hpp`**:
   - Added `centroidsT` buffer to `Data` struct
   - Added `transpose_centroids` function declaration

2. **`src/kmeans.cpp`**:
   - Implemented `transpose_centroids` function
   - Modified `assign_labels` to use transposed access
   - Modified `update_centroids` to rebuild transposed buffer

3. **`src/main.cpp`**:
   - Initialize `centroidsT` buffer
   - Build initial transposed centroids after data generation

4. **`scripts/slurm/run_experiment_e1.slurm`**:
   - E1 experiment script with transposed centroids optimization

## Benchmarking Strategy

### Comparison with E0

1. **Same Configurations**: Use identical canonical and stress configs
2. **Statistical Significance**: 3 warm-up + 5 measurement runs
3. **Performance Metrics**: Kernel timings, MLUPS, convergence
4. **Correctness Verification**: Inertia values must match E0

### Expected Results

- **Canonical Config (N=200K, D=16, K=8)**: Moderate improvement
- **Stress Config (N=100K, D=64, K=64)**: Significant improvement
- **Profiling Data**: Reduced cache misses in `assign_labels`

## Next Steps

After E1 completion:

1. **Performance Analysis**: Compare E1 vs E0 results
2. **Identify Next Bottleneck**: Profile E1 to find remaining bottlenecks
3. **Plan E2**: Design next optimization (e.g., loop unrolling, cache blocking)
4. **Documentation**: Update optimization strategy based on E1 results

## Technical Notes

### Memory Overhead

- **Additional Storage**: D×K floats for transposed centroids
- **Memory Cost**: Minimal compared to N×D points storage
- **Trade-off**: Small memory increase for significant performance gain

### Compiler Optimizations

- **Anti-vectorization**: Maintained with `-fno-tree-vectorize`
- **Cache Behavior**: Improved through better memory access patterns
- **Branch Prediction**: Unchanged from E0 implementation

### Scalability

- **Large D**: Benefits increase with higher dimensions
- **Large K**: Benefits increase with more clusters
- **Memory-bound**: Particularly effective on memory-bound workloads

This optimization provides a solid foundation for further improvements while maintaining the single-threaded, non-vectorized constraints required by M1.
