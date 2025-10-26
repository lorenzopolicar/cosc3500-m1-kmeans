# Parallel K-Means Design Document

## Overview

This document outlines the parallelization strategy for K-Means clustering, building upon the serial optimizations from Milestone 1.

## Algorithm Analysis

### K-Means Lloyd's Algorithm

1. **Initialize**: Select K initial centroids
2. **Repeat until convergence**:
   - **Assign**: Each point finds its nearest centroid
   - **Update**: Recompute centroids as mean of assigned points

### Parallelization Opportunities

#### Assign Phase (86-91% of runtime)
- **Embarrassingly parallel**: Each point's assignment is independent
- **Compute pattern**: N×K×D distance calculations
- **Memory pattern**: Read-heavy (points and centroids), write minimal (labels only)
- **Parallelization strategy**:
  - Data parallelism over points
  - Each thread/GPU thread processes one or more points
  - No synchronization needed during computation

#### Update Phase (1-2% of runtime)
- **Reduction pattern**: Sum points per cluster, then divide
- **Challenge**: Potential race conditions when accumulating
- **Parallelization strategies**:
  - Atomic operations (simple but potentially slow)
  - Private accumulation + reduction
  - Segmented reduction by cluster

## Baseline Serial Performance (from M1)

Best configuration: **E2 Micro-optimizations**
- Canonical (N=200K, D=16, K=8): 18.604ms
- Stress (N=100K, D=64, K=64): 292.040ms

Key optimizations carried forward:
- Transposed centroids for cache locality
- Branchless minimum finding
- Invariant hoisting
- Strided pointer arithmetic

## Parallel Implementation Strategy

### Phase 1: Direct Parallelization

#### CUDA Approach
```cuda
// Each thread handles one point
__global__ void assign_labels_basic(points, centroids, labels, N, D, K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Find nearest centroid for point[tid]
    labels[tid] = find_nearest(points[tid], centroids, D, K);
}
```

#### OpenMP Approach
```cpp
#pragma omp parallel for
for (int i = 0; i < N; ++i) {
    labels[i] = find_nearest(points[i], centroids, D, K);
}
```

### Phase 2: Memory Optimization

#### CUDA Optimizations
1. **Shared Memory**: Load centroids into shared memory (if K×D fits)
2. **Coalesced Access**: Ensure contiguous threads access contiguous memory
3. **Texture Memory**: Use texture cache for read-only centroids
4. **Constant Memory**: For small K, use constant memory

#### OpenMP Optimizations
1. **NUMA Awareness**: First-touch policy for data initialization
2. **Cache Blocking**: Process points in cache-friendly chunks
3. **False Sharing Avoidance**: Pad data structures
4. **Thread Affinity**: Pin threads to cores

### Phase 3: Advanced Techniques

#### CUDA Advanced
1. **Warp-level Operations**: Use warp shuffles for reductions
2. **Persistent Kernels**: Keep data in registers across iterations
3. **Multi-Stream Execution**: Overlap compute and memory transfers
4. **Mixed Precision**: FP16 for distances, FP32 for accumulation

#### OpenMP Advanced
1. **SIMD Directives**: Explicit vectorization hints
2. **Task Parallelism**: Parallel initialization and I/O
3. **Nested Parallelism**: For very large K
4. **Custom Reduction**: Optimized cluster accumulation

## Memory Requirements

### CUDA Memory Calculation
```
Points:     N × D × 4 bytes
Centroids:  K × D × 4 bytes
Labels:     N × 4 bytes
Workspace:  K × D × 4 bytes (for reduction)

Total GPU:  4 × (N×D + 2×K×D + N) bytes
```

Example for N=1M, D=64, K=64:
- Points: 256 MB
- Centroids: 16 KB
- Labels: 4 MB
- Total: ~260 MB (well within modern GPU capacity)

### Shared Memory Constraints
- A100: 164 KB shared memory per SM
- Maximum centroids in shared: K×D×4 < 164KB
- For D=64: K_max = 640 (more than sufficient)

## Expected Performance

### Target Metrics

#### CUDA (A100 GPU)
- Memory bandwidth: 1555 GB/s theoretical
- Compute: 19.5 TFLOPS (FP32)
- Expected speedup: 50-100x over serial

#### OpenMP (32-core CPU)
- Theoretical speedup: 32x (Amdahl's law limited)
- Practical speedup: 15-25x (memory bandwidth limited)

### Bottleneck Analysis

**Memory Bandwidth Bound**:
- Assign phase: (N×D + K×D) × 4 bytes read per iteration
- At N=1M, D=64, K=64: ~260 MB per iteration
- Need ~26 GB/s for 100 iterations/second

**Compute Bound** (unlikely for standard K-Means):
- N×K×D FLOPs per iteration
- At above config: 4 GFLOPs per iteration
- Modern GPUs: >10 TFLOPS (compute not limiting)

## Implementation Phases

### Week 1: Foundation
1. ✅ Port serial E2 to baseline parallel
2. ✅ Verify correctness against serial
3. ✅ Establish performance baseline
4. □ Profile and identify bottlenecks

### Week 2: Optimization
1. □ Implement memory optimizations
2. □ Add advanced kernel variants
3. □ Tune launch configurations
4. □ Compare different approaches

### Week 3: Analysis
1. □ Scaling studies (strong and weak)
2. □ Cross-platform comparison
3. □ Document best practices
4. □ Prepare presentation

## Risk Mitigation

### Technical Risks
- **GPU Memory Overflow**: Implement chunking for very large N
- **Atomic Contention**: Use hierarchical reduction
- **Divergent Branches**: Ensure coalesced execution
- **Numerical Precision**: Maintain double accumulation

### Performance Risks
- **Low Occupancy**: Tune block sizes dynamically
- **Memory Bottleneck**: Optimize access patterns
- **Launch Overhead**: Batch small problems
- **CPU-GPU Transfer**: Minimize and overlap

## Success Criteria

### Minimum Requirements
- ✅ Functional parallel implementation
- □ 10x speedup over serial
- □ Correct convergence
- □ Handles all test configurations

### Target Goals
- □ 50x+ speedup on large problems
- □ Both CUDA and OpenMP working
- □ Comprehensive scaling analysis
- □ Novel optimization demonstrated

## Key Design Decisions

1. **Start with CUDA**: Greater speedup potential
2. **Preserve E2 optimizations**: Proven 12% improvement
3. **Focus on assign kernel**: 90% of runtime
4. **Use atomic operations initially**: Simplicity over performance
5. **Profile continuously**: Data-driven optimization

## Next Steps

1. Implement baseline CUDA kernel
2. Add timing infrastructure
3. Verify correctness
4. Profile with NSight Compute
5. Identify optimization opportunities

---

*Last Updated: 2025-10-25*