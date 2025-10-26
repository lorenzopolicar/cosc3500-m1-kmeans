# COSC3500 Milestone 2 - Parallel K-Means Implementation Plan

## Executive Summary

This document provides a comprehensive plan for extending the Milestone 1 serial K-Means implementation to a parallel version for Milestone 2. Building upon the successful E2 optimization (7.7-12.6% improvement) and conditional E5 success (32.4% on stress configs), we will implement parallel strategies focusing on **CUDA** as the primary technology, with potential **OpenMP** comparison.

## Current State Analysis (M1 Recap)

### Successful Serial Optimizations
- **E0 Baseline**: Reference implementation (20.157ms canonical, 334.078ms stress)
- **E1 Transposed Centroids**: 3-3.6% improvement via memory layout optimization
- **E2 Micro-optimizations**: 7.7-12.6% improvement (best reliable baseline)
- **E5 Register Blocking**: 32.4% improvement on stress configs (conditional success)

### Key Performance Insights
- **Bottleneck**: assign_labels kernel consumes 86-91% of execution time
- **Memory Bound**: Performance limited by memory access patterns, not computation
- **Problem Size Dependent**: Optimizations have different effectiveness at different scales
- **Cache Effects**: Working set size (K×D×4 bytes) critical for performance

### Repository Structure
```
cosc3500-m1-kmeans/          [Existing M1 work]
├── src/                     [Serial implementation]
├── include/                 [Data structures]
├── docs-m1/                 [M1 documentation]
├── bench/                   [M1 benchmark data]
├── plots-m1/                [M1 analysis plots]
├── scripts-m1/              [M1 scripts]
└── m1.md                    [M1 scope document]

[NEW FOR M2]
├── m2/                      [M2 root directory]
│   ├── src/                 [Parallel implementations]
│   │   ├── cuda/           [CUDA kernels]
│   │   ├── openmp/         [OpenMP version]
│   │   └── baseline/       [Serial reference]
│   ├── include/            [Parallel headers]
│   ├── Makefile            [M2 build system]
│   └── README.md           [M2 overview]
├── m2-docs/                [M2 documentation]
│   ├── experiments/        [Per-experiment docs]
│   ├── design.md           [Parallel design doc]
│   ├── implementation.md   [Technical details]
│   └── results.md          [Performance analysis]
├── m2-bench/               [M2 benchmark data]
│   ├── cuda/               [CUDA results]
│   ├── openmp/             [OpenMP results]
│   └── comparison/         [Cross-platform]
├── m2-plots/               [M2 visualizations]
├── m2-scripts/             [M2 automation]
│   ├── benchmark/          [Benchmark runners]
│   ├── slurm/              [HPC job scripts]
│   └── analysis/           [Analysis tools]
└── m2.md                   [M2 scope document]
```

## Milestone 2 Strategy

### Primary Approach: CUDA Implementation

#### Why CUDA First?
1. **Order of Magnitude Speedup**: GPUs typically provide 10-100x speedup for data-parallel workloads
2. **Natural Fit**: K-Means is embarrassingly parallel in the assignment phase
3. **Available Hardware**: UQ clusters have A100 GPUs (rangpur nodes)
4. **Clear Parallelization**: Each point's nearest centroid can be computed independently

#### CUDA Implementation Phases

**Phase 1: Basic CUDA Port (Week 1)**
- Port E2 optimized serial code to CUDA
- One thread per point for assign_labels
- Shared memory for centroids (if K×D fits)
- Basic reduction for update_centroids
- Goal: Functional GPU implementation with >10x speedup

**Phase 2: Optimization (Week 2)**
- Coalesced memory access patterns
- Warp-level primitives for reductions
- Texture memory for centroids (read-only cache)
- Multiple kernels vs fused kernels comparison
- Goal: >50x speedup over serial baseline

**Phase 3: Advanced Techniques (Week 3)**
- Multi-GPU support (if available)
- Overlapped compute/transfer with streams
- Mixed precision investigation (FP16 distances)
- Dynamic parallelism for adaptive clustering
- Goal: Push limits of single-node performance

### Secondary Approach: OpenMP Comparison

#### OpenMP Implementation (Week 2-3)
- Parallel for over points in assign_labels
- Reduction for centroid updates
- NUMA awareness for memory allocation
- Comparison with CUDA for different problem sizes
- Goal: Understand CPU vs GPU trade-offs

### Hybrid Approach (Optional, Week 4)
- CPU preprocessing (data loading/validation)
- GPU main computation
- CPU post-processing (convergence check)
- Asynchronous execution overlap

## Technical Implementation Details

### CUDA Kernel Design

#### Assign Labels Kernel
```cuda
__global__ void assign_labels_cuda(
    const float* points,      // N×D
    const float* centroids,   // K×D or D×K (transposed)
    int* labels,              // N
    int N, int D, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Load point to registers/shared memory
    // Compute distances to all centroids
    // Find minimum and update label
}
```

#### Update Centroids Kernel
```cuda
__global__ void update_centroids_cuda(
    const float* points,      // N×D
    const int* labels,        // N
    float* centroids,         // K×D
    int* counts,              // K
    int N, int D, int K
) {
    // Parallel reduction per centroid
    // Atomic operations for accumulation
    // Or segmented reduction approach
}
```

### Memory Management Strategy

#### Device Memory Layout
- **Points**: Keep in global memory (too large for shared)
- **Centroids**: Load to shared memory per block (if K×D < 48KB)
- **Labels**: Global memory with coalesced access
- **Workspace**: Pre-allocated buffers for reductions

#### Transfer Optimization
- Pinned host memory for faster transfers
- Asynchronous copies with streams
- Minimize host-device communication
- Keep data on GPU across iterations

### Parallelization Patterns

#### Data Parallelism (Primary)
- Each thread processes one point
- Block size: 256-512 threads (tunable)
- Grid size: (N + blockSize - 1) / blockSize

#### Task Parallelism (Secondary)
- Stream 1: Assign labels
- Stream 2: Update centroids
- Stream 3: Compute inertia
- Overlap where dependencies allow

## Benchmarking Plan

### Test Configurations

#### Small Scale (CPU competitive)
- N=10,000, D=16, K=8
- N=50,000, D=32, K=16

#### Medium Scale (GPU advantage)
- N=100,000, D=64, K=32
- N=500,000, D=32, K=64

#### Large Scale (GPU dominant)
- N=1,000,000, D=64, K=64
- N=5,000,000, D=128, K=128

#### Stress Test (memory limits)
- N=10,000,000, D=256, K=256
- Test out-of-core algorithms if needed

### Performance Metrics

#### Primary Metrics
- **Speedup**: Time(serial) / Time(parallel)
- **Efficiency**: Speedup / Number_of_processors
- **Throughput**: MLUPS (Million Label Updates Per Second)
- **Scalability**: Strong and weak scaling analysis

#### Secondary Metrics
- **Memory Bandwidth Utilization**: GB/s achieved vs theoretical
- **Occupancy**: Active warps / maximum warps
- **Transfer Overhead**: Data movement time / compute time
- **Energy Efficiency**: Joules per iteration (if measurable)

### Benchmark Protocol

1. **Warm-up Phase**: 3 iterations to stabilize GPU clocks
2. **Measurement Phase**: 10 iterations, report median
3. **Validation**: Compare inertia with serial implementation
4. **Profiling**: NSight Compute for CUDA, VTune for OpenMP
5. **Statistical Analysis**: Error bars, confidence intervals

## Experimental Design

### Experiment Structure

#### E0: Baseline Parallel
- Direct port of serial E2 to CUDA/OpenMP
- No optimizations beyond basic parallelization
- Establish parallel baseline performance

#### E1: Memory Optimization
- Coalesced access patterns
- Shared memory for centroids
- Texture/constant memory evaluation

#### E2: Computation Optimization
- Warp-level primitives
- Fast math intrinsics
- Mixed precision exploration

#### E3: Overlap and Streams
- Multi-stream execution
- Compute-transfer overlap
- Kernel fusion vs separation

#### E4: Algorithm Variants
- Mini-batch K-Means
- Elkan's algorithm acceleration
- Approximate nearest neighbor

#### E5: Scaling Analysis
- Strong scaling (fixed problem, vary resources)
- Weak scaling (scale problem with resources)
- Amdahl's law validation
- Gustafson's law analysis

### Documentation Strategy

For each experiment:
1. **Hypothesis**: What we expect to improve and why
2. **Implementation**: Code changes and techniques used
3. **Results**: Performance measurements and analysis
4. **Reflection**: What worked, what didn't, and why
5. **Next Steps**: How results inform subsequent experiments

## Development Workflow

### Phase 1: Setup and Baseline (Days 1-3)
- [ ] Create M2 directory structure
- [ ] Port E2 serial code to m2/src/baseline
- [ ] Implement basic CUDA assign_labels kernel
- [ ] Implement basic CUDA update_centroids kernel
- [ ] Verify correctness against serial implementation
- [ ] Document initial results in m2-docs/experiments/e0_baseline.md

### Phase 2: Core Optimization (Days 4-8)
- [ ] Implement shared memory optimization
- [ ] Optimize memory access patterns
- [ ] Implement efficient reductions
- [ ] Profile and identify bottlenecks
- [ ] Document each optimization in m2-docs/experiments/

### Phase 3: Advanced Features (Days 9-12)
- [ ] Implement OpenMP version for comparison
- [ ] Explore advanced CUDA features
- [ ] Conduct scaling studies
- [ ] Generate comprehensive plots
- [ ] Complete performance analysis

### Phase 4: Documentation and Video (Days 13-14)
- [ ] Consolidate all results
- [ ] Create presentation slides
- [ ] Record 10-minute video
- [ ] Final code cleanup
- [ ] Submit deliverables

## Risk Mitigation

### Technical Risks
1. **GPU Memory Limits**: Pre-calculate memory requirements, implement chunking if needed
2. **Numerical Precision**: Maintain double precision for accumulation, validate results
3. **Load Imbalance**: Consider dynamic scheduling for OpenMP, investigate point redistribution
4. **Convergence Issues**: Implement deterministic parallel reduction, careful with atomics

### Schedule Risks
1. **Cluster Access**: Submit jobs early, have local fallback
2. **Debugging Time**: Budget 2x time for parallel debugging
3. **Performance Tuning**: Set minimum acceptable speedup early
4. **Documentation**: Write as you go, not at the end

## Success Criteria

### Minimum Viable Product
- [ ] Functional CUDA implementation
- [ ] 10x speedup over serial baseline
- [ ] Correct convergence (matches serial inertia)
- [ ] Clean, documented code
- [ ] Basic performance analysis

### Target Goals
- [ ] 50x+ speedup on large problems
- [ ] Both CUDA and OpenMP implementations
- [ ] Comprehensive scaling analysis
- [ ] Advanced optimization techniques
- [ ] Professional presentation quality

### Stretch Goals
- [ ] Multi-GPU implementation
- [ ] Hybrid CPU-GPU approach
- [ ] Novel optimization technique
- [ ] Publication-quality results
- [ ] Theoretical performance model

## Key Commands and Scripts

### Building
```bash
# CUDA version
cd m2
make cuda

# OpenMP version
make openmp

# All versions
make all

# With specific compute capability
make cuda SM=80  # for A100
```

### Benchmarking
```bash
# Run CUDA benchmarks
./m2-scripts/benchmark/run_cuda_bench.sh N D K iters

# Run OpenMP benchmarks
./m2-scripts/benchmark/run_openmp_bench.sh N D K iters threads

# Run comparison suite
./m2-scripts/benchmark/run_comparison.sh

# Submit to Slurm
sbatch m2-scripts/slurm/cuda_experiment.slurm
```

### Analysis
```bash
# Generate plots for experiment
python m2-scripts/analysis/plot_experiment.py cuda e1

# Compare implementations
python m2-scripts/analysis/compare_implementations.py

# Generate scaling plots
python m2-scripts/analysis/plot_scaling.py
```

### Profiling
```bash
# CUDA profiling with NSight
nsys profile --stats=true ./kmeans_cuda ...

# OpenMP profiling with VTune
vtune -collect hotspots ./kmeans_openmp ...

# Memory analysis
cuda-memcheck ./kmeans_cuda ...
```

## References and Resources

### CUDA Resources
- CUDA Programming Guide
- CUDA Best Practices Guide
- NSight Compute Documentation
- CUB Library for reductions

### OpenMP Resources
- OpenMP 5.0 Specification
- Intel OpenMP Developer Guide
- NUMA Optimization Guide

### K-Means Specific
- "Scalable K-Means++" (2012)
- "Using the Triangle Inequality to Accelerate K-Means" (Elkan, 2003)
- "Web-Scale K-Means Clustering" (Sculley, 2010)

### UQ HPC Documentation
- Rangpur User Guide
- Slurm on Rangpur
- GPU Computing on HPC

## Checklist for Success

### Pre-Development
- [x] Analyze M1 results thoroughly
- [x] Design parallel strategy
- [x] Create directory structure plan
- [ ] Set up development environment
- [ ] Test CUDA toolchain on cluster

### During Development
- [ ] Maintain git history with meaningful commits
- [ ] Document each experiment immediately
- [ ] Profile early and often
- [ ] Validate correctness continuously
- [ ] Keep benchmark data organized

### Pre-Submission
- [ ] Code review and cleanup
- [ ] Ensure reproducibility
- [ ] Complete all documentation
- [ ] Generate all plots
- [ ] Practice presentation
- [ ] Record and edit video

### Submission
- [ ] Video under 10 minutes
- [ ] Face visible throughout
- [ ] All code in repository
- [ ] Results reproducible
- [ ] Ready for interview

## Final Notes

This plan builds upon the solid foundation of M1, leveraging the insights gained from serial optimization to inform parallel strategy. The focus on CUDA with OpenMP comparison provides both depth (single technology mastery) and breadth (technology comparison) as recommended for high marks.

Key success factors:
1. Start simple, optimize incrementally
2. Measure everything, assume nothing
3. Document as you go
4. Validate correctness at every step
5. Focus on demonstrated understanding over raw performance

Remember: The goal is not just speed, but understanding and communicating the principles of parallel computing through the lens of K-Means clustering.

---

*Last Updated: 2025-10-25*
*Author: Assistant to Lorenzo Policar*
*Course: COSC3500 High-Performance Computing*