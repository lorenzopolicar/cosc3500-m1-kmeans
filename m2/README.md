# M2 CUDA K-Means Implementation

## Quick Start on Cluster

### 1. Load CUDA Module
```bash
module load cuda/11.1
```

### 2. Build and Test
```bash
cd m2
./run_first_test.sh
```

### 3. Submit SLURM Job (Quick Test)
```bash
sbatch ../m2-scripts/slurm/test_cuda.slurm
```

### 4. Submit Full Benchmark
```bash
sbatch ../m2-scripts/slurm/cuda_baseline.slurm
```

## Files Created

### Core Implementation
- `src/cuda/kmeans_cuda.cu` - CUDA kernels (assign, update, inertia)
- `src/cuda/main_cuda.cu` - Main program with benchmarking
- `src/common/kmeans_common.cpp` - Shared utilities
- `include/kmeans_cuda.cuh` - CUDA headers
- `include/kmeans_common.hpp` - Common interfaces

### Key Features
- **Basic kernel**: Global memory only
- **Shared memory kernel**: Centroids in shared memory (if they fit)
- **Transposed kernel**: Better memory coalescing
- **Atomic updates**: For centroid accumulation
- **Output format**: Same CSV format as M1 for easy comparison

## Expected Performance

Based on M1 baseline (E2):
- Canonical (N=200K, D=16, K=8): 18.6ms → Target <1ms (20x speedup)
- Stress (N=100K, D=64, K=64): 292ms → Target <6ms (50x speedup)

## Troubleshooting

### Build Errors
- Check CUDA module is loaded: `module list`
- Verify CUDA version: `nvcc --version`

### Runtime Errors
- Check GPU availability: `nvidia-smi`
- Verify memory: Start with smaller N

### Performance Issues
- Profile with: `nvprof ./kmeans_cuda ...`
- Check shared memory usage in output
- Verify coalesced access patterns

## Next Steps

1. ✅ Build and test basic CUDA implementation
2. Run benchmarks and collect data
3. Profile with NSight Compute
4. Implement OpenMP version for comparison
5. Optimize based on profiling results

## Output Files

Each run generates:
- `*.csv` - Timing data (iteration, assign_ms, update_ms, total_ms)
- `*_meta.txt` - Configuration and system info
- `*_inertia.txt` - Convergence tracking

Same format as M1 for easy comparison!