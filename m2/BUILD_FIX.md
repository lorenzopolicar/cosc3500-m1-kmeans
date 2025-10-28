# Build Fix Applied

## Issue 1: OpenMP Build Failure
OpenMP build was failing with "undefined reference to `main`" because object files weren't being compiled.

### Root Cause
The C++ compilation rule in the Makefile didn't have OpenMP flags set, so the `.cpp` files weren't being compiled into `.o` object files before linking.

### Fix Applied
Updated Makefile line 194 to include OpenMP flags in the C++ compilation rule:
```makefile
$(CXX) $(CXXFLAGS_BASE) $(OPT_FLAGS) $(OPENMP_FLAGS) $(INCLUDES) -c $< -o $@
```

## Issue 2: SLURM CPU Count Error
SLURM job submission was failing with "CPU count per node can not be satisfied" error.

### Root Cause
The SLURM scripts were requesting 32 CPUs (`--cpus-per-task=32`), but the cluster nodes don't have that many cores available.

### Fix Applied
Updated all SLURM scripts to request only 16 CPUs:
- `m2-scripts/slurm/openmp_baseline.slurm`: Changed from 32 to 16 CPUs
- `m2-scripts/slurm/comparison_cpu_gpu.slurm`: Changed from 32 to 16 CPUs
- Thread scaling tests now use: 1, 2, 4, 8, 16 (instead of up to 32)

## Verification
After these fixes, the build and job submission should work:
```bash
# Build
make clean
make openmp

# Submit job
sbatch m2-scripts/slurm/openmp_baseline.slurm
```

## Status
✅ All fixes applied and ready to use on cluster