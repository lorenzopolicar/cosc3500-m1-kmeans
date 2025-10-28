# M2 - Quick Reference Card

## 🚀 What to Run on Cluster

### Step 1: Test OpenMP Build
```bash
cd ~/cosc3500/milestone-1/cosc3500-m1-kmeans/m2
make openmp
./kmeans_openmp -N 1000 -D 8 -K 4 -I 5 -S 42 -T 4
```

### Step 2: Submit OpenMP Benchmark
```bash
cd ~/cosc3500/milestone-1/cosc3500-m1-kmeans
sbatch m2-scripts/slurm/openmp_baseline.slurm
```

### Step 3: Check Status
```bash
squeue -u $USER
# Wait ~15 minutes
cat openmp_baseline_*.out
```

---

## 📊 Current Results

### CUDA (Done ✅)
- **Canonical**: 9.6x speedup
- **Stress**: 76.7x speedup (⭐ Highlight this!)
- **Large**: ~70x speedup

### OpenMP (To Get)
- Expected: 12-16x speedup with 16 threads
- Will enable CPU vs GPU comparison

---

## 📁 Key Files

### Documentation
- **Full CUDA docs**: `m2-docs/cuda_implementation_complete.md`
- **Performance results**: `m2-docs/experiments/e0_cuda_baseline_results.md`
- **Implementation summary**: `M2_IMPLEMENTATION_SUMMARY.md`
- **Execution guide**: `m2/READY_TO_RUN.md`

### Code
- **CUDA**: `m2/src/cuda/`
- **OpenMP**: `m2/src/openmp/`
- **Common**: `m2/src/common/`

### Results
- **CUDA benchmarks**: `m2-bench/cuda/baseline_20251026_184044/`
- **OpenMP benchmarks**: `m2-bench/openmp/` (after running)

---

## 🎯 For Presentation

### Key Points
1. **Technology comparison**: CUDA (76.7x) vs OpenMP (~15-20x)
2. **Optimization techniques**: Shared memory, thread-private buffers
3. **Scaling analysis**: Amdahl's and Gustafson's laws
4. **Correctness**: Verified with inertia convergence

### Figures Needed
1. Speedup bar chart (Serial, OpenMP, CUDA)
2. OpenMP scaling curve (1-32 threads)
3. Kernel breakdown (assign vs update time)

---

## ⚠️ Common Issues

**Build fails**: Check `module load compiler-rt/latest`
**SLURM fails**: Time limit too high (use ≤15 min)
**Wrong results**: Check seed consistency, inertia monotonic

---

## ✅ Checklist

- [x] CUDA implemented and verified
- [x] OpenMP implemented
- [x] Build system working
- [x] SLURM scripts fixed
- [x] Documentation complete
- [ ] **TO DO**: Run OpenMP benchmarks
- [ ] **TO DO**: Create presentation
- [ ] **TO DO**: Record video