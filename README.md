# COSC3500 Milestone 1 — K-Means (Lloyd's)

**Scope:** Single-threaded C++17 baseline (no SIMD, no threads/OpenMP, no CUDA).  
**Focus:** Serial optimisation of two kernels — `assign` (N×K×D distance loop) and `update` (K×D reductions).  
**Benchmarking:** UQ cluster (Rangpur), fixed node type via Slurm `--constraint`, warm-ups + repeats, median times, and **kernel-split** timing. I/O excluded.

## Plan (high level)
- **E0 Baseline:** Correct, timed split of assign/update; reproducible datasets; inertia per iteration.
- **E1–E8 Optimisations:** centroid transpose (contiguous access), branchless argmin, hoist invariants, accumulator precision policy (`double` sums), K-tiling (cache), two-pass update (streaming writes), norms trick (optional), AoS vs SoA comparison. Each step: hypothesise → change → re-measure (time + cachegrind) → log.
- **Benchmark Matrix:** N∈{1e5, 5e5, 1e6}, D∈{16, 32, 64}, K∈{8, 32, 64}, iters=20; 1 warm-up + ≥5 repeats → median.
- **Profilers:** gprof; valgrind/cachegrind (report misses per label-update).  
- **M2 outlook (not for M1 implementation):** data-parallel over points + per-iteration all-reduce of K×D sums + K counts; broadcast centroids.

## Guardrails
- ❌ No `<immintrin.h>`, `#pragma omp`, `<execution>`, `std::thread`, CUDA, or BLAS/Eigen in hot loops.
- ✅ Prefer `-O2` for primary results (optionally add anti-autovec flags; document in report).
- ✅ Exclude data generation and I/O from timings.
- ✅ Keep an AI tooling log (`docs/ai_log.md`).

## Quick notes
- Build system, profiling scripts, and Slurm job will be added in later steps.
