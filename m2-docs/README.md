# COSC3500 Milestone 2 - Documentation

This directory contains comprehensive documentation for the parallel K-Means implementation.

## Directory Structure

```
m2-docs/
├── README.md              (this file)
├── design/
│   ├── parallel_design.md     - Parallelization strategy and design decisions
│   ├── cuda_architecture.md   - CUDA implementation details
│   └── openmp_strategy.md     - OpenMP implementation approach
├── experiments/
│   ├── e0_baseline.md         - Baseline parallel implementation
│   ├── e1_memory_opt.md       - Memory optimization experiments
│   ├── e2_computation_opt.md  - Computation optimization experiments
│   ├── e3_streaming.md        - Stream and overlap experiments
│   └── e4_scaling.md          - Scaling analysis
└── profiling/
    ├── cuda_profiling.md      - CUDA profiling with NSight
    ├── openmp_profiling.md    - OpenMP profiling results
    └── bottleneck_analysis.md - Performance bottleneck identification
```

## Documentation Guidelines

### For Each Experiment

1. **Hypothesis**: What we expect to improve and why
2. **Implementation**: Code changes and techniques used
3. **Methodology**: How the experiment was conducted
4. **Results**: Performance measurements and analysis
5. **Reflection**: What worked, what didn't, and why
6. **Next Steps**: How results inform subsequent experiments

### Performance Metrics to Track

- **Execution Time**: Total, assign kernel, update kernel
- **Speedup**: Relative to serial baseline
- **Efficiency**: Speedup / number of processors
- **MLUPS**: Million Label Updates Per Second
- **Memory Bandwidth**: GB/s achieved vs theoretical
- **GPU Occupancy**: For CUDA implementations
- **Scalability**: Strong and weak scaling

## Key Files

### Design Documents

- `parallel_design.md`: Overall parallelization strategy
- `cuda_architecture.md`: CUDA kernel design and optimization
- `openmp_strategy.md`: OpenMP parallelization approach

### Experiment Logs

Track each optimization attempt with:
- Configuration tested
- Performance results
- Code snippets
- Profiling data
- Analysis and conclusions

### Profiling Results

- NSight Compute reports for CUDA
- gprof/perf results for CPU code
- Memory access patterns
- Bottleneck identification

## Workflow

1. **Plan** the experiment in the appropriate design document
2. **Implement** the optimization in code
3. **Benchmark** using the scripts in `m2-scripts/`
4. **Profile** to understand performance characteristics
5. **Document** results in the experiments directory
6. **Analyze** and plan next steps

## Tips for Documentation

- Include relevant code snippets
- Use tables for performance comparisons
- Add graphs/plots when helpful
- Link to raw data in `m2-bench/`
- Be specific about hardware/software configuration
- Document both successes and failures

## References

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [OpenMP Specification](https://www.openmp.org/specifications/)
- [K-Means Clustering Papers](../references/)

---

*Last Updated: 2025-10-25*