# Experiment Summary Script

## Overview

The `summarize_experiment.sh` script provides comprehensive analysis of benchmark experiment results, automatically generating aggregated statistics and performance summaries.

## Usage

### Basic Usage

```bash
# Generate summary for experiment E0
./scripts/summarize_experiment.sh 0

# Generate summary for experiment E1
./scripts/summariment.sh 1

# Generate summary with custom output file
./scripts/summarize_experiment.sh 0 my_summary.md
```

### What It Does

1. **Scans all benchmark artifacts** in `bench/e{X}/`
2. **Aggregates performance data** across multiple runs
3. **Calculates statistical measures**: median, mean, standard deviation
4. **Analyzes convergence behavior** from inertia data
5. **Summarizes profiling results** (gprof, cachegrind)
6. **Provides complete file inventory** and system information

### Output Format

The script generates a Markdown file (`summary_e{X}.md`) containing:

- **System Information**: Build environment, compiler, Git state
- **Performance Analysis**: Timing statistics for each configuration
- **Convergence Analysis**: Inertia progression and monotonicity
- **Profiling Results**: Function-level and cache performance data
- **File Summary**: Complete inventory of generated artifacts

### Statistical Analysis

For each configuration, the script provides:

- **Timing Statistics**: Median, mean, and standard deviation
- **Kernel Breakdown**: Assign vs. update kernel percentages
- **Run Analysis**: Distinguishes between warm-up and measurement runs
- **Cross-Run Comparison**: Statistical significance across multiple executions

### Example Output

```markdown
### Configuration: times_N200000_D16_K8_iters10_seed1
**Measurement Runs:** 5 (runs 4-8)

**Timing Statistics (milliseconds):**
- Assign Kernel:
  - Median: 77.51623
  - Mean: 77.584
  - Std Dev: 1.624743
- Update Kernel:
  - Median: 9.834630
  - Mean: 9.730
  - Std Dev: .172707
- Total Time:
  - Median: 87.40717
  - Mean: 87.314
  - Std Dev: 1.579459

**Time Distribution:**
- Assign Kernel: 88.7%
- Update Kernel: 11.3%
```

### Dependencies

- **bc calculator**: For statistical calculations (install with `brew install bc` on macOS)
- **Standard Unix tools**: `find`, `grep`, `sed`, `awk`, `sort`

### Integration with Benchmarking

This script is designed to run **after** completing a benchmark experiment:

1. Run your experiment: `./scripts/run_experiment.sh 0`
2. Generate summary: `./scripts/summarize_experiment.sh 0`
3. Review results: `cat bench/e0/summary_e0.md`

### Customization

The script automatically detects:
- Run-based configurations (`_run{N}` suffix)
- Single-run configurations (no suffix)
- Profiling data availability
- System information completeness

### Error Handling

- Gracefully handles missing files
- Reports "N/A" for unavailable statistics
- Provides informative error messages
- Continues processing despite individual file issues

### Academic Use

This summary provides:
- **Reproducibility**: Complete experiment documentation
- **Statistical rigor**: Proper aggregation of multiple runs
- **Performance analysis**: Kernel-level timing breakdown
- **Convergence validation**: Algorithm correctness verification
- **System transparency**: Full build and runtime environment
