# COSC3500 M1 K-Means Plotting System

## Overview

The plotting system provides automated generation of publication-quality visualizations for K-Means benchmark experiments. Built with Python, matplotlib, and pandas, it automatically analyzes benchmark data and produces comprehensive performance and convergence plots.

## Architecture

### Core Components

1. **Analysis Script**: `scripts/analyze_experiment.py`
2. **Data Processing**: Automatic CSV parsing and statistical aggregation
3. **Visualization Engine**: matplotlib-based plotting with professional styling
4. **Output Management**: High-resolution PNG files in `plots/` directory

### Dependencies

```bash
# Core requirements
pandas>=1.3.0      # Data manipulation and analysis
matplotlib>=3.5.0  # Plotting and visualization
numpy>=1.21.0      # Numerical computing
pathlib2>=2.3.0    # Path handling (Python < 3.4)
```

## Installation

### Virtual Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Verification

```bash
# Test installation
python scripts/analyze_experiment.py --help
```

## Usage

### Basic Commands

```bash
# Analyze entire experiment
python scripts/analyze_experiment.py 0

# Analyze specific configuration
python scripts/analyze_experiment.py 0 "N200000_D16_K8_iters10_seed1"

# Generate plots only (skip summary report)
python scripts/analyze_experiment.py 0 --plots-only
```

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `experiment` | Experiment number (required) | `0`, `1`, `2` |
| `tag` | Configuration tag (optional) | `"N200000_D16_K8_iters10_seed1"` |
| `--plots-only` | Skip summary report | `--plots-only` |

## Generated Plots

### 1. Per-Iteration Kernel Timings

**File Pattern**: `{experiment}_periter_timings_{tag}.png`

**Content**:
- **Top Panel**: Individual kernel timings (assign/update) with median lines
- **Bottom Panel**: Total time per iteration
- **Features**: Transparent individual runs, solid median lines, grid overlay

**Use Case**: Analyze iteration-by-iteration performance patterns and identify convergence behavior.

**Example Output**:
```
plots/e0_periter_timings_N200000_D16_K8_iters10_seed1.png
```

### 2. Per-Iteration MLUPS

**File Pattern**: `{experiment}_periter_mlups_{tag}.png`

**Content**:
- **Y-axis**: MLUPS (Million Label Updates Per Second)
- **X-axis**: Iteration number
- **Features**: Individual runs (transparent), median line (solid), configuration parameters

**Calculation**: `MLUPS = (N × K) / (assign_ms / 1000) / 1e6`

**Use Case**: Performance analysis and optimization effectiveness measurement.

**Example Output**:
```
plots/e0_periter_mlups_N200000_D16_K8_iters10_seed1.png
```

### 3. Convergence Analysis

**File Pattern**: `{experiment}_inertia_{tag}.png`

**Content**:
- **Y-axis**: Inertia (log scale)
- **X-axis**: Iteration number
- **Features**: Individual runs, median line, convergence metrics box
- **Metrics**: Initial/final inertia, improvement percentage

**Use Case**: Validate algorithm correctness and measure convergence quality.

**Example Output**:
```
plots/e0_inertia_N200000_D16_K8_iters10_seed1.png
```

### 4. Median Kernel Costs

**File Pattern**: `{experiment}_median_kernels_{tag}.png`

**Content**:
- **Left Panel**: Per-iteration kernel breakdown (bar chart)
- **Right Panel**: Overall median comparison
- **Features**: Side-by-side bars, value labels, configuration info

**Use Case**: Kernel performance comparison and bottleneck identification.

**Example Output**:
```
plots/e0_median_kernels_N200000_D16_K8_iters10_seed1.png
```

## Data Processing

### Automatic Detection

The script automatically:

1. **Scans Experiment Directory**: Finds all `times_*.csv` and `inertia_*.csv` files
2. **Groups by Configuration**: Identifies unique parameter combinations
3. **Separates Runs**: Distinguishes warm-up (runs 1-3) from measurement (runs 4-8)
4. **Extracts Parameters**: Parses N, D, K, iters from filenames

### Statistical Aggregation

For each configuration:

- **Median Calculation**: Per-iteration median across measurement runs
- **Performance Metrics**: MLUPS, time distribution percentages
- **Convergence Analysis**: Inertia improvement and monotonicity
- **Error Handling**: Graceful degradation for missing or corrupted data

### File Naming Convention

```
times_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.csv
inertia_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.csv
```

**Example**:
```
times_N200000_D16_K8_iters10_seed1_run4.csv
inertia_N200000_D16_K8_iters10_seed1_run4.csv
```

## Plot Customization

### Matplotlib Styling

```python
# Default style configuration
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
```

### Customization Options

To modify plot appearance, edit `scripts/analyze_experiment.py`:

```python
# Change figure size
plt.rcParams['figure.figsize'] = (12, 8)

# Modify colors
ax.plot(data, color='red', linewidth=3)

# Adjust grid
ax.grid(True, alpha=0.5, linestyle='--')

# Custom fonts
plt.rcParams['font.family'] = 'serif'
```

## Output Management

### Directory Structure

```
plots/
├── e0_periter_timings_N200000_D16_K8_iters10_seed1.png
├── e0_periter_mlups_N200000_D16_K8_iters10_seed1.png
├── e0_inertia_N200000_D16_K8_iters10_seed1.png
├── e0_median_kernels_N200000_D16_K8_iters10_seed1.png
├── e0_periter_timings_N100000_D64_K64_iters10_seed1.png
├── e0_periter_mlups_N100000_D64_K64_iters10_seed1.png
├── e0_inertia_N100000_D64_K64_iters10_seed1.png
└── e0_median_kernels_N100000_D64_K64_iters10_seed1.png
```

### File Specifications

- **Format**: PNG (Portable Network Graphics)
- **Resolution**: 300 DPI (print quality)
- **Color Space**: RGB
- **Compression**: Lossless

## Analysis Workflow

### Complete Experiment Analysis

```bash
# 1. Run benchmark experiment
./scripts/run_experiment.sh 0

# 2. Generate all visualizations
python scripts/analyze_experiment.py 0

# 3. Review generated plots
ls -la plots/
```

### Selective Analysis

```bash
# Analyze only canonical configuration
python scripts/analyze_experiment.py 0 "N200000_D16_K8_iters10_seed1"

# Generate plots for specific optimization
python scripts/analyze_experiment.py 1 "N200000_D16_K8_iters10_seed1"
```

### Batch Processing

```bash
# Analyze multiple experiments
for exp in 0 1 2 3; do
    python scripts/analyze_experiment.py $exp
done
```

## Performance Metrics

### MLUPS Calculation

**Formula**: `MLUPS = (N × K) / (assign_ms / 1000) / 1e6`

**Components**:
- **N**: Number of data points
- **K**: Number of clusters
- **assign_ms**: Assignment kernel time in milliseconds
- **1e6**: Conversion to millions

**Example**:
```
N=200,000, K=8, assign_ms=7.680
MLUPS = (200,000 × 8) / (7.680 / 1000) / 1,000,000 = 208.33
```

### Time Distribution

**Calculation**: `percentage = (kernel_time / total_time) × 100`

**Example**:
```
assign_ms = 7.680, total_ms = 8.652
assign_pct = (7.680 / 8.652) × 100 = 88.8%
```

## Error Handling

### Graceful Degradation

The script handles various error conditions:

1. **Missing Files**: Continues processing available data
2. **Corrupted Data**: Skips problematic files with warnings
3. **Invalid Parameters**: Reports parsing errors and continues
4. **Empty Datasets**: Provides informative error messages

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No plots generated | Missing dependencies | Install requirements.txt |
| Empty plots | No measurement runs | Check run numbering (4-8) |
| Parameter parsing errors | Filename format | Verify naming convention |
| Memory errors | Large datasets | Reduce problem size for testing |

## Integration with Benchmarking

### Workflow Integration

The plotting system integrates seamlessly with the benchmarking infrastructure:

1. **Data Generation**: `run_experiment.sh` creates CSV files
2. **Analysis**: `analyze_experiment.py` processes and visualizes
3. **Documentation**: Plots provide visual evidence for optimization claims

### Academic Use

Generated plots support:

- **Performance Analysis**: Quantitative comparison of optimizations
- **Convergence Validation**: Algorithm correctness verification
- **Scalability Studies**: Performance across problem sizes
- **Optimization Tracking**: Progress through E0-E8 experiments

## Advanced Features

### Custom Plot Types

To add new plot types, extend the `ExperimentAnalyzer` class:

```python
def generate_custom_plot(self, tag: str):
    """Generate custom visualization."""
    # Implementation here
    pass
```

### Data Export

The script can be modified to export processed data:

```python
# Export aggregated statistics
median_data.to_csv(f"analysis_{tag}.csv", index=False)

# Export performance summary
summary_df.to_csv(f"performance_{tag}.csv", index=False)
```

### Batch Analysis

For multiple experiments, create a wrapper script:

```bash
#!/bin/bash
# analyze_all_experiments.sh
for exp in {0..8}; do
    if [[ -d "bench/e$exp" ]]; then
        echo "Analyzing experiment $exp..."
        python scripts/analyze_experiment.py $exp
    fi
done
```

## Best Practices

### For Reproducibility

1. **Version Control**: Commit plots with experiment data
2. **Naming Consistency**: Use consistent tag formats
3. **Documentation**: Include plot descriptions in reports
4. **Parameter Recording**: Document all configuration parameters

### For Analysis

1. **Multiple Runs**: Always use statistical aggregation
2. **Warm-up Exclusion**: Focus on measurement runs only
3. **Cross-Comparison**: Compare across configurations
4. **Performance Tracking**: Monitor MLUPS improvements

### For Publication

1. **High Resolution**: Use 300 DPI for print quality
2. **Clear Labels**: Ensure axis labels and titles are readable
3. **Color Accessibility**: Use distinguishable colors
4. **Consistent Styling**: Maintain uniform appearance across plots

## Troubleshooting

### Common Problems

**No plots generated**:
```bash
# Check dependencies
pip list | grep -E "(pandas|matplotlib|numpy)"

# Verify data files
ls -la bench/e0/
```

**Empty plots**:
```bash
# Check run numbering
grep "run" bench/e0/times_*.csv | head -5

# Verify measurement runs exist
python -c "import pandas as pd; df=pd.read_csv('bench/e0/times_N200000_D16_K8_iters10_seed1_run4.csv'); print(df.shape)"
```

**Parameter parsing errors**:
```bash
# Check filename format
ls bench/e0/times_*.csv | head -3

# Verify naming convention
python scripts/analyze_experiment.py 0 --debug
```

### Debug Mode

Add debug output to the script:

```python
# In _extract_config_params method
print(f"Debug: Parsing tag '{tag}'")
print(f"Debug: N match: {re.search(r'N(\d+)', tag)}")
```

## Conclusion

The plotting system provides:

1. **Automated Analysis**: No manual data processing required
2. **Professional Quality**: Publication-ready visualizations
3. **Statistical Rigor**: Proper aggregation of multiple runs
4. **Flexible Usage**: Support for various analysis scenarios
5. **Academic Integration**: Seamless workflow with benchmarking

This system enables systematic analysis of optimization experiments and provides the visual foundation for academic reporting and optimization tracking.
