# Plotting System Quick Reference

## 🚀 Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Analyze experiment E0
python scripts/analyze_experiment.py 0

# Generate plots for specific configuration
python scripts/analyiment.py 0 "N200000_D16_K8_iters10_seed1"
```

## 📊 Generated Plots

| Plot Type | File Pattern | Description |
|-----------|--------------|-------------|
| **Kernel Timings** | `{exp}_periter_timings_{tag}.png` | Per-iteration assign/update times |
| **MLUPS** | `{exp}_periter_mlups_{tag}.png` | Performance in MLUPS vs iteration |
| **Convergence** | `{exp}_inertia_{tag}.png` | Inertia progression (log scale) |
| **Kernel Costs** | `{exp}_median_kernels_{tag}.png` | Median kernel performance comparison |

## 🔧 Common Commands

### Basic Analysis
```bash
# Full experiment analysis
python scripts/analyze_experiment.py 0

# Specific configuration only
python scripts/analyze_experiment.py 0 "N200000_D16_K8_iters10_seed1"

# Plots only (skip summary)
python scripts/analyze_experiment.py 0 --plots-only
```

### Batch Processing
```bash
# Analyze multiple experiments
for exp in 0 1 2 3; do
    python scripts/analyze_experiment.py $exp
done

# All experiments in range
for exp in {0..8}; do
    if [[ -d "bench/e$exp" ]]; then
        python scripts/analyze_experiment.py $exp
    fi
done
```

## 📁 File Structure

```
plots/
├── e0_periter_timings_N200000_D16_K8_iters10_seed1.png
├── e0_periter_mlups_N200000_D16_K8_iters10_seed1.png
├── e0_inertia_N200000_D16_K8_iters10_seed1.png
├── e0_median_kernels_N200000_D16_K8_iters10_seed1.png
└── ... (similar for other configurations)
```

## 📈 Performance Metrics

### MLUPS Calculation
```
MLUPS = (N × K) / (assign_ms / 1000) / 1e6

Example: N=200K, K=8, assign_ms=7.680
MLUPS = (200,000 × 8) / (7.680 / 1000) / 1,000,000 = 208.33
```

### Time Distribution
```
assign_pct = (assign_ms / total_ms) × 100
update_pct = (update_ms / total_ms) × 100

Example: assign=7.680, total=8.652
assign_pct = (7.680 / 8.652) × 100 = 88.8%
```

## 🎯 Configuration Tags

### Canonical Config
```
N200000_D16_K8_iters10_seed1
- N: 200,000 points
- D: 16 dimensions  
- K: 8 clusters
- iters: 10 iterations
- seed: 1
```

### Stress Config
```
N100000_D64_K64_iters10_seed1
- N: 100,000 points
- D: 64 dimensions
- K: 64 clusters
- iters: 10 iterations
- seed: 1
```

## 🔍 Data Processing

### Automatic Detection
- **Warm-up runs**: 1-3 (excluded from analysis)
- **Measurement runs**: 4-8 (used for statistics)
- **Statistical aggregation**: Median across measurement runs
- **Parameter extraction**: N, D, K, iters from filenames

### File Requirements
```
times_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.csv
inertia_N{N}_D{D}_K{K}_iters{iters}_seed{seed}_run{run}.csv
```

## 🛠️ Troubleshooting

### Common Issues
```bash
# No plots generated
pip install -r requirements.txt

# Empty plots
ls bench/e0/times_*_run[4-8].csv

# Parameter parsing errors
python scripts/analyze_experiment.py 0 --help
```

### Debug Commands
```bash
# Check data files
ls -la bench/e0/

# Verify CSV content
head -5 bench/e0/times_N200000_D16_K8_iters10_seed1_run4.csv

# Check run numbering
grep "run" bench/e0/times_*.csv | head -10
```

## 📋 Workflow Integration

### Complete Pipeline
```bash
# 1. Run benchmark
./scripts/run_experiment.sh 0

# 2. Generate plots
python scripts/analyze_experiment.py 0

# 3. Review results
ls -la plots/
open plots/  # macOS
```

### For Optimization Experiments
```bash
# Baseline (E0)
python scripts/analyze_experiment.py 0

# Optimization variant (E1)
python scripts/analyze_experiment.py 1

# Compare configurations
python scripts/analyze_experiment.py 0 "N200000_D16_K8_iters10_seed1"
python scripts/analyze_experiment.py 1 "N200000_D16_K8_iters10_seed1"
```

## 🎨 Customization

### Modify Plot Appearance
Edit `scripts/analyze_experiment.py`:

```python
# Change figure size
plt.rcParams['figure.figsize'] = (12, 8)

# Modify colors
ax.plot(data, color='red', linewidth=3)

# Adjust grid
ax.grid(True, alpha=0.5, linestyle='--')
```

### Add New Plot Types
```python
def generate_custom_plot(self, tag: str):
    """Generate custom visualization."""
    # Implementation here
    pass
```

## 📚 Academic Use

### Performance Analysis
- **MLUPS tracking**: Monitor optimization improvements
- **Kernel breakdown**: Identify bottlenecks
- **Scalability**: Performance across problem sizes

### Convergence Validation
- **Monotonicity**: Ensure algorithm correctness
- **Improvement metrics**: Quantify convergence quality
- **Iteration analysis**: Understand convergence behavior

### Publication Support
- **High resolution**: 300 DPI for print quality
- **Professional styling**: Publication-ready appearance
- **Statistical rigor**: Proper aggregation of multiple runs

## 🚀 Pro Tips

1. **Use consistent naming**: Maintain tag format across experiments
2. **Version control plots**: Commit with experiment data
3. **Batch analysis**: Automate multiple experiment processing
4. **Custom metrics**: Extend script for specific analysis needs
5. **Cross-comparison**: Compare E0-E8 for optimization tracking
