# Slurm Job Scripts for K-Means Experiments

This directory contains Slurm job scripts for running K-Means benchmark experiments on the Rangpur HPC cluster.

## 📁 Available Scripts

### **Ready-to-Use Scripts**

| Script | Description | Optimization |
|--------|-------------|--------------|
| `run_experiment_e0.slurm` | E0 baseline experiment | No optimizations |
| `run_experiment_e1.slurm` | E1 transposed centroids | Memory layout optimization |
| `run_experiment_template.slurm` | Template for new experiments | Customizable |

### **Template Script**

The `run_experiment_template.slurm` provides a base for creating new experiment scripts. Copy and modify it for E2, E3, etc.

## 🚀 Quick Start

### **1. Submit E0 Baseline**

```bash
# Submit the baseline experiment
sbatch run_experiment_e0.slurm

# Check job status
squeue -u $USER

# Monitor job output
tail -f slurm-<job_id>.out
```

### **2. Submit E1 Optimization**

```bash
# Submit the transposed centroids experiment
sbatch run_experiment_e1.slurm

# Check job status
squeue -u $USER
```

### **3. Create New Experiment Script**

```bash
# Copy template
cp run_experiment_template.slurm run_experiment_e2.slurm

# Edit the configuration section
nano run_experiment_e2.slurm

# Submit the new experiment
sbatch run_experiment_e2.slurm
```

## ⚙️ Configuration

### **Modifying Template Scripts**

Edit the configuration section in any script:

```bash
# Experiment number (0, 1, 2, 3, etc.)
EXPERIMENT_NUM=2

# Build definitions for optimization variants
BUILD_DEFS="-DUNROLL=4 -DALIGNED=1"
```

### **Common Optimization Flags**

| Optimization | Build Definition | Description |
|--------------|------------------|-------------|
| **Baseline** | `""` | No optimizations |
| **Transposed C** | `"-DTRANSPOSED_C=1"` | Memory layout optimization |
| **Loop Unrolling** | `"-DUNROLL=4"` | Unroll loops by factor of 4 |
| **Cache Blocking** | `"-DBLOCK_SIZE=32"` | 32×32 cache blocking |
| **Memory Alignment** | `"-DALIGNED=1"` | Aligned memory allocation |
| **Multiple** | `"-DTRANSPOSED_C=1 -DUNROLL=4"` | Combine optimizations |

## 📋 Job Submission

### **Basic Submission**

```bash
# Submit job
sbatch run_experiment_e0.slurm

# Job ID will be displayed
# Submitted batch job 12345
```

### **Check Job Status**

```bash
# View all your jobs
squeue -u $USER

# View specific job
squeue -j 12345

# View job details
scontrol show job 12345
```

### **Monitor Job Progress**

```bash
# Follow job output in real-time
tail -f slurm-12345.out

# Check job output after completion
cat slurm-12345.out
```

### **Cancel Job (if needed)**

```bash
# Cancel specific job
scancel 12345

# Cancel all your jobs
scancel -u $USER
```

## 📊 Expected Output

### **Successful Job Completion**

```
==========================================
E0 Experiment completed successfully!
==========================================
Generated files in bench/e0/:
54 files generated

Key output files:
✅ System information captured
✅ Canonical config timing data
✅ Stress config timing data

Experiment results are ready for analysis!
Next steps:
  1. Generate plots: python scripts/analyze_experiment.py 0
  2. Review summary: cat bench/e0/summary_e0.md
  3. Check plots: ls -la plots/
```

### **File Generation**

Each experiment creates:
- **System information**: `sysinfo_e{X}.txt`
- **Timing data**: `times_*.csv` (8 runs per config)
- **Convergence data**: `inertia_*.csv` (8 runs per config)
- **Metadata**: `meta_*.txt` (8 runs per config)
- **Profiling data**: `gprof_e{X}.txt`, `cachegrind_e{X}.txt` (if available)

## 🔧 Customization

### **Modifying Time Limits**

Change the `#SBATCH --time` directive:

```bash
# 1 hour
#SBATCH --time=0-01:00:00

# 30 minutes
#SBATCH --time=0-00:30:00

# 2 hours
#SBATCH --time=0-02:00:00
```

### **Adding Module Loads**

Uncomment and modify module loads:

```bash
# Load specific compiler version
module load gcc/11.2.0

# Load profiling tools
module load valgrind/3.18.1

# Load other tools as needed
module load intel/2021.4.0
```

### **Modifying Resource Requirements**

Adjust Slurm directives:

```bash
# Request more memory
#SBATCH --mem=8G

# Request specific node type
#SBATCH --constraint=skylake

# Request specific partition
#SBATCH --partition=cosc3500-debug
```

## 📈 Workflow Integration

### **Complete Experiment Pipeline**

```bash
# 1. Submit baseline experiment
sbatch run_experiment_e0.slurm

# 2. Wait for completion, then submit optimization
sbatch run_experiment_e1.slurm

# 3. After completion, generate plots
python scripts/analyze_experiment.py 0
python scripts/analyze_experiment.py 1

# 4. Compare results
ls -la plots/
```

### **Batch Submission**

Create a batch submission script:

```bash
#!/bin/bash
# submit_all_experiments.sh

# Submit baseline
echo "Submitting E0 baseline..."
sbatch run_experiment_e0.slurm

# Submit optimizations
echo "Submitting E1 transposed centroids..."
sbatch run_experiment_e1.slurm

echo "All experiments submitted!"
echo "Monitor with: squeue -u $USER"
```

## 🚨 Troubleshooting

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **Job fails immediately** | Check script syntax: `bash -n script.slurm` |
| **Build fails** | Verify Makefile and dependencies |
| **Permission denied** | Make script executable: `chmod +x script.slurm` |
| **Job times out** | Increase `--time` limit |
| **Out of memory** | Increase `--mem` or reduce problem size |

### **Debug Commands**

```bash
# Check script syntax
bash -n run_experiment_e0.slurm

# Test script locally (without sbatch)
bash run_experiment_e0.slurm

# Check job output
cat slurm-<job_id>.out

# Check job history
sacct -j <job_id> --format=JobID,JobName,State,ExitCode
```

### **Getting Help**

```bash
# View available partitions
sinfo

# View queue status
squeue

# Check your job limits
sacctmgr show user $USER

# View cluster information
sinfo -o "%20P %5D %14F %8z %10m %10d %11l %N"
```

## 📚 Best Practices

### **Before Submission**

1. **Test locally**: Run `make clean && make release` locally first
2. **Check paths**: Ensure all script paths are correct
3. **Verify dependencies**: Confirm required modules are available
4. **Set time limits**: Estimate runtime and add buffer

### **During Execution**

1. **Monitor progress**: Use `tail -f slurm-<job_id>.out`
2. **Check status**: Regular `squeue -u $USER` checks
3. **Resource usage**: Monitor with `scontrol show job <job_id>`

### **After Completion**

1. **Verify output**: Check all expected files were created
2. **Generate plots**: Run analysis scripts
3. **Clean up**: Remove temporary Slurm output files
4. **Document results**: Note any issues or observations

## 🎯 Next Steps

After running experiments:

1. **Generate visualizations**: `python scripts/analyze_experiment.py <exp_num>`
2. **Review summaries**: Check generated summary files
3. **Compare results**: Analyze performance improvements
4. **Plan next optimization**: Design E2, E3, etc. experiments

## 📞 Support

For cluster-specific issues:
- Check Rangpur documentation
- Contact cluster administrators
- Review Slurm documentation: https://slurm.schedmd.com/

For experiment-specific issues:
- Review script output and error messages
- Check generated log files
- Verify configuration parameters
