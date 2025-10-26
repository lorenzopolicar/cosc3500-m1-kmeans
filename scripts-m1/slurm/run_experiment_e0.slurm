#!/bin/bash
#SBATCH --job-name=kmeans_e0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:01:00
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500

# Load any required modules (uncomment if needed)
# module load gcc/11.2.0
# module load valgrind/3.18.1

# Set environment variables for the experiment
export EXPERIMENT=e0
export ANTIVEC=1

# Print job information
echo "=========================================="
echo "K-Means E0 Baseline Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_PARTITION"
echo "Time limit: $SLURM_TIMELIMIT"
echo "=========================================="

# Print system information
echo "System Information:"
uname -a
echo ""

# Print compiler information
echo "Compiler Information:"
gcc --version
echo ""

# Change to project directory (adjust path as needed)
cd $SLURM_SUBMIT_DIR

# Clean and build the project
echo "Building project..."
make clean
make release
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi
echo "Build completed successfully!"
echo ""

# Run the experiment using the run_experiment script
echo "Running E0 baseline experiment..."
echo "This will generate:"
echo "  - Timing data (3 warm-up + 5 measurement runs)"
echo "  - Convergence analysis"
echo "  - System information capture"
echo "  - Profiling data (if available)"
echo ""

# Run the experiment script
./scripts/run_experiment.sh 0

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "E0 Experiment completed successfully!"
    echo "=========================================="
    
    # List generated files
    echo "Generated files in bench/e0/:"
    ls -la bench/e0/ | wc -l
    echo "files generated"
    
    # Show file structure
    echo ""
    echo "File structure:"
    find bench/e0/ -type f | head -20
    
    # Check for key output files
    echo ""
    echo "Key output files:"
    if [ -f "bench/e0/sysinfo_e0.txt" ]; then
        echo "✅ System information captured"
    else
        echo "❌ System information missing"
    fi
    
    if [ -f "bench/e0/times_N200000_D16_K8_iters10_seed1_run4.csv" ]; then
        echo "✅ Canonical config timing data"
    else
        echo "❌ Canonical config timing data missing"
    fi
    
    if [ -f "bench/e0/times_N100000_D64_K64_iters10_seed1_run4.csv" ]; then
        echo "✅ Stress config timing data"
    else
        echo "❌ Stress config timing data missing"
    fi
    
    echo ""
    echo "Experiment results are ready for analysis!"
    echo "Next steps:"
    echo "  1. Generate plots: python scripts/analyze_experiment.py 0"
    echo "  2. Review summary: cat bench/e0/summary_e0.md"
    echo "  3. Check plots: ls -la plots/"
    
else
    echo ""
    echo "=========================================="
    echo "ERROR: E0 Experiment failed!"
    echo "=========================================="
    echo "Check the output above for error messages."
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
