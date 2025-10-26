#!/bin/bash
#SBATCH --job-name=kmeans_e5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:01:00
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500

# ============================================================================
# K-Means E5 Experiment: K-Register Blocking Optimization
# ============================================================================
# 
# This experiment implements cache optimization through K-register blocking
# to reduce per-point overhead by evaluating TK centroids at once using
# multiple accumulators, reusing the loaded point value px[d] across TK
# centroids for better temporal locality.
#
# ============================================================================

# Experiment configuration
EXPERIMENT_NUM=5
BUILD_DEFS="-DTRANSPOSED_C=1 -DHOIST=1 -DBRANCHLESS=1 -DSTRIDE_PTR=1 -DTK=4"

# Load any required modules (uncomment if needed)
# module load gcc/11.2.0
# module load valgrind/3.18.1

# Set environment variables for the experiment
export EXPERIMENT=e${EXPERIMENT_NUM}
export ANTIVEC=1

# Print job information
echo "=========================================="
echo "K-Means E5: K-Register Blocking Optimization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_PARTITION"
echo "Time limit: $SLURM_TIMELIMIT"
echo "Experiment: $EXPERIMENT"
echo "Build definitions: $BUILD_DEFS"
echo "Optimizations:"
echo "  - TRANSPOSED_C: Transposed centroids (from E1)"
echo "  - HOIST: Invariant hoisting (from E2)"
echo "  - BRANCHLESS: Branchless argmin (from E2)"
echo "  - STRIDE_PTR: Strided pointer arithmetic (from E2)"
echo "  - TK: K-register blocking optimization"
echo "=========================================="

# Print system information
echo "System Information:"
uname -a
echo ""

# Print compiler information
echo "Compiler Information:"
gcc --version
echo ""

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Clean and build the project
echo "Building project with E5 K-register blocking optimization..."
make clean
ANTIVEC=1 DEFS="$BUILD_DEFS" make release

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi
echo "Build completed successfully!"
echo ""

# Run the experiment using the run_experiment script
echo "Running E5 K-register blocking experiment..."
echo "Optimizations:"
echo "  - K-register blocking: Process 4 centroids at once with register accumulators"
echo "  - Temporal locality: Reuse px[d] across TK centroids for better cache efficiency"
echo "  - Combined with E1+E2: Transposed centroids + micro-optimizations"
echo ""

echo "This will generate:"
echo "  - Timing data (3 warm-up + 5 measurement runs)"
echo "  - Convergence analysis"
echo "  - System information capture"
echo "  - Profiling data (if available)"
echo ""

# Run the experiment script
./scripts/run_experiment.sh ${EXPERIMENT_NUM} "$BUILD_DEFS"

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "E5 Experiment completed successfully!"
    echo "=========================================="
    
    # List generated files
    echo "Generated files in bench/e${EXPERIMENT_NUM}/:"
    ls -la bench/e${EXPERIMENT_NUM}/ | wc -l
    echo "files generated"
    
    # Show file structure
    echo ""
    echo "File structure:"
    find bench/e${EXPERIMENT_NUM}/ -type f | head -20
    
    # Check for key output files
    echo ""
    echo "Key output files:"
    if [ -f "bench/e${EXPERIMENT_NUM}/sysinfo_e${EXPERIMENT_NUM}.txt" ]; then
        echo "✅ System information captured"
    else
        echo "❌ System information missing"
    fi
    
    if [ -f "bench/e${EXPERIMENT_NUM}/times_N200000_D16_K8_iters10_seed1_run4.csv" ]; then
        echo "✅ Canonical config timing data"
    else
        echo "❌ Canonical config timing data missing"
    fi
    
    if [ -f "bench/e${EXPERIMENT_NUM}/times_N100000_D64_K64_iters10_seed1_run4.csv" ]; then
        echo "✅ Stress config timing data"
    else
        echo "❌ Stress config timing data missing"
    fi
    
    echo ""
    echo "E5 experiment results are ready for analysis!"
    echo "Next steps:"
    echo "  1. Generate plots: python scripts/analyze_experiment.py ${EXPERIMENT_NUM}"
    echo "  2. Review summary: cat bench/e${EXPERIMENT_NUM}/summary_e${EXPERIMENT_NUM}.md"
    echo "  3. Check plots: ls -la plots/"
    echo "  4. Compare with E0/E1/E2: python scripts/analyze_experiment.py 0 1 2 5"
    
else
    echo ""
    echo "=========================================="
    echo "ERROR: E5 Experiment failed!"
    echo "=========================================="
    echo "Check the output above for error messages."
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
