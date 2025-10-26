#!/bin/bash
#SBATCH --job-name=kmeans_e{X}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:01:00
#SBATCH --partition=cosc3500
#SBATCH --account=cosc3500

# ============================================================================
# K-Means Experiment Slurm Job Script Template
# ============================================================================
# 
# USAGE:
# 1. Copy this file: cp run_experiment_template.slurm run_experiment_e{X}.slurm
# 2. Replace {X} with experiment number (0, 1, 2, etc.)
# 3. Modify EXPERIMENT_NUM and BUILD_DEFS as needed
# 4. Submit: sbatch run_experiment_e{X}.slurm
#
# EXAMPLE:
# - For E0 baseline: EXPERIMENT_NUM=0, BUILD_DEFS=""
# - For E1 optimization: EXPERIMENT_NUM=1, BUILD_DEFS="-DTRANSPOSED_C=1"
# - For E2 optimization: EXPERIMENT_NUM=2, BUILD_DEFS="-DUNROLL=4"
#
# ============================================================================

# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================

# Experiment number (0, 1, 2, 3, etc.)
EXPERIMENT_NUM=0

# Build definitions for optimization variants (leave empty for baseline)
# Examples:
# - Baseline: BUILD_DEFS=""
# - Transposed centroids: BUILD_DEFS="-DTRANSPOSED_C=1"
# - Loop unrolling: BUILD_DEFS="-DUNROLL=4"
# - Cache blocking: BUILD_DEFS="-DBLOCK_SIZE=32"
# - Memory alignment: BUILD_DEFS="-DALIGNED=1"
# - Multiple optimizations: BUILD_DEFS="-DTRANSPOSED_C=1 -DUNROLL=4"
BUILD_DEFS=""

# ============================================================================
# JOB CONFIGURATION (usually don't need to change)
# ============================================================================

# Load any required modules (uncomment if needed)
# module load gcc/11.2.0
# module load valgrind/3.18.1

# Set environment variables for the experiment
export EXPERIMENT=e${EXPERIMENT_NUM}
export ANTIVEC=1

# ============================================================================
# JOB EXECUTION
# ============================================================================

# Print job information
echo "=========================================="
echo "K-Means E${EXPERIMENT_NUM} Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_PARTITION"
echo "Time limit: $SLURM_TIMELIMIT"
echo "Experiment: $EXPERIMENT"
echo "Build definitions: $BUILD_DEFS"
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
echo "Building project..."
make clean

if [ -n "$BUILD_DEFS" ]; then
    echo "Building with definitions: $BUILD_DEFS"
    ANTIVEC=1 DEFS="$BUILD_DEFS" make release
else
    echo "Building baseline version"
    ANTIVEC=1 make release
fi

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi
echo "Build completed successfully!"
echo ""

# Run the experiment using the run_experiment script
echo "Running E${EXPERIMENT_NUM} experiment..."
if [ -n "$BUILD_DEFS" ]; then
    echo "Optimization variant: $BUILD_DEFS"
else
    echo "Baseline version (no optimizations)"
fi
echo ""

echo "This will generate:"
echo "  - Timing data (3 warm-up + 5 measurement runs)"
echo "  - Convergence analysis"
echo "  - System information capture"
echo "  - Profiling data (if available)"
echo ""

# Run the experiment script
if [ -n "$BUILD_DEFS" ]; then
    ./scripts/run_experiment.sh ${EXPERIMENT_NUM} "$BUILD_DEFS"
else
    ./scripts/run_experiment.sh ${EXPERIMENT_NUM}
fi

# Check if experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "E${EXPERIMENT_NUM} Experiment completed successfully!"
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
    echo "Experiment results are ready for analysis!"
    echo "Next steps:"
    echo "  1. Generate plots: python scripts/analyze_experiment.py ${EXPERIMENT_NUM}"
    echo "  2. Review summary: cat bench/e${EXPERIMENT_NUM}/summary_e${EXPERIMENT_NUM}.md"
    echo "  3. Check plots: ls -la plots/"
    
else
    echo ""
    echo "=========================================="
    echo "ERROR: E${EXPERIMENT_NUM} Experiment failed!"
    echo "=========================================="
    echo "Check the output above for error messages."
    exit 1
fi

echo ""
echo "Job completed at: $(date)"
