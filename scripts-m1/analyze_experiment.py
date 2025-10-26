#!/usr/bin/env python3
"""
COSC3500 M1 K-Means Experiment Analyzer

This script analyzes benchmark experiment results and generates comprehensive visualizations
including per-iteration timings, MLUPS, convergence, and kernel cost analysis.

Usage:
    python scripts/analyze_experiment.py <experiment_number> [tag]
    
Example:
    python scripts/analyze_experiment.py 0 canonical
    python scripts/analyze_experiment.py 0 stress
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import re

# Set matplotlib style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ExperimentAnalyzer:
    """Analyzes K-Means benchmark experiment results and generates visualizations."""
    
    def __init__(self, experiment_dir: str):
        """Initialize analyzer with experiment directory."""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = self.experiment_dir.name
        
        # Create experiment-specific subdirectory in plots/
        self.plots_dir = Path("plots") / self.experiment_name
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep the main plots directory for backward compatibility
        self.main_plots_dir = Path("plots")
        self.main_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.timing_data: Dict[str, pd.DataFrame] = {}
        self.inertia_data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        
        # Load all data
        self._load_experiment_data()
    
    def _load_experiment_data(self):
        """Load all timing and inertia data from the experiment directory."""
        print(f"Loading data from {self.experiment_dir}...")
        
        # Find all timing files
        timing_files = list(self.experiment_dir.glob("times_*.csv"))
        inertia_files = list(self.experiment_dir.glob("inertia_*.csv"))
        
        print(f"Found {len(timing_files)} timing files and {len(inertia_files)} inertia files")
        
        # Group files by configuration (tag)
        for timing_file in timing_files:
            tag = self._extract_tag_from_filename(timing_file.name)
            if tag not in self.timing_data:
                self.timing_data[tag] = []
                self.inertia_data[tag] = []
                self.metadata[tag] = {}
            
            # Load timing data
            try:
                df = pd.read_csv(timing_file)
                df['run'] = self._extract_run_number(timing_file.name)
                self.timing_data[tag].append(df)
            except Exception as e:
                print(f"Warning: Could not load {timing_file}: {e}")
        
        # Load inertia data
        for inertia_file in inertia_files:
            tag = self._extract_tag_from_filename(inertia_file.name)
            if tag in self.inertia_data:
                try:
                    df = pd.read_csv(inertia_file)
                    df['run'] = self._extract_run_number(inertia_file.name)
                    self.inertia_data[tag].append(df)
                except Exception as e:
                    print(f"Warning: Could not load {inertia_file}: {e}")
        
        # Convert lists to DataFrames
        for tag in list(self.timing_data.keys()):
            if self.timing_data[tag]:
                self.timing_data[tag] = pd.concat(self.timing_data[tag], ignore_index=True)
                self.timing_data[tag] = self.timing_data[tag].sort_values(['run', 'iter'])
            if self.inertia_data[tag]:
                self.inertia_data[tag] = pd.concat(self.inertia_data[tag], ignore_index=True)
                self.inertia_data[tag] = self.inertia_data[tag].sort_values(['run', 'iter'])
        
        print(f"Loaded data for tags: {list(self.timing_data.keys())}")
    
    def _extract_tag_from_filename(self, filename: str) -> str:
        """Extract configuration tag from filename."""
        # Remove file extension and run number
        base = filename.replace('.csv', '').replace('.txt', '')
        # Remove run number suffix
        base = re.sub(r'_run\d+$', '', base)
        # Extract the configuration part
        if base.startswith('times_'):
            return base[6:]  # Remove 'times_' prefix
        elif base.startswith('inertia_'):
            return base[8:]  # Remove 'inertia_' prefix
        else:
            return base
    
    def _extract_run_number(self, filename: str) -> int:
        """Extract run number from filename."""
        match = re.search(r'_run(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    def _extract_config_params(self, tag: str) -> Dict[str, int]:
        """Extract N, D, K, iters from tag."""
        # Parse tag like "N200000_D16_K8_iters10_seed1"
        params = {}
        try:
            params['N'] = int(re.search(r'N(\d+)', tag).group(1))
            params['D'] = int(re.search(r'D(\d+)', tag).group(1))
            params['K'] = int(re.search(r'K(\d+)', tag).group(1))
            params['iters'] = int(re.search(r'iters(\d+)', tag).group(1))
        except (AttributeError, ValueError):
            print(f"Warning: Could not parse parameters from tag: {tag}")
            params = {'N': 0, 'D': 0, 'K': 0, 'iters': 0}
        return params
    
    def generate_per_iteration_timings(self, tag: str):
        """Generate per-iteration kernel timings plot."""
        if tag not in self.timing_data or self.timing_data[tag].empty:
            print(f"No timing data found for tag: {tag}")
            return
        
        df = self.timing_data[tag]
        
        # Separate warm-up and measurement runs
        warmup_runs = df[df['run'] <= 3]
        measurement_runs = df[df['run'] > 3]
        
        if measurement_runs.empty:
            print(f"No measurement runs found for tag: {tag}")
            return
        
        # Calculate median across measurement runs for each iteration
        median_data = measurement_runs.groupby('iter').agg({
            'assign_ms': 'median',
            'update_ms': 'median',
            'total_ms': 'median'
        }).reset_index()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Individual runs (transparent)
        for run in measurement_runs['run'].unique():
            run_data = measurement_runs[measurement_runs['run'] == run]
            ax1.plot(run_data['iter'], run_data['assign_ms'], 'b-', alpha=0.2, linewidth=0.5)
            ax1.plot(run_data['iter'], run_data['update_ms'], 'r-', alpha=0.2, linewidth=0.5)
        
        # Plot 2: Median lines (solid)
        ax1.plot(median_data['iter'], median_data['assign_ms'], 'b-', linewidth=2, label='Assign Kernel (median)')
        ax1.plot(median_data['iter'], median_data['update_ms'], 'r-', linewidth=2, label='Update Kernel (median)')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'Per-Iteration Kernel Timings - {tag}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 3: Total time
        ax2.plot(median_data['iter'], median_data['total_ms'], 'g-', linewidth=2, label='Total Time (median)')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Time (ms)')
        ax2.set_title(f'Per-Iteration Total Time - {tag}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot in experiment subdirectory
        output_file = self.plots_dir / f"periter_timings_{tag}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved per-iteration timings plot: {output_file}")
        plt.close()
    
    def generate_per_iteration_mlups(self, tag: str):
        """Generate per-iteration MLUPS plot."""
        if tag not in self.timing_data or self.timing_data[tag].empty:
            print(f"No timing data found for tag: {tag}")
            return
        
        df = self.timing_data[tag]
        params = self._extract_config_params(tag)
        
        if params['N'] == 0:
            print(f"Could not extract parameters for tag: {tag}")
            return
        
        # Calculate MLUPS for each iteration
        df['mlups'] = (params['N'] * params['K']) / (df['assign_ms'] / 1000) / 1e6
        
        # Separate warm-up and measurement runs
        measurement_runs = df[df['run'] > 3]
        
        if measurement_runs.empty:
            print(f"No measurement runs found for tag: {tag}")
            return
        
        # Calculate median MLUPS across measurement runs for each iteration
        median_mlups = measurement_runs.groupby('iter')['mlups'].median().reset_index()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs (transparent)
        for run in measurement_runs['run'].unique():
            run_data = measurement_runs[measurement_runs['run'] == run]
            ax.plot(run_data['iter'], run_data['mlups'], 'b-', alpha=0.2, linewidth=0.5)
        
        # Plot median line (solid)
        ax.plot(median_mlups['iter'], median_mlups['mlups'], 'b-', linewidth=2, label='MLUPS (median)')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MLUPS (Million Label Updates Per Second)')
        ax.set_title(f'Per-Iteration MLUPS - {tag}\nN={params["N"]:,}, D={params["D"]}, K={params["K"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot in experiment subdirectory
        output_file = self.plots_dir / f"periter_mlups_{tag}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved per-iteration MLUPS plot: {output_file}")
        plt.close()
    
    def generate_convergence_plot(self, tag: str):
        """Generate convergence plot showing inertia vs iteration."""
        if tag not in self.inertia_data or self.inertia_data[tag].empty:
            print(f"No inertia data found for tag: {tag}")
            return
        
        df = self.inertia_data[tag]
        
        # Separate warm-up and measurement runs
        warmup_runs = df[df['run'] <= 3]
        measurement_runs = df[df['run'] > 3]
        
        if measurement_runs.empty:
            print(f"No measurement runs found for tag: {tag}")
            return
        
        # Calculate median inertia across measurement runs for each iteration
        median_inertia = measurement_runs.groupby('iter')['inertia'].median().reset_index()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual runs (transparent)
        for run in measurement_runs['run'].unique():
            run_data = measurement_runs[measurement_runs['run'] == run]
            ax.plot(run_data['iter'], run_data['inertia'], 'b-', alpha=0.2, linewidth=0.5)
        
        # Plot median line (solid)
        ax.plot(median_inertia['iter'], median_inertia['inertia'], 'b-', linewidth=2, label='Inertia (median)')
        
        # Add convergence metrics
        initial_inertia = median_inertia.iloc[0]['inertia']
        final_inertia = median_inertia.iloc[-1]['inertia']
        improvement = ((initial_inertia - final_inertia) / initial_inertia) * 100
        
        ax.text(0.02, 0.98, f'Initial: {initial_inertia:.2e}\nFinal: {final_inertia:.2e}\nImprovement: {improvement:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Inertia (Sum of Squared Distances)')
        ax.set_title(f'Convergence - {tag}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for better visualization
        
        plt.tight_layout()
        
        # Save plot in experiment subdirectory
        output_file = self.plots_dir / f"inertia_{tag}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot: {output_file}")
        plt.close()
    
    def generate_median_kernel_cost(self, tag: str):
        """Generate median kernel cost bar chart."""
        if tag not in self.timing_data or self.timing_data[tag].empty:
            print(f"No timing data found for tag: {tag}")
            return
        
        df = self.timing_data[tag]
        params = self._extract_config_params(tag)
        
        # Separate warm-up and measurement runs
        measurement_runs = df[df['run'] > 3]
        
        if measurement_runs.empty:
            print(f"No measurement runs found for tag: {tag}")
            return
        
        # Calculate median per-iteration times across measurement runs
        median_per_iter = measurement_runs.groupby('iter').agg({
            'assign_ms': 'median',
            'update_ms': 'median'
        }).reset_index()
        
        # Calculate overall median
        overall_assign_median = median_per_iter['assign_ms'].median()
        overall_update_median = median_per_iter['update_ms'].median()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Per-iteration breakdown
        x = np.arange(len(median_per_iter))
        width = 0.35
        
        ax1.bar(x - width/2, median_per_iter['assign_ms'], width, label='Assign Kernel', alpha=0.8)
        ax1.bar(x + width/2, median_per_iter['update_ms'], width, label='Update Kernel', alpha=0.8)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title(f'Per-Iteration Kernel Costs - {tag}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(median_per_iter['iter'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overall median comparison
        kernels = ['Assign', 'Update']
        medians = [overall_assign_median, overall_update_median]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax2.bar(kernels, medians, color=colors, alpha=0.8)
        ax2.set_ylabel('Median Time (ms)')
        ax2.set_title(f'Overall Median Kernel Costs - {tag}')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, median in zip(bars, medians):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{median:.2f}', ha='center', va='bottom')
        
        # Add configuration info
        fig.suptitle(f'Kernel Cost Analysis - {tag}\nN={params["N"]:,}, D={params["D"]}, K={params["K"]}, iters={params["iters"]}', 
                    fontsize=14, y=0.95)
        
        plt.tight_layout()
        
        # Save plot in experiment subdirectory
        output_file = self.plots_dir / f"median_kernels_{tag}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved median kernel cost plot: {output_file}")
        plt.close()
    
    def generate_summary_report(self, tag: str = None):
        """Generate a summary report for the experiment."""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        print(f"{'='*60}")
        
        # Capture the summary output
        import io
        from contextlib import redirect_stdout
        
        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            if tag and tag in self.timing_data:
                self._print_tag_summary(tag)
            else:
                for tag in self.timing_data.keys():
                    self._print_tag_summary(tag)
        
        # Print to console
        print(summary_buffer.getvalue())
        
        # Save to plots subdirectory
        summary_file = self.plots_dir / f"summary_{tag or 'all'}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"EXPERIMENT SUMMARY: {self.experiment_name}\n")
            f.write("="*60 + "\n")
            f.write(summary_buffer.getvalue())
        print(f"Saved summary report: {summary_file}")
    
    def _print_tag_summary(self, tag: str):
        """Print summary for a specific tag."""
        if tag not in self.timing_data or self.timing_data[tag].empty:
            return
        
        df = self.timing_data[tag]
        params = self._extract_config_params(tag)
        
        # Separate warm-up and measurement runs
        measurement_runs = df[df['run'] > 3]
        
        if measurement_runs.empty:
            return
        
        # Calculate statistics
        median_per_iter = measurement_runs.groupby('iter').agg({
            'assign_ms': 'median',
            'update_ms': 'median',
            'total_ms': 'median'
        }).reset_index()
        
        overall_assign_median = median_per_iter['assign_ms'].median()
        overall_update_median = median_per_iter['update_ms'].median()
        overall_total_median = median_per_iter['total_ms'].median()
        
        # Calculate MLUPS
        mlups = (params['N'] * params['K']) / (overall_assign_median / 1000) / 1e6
        
        print(f"\nConfiguration: {tag}")
        print(f"  Parameters: N={params['N']:,}, D={params['D']}, K={params['K']}, iters={params['iters']}")
        print(f"  Measurement runs: {len(measurement_runs['run'].unique())}")
        print(f"  Overall median times:")
        print(f"    Assign kernel: {overall_assign_median:.3f} ms")
        print(f"    Update kernel: {overall_update_median:.3f} ms")
        print(f"    Total time: {overall_total_median:.3f} ms")
        print(f"  Performance: {mlups:.2f} MLUPS")
        
        # Time distribution
        assign_pct = (overall_assign_median / overall_total_median) * 100
        update_pct = (overall_update_median / overall_total_median) * 100
        print(f"  Time distribution: Assign {assign_pct:.1f}%, Update {update_pct:.1f}%")
    
    def generate_all_plots(self, tag: str = None):
        """Generate all plots for the experiment."""
        tags_to_process = [tag] if tag else list(self.timing_data.keys())
        
        for t in tags_to_process:
            if t in self.timing_data:
                print(f"\nGenerating plots for tag: {t}")
                self.generate_per_iteration_timings(t)
                self.generate_per_iteration_mlups(t)
                self.generate_convergence_plot(t)
                self.generate_median_kernel_cost(t)
            else:
                print(f"Warning: No data found for tag: {t}")


def main():
    """Main function to run the experiment analyzer."""
    parser = argparse.ArgumentParser(description='Analyze K-Means benchmark experiment results')
    parser.add_argument('experiment', type=str, help='Experiment number (e.g., 0, 1, 2)')
    parser.add_argument('tag', nargs='?', type=str, help='Specific configuration tag to analyze (optional)')
    parser.add_argument('--plots-only', action='store_true', help='Generate plots only, skip summary report')
    
    args = parser.parse_args()
    
    # Construct experiment directory path
    experiment_dir = f"bench/e{args.experiment}"
    
    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory '{experiment_dir}' not found.")
        sys.exit(1)
    
    # Create analyzer and generate results
    analyzer = ExperimentAnalyzer(experiment_dir)
    
    if not args.plots_only:
        analyzer.generate_summary_report(args.tag)
    
    analyzer.generate_all_plots(args.tag)
    
    print(f"\nAnalysis complete! Check the experiment-specific directory for generated files:")
    print(f"  - plots/e{args.experiment}/ (experiment analysis and plots)")


if __name__ == "__main__":
    main()
