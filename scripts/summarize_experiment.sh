#!/bin/bash

# COSC3500 M1 K-Means Experiment Summary Generator
# This script analyzes all benchmark artifacts and provides aggregated statistics

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to calculate median
median() {
    local numbers=("$@")
    local count=${#numbers[@]}
    if [[ $count -eq 0 ]]; then
        echo "N/A"
        return
    fi
    
    # Sort numbers and find median
    IFS=$'\n' sorted=($(sort -n <<<"${numbers[*]}"))
    unset IFS
    
    if [[ $((count % 2)) -eq 0 ]]; then
        # Even number of elements, average of middle two
        local mid1=$((count / 2 - 1))
        local mid2=$((count / 2))
        local val1=${sorted[$mid1]}
        local val2=${sorted[$mid2]}
        echo "scale=3; ($val1 + $val2) / 2" | bc -l 2>/dev/null || echo "N/A"
    else
        # Odd number of elements, middle element
        local mid=$((count / 2))
        echo "${sorted[$mid]}"
    fi
}

# Function to calculate mean
mean() {
    local numbers=("$@")
    local count=${#numbers[@]}
    if [[ $count -eq 0 ]]; then
        echo "N/A"
        return
    fi
    
    local sum=0
    for num in "${numbers[@]}"; do
        sum=$(echo "scale=3; $sum + $num" | bc -l 2>/dev/null || echo "0")
    done
    
    echo "scale=3; $sum / $count" | bc -l 2>/dev/null || echo "N/A"
}

# Function to calculate standard deviation
stddev() {
    local numbers=("$@")
    local count=${#numbers[@]}
    if [[ $count -lt 2 ]]; then
        echo "N/A"
        return
    fi
    
    local avg=$(mean "${numbers[@]}")
    if [[ "$avg" == "N/A" ]]; then
        echo "N/A"
        return
    fi
    
    local sum_sq=0
    for num in "${numbers[@]}"; do
        local diff=$(echo "scale=6; $num - $avg" | bc -l 2>/dev/null || echo "0")
        local diff_sq=$(echo "scale=6; $diff * $diff" | bc -l 2>/dev/null || echo "0")
        sum_sq=$(echo "scale=6; $sum_sq + $diff_sq" | bc -l 2>/dev/null || echo "0")
    done
    
    local variance=$(echo "scale=6; $sum_sq / ($count - 1)" | bc -l 2>/dev/null || echo "0")
    echo "scale=3; sqrt($variance)" | bc -l 2>/dev/null || echo "N/A"
}

# Function to extract timing data from CSV
extract_timings() {
    local csv_file="$1"
    local column="$2"
    
    if [[ ! -f "$csv_file" ]]; then
        echo "N/A"
        return
    fi
    
    # Skip header and extract column values
    tail -n +2 "$csv_file" | cut -d',' -f"$column" | tr '\n' ' '
}

# Function to analyze convergence
analyze_convergence() {
    local inertia_file="$1"
    
    if [[ ! -f "$inertia_file" ]]; then
        echo "N/A"
        return
    fi
    
            # Extract inertia values (skip header)
        local inertia_values=()
        while IFS= read -r line; do
            inertia_values+=("$line")
        done < <(tail -n +2 "$inertia_file" | cut -d',' -f2)
        
        local count=${#inertia_values[@]}
        
        if [[ $count -lt 2 ]]; then
            echo "N/A"
            return
        fi
        
        # Check for monotonicity
        local monotonic=true
        local prev=${inertia_values[0]}
        for val in "${inertia_values[@]:1}"; do
            if [[ "$val" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$prev" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                if (( $(echo "$val > $prev + 0.000001" | bc -l 2>/dev/null || echo "0") )); then
                    monotonic=false
                    break
                fi
            fi
            prev=$val
        done
        
        # Calculate convergence metrics
        local initial=${inertia_values[0]}
        local final=${inertia_values[$((${#inertia_values[@]}-1))]}
        local improvement="N/A"
        if [[ "$initial" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$final" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$initial" != "0" ]]; then
            # Handle scientific notation by converting to decimal
            local initial_decimal=$(echo "$initial" | sed 's/e+/*10^/' | sed 's/e-/*10^-/' | bc -l 2>/dev/null || echo "$initial")
            local final_decimal=$(echo "$final" | sed 's/e+/*10^/' | sed 's/e-/*10^-/' | bc -l 2>/dev/null || echo "$final")
            
            if [[ "$initial_decimal" != "0" ]]; then
                improvement=$(echo "scale=2; (($initial_decimal - $final_decimal) / $initial_decimal) * 100" | bc -l 2>/dev/null || echo "N/A")
            fi
        fi
        
        echo "Initial: $initial, Final: $final, Improvement: ${improvement}%, Monotonic: $monotonic"
}

# Main function to generate summary
generate_summary() {
    local experiment_dir="$1"
    local summary_file="$2"
    
    print_status "Generating summary for experiment: $experiment_dir"
    
    # Check if experiment directory exists
    if [[ ! -d "$experiment_dir" ]]; then
        print_error "Experiment directory not found: $experiment_dir"
        exit 1
    fi
    
    # Initialize summary file
    cat > "$summary_file" << EOF
# COSC3500 M1 K-Means Experiment Summary
# Generated: $(date)
# Experiment: $experiment_dir

EOF
    
    # 1. System Information
    print_status "Analyzing system information..."
    local sysinfo_files=($(find "$experiment_dir" -name "sysinfo_e*.txt" 2>/dev/null))
    if [[ ${#sysinfo_files[@]} -gt 0 ]]; then
        local sysinfo_file=${sysinfo_files[0]}
        echo "## System Information" >> "$summary_file"
        echo "Source: $(basename "$sysinfo_file")" >> "$summary_file"
        echo "" >> "$summary_file"
        
        # Extract key system info
        if [[ -f "$sysinfo_file" ]]; then
            echo "### Build Environment" >> "$summary_file"
            grep -E "^(Compiler|Makefile|Environment|Git)" "$sysinfo_file" 2>/dev/null | sed 's/^/- /' >> "$summary_file" || echo "- No build environment info found" >> "$summary_file"
            echo "" >> "$summary_file"
            
            echo "### System Details" >> "$summary_file"
            grep -E "^(System|CPU|Memory)" "$sysinfo_file" 2>/dev/null | sed 's/^/- /' >> "$summary_file" || echo "- System: $(grep '^System:' "$sysinfo_file" 2>/dev/null | sed 's/^System: /- /')" >> "$summary_file"
            echo "" >> "$summary_file"
            
            # Slurm info if available
            if grep -q "SLURM" "$sysinfo_file" 2>/dev/null; then
                echo "### HPC Cluster Information" >> "$summary_file"
                grep "SLURM" "$sysinfo_file" 2>/dev/null | sed 's/^/- /' >> "$summary_file"
                echo "" >> "$summary_file"
            fi
        fi
    else
        echo "## System Information" >> "$summary_file"
        echo "No system information file found." >> "$summary_file"
        echo "" >> "$summary_file"
    fi
    
    # 2. Configuration Analysis
    print_status "Analyzing experiment configurations..."
    echo "## Experiment Configurations" >> "$summary_file"
    
    # Find all metadata files
    local meta_files=($(find "$experiment_dir" -name "meta_*.txt" | sort))
    if [[ ${#meta_files[@]} -gt 0 ]]; then
        local first_meta=${meta_files[0]}
        echo "### Data Generation Parameters" >> "$summary_file"
        grep "Data Params" "$first_meta" | sed 's/^/- /' >> "$summary_file"
        echo "" >> "$summary_file"
        
        echo "### Build Configuration" >> "$summary_file"
        grep "ANTIVEC\|DEFS\|Build definitions" "$first_meta" | sed 's/^/- /' >> "$summary_file"
        echo "" >> "$summary_file"
    fi
    
    # 3. Performance Analysis
    print_status "Analyzing performance data..."
    echo "## Performance Analysis" >> "$summary_file"
    
    # Group files by configuration
    local configs=()
    for file in $(find "$experiment_dir" -name "times_*.csv" | sort); do
        local basename=$(basename "$file" .csv)
        local config=$(echo "$basename" | sed 's/_run[0-9]*$//')
        if [[ ! " ${configs[@]} " =~ " ${config} " ]]; then
            configs+=("$config")
        fi
    done
    
    # Debug: print found configurations
    print_status "Found configurations: ${configs[*]}"
    
    for config in "${configs[@]}"; do
        echo "### Configuration: $config" >> "$summary_file"
        
        # Find all runs for this configuration
        local run_files=($(find "$experiment_dir" -name "${config}_run*.csv" | sort))
        local single_run_files=($(find "$experiment_dir" -name "${config}.csv" | sort))
        local measurement_runs=()
        
        # Handle both run-based and single files
        if [[ ${#run_files[@]} -gt 0 ]]; then
            # Separate warm-up and measurement runs
            for file in "${run_files[@]}"; do
                local run_num=$(echo "$file" | grep -o '_run[0-9]*' | sed 's/_run//')
                if [[ $run_num -gt 3 ]]; then
                    measurement_runs+=("$file")
                fi
            done
        elif [[ ${#single_run_files[@]} -gt 0 ]]; then
            # Single file without run numbers
            measurement_runs=("${single_run_files[@]}")
        fi
        
        if [[ ${#measurement_runs[@]} -gt 0 ]]; then
            # Determine run type description
            local run_description=""
            if [[ ${#run_files[@]} -gt 0 ]]; then
                run_description=" (runs 4-8)"
            else
                run_description=" (single run)"
            fi
            
            echo "**Measurement Runs:** ${#measurement_runs[@]}$run_description" >> "$summary_file"
            echo "" >> "$summary_file"
            
            # Extract timing data for each run
            local assign_times=()
            local update_times=()
            local total_times=()
            
            for run_file in "${measurement_runs[@]}"; do
                # Calculate total time per run (sum of all iterations)
                local run_assign=$(tail -n +2 "$run_file" | cut -d',' -f2 | tr '\n' '+' | sed 's/+$//' | bc -l 2>/dev/null || echo "0")
                local run_update=$(tail -n +2 "$run_file" | cut -d',' -f3 | tr '\n' '+' | sed 's/+$//' | bc -l 2>/dev/null || echo "0")
                local run_total=$(tail -n +2 "$run_file" | cut -d',' -f4 | tr '\n' '+' | sed 's/+$//' | bc -l 2>/dev/null || echo "0")
                
                # Validate that we got numeric values
                if [[ "$run_assign" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$run_update" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$run_total" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                    assign_times+=("$run_assign")
                    update_times+=("$run_update")
                    total_times+=("$run_total")
                else
                    print_warning "Invalid timing data in $run_file: assign=$run_assign, update=$run_update, total=$run_total"
                fi
            done
            
            # Calculate statistics
            local assign_median=$(median "${assign_times[@]}")
            local assign_mean=$(mean "${assign_times[@]}")
            local assign_std=$(stddev "${assign_times[@]}")
            
            local update_median=$(median "${update_times[@]}")
            local update_mean=$(mean "${update_times[@]}")
            local update_std=$(stddev "${update_times[@]}")
            
            local total_median=$(median "${total_times[@]}")
            local total_mean=$(mean "${total_times[@]}")
            local total_std=$(stddev "${total_times[@]}")
            
            echo "**Timing Statistics (milliseconds):**" >> "$summary_file"
            echo "- Assign Kernel:" >> "$summary_file"
            echo "  - Median: $assign_median" >> "$summary_file"
            echo "  - Mean: $assign_mean" >> "$summary_file"
            echo "  - Std Dev: $assign_std" >> "$summary_file"
            echo "- Update Kernel:" >> "$summary_file"
            echo "  - Median: $update_median" >> "$summary_file"
            echo "  - Mean: $update_mean" >> "$summary_file"
            echo "  - Std Dev: $update_std" >> "$summary_file"
            echo "- Total Time:" >> "$summary_file"
            echo "  - Median: $total_median" >> "$summary_file"
            echo "  - Mean: $total_mean" >> "$summary_file"
            echo "  - Std Dev: $total_std" >> "$summary_file"
            echo "" >> "$summary_file"
            
            # Calculate percentage breakdown
            if [[ "$total_median" != "N/A" && "$total_median" != "0" ]]; then
                local assign_pct=$(echo "scale=1; ($assign_median / $total_median) * 100" | bc -l 2>/dev/null || echo "N/A")
                local update_pct=$(echo "scale=1; ($update_median / $total_median) * 100" | bc -l 2>/dev/null || echo "N/A")
                
                # Validate percentages and ensure they add up to 100%
                if [[ "$assign_pct" != "N/A" && "$update_pct" != "N/A" ]]; then
                    local total_pct=$(echo "scale=1; $assign_pct + $update_pct" | bc -l 2>/dev/null || echo "N/A")
                    echo "**Time Distribution:**" >> "$summary_file"
                    echo "- Assign Kernel: ${assign_pct}%" >> "$summary_file"
                    echo "- Update Kernel: ${update_pct}%" >> "$summary_file"
                    echo "- Total: ${total_pct}%" >> "$summary_file"
                    echo "" >> "$summary_file"
                else
                    echo "**Time Distribution:**" >> "$summary_file"
                    echo "- Assign Kernel: ${assign_pct}" >> "$summary_file"
                    echo "- Update Kernel: ${update_pct}" >> "$summary_file"
                    echo "" >> "$summary_file"
                fi
            fi
        else
            echo "**No measurement runs found for this configuration.**" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
        
        # Analyze convergence for this configuration
        # Convert times_ config to inertia_ config
        local inertia_config=$(echo "$config" | sed 's/^times_/inertia_/')
        local inertia_file=$(find "$experiment_dir" -name "${inertia_config}_run*.csv" | head -1)
        if [[ -z "$inertia_file" ]]; then
            inertia_file=$(find "$experiment_dir" -name "${inertia_config}.csv" | head -1)
        fi
        if [[ -n "$inertia_file" ]]; then
            echo "**Convergence Analysis:**" >> "$summary_file"
            local convergence=$(analyze_convergence "$inertia_file")
            echo "- $convergence" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
    done
    
    # 4. Profiling Results
    print_status "Analyzing profiling results..."
    echo "## Profiling Results" >> "$summary_file"
    
    local profiling_found=false
    
    # gprof results
    local gprof_files=($(find "$experiment_dir" -name "gprof_e*.txt" 2>/dev/null))
    if [[ ${#gprof_files[@]} -gt 0 ]]; then
        local gprof_file=${gprof_files[0]}
        echo "### Function-Level Profiling (gprof)" >> "$summary_file"
        echo "Source: $(basename "$gprof_file")" >> "$summary_file"
        echo "" >> "$summary_file"
        
        # Extract top functions by time
        if [[ -f "$gprof_file" ]]; then
            echo "**Top Functions by Time:**" >> "$summary_file"
            grep -A 20 "time   seconds" "$gprof_file" 2>/dev/null | head -25 | sed 's/^/- /' >> "$summary_file" || echo "- No function timing data found" >> "$summary_file"
            echo "" >> "$summary_file"
            profiling_found=true
        fi
    fi
    
    # cachegrind results
    local cachegrind_files=($(find "$experiment_dir" -name "cachegrind_e*.txt" 2>/dev/null))
    if [[ ${#cachegrind_files[@]} -gt 0 ]]; then
        local cachegrind_file=${cachegrind_files[0]}
        echo "### Cache Performance Analysis (cachegrind)" >> "$summary_file"
        echo "Source: $(basename "$cachegrind_file")" >> "$summary_file"
        echo "" >> "$summary_file"
        
        if [[ -f "$cachegrind_file" ]]; then
            echo "**Cache Miss Summary:**" >> "$summary_file"
            grep -E "(D1|LL) misses" "$cachegrind_file" 2>/dev/null | sed 's/^/- /' >> "$summary_file" || echo "- No cache miss data found" >> "$summary_file"
            echo "" >> "$summary_file"
            profiling_found=true
        fi
    fi
    
    # Normalized cache statistics
    local cache_stats_files=($(find "$experiment_dir" -name "cache_stats_e*.txt" 2>/dev/null))
    if [[ ${#cache_stats_files[@]} -gt 0 ]]; then
        local cache_stats_file=${cache_stats_files[0]}
        echo "### Normalized Cache Statistics" >> "$summary_file"
        echo "Source: $(basename "$cache_stats_file")" >> "$summary_file"
        echo "" >> "$summary_file"
        
        if [[ -f "$cache_stats_file" ]]; then
            cat "$cache_stats_file" | sed 's/^/- /' >> "$summary_file"
            echo "" >> "$summary_file"
            profiling_found=true
        fi
    fi
    
    # macOS sample profiling
    local sample_files=($(find "$experiment_dir" -name "*sample*" -not -name "*.md" 2>/dev/null))
    if [[ ${#sample_files[@]} -gt 0 ]]; then
        local sample_file=${sample_files[0]}
        echo "### macOS Sample Profiling" >> "$summary_file"
        echo "Source: $(basename "$sample_file")" >> "$summary_file"
        echo "" >> "$summary_file"
        
        if [[ -f "$sample_file" ]]; then
            echo "**Sample Profile Data:**" >> "$summary_file"
            head -20 "$sample_file" | sed 's/^/- /' >> "$summary_file"
            echo "" >> "$summary_file"
            profiling_found=true
        fi
    fi
    
    if [[ "$profiling_found" == false ]]; then
        echo "No profiling data found for this experiment." >> "$summary_file"
        echo "" >> "$summary_file"
    fi
    
    # 5. File Summary
    print_status "Generating file summary..."
    echo "## File Summary" >> "$summary_file"
    
    local total_files=$(find "$experiment_dir" -type f | wc -l)
    local timing_files=$(find "$experiment_dir" -name "times_*.csv" | wc -l)
    local inertia_files=$(find "$experiment_dir" -name "inertia_*.csv" | wc -l)
    local meta_files=$(find "$experiment_dir" -name "meta_*.txt" | wc -l)
    local profiling_files=$(find "$experiment_dir" -name "*gprof*" -o -name "*cachegrind*" -o -name "*cache_stats*" | wc -l)
    
    echo "**Total Files:** $total_files" >> "$summary_file"
    echo "**Timing Files:** $timing_files" >> "$summary_file"
    echo "**Inertia Files:** $inertia_files" >> "$summary_file"
    echo "**Metadata Files:** $meta_files" >> "$summary_file"
    echo "**Profiling Files:** $profiling_files" >> "$summary_file"
    echo "" >> "$summary_file"
    
    echo "**File Structure:**" >> "$summary_file"
    find "$experiment_dir" -type f | sort | sed 's/^/- /' >> "$summary_file"
    
    print_success "Summary generated successfully: $summary_file"
}

# Main execution
main() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 <experiment_number> [output_file]"
        echo "Example: $0 0"
        echo "Example: $0 0 summary_e0.md"
        exit 1
    fi
    
    local experiment_num="$1"
    local experiment_dir="bench/e${experiment_num}"
    
    # Default output file
    local output_file="${experiment_dir}/summary_e${experiment_num}.md"
    if [[ $# -ge 2 ]]; then
        output_file="$2"
    fi
    
    # Check if bc is available for calculations
    if ! command -v bc &> /dev/null; then
        print_warning "bc calculator not found. Some statistics may show 'N/A'."
        print_warning "Install bc for full statistical analysis: brew install bc (macOS) or apt-get install bc (Ubuntu)"
    fi
    
    # Generate summary
    generate_summary "$experiment_dir" "$output_file"
    
    # Display summary location
    echo ""
    print_success "Experiment summary saved to: $output_file"
    echo ""
    print_status "Summary includes:"
    echo "  - System and build information"
    echo "  - Performance statistics (median, mean, std dev)"
    echo "  - Convergence analysis"
    echo "  - Profiling results"
    echo "  - Complete file inventory"
}

# Run main function
main "$@"
