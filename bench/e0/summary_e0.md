# COSC3500 M1 K-Means Experiment Summary
# Generated: Wed  3 Sep 2025 16:01:06 AEST
# Experiment: bench/e0

## System Information
Source: sysinfo_e0.txt

### Build Environment
- Compiler: Apple clang version 17.0.0 (clang-1700.0.13.5)
- Makefile CXXFLAGS_BASE: -std=c++17 -O2 -Wall -Wextra -Wshadow -Wconversion -Iinclude
- Makefile ANTIVEC_FLAGS: 
- Environment CXX: g++
- Environment CXXFLAGS: not set
- Environment ANTIVEC: not set
- Environment DEFS: not set
- Git SHA: 7905c6c
- Git branch: main
- Git status: 13 files modified

### System Details
- System: Darwin Mac-3.modem 24.6.0 Darwin Kernel Version 24.6.0: Mon Jul 14 11:30:40 PDT 2025; root:xnu-11417.140.69~1/RELEASE_ARM64_T6041 arm64

## Experiment Configurations
### Data Generation Parameters
- Data Params: noise_std=1.5, center_range=3.0, init=random

### Build Configuration
- ANTIVEC: 1

## Performance Analysis
### Configuration: times_N100000_D64_K64_iters10_seed1
**Measurement Runs:** 5 (runs 4-8)

**Timing Statistics (milliseconds):**
- Assign Kernel:
  - Median: 1492.965
  - Mean: 1491.789
  - Std Dev: 4.364521
- Update Kernel:
  - Median: 17.02953
  - Mean: 17.064
  - Std Dev: .231997
- Total Time:
  - Median: 1509.705
  - Mean: 1508.853
  - Std Dev: 4.215427

**Time Distribution:**
- Assign Kernel: 90.0%
- Update Kernel: 0%

**Convergence Analysis:**
- Initial: 2.98381e+07, Final: 1.62581e+07, Improvement: N/A%, Monotonic: true

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
- Assign Kernel: 80.0%
- Update Kernel: 10.0%

**Convergence Analysis:**
- Initial: 1.32646e+07, Final: 7.98832e+06, Improvement: N/A%, Monotonic: true

### Configuration: times_N500000_D32_K32_iters20_seed1
**Measurement Runs:** 1 (single run)

**Timing Statistics (milliseconds):**
- Assign Kernel:
  - Median: 1808.3438
  - Mean: 1808.343
  - Std Dev: N/A
- Update Kernel:
  - Median: 35.33208
  - Mean: 35.332
  - Std Dev: N/A
- Total Time:
  - Median: 1843.6758
  - Mean: 1843.675
  - Std Dev: N/A

**Time Distribution:**
- Assign Kernel: 90.0%
- Update Kernel: 0%

**Convergence Analysis:**
- Initial: 7.4598e+07, Final: 3.88431e+07, Improvement: N/A%, Monotonic: true

## Profiling Results
### macOS Sample Profiling
Source: sample_stress_config.txt

**Sample Profile Data:**
- Sampling process 33503 for 5 seconds with 1 millisecond of run time between samples
- Sampling completed, processing symbols...
- Sample analysis of process 33503 written to file /tmp/kmeans_2025-09-03_153659_xftk.sample.txt
- 
- Analysis of sampling kmeans (pid 33503) every 1 millisecond
- Process:         kmeans [33503]
- Path:            /Users/USER/*/kmeans
- Load Address:    0x1025d4000
- Identifier:      kmeans
- Version:         0
- Code Type:       ARM64
- Platform:        macOS
- Parent Process:  bash [33278]
- Target Type:     live task
- 
- Date/Time:       2025-09-03 15:36:59.949 +1000
- Launch Time:     2025-09-03 15:36:57.898 +1000
- OS Version:      macOS 15.6.1 (24G90)
- Report Version:  7
- Analysis Tool:   /usr/bin/sample

## File Summary
**Total Files:**       54
**Timing Files:**       17
**Inertia Files:**       17
**Metadata Files:**       17
**Profiling Files:**        0

**File Structure:**
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run1.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run2.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run3.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run4.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run5.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run6.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run7.csv
- bench/e0/inertia_N100000_D64_K64_iters10_seed1_run8.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run1.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run2.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run3.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run4.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run5.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run6.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run7.csv
- bench/e0/inertia_N200000_D16_K8_iters10_seed1_run8.csv
- bench/e0/inertia_N500000_D32_K32_iters20_seed1.csv
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run1.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run2.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run3.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run4.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run5.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run6.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run7.txt
- bench/e0/meta_N100000_D64_K64_iters10_seed1_run8.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run1.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run2.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run3.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run4.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run5.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run6.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run7.txt
- bench/e0/meta_N200000_D16_K8_iters10_seed1_run8.txt
- bench/e0/meta_N500000_D32_K32_iters20_seed1.txt
- bench/e0/sample_stress_config.txt
- bench/e0/summary_e0.md
- bench/e0/sysinfo_e0.txt
- bench/e0/times_N100000_D64_K64_iters10_seed1_run1.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run2.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run3.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run4.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run5.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run6.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run7.csv
- bench/e0/times_N100000_D64_K64_iters10_seed1_run8.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run1.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run2.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run3.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run4.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run5.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run6.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run7.csv
- bench/e0/times_N200000_D16_K8_iters10_seed1_run8.csv
- bench/e0/times_N500000_D32_K32_iters20_seed1.csv
