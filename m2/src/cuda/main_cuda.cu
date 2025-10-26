#include "../../include/kmeans_cuda.cuh"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstring>

// Function to write metadata file (same format as M1)
void write_metadata(const std::string& base_filename, const KMeansConfig& config) {
    std::string meta_filename = base_filename.substr(0, base_filename.find_last_of('.')) + "_meta.txt";
    std::ofstream meta_file(meta_filename);

    if (!meta_file.is_open()) {
        std::cerr << "Warning: Could not create metadata file: " << meta_filename << std::endl;
        return;
    }

    meta_file << "K-Means CUDA Implementation Metadata\n";
    meta_file << "=====================================\n";
    meta_file << "Date: " << __DATE__ << " " << __TIME__ << "\n";
    meta_file << "CUDA Version: " << CUDART_VERSION << "\n";
    meta_file << "Implementation: CUDA\n";
    meta_file << "\nConfiguration:\n";
    meta_file << "  N (points): " << config.N << "\n";
    meta_file << "  D (dimensions): " << config.D << "\n";
    meta_file << "  K (clusters): " << config.K << "\n";
    meta_file << "  Max iterations: " << config.max_iters << "\n";
    meta_file << "  Random seed: " << config.seed << "\n";
    meta_file << "  CUDA block size: " << config.block_size << "\n";
    meta_file << "  Warmup iterations: " << config.warmup_iters << "\n";
    meta_file << "  Benchmark iterations: " << config.bench_iters << "\n";

    // Add GPU information
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    meta_file << "\nGPU Information:\n";
    meta_file << "  Device: " << props.name << "\n";
    meta_file << "  Compute capability: " << props.major << "." << props.minor << "\n";
    meta_file << "  Global memory: " << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    meta_file << "  SMs: " << props.multiProcessorCount << "\n";

    meta_file.close();
}

// Function to write timing results to CSV (same format as M1)
void write_timing_csv(const std::string& filename, const KMeansTimers& timers) {
    std::ofstream csv_file(filename);

    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not create CSV file: " << filename << std::endl;
        return;
    }

    // Header (same as M1)
    csv_file << "iteration,assign_ms,update_ms,total_ms\n";

    // Write per-iteration data
    for (size_t i = 0; i < timers.iter_assign_times.size(); ++i) {
        csv_file << i + 1 << ","
                 << std::fixed << std::setprecision(6)
                 << timers.iter_assign_times[i] << ","
                 << timers.iter_update_times[i] << ","
                 << timers.iter_total_times[i] << "\n";
    }

    csv_file.close();
}

// Function to write inertia log (same format as M1)
void write_inertia_log(const std::string& base_filename, const std::vector<double>& inertias) {
    std::string inertia_filename = base_filename.substr(0, base_filename.find_last_of('.')) + "_inertia.txt";
    std::ofstream inertia_file(inertia_filename);

    if (!inertia_file.is_open()) {
        std::cerr << "Warning: Could not create inertia file: " << inertia_filename << std::endl;
        return;
    }

    inertia_file << "# K-Means Inertia per Iteration\n";
    inertia_file << "# iteration,inertia\n";

    for (size_t i = 0; i < inertias.size(); ++i) {
        inertia_file << i + 1 << "," << std::scientific << std::setprecision(10) << inertias[i] << "\n";
    }

    inertia_file.close();
}

int main(int argc, char** argv) {
    // Parse command-line arguments
    KMeansConfig config = KMeansUtils::parse_arguments(argc, argv);

    // Print configuration
    std::cout << "=== CUDA K-Means Configuration ===" << std::endl;
    std::cout << "N: " << config.N << " points" << std::endl;
    std::cout << "D: " << config.D << " dimensions" << std::endl;
    std::cout << "K: " << config.K << " clusters" << std::endl;
    std::cout << "Max iterations: " << config.max_iters << std::endl;
    std::cout << "Random seed: " << config.seed << std::endl;
    std::cout << "CUDA block size: " << config.block_size << std::endl;
    std::cout << "Warmup iterations: " << config.warmup_iters << std::endl;
    std::cout << "Benchmark iterations: " << config.bench_iters << std::endl;

    // Print GPU information
    KMeansCUDA cuda_impl;
    cuda_impl.print_device_info();
    std::cout << std::endl;

    // Create data structure
    KMeansData data(config.N, config.D, config.K);

    // Generate or load data
    if (config.input_file.empty()) {
        std::cout << "Generating synthetic data..." << std::endl;
        KMeansUtils::generate_synthetic_data(data, config.seed);
    } else {
        std::cout << "Loading data from: " << config.input_file << std::endl;
        KMeansUtils::load_data(data, config.input_file);
    }

    // Warmup runs (not measured)
    if (config.warmup_iters > 0) {
        std::cout << "\nPerforming " << config.warmup_iters << " warmup runs..." << std::endl;
        for (int w = 0; w < config.warmup_iters; ++w) {
            KMeansData warmup_data(config.N, config.D, config.K);
            warmup_data.points = data.points;  // Copy input data
            KMeansTimers warmup_timers;

            cuda_impl.run(warmup_data, config, warmup_timers);

            std::cout << "  Warmup " << (w + 1) << ": "
                      << std::fixed << std::setprecision(3) << warmup_timers.total_time << " ms" << std::endl;
        }
    }

    // Benchmark runs
    std::cout << "\nPerforming " << config.bench_iters << " benchmark runs..." << std::endl;
    std::vector<KMeansTimers> all_timers;
    std::vector<double> final_inertias;
    std::vector<std::vector<double>> all_inertias;  // For tracking convergence

    for (int b = 0; b < config.bench_iters; ++b) {
        // Create fresh data copy for each run
        KMeansData run_data(config.N, config.D, config.K);
        run_data.points = data.points;

        KMeansTimers run_timers;

        // Track inertias for this run
        std::vector<double> run_inertias;

        // Custom run loop to track inertias (similar to M1)
        Timer total_timer;
        total_timer.start();

        // Initialize
        Timer init_timer;
        init_timer.start();
        cuda_impl.initialize(run_data, config);
        run_timers.init_time = init_timer.elapsed_ms();

        double prev_inertia = std::numeric_limits<double>::max();

        // Main iteration loop
        for (size_t iter = 0; iter < config.max_iters; ++iter) {
            Timer iter_timer;
            iter_timer.start();

            // Assignment step
            Timer assign_timer;
            assign_timer.start();
            cuda_impl.assign_labels(run_data);
            cudaDeviceSynchronize();  // Ensure kernel completion
            double assign_time = assign_timer.elapsed_ms();
            run_timers.iter_assign_times.push_back(assign_time);

            // Update step
            Timer update_timer;
            update_timer.start();
            cuda_impl.update_centroids(run_data);
            cudaDeviceSynchronize();  // Ensure kernel completion
            double update_time = update_timer.elapsed_ms();
            run_timers.iter_update_times.push_back(update_time);

            double iter_time = iter_timer.elapsed_ms();
            run_timers.iter_total_times.push_back(iter_time);

            // Compute inertia
            double inertia = cuda_impl.compute_inertia(run_data);
            run_inertias.push_back(inertia);

            if (config.verbose && b == 0) {  // Only print for first benchmark run
                std::cout << "  Iteration " << std::setw(2) << (iter + 1) << ": "
                          << "assign=" << std::fixed << std::setprecision(3) << assign_time << "ms, "
                          << "update=" << std::setprecision(3) << update_time << "ms, "
                          << "inertia=" << std::scientific << std::setprecision(6) << inertia << std::endl;
            }

            // Check convergence
            if (std::abs(prev_inertia - inertia) < config.tolerance) {
                if (config.verbose && b == 0) {
                    std::cout << "  Converged at iteration " << iter + 1 << std::endl;
                }
                break;
            }
            prev_inertia = inertia;
        }

        run_timers.total_time = total_timer.elapsed_ms();

        // Calculate aggregate times
        run_timers.assign_time = std::accumulate(run_timers.iter_assign_times.begin(),
                                                 run_timers.iter_assign_times.end(), 0.0);
        run_timers.update_time = std::accumulate(run_timers.iter_update_times.begin(),
                                                 run_timers.iter_update_times.end(), 0.0);

        all_timers.push_back(run_timers);
        final_inertias.push_back(run_inertias.back());
        all_inertias.push_back(run_inertias);

        std::cout << "  Run " << (b + 1) << ": "
                  << std::fixed << std::setprecision(3) << run_timers.total_time << " ms, "
                  << "final inertia=" << std::scientific << std::setprecision(6) << run_inertias.back() << std::endl;
    }

    // Calculate statistics (median values like M1)
    std::cout << "\n=== Performance Summary ===" << std::endl;

    // Collect all assign times
    std::vector<double> all_assign_times;
    std::vector<double> all_update_times;
    std::vector<double> all_total_times;

    for (const auto& timer : all_timers) {
        all_total_times.push_back(timer.total_time);
        // Use median of each run's iterations
        std::vector<double> sorted_assign = timer.iter_assign_times;
        std::vector<double> sorted_update = timer.iter_update_times;
        std::sort(sorted_assign.begin(), sorted_assign.end());
        std::sort(sorted_update.begin(), sorted_update.end());
        if (!sorted_assign.empty()) {
            all_assign_times.push_back(sorted_assign[sorted_assign.size() / 2]);
            all_update_times.push_back(sorted_update[sorted_update.size() / 2]);
        }
    }

    // Calculate median across runs
    std::sort(all_assign_times.begin(), all_assign_times.end());
    std::sort(all_update_times.begin(), all_update_times.end());
    std::sort(all_total_times.begin(), all_total_times.end());

    double median_assign = all_assign_times[all_assign_times.size() / 2];
    double median_update = all_update_times[all_update_times.size() / 2];
    double median_total = all_total_times[all_total_times.size() / 2];

    std::cout << "Median total time: " << std::fixed << std::setprecision(3) << median_total << " ms" << std::endl;
    std::cout << "Median assign time: " << median_assign << " ms" << std::endl;
    std::cout << "Median update time: " << median_update << " ms" << std::endl;

    // Calculate MLUPS
    double mlups = KMeansUtils::calculate_mlups(config.N, config.K, config.D, median_assign);
    std::cout << "MLUPS: " << std::scientific << std::setprecision(3) << mlups << std::endl;

    // Report final inertias
    std::sort(final_inertias.begin(), final_inertias.end());
    std::cout << "Median final inertia: " << std::scientific << std::setprecision(10)
              << final_inertias[final_inertias.size() / 2] << std::endl;

    std::cout << "Iterations per run: " << all_timers[0].iter_assign_times.size() << std::endl;

    // Write output files if requested (using median run data)
    if (!config.output_file.empty()) {
        std::cout << "\nWriting results to: " << config.output_file << std::endl;

        // Write timing CSV (median run)
        size_t median_run_idx = all_timers.size() / 2;
        write_timing_csv(config.output_file, all_timers[median_run_idx]);

        // Write metadata
        write_metadata(config.output_file, config);

        // Write inertia log (median run)
        write_inertia_log(config.output_file, all_inertias[median_run_idx]);

        // Save final results if needed
        if (config.validate) {
            std::string results_file = config.output_file.substr(0, config.output_file.find_last_of('.')) + "_labels.csv";
            // Copy labels back from GPU for the last run
            KMeansData final_data(config.N, config.D, config.K);
            final_data.points = data.points;
            cuda_impl.initialize(final_data, config);
            // Run one more time to get final labels
            KMeansTimers final_timer;
            cuda_impl.run(final_data, config, final_timer);
            KMeansUtils::save_results(final_data, results_file);
        }
    }

    std::cout << "\nCUDA K-Means completed successfully!" << std::endl;

    return 0;
}