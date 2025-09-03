#include "kernels.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cstdlib>
#include <cstring>

// Forward declarations
void capture_metadata(const std::string& experiment, size_t n, size_t d, size_t k, size_t iters, unsigned seed);

// Parse command line arguments
bool parse_args(int argc, char* argv[], size_t& n, size_t& d, size_t& k, size_t& iters, unsigned& seed) {
    // Default values
    n = 200000;
    d = 16;
    k = 8;
    iters = 20;
    seed = 1;
    
    // Parse arguments
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) return false;
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--n") n = std::stoul(value);
        else if (arg == "--d") d = std::stoul(value);
        else if (arg == "--k") k = std::stoul(value);
        else if (arg == "--iters") iters = std::stoul(value);
        else if (arg == "--seed") seed = static_cast<unsigned>(std::stoul(value));
        else return false;
    }
    
    return true;
}

// Generate synthetic data with Gaussian blobs
void generate_data(Data& data, unsigned seed) {
    std::mt19937 gen(seed);
    
    // More challenging parameters for clustering
    float noise_std = 1.5f;  // Increased noise
    float center_range = 3.0f;  // Reduced center range (closer clusters)
    
    std::normal_distribution<float> noise_dist(0.0f, noise_std);
    std::uniform_real_distribution<float> center_dist(-center_range, center_range);
    std::uniform_int_distribution<int> cluster_dist(0, static_cast<int>(data.K - 1));
    
    // Generate K random centers in smaller range (closer together)
    std::vector<std::vector<float>> centers(data.K, std::vector<float>(data.D));
    for (size_t k = 0; k < data.K; ++k) {
        for (size_t dim = 0; dim < data.D; ++dim) {
            centers[k][dim] = center_dist(gen);
        }
    }
    
    // Generate N points: assign to random center + increased noise
    for (size_t i = 0; i < data.N; ++i) {
        int cluster = cluster_dist(gen);
        for (size_t dim = 0; dim < data.D; ++dim) {
            data.points[i * data.D + dim] = centers[static_cast<size_t>(cluster)][dim] + noise_dist(gen);
        }
    }
    
    // Initialize centroids to RANDOM positions (not the true centers)
    // This makes clustering much more challenging
    std::uniform_real_distribution<float> centroid_dist(-8.0f, 8.0f);
    for (size_t k = 0; k < data.K; ++k) {
        for (size_t dim = 0; dim < data.D; ++dim) {
            data.centroids[k * data.D + dim] = centroid_dist(gen);
        }
    }
}

// Capture run metadata for benchmarking
void capture_metadata(const std::string& experiment, size_t n, size_t d, size_t k, size_t iters, unsigned seed, const std::string& run_suffix = "") {
    // Create experiment directory
    std::string exp_dir = "bench/" + experiment;
    std::filesystem::create_directories(exp_dir);
    
    // Create metadata filename
    std::string meta_filename = exp_dir + "/meta_N" + std::to_string(n) + "_D" + std::to_string(d) + 
                               "_K" + std::to_string(k) + "_iters" + std::to_string(iters) + 
                               "_seed" + std::to_string(seed) + run_suffix + ".txt";
    
    std::ofstream meta_file(meta_filename);
    if (meta_file.is_open()) {
        meta_file << "Experiment: " << experiment << "\n";
        meta_file << "Timestamp: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n";
        
        // Git SHA if available
        std::string git_sha = "unknown";
        FILE* git_pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
        if (git_pipe) {
            char buffer[128];
            if (fgets(buffer, sizeof(buffer), git_pipe) != nullptr) {
                git_sha = std::string(buffer);
                git_sha.erase(git_sha.find_last_not_of(" \n\r\t") + 1);
            }
            pclose(git_pipe);
        }
        meta_file << "Git SHA: " << git_sha << "\n";
        
        // Compiler and flags
        const char* cxx = std::getenv("CXX");
        const char* cxxflags = std::getenv("CXXFLAGS");
        meta_file << "Compiler: " << (cxx ? cxx : "c++") << "\n";
        meta_file << "Flags: " << (cxxflags ? cxxflags : "(default)") << "\n";
        
        // ANTIVEC value
        const char* antivec = std::getenv("ANTIVEC");
        meta_file << "ANTIVEC: " << (antivec ? antivec : "not set") << "\n";
        
        // CLI arguments
        meta_file << "CLI Args: --n " << n << " --d " << d << " --k " << k 
                  << " --iters " << iters << " --seed " << seed << "\n";
        
        // Data generation parameters
        meta_file << "Data Params: noise_std=1.5, center_range=3.0, init=random\n";
        
        meta_file.close();
    }
}

int main(int argc, char* argv[]) {
    size_t n, d, k, iters;
    unsigned seed;
    
    if (!parse_args(argc, argv, n, d, k, iters, seed)) {
        std::cerr << "Usage: " << argv[0] << " [--n N] [--d D] [--k K] [--iters ITERS] [--seed SEED]" << std::endl;
        return 1;
    }
    
    // Initialize data structure
    Data data;
    data.N = n;
    data.D = d;
    data.K = k;
    data.points.resize(n * d);
    data.centroids.resize(k * d);
    data.labels.resize(n);
    
    // Generate synthetic data
    generate_data(data, seed);
    
    // Capture metadata for benchmarking
    const char* exp_env = std::getenv("EXPERIMENT");
    std::string experiment = exp_env ? exp_env : "e0";
    
    // Get run number for multiple runs (warm-up + measurement)
    const char* run_env = std::getenv("RUN_NUM");
    std::string run_suffix = run_env ? "_run" + std::string(run_env) : "";
    
    capture_metadata(experiment, n, d, k, iters, seed, run_suffix);
    
    // Prepare timing vectors
    std::vector<double> assign_ms(iters);
    std::vector<double> update_ms(iters);
    std::vector<double> inertia_vals(iters);
    
    // Run Lloyd iterations
    for (size_t iter = 0; iter < iters; ++iter) {
        // Time assign kernel
        auto assign_start = std::chrono::steady_clock::now();
        assign_labels(data);
        auto assign_end = std::chrono::steady_clock::now();
        assign_ms[iter] = std::chrono::duration<double, std::milli>(assign_end - assign_start).count();
        
        // Time update kernel
        auto update_start = std::chrono::steady_clock::now();
        update_centroids(data);
        auto update_end = std::chrono::steady_clock::now();
        update_ms[iter] = std::chrono::duration<double, std::milli>(update_end - update_start).count();
        
        // Compute inertia (not timed)
        inertia_vals[iter] = inertia(data);
        
        // Check monotonicity
        if (iter > 0 && inertia_vals[iter] > inertia_vals[iter-1] + 1e-6) {
            std::cerr << "Warning: inertia increased at iteration " << iter 
                      << " (" << inertia_vals[iter-1] << " -> " << inertia_vals[iter] << ")" << std::endl;
        }
    }
    
    // Write timing CSV
    std::string times_filename = "bench/" + experiment + "/times_N" + std::to_string(n) + "_D" + std::to_string(d) + 
                                "_K" + std::to_string(k) + "_iters" + std::to_string(iters) + 
                                "_seed" + std::to_string(seed) + run_suffix + ".csv";
    std::ofstream times_file(times_filename);
    times_file << "iter,assign_ms,update_ms,total_ms\n";
    for (size_t iter = 0; iter < iters; ++iter) {
        double total_ms = assign_ms[iter] + update_ms[iter];
        times_file << iter << "," << assign_ms[iter] << "," << update_ms[iter] << "," << total_ms << "\n";
    }
    times_file.close();
    
    // Write inertia CSV
    std::string inertia_filename = "bench/" + experiment + "/inertia_N" + std::to_string(n) + "_D" + std::to_string(d) + 
                                  "_K" + std::to_string(k) + "_iters" + std::to_string(iters) + 
                                  "_seed" + std::to_string(seed) + run_suffix + ".csv";
    std::ofstream inertia_file(inertia_filename);
    inertia_file << "iter,inertia,N,D,K,iters,seed\n";
    for (size_t iter = 0; iter < iters; ++iter) {
        inertia_file << iter << "," << inertia_vals[iter] << "," << n << "," << d << "," 
                     << k << "," << iters << "," << seed << "\n";
    }
    inertia_file.close();
    
    // Print summary
    double total_assign = 0.0, total_update = 0.0;
    for (size_t iter = 0; iter < iters; ++iter) {
        total_assign += assign_ms[iter];
        total_update += update_ms[iter];
    }
    double total_time = total_assign + total_update;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Final inertia: " << inertia_vals[iters-1] << std::endl;
    std::cout << "Time split: assign=" << (total_assign/total_time*100) << "%, update=" 
              << (total_update/total_time*100) << "%" << std::endl;
    
    return 0;
}
