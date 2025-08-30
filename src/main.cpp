#include "kernels.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>
#include <iomanip>

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
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> center_dist(-5.0f, 5.0f);
    std::uniform_int_distribution<int> cluster_dist(0, static_cast<int>(data.K - 1));
    
    // Generate K random centers in [-5,5]^D
    std::vector<std::vector<float>> centers(data.K, std::vector<float>(data.D));
    for (size_t k = 0; k < data.K; ++k) {
        for (size_t dim = 0; dim < data.D; ++dim) {
            centers[k][dim] = center_dist(gen);
        }
    }
    
    // Generate N points: assign to random center + N(0,1) noise
    for (size_t i = 0; i < data.N; ++i) {
        int cluster = cluster_dist(gen);
        for (size_t dim = 0; dim < data.D; ++dim) {
            data.points[i * data.D + dim] = centers[static_cast<size_t>(cluster)][dim] + noise_dist(gen);
        }
    }
    
    // Initialize centroids to the generated centers
    for (size_t k = 0; k < data.K; ++k) {
        for (size_t dim = 0; dim < data.D; ++dim) {
            data.centroids[k * data.D + dim] = centers[k][dim];
        }
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
    
    // Ensure bench directory exists
    std::filesystem::create_directories("bench");
    
    // Write timing CSV
    std::string times_filename = "bench/times_N" + std::to_string(n) + "_D" + std::to_string(d) + 
                                "_K" + std::to_string(k) + "_iters" + std::to_string(iters) + 
                                "_seed" + std::to_string(seed) + ".csv";
    std::ofstream times_file(times_filename);
    times_file << "iter,assign_ms,update_ms,total_ms\n";
    for (size_t iter = 0; iter < iters; ++iter) {
        double total_ms = assign_ms[iter] + update_ms[iter];
        times_file << iter << "," << assign_ms[iter] << "," << update_ms[iter] << "," << total_ms << "\n";
    }
    times_file.close();
    
    // Write inertia CSV
    std::string inertia_filename = "bench/inertia_N" + std::to_string(n) + "_D" + std::to_string(d) + 
                                  "_K" + std::to_string(k) + "_iters" + std::to_string(iters) + 
                                  "_seed" + std::to_string(seed) + ".csv";
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
