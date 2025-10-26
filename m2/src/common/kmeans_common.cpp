#include "../../include/kmeans_common.hpp"
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <numeric>

// Destructor implementation
KMeansData::~KMeansData() {
    // GPU memory cleanup handled by CUDA implementation
    // Host memory automatically cleaned up by vector destructors
}

// Base implementation of run method
void KMeansImplementation::run(KMeansData& data, const KMeansConfig& config, KMeansTimers& timers) {
    Timer total_timer, iter_timer;

    total_timer.start();

    // Initialize
    Timer init_timer;
    init_timer.start();
    initialize(data, config);
    timers.init_time = init_timer.elapsed_ms();

    if (config.verbose) {
        std::cout << "Starting " << name() << " K-Means with N=" << data.N
                  << ", D=" << data.D << ", K=" << data.K << std::endl;
    }

    double prev_inertia = std::numeric_limits<double>::max();

    // Main iteration loop
    for (size_t iter = 0; iter < config.max_iters; ++iter) {
        iter_timer.start();

        // Assignment step
        Timer assign_timer;
        assign_timer.start();
        assign_labels(data);
        double assign_time = assign_timer.elapsed_ms();
        timers.iter_assign_times.push_back(assign_time);

        // Update step
        Timer update_timer;
        update_timer.start();
        update_centroids(data);
        double update_time = update_timer.elapsed_ms();
        timers.iter_update_times.push_back(update_time);

        double iter_time = iter_timer.elapsed_ms();
        timers.iter_total_times.push_back(iter_time);

        // Compute inertia for convergence check
        double inertia = compute_inertia(data);

        if (config.verbose) {
            std::cout << "Iteration " << iter + 1 << ": "
                      << "assign=" << std::fixed << std::setprecision(3) << assign_time << "ms, "
                      << "update=" << update_time << "ms, "
                      << "inertia=" << std::scientific << std::setprecision(6) << inertia << std::endl;
        }

        // Check convergence
        if (std::abs(prev_inertia - inertia) < config.tolerance) {
            if (config.verbose) {
                std::cout << "Converged at iteration " << iter + 1 << std::endl;
            }
            break;
        }
        prev_inertia = inertia;
    }

    timers.total_time = total_timer.elapsed_ms();

    // Calculate aggregate times
    timers.assign_time = std::accumulate(timers.iter_assign_times.begin(),
                                         timers.iter_assign_times.end(), 0.0);
    timers.update_time = std::accumulate(timers.iter_update_times.begin(),
                                         timers.iter_update_times.end(), 0.0);
}

namespace KMeansUtils {

void generate_synthetic_data(KMeansData& data, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate random Gaussian blobs
    for (size_t i = 0; i < data.N * data.D; ++i) {
        data.points[i] = dist(rng);
    }

    // Scale and shift some clusters to create separation
    for (size_t i = 0; i < data.N; ++i) {
        int cluster = i % data.K;  // Assign rough clusters for data generation
        float scale = 0.1f;
        float shift = static_cast<float>(cluster * 2.0);

        for (size_t d = 0; d < data.D; ++d) {
            data.points[i * data.D + d] = data.points[i * data.D + d] * scale + shift;
        }
    }
}

void load_data(KMeansData& data, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Simple CSV format: each row is a point
    std::string line;
    size_t point_idx = 0;

    while (std::getline(file, line) && point_idx < data.N) {
        std::stringstream ss(line);
        std::string value;
        size_t dim_idx = 0;

        while (std::getline(ss, value, ',') && dim_idx < data.D) {
            data.points[point_idx * data.D + dim_idx] = std::stof(value);
            dim_idx++;
        }
        point_idx++;
    }
}

void save_results(const KMeansData& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    // Write header
    file << "point_id,label,inertia_contribution" << std::endl;

    // Write labels and distances
    for (size_t i = 0; i < data.N; ++i) {
        int label = data.labels[i];

        // Calculate distance to assigned centroid
        float dist = 0.0f;
        for (size_t d = 0; d < data.D; ++d) {
            float diff = data.points[i * data.D + d] - data.centroids[label * data.D + d];
            dist += diff * diff;
        }

        file << i << "," << label << "," << dist << std::endl;
    }
}

void random_init(KMeansData& data, int seed) {
    std::mt19937 rng(seed);
    std::vector<int> indices(data.N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Select first K points as initial centroids
    for (size_t k = 0; k < data.K; ++k) {
        size_t point_idx = indices[k];
        for (size_t d = 0; d < data.D; ++d) {
            data.centroids[k * data.D + d] = data.points[point_idx * data.D + d];
        }
    }
}

void kmeans_plusplus_init(KMeansData& data, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    // Choose first centroid randomly
    std::uniform_int_distribution<size_t> first_choice(0, data.N - 1);
    size_t first_idx = first_choice(rng);
    for (size_t d = 0; d < data.D; ++d) {
        data.centroids[0 * data.D + d] = data.points[first_idx * data.D + d];
    }

    // For each remaining centroid
    std::vector<float> min_distances(data.N, std::numeric_limits<float>::max());

    for (size_t k = 1; k < data.K; ++k) {
        // Update minimum distances
        for (size_t i = 0; i < data.N; ++i) {
            float dist = 0.0f;
            for (size_t d = 0; d < data.D; ++d) {
                float diff = data.points[i * data.D + d] - data.centroids[(k-1) * data.D + d];
                dist += diff * diff;
            }
            min_distances[i] = std::min(min_distances[i], dist);
        }

        // Choose next centroid with probability proportional to squared distance
        float total_dist = std::accumulate(min_distances.begin(), min_distances.end(), 0.0f);
        float threshold = uniform(rng) * total_dist;
        float cumsum = 0.0f;

        size_t chosen_idx = 0;
        for (size_t i = 0; i < data.N; ++i) {
            cumsum += min_distances[i];
            if (cumsum >= threshold) {
                chosen_idx = i;
                break;
            }
        }

        // Copy chosen point as new centroid
        for (size_t d = 0; d < data.D; ++d) {
            data.centroids[k * data.D + d] = data.points[chosen_idx * data.D + d];
        }
    }
}

bool validate_results(const KMeansData& result1, const KMeansData& result2, float tolerance) {
    if (result1.N != result2.N || result1.K != result2.K) {
        return false;
    }

    // Compare inertias
    double inertia1 = compute_inertia_reference(result1);
    double inertia2 = compute_inertia_reference(result2);

    if (std::abs(inertia1 - inertia2) > tolerance) {
        std::cerr << "Inertia mismatch: " << inertia1 << " vs " << inertia2
                  << " (diff: " << std::abs(inertia1 - inertia2) << ")" << std::endl;
        return false;
    }

    return true;
}

double compute_inertia_reference(const KMeansData& data) {
    double total_inertia = 0.0;

    for (size_t i = 0; i < data.N; ++i) {
        int label = data.labels[i];
        double dist = 0.0;

        for (size_t d = 0; d < data.D; ++d) {
            double diff = data.points[i * data.D + d] - data.centroids[label * data.D + d];
            dist += diff * diff;
        }

        total_inertia += dist;
    }

    return total_inertia;
}

void transpose_centroids(KMeansData& data) {
    // Transpose from [K×D] to [D×K]
    for (size_t k = 0; k < data.K; ++k) {
        for (size_t d = 0; d < data.D; ++d) {
            data.centroidsT[d * data.K + k] = data.centroids[k * data.D + d];
        }
    }
}

double calculate_mlups(size_t N, size_t K, size_t D, double time_ms) {
    double operations = static_cast<double>(N) * static_cast<double>(K) * static_cast<double>(D);
    double time_s = time_ms / 1000.0;
    return (operations / 1e6) / time_s;  // Million label updates per second
}

void print_performance_summary(const KMeansConfig& config, const KMeansTimers& timers) {
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << timers.total_time << " ms" << std::endl;
    std::cout << "Init time: " << timers.init_time << " ms" << std::endl;
    std::cout << "Assign time: " << timers.assign_time << " ms" << std::endl;
    std::cout << "Update time: " << timers.update_time << " ms" << std::endl;

    if (timers.transfer_time > 0) {
        std::cout << "Transfer time: " << timers.transfer_time << " ms" << std::endl;
    }

    // Calculate median times
    if (!timers.iter_assign_times.empty()) {
        std::vector<double> assign_sorted = timers.iter_assign_times;
        std::sort(assign_sorted.begin(), assign_sorted.end());
        double median_assign = assign_sorted[assign_sorted.size() / 2];

        std::vector<double> update_sorted = timers.iter_update_times;
        std::sort(update_sorted.begin(), update_sorted.end());
        double median_update = update_sorted[update_sorted.size() / 2];

        std::cout << "Median assign: " << median_assign << " ms" << std::endl;
        std::cout << "Median update: " << median_update << " ms" << std::endl;

        double mlups = calculate_mlups(config.N, config.K, config.D, median_assign);
        std::cout << "MLUPS: " << std::scientific << std::setprecision(2) << mlups << std::endl;
    }

    std::cout << "Iterations: " << timers.iter_assign_times.size() << std::endl;
}

KMeansConfig parse_arguments(int argc, char** argv) {
    KMeansConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "-N" || arg == "--points") && i + 1 < argc) {
            config.N = std::stoul(argv[++i]);
        } else if ((arg == "-D" || arg == "--dimensions") && i + 1 < argc) {
            config.D = std::stoul(argv[++i]);
        } else if ((arg == "-K" || arg == "--clusters") && i + 1 < argc) {
            config.K = std::stoul(argv[++i]);
        } else if ((arg == "-I" || arg == "--iterations") && i + 1 < argc) {
            config.max_iters = std::stoul(argv[++i]);
        } else if ((arg == "-S" || arg == "--seed") && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        } else if ((arg == "-T" || arg == "--threads") && i + 1 < argc) {
            config.num_threads = std::stoi(argv[++i]);
        } else if ((arg == "-B" || arg == "--block-size") && i + 1 < argc) {
            config.block_size = std::stoi(argv[++i]);
        } else if ((arg == "--warmup") && i + 1 < argc) {
            config.warmup_iters = std::stoi(argv[++i]);
        } else if ((arg == "--bench") && i + 1 < argc) {
            config.bench_iters = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            config.input_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            config.verbose = true;
        } else if (arg == "--validate") {
            config.validate = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -N, --points <num>       Number of points (default: 100000)" << std::endl;
            std::cout << "  -D, --dimensions <num>   Number of dimensions (default: 64)" << std::endl;
            std::cout << "  -K, --clusters <num>     Number of clusters (default: 32)" << std::endl;
            std::cout << "  -I, --iterations <num>   Maximum iterations (default: 100)" << std::endl;
            std::cout << "  -S, --seed <num>         Random seed (default: 42)" << std::endl;
            std::cout << "  -T, --threads <num>      Number of threads (OpenMP)" << std::endl;
            std::cout << "  -B, --block-size <num>   CUDA block size (default: 256)" << std::endl;
            std::cout << "  --warmup <num>           Warmup iterations (default: 3)" << std::endl;
            std::cout << "  --bench <num>            Benchmark iterations (default: 5)" << std::endl;
            std::cout << "  -i, --input <file>       Input data file" << std::endl;
            std::cout << "  -o, --output <file>      Output results file" << std::endl;
            std::cout << "  -v, --verbose            Verbose output" << std::endl;
            std::cout << "  --validate               Validate against reference" << std::endl;
            std::cout << "  -h, --help               Show this help" << std::endl;
            exit(0);
        }
    }

    return config;
}

} // namespace KMeansUtils