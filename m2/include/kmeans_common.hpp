#pragma once

#include <vector>
#include <chrono>
#include <string>

// Data structure for K-means clustering (shared across all implementations)
struct KMeansData {
    // Core data arrays
    std::vector<float> points;      // N points of dimension D (row-major: [i*D + d])
    std::vector<float> centroids;   // K centroids of dimension D (row-major: [k*D + d])
    std::vector<int> labels;        // N labels (0 to K-1)

    // Dimensions
    size_t N;  // Number of points
    size_t D;  // Number of dimensions
    size_t K;  // Number of clusters

    // For GPU implementations
    float* d_points = nullptr;      // Device points
    float* d_centroids = nullptr;   // Device centroids
    int* d_labels = nullptr;        // Device labels

    // For optimized memory layouts
    std::vector<float> centroidsT;  // Transposed centroids [D×K] for better access patterns
    float* d_centroidsT = nullptr;  // Device transposed centroids

    // Constructor
    KMeansData(size_t n, size_t d, size_t k) : N(n), D(d), K(k) {
        points.resize(N * D);
        centroids.resize(K * D);
        labels.resize(N);
        centroidsT.resize(K * D);  // Same size, different layout
    }

    // Destructor (will handle GPU cleanup if needed)
    ~KMeansData();
};

// Timing structure for performance measurement
struct KMeansTimers {
    double init_time = 0.0;         // Initialization/setup time
    double assign_time = 0.0;       // Assignment kernel time
    double update_time = 0.0;       // Update kernel time
    double total_time = 0.0;        // Total execution time
    double transfer_time = 0.0;     // GPU memory transfer time (if applicable)

    // Iteration-specific timers
    std::vector<double> iter_assign_times;
    std::vector<double> iter_update_times;
    std::vector<double> iter_total_times;

    void reset() {
        init_time = assign_time = update_time = total_time = transfer_time = 0.0;
        iter_assign_times.clear();
        iter_update_times.clear();
        iter_total_times.clear();
    }
};

// Common configuration parameters
struct KMeansConfig {
    size_t N = 100000;           // Number of points
    size_t D = 64;               // Number of dimensions
    size_t K = 32;               // Number of clusters
    size_t max_iters = 100;      // Maximum iterations
    int seed = 42;               // Random seed
    float tolerance = 1e-6;      // Convergence tolerance

    // Parallel configuration
    int num_threads = 4;         // OpenMP threads
    int block_size = 256;        // CUDA block size

    // I/O configuration
    std::string input_file = "";      // Input data file (empty = generate)
    std::string output_file = "";     // Output results file
    bool verbose = false;             // Verbose output
    bool validate = false;            // Validate against reference

    // Benchmark configuration
    int warmup_iters = 3;        // Warmup iterations
    int bench_iters = 5;         // Benchmark iterations
};

// Interface for K-means implementations
class KMeansImplementation {
public:
    virtual ~KMeansImplementation() = default;

    // Core operations
    virtual void initialize(KMeansData& data, const KMeansConfig& config) = 0;
    virtual void assign_labels(KMeansData& data) = 0;
    virtual void update_centroids(KMeansData& data) = 0;
    virtual double compute_inertia(const KMeansData& data) = 0;

    // Full clustering algorithm
    virtual void run(KMeansData& data, const KMeansConfig& config, KMeansTimers& timers);

    // Implementation name
    virtual std::string name() const = 0;
};

// Utility functions (shared across all implementations)
namespace KMeansUtils {
    // Data generation and I/O
    void generate_synthetic_data(KMeansData& data, int seed);
    void load_data(KMeansData& data, const std::string& filename);
    void save_results(const KMeansData& data, const std::string& filename);

    // Initialization methods
    void random_init(KMeansData& data, int seed);
    void kmeans_plusplus_init(KMeansData& data, int seed);

    // Validation and correctness checking
    bool validate_results(const KMeansData& result1, const KMeansData& result2, float tolerance = 1e-5);
    double compute_inertia_reference(const KMeansData& data);

    // Memory layout optimization
    void transpose_centroids(KMeansData& data);  // [K×D] to [D×K]

    // Performance metrics
    double calculate_mlups(size_t N, size_t K, size_t D, double time_ms);
    void print_performance_summary(const KMeansConfig& config, const KMeansTimers& timers);

    // Command-line parsing
    KMeansConfig parse_arguments(int argc, char** argv);
}

// Timer utility class
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }

    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }
};