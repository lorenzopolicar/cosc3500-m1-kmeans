#include "../../include/kmeans_openmp.hpp"
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>

// =============================================================================
// OPENMP K-MEANS IMPLEMENTATION
// =============================================================================

void KMeansOpenMP::initialize(KMeansData& data, const KMeansConfig& config) {
    // Random initialization (serial)
    KMeansUtils::random_init(data, config.seed);

    // Transpose centroids for better cache locality (from M1 E1 optimization)
    KMeansUtils::transpose_centroids(data);

    // Print OpenMP configuration
    #pragma omp parallel
    {
        #pragma omp single
        {
            int actual_threads = omp_get_num_threads();
            if (config.verbose) {
                std::cout << "OpenMP configuration: " << actual_threads << " threads" << std::endl;
            }
        }
    }
}

void KMeansOpenMP::assign_labels(KMeansData& data) {
    // Data-parallel over points (from M1 E2 optimizations)
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    const float* __restrict__ centroidsT = data.centroidsT.data();  // Transposed layout
    int* __restrict__ labels = data.labels.data();

    // Parallel loop over all points
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const float* px = &points[i * D];
        float min_dist = std::numeric_limits<float>::max();
        int best_label = 0;

        // Find nearest centroid
        for (size_t k = 0; k < K; ++k) {
            float dist = 0.0f;

            // Compute squared Euclidean distance
            // Using transposed layout: centroidsT[d * K + k]
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - centroidsT[d * K + k];
                dist += diff * diff;
            }

            // Branchless minimum update (M1 E2 optimization)
            bool is_closer = (dist < min_dist);
            min_dist = is_closer ? dist : min_dist;
            best_label = is_closer ? static_cast<int>(k) : best_label;
        }

        labels[i] = best_label;
    }
}

void KMeansOpenMP::update_centroids(KMeansData& data) {
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    const int* __restrict__ labels = data.labels.data();
    float* __restrict__ centroids = data.centroids.data();

    // Thread-local storage for accumulation
    // Each thread accumulates into its own buffer to avoid false sharing
    int nthreads = omp_get_max_threads();
    std::vector<double> thread_sums(nthreads * K * D, 0.0);
    std::vector<int> thread_counts(nthreads * K, 0);

    // Parallel accumulation phase
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double* local_sums = &thread_sums[tid * K * D];
        int* local_counts = &thread_counts[tid * K];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            int label = labels[i];
            const float* px = &points[i * D];

            local_counts[label]++;
            for (size_t d = 0; d < D; ++d) {
                local_sums[label * D + d] += static_cast<double>(px[d]);
            }
        }
    }

    // Serial reduction phase (combine thread-local results)
    std::vector<double> global_sums(K * D, 0.0);
    std::vector<int> global_counts(K, 0);

    for (int t = 0; t < nthreads; ++t) {
        const double* local_sums = &thread_sums[t * K * D];
        const int* local_counts = &thread_counts[t * K];

        for (size_t k = 0; k < K; ++k) {
            global_counts[k] += local_counts[k];
            for (size_t d = 0; d < D; ++d) {
                global_sums[k * D + d] += local_sums[k * D + d];
            }
        }
    }

    // Compute new centroids (parallel over clusters)
    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < K; ++k) {
        if (global_counts[k] > 0) {
            double inv_count = 1.0 / global_counts[k];
            for (size_t d = 0; d < D; ++d) {
                centroids[k * D + d] = static_cast<float>(global_sums[k * D + d] * inv_count);
            }
        }
    }

    // Update transposed centroids
    KMeansUtils::transpose_centroids(data);
}

double KMeansOpenMP::compute_inertia(const KMeansData& data) {
    const size_t N = data.N;
    const size_t D = data.D;
    const float* __restrict__ points = data.points.data();
    const float* __restrict__ centroids = data.centroids.data();
    const int* __restrict__ labels = data.labels.data();

    double total_inertia = 0.0;

    // Parallel reduction over all points
    #pragma omp parallel for reduction(+:total_inertia) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        int label = labels[i];
        const float* px = &points[i * D];
        const float* cx = &centroids[label * D];

        double dist = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = static_cast<double>(px[d]) - static_cast<double>(cx[d]);
            dist += diff * diff;
        }

        total_inertia += dist;
    }

    return total_inertia;
}