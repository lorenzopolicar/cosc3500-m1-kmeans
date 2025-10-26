#include "../../include/kmeans_common.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <vector>

// Serial K-Means implementation (based on M1 E2 optimizations)
class KMeansSerial : public KMeansImplementation {
public:
    void initialize(KMeansData& data, const KMeansConfig& config) override {
        // Random initialization
        KMeansUtils::random_init(data, config.seed);

        // Transpose centroids for better cache locality
        KMeansUtils::transpose_centroids(data);
    }

    void assign_labels(KMeansData& data) override {
        // E2 optimizations: hoisted invariants, branchless min, strided pointers
        const size_t N = data.N;
        const size_t D = data.D;
        const size_t K = data.K;
        const float* __restrict__ points = data.points.data();
        const float* __restrict__ centroidsT = data.centroidsT.data();  // Using transposed
        int* __restrict__ labels = data.labels.data();

        // Process each point
        for (size_t i = 0; i < N; ++i) {
            const float* __restrict__ px = &points[i * D];
            float min_dist = std::numeric_limits<float>::max();
            int best_label = 0;

            // Check each centroid
            for (size_t k = 0; k < K; ++k) {
                float dist = 0.0f;

                // Compute squared L2 distance
                // Using transposed layout for better cache locality
                for (size_t d = 0; d < D; ++d) {
                    float diff = px[d] - centroidsT[d * K + k];
                    dist += diff * diff;
                }

                // Branchless minimum update (E2 optimization)
                bool is_closer = (dist < min_dist);
                min_dist = is_closer ? dist : min_dist;
                best_label = is_closer ? static_cast<int>(k) : best_label;
            }

            labels[i] = best_label;
        }
    }

    void update_centroids(KMeansData& data) override {
        const size_t N = data.N;
        const size_t D = data.D;
        const size_t K = data.K;
        const float* __restrict__ points = data.points.data();
        const int* __restrict__ labels = data.labels.data();
        float* __restrict__ centroids = data.centroids.data();

        // Use double precision for accumulation (prevents numerical errors)
        std::vector<double> sums(K * D, 0.0);
        std::vector<int> counts(K, 0);

        // Accumulate points into their assigned centroids
        for (size_t i = 0; i < N; ++i) {
            int label = labels[i];
            counts[label]++;

            const float* __restrict__ px = &points[i * D];
            double* __restrict__ sum = &sums[label * D];

            for (size_t d = 0; d < D; ++d) {
                sum[d] += static_cast<double>(px[d]);
            }
        }

        // Compute new centroids (mean of assigned points)
        for (size_t k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                const double* __restrict__ sum = &sums[k * D];
                float* __restrict__ centroid = &centroids[k * D];
                double inv_count = 1.0 / counts[k];

                for (size_t d = 0; d < D; ++d) {
                    centroid[d] = static_cast<float>(sum[d] * inv_count);
                }
            }
        }

        // Update transposed centroids
        KMeansUtils::transpose_centroids(data);
    }

    double compute_inertia(const KMeansData& data) override {
        const size_t N = data.N;
        const size_t D = data.D;
        const float* __restrict__ points = data.points.data();
        const float* __restrict__ centroids = data.centroids.data();
        const int* __restrict__ labels = data.labels.data();

        double total_inertia = 0.0;

        for (size_t i = 0; i < N; ++i) {
            int label = labels[i];
            const float* __restrict__ px = &points[i * D];
            const float* __restrict__ cx = &centroids[label * D];

            double dist = 0.0;
            for (size_t d = 0; d < D; ++d) {
                double diff = static_cast<double>(px[d]) - static_cast<double>(cx[d]);
                dist += diff * diff;
            }

            total_inertia += dist;
        }

        return total_inertia;
    }

    std::string name() const override {
        return "Serial (E2 Optimized)";
    }
};