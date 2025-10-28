#pragma once

#include "kmeans_common.hpp"
#include <omp.h>

// =============================================================================
// OPENMP K-MEANS IMPLEMENTATION
// =============================================================================

class KMeansOpenMP : public KMeansImplementation {
private:
    int num_threads;

public:
    explicit KMeansOpenMP(int threads = 4) : num_threads(threads) {
        omp_set_num_threads(num_threads);
    }

    void initialize(KMeansData& data, const KMeansConfig& config) override;
    void assign_labels(KMeansData& data) override;
    void update_centroids(KMeansData& data) override;
    double compute_inertia(const KMeansData& data) override;

    std::string name() const override {
        return "OpenMP (" + std::to_string(num_threads) + " threads)";
    }

    void set_num_threads(int threads) {
        num_threads = threads;
        omp_set_num_threads(num_threads);
    }
};