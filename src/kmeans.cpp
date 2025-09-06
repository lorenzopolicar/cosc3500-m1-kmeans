#include "kernels.hpp"
#include <iostream>
#include <algorithm>
#include <limits>

#ifdef TRANSPOSED_C
// Helper function to transpose centroids from [K×D] to [D×K]
void transpose_centroids(Data& data) {
    for (size_t d = 0; d < data.D; ++d) {
        for (size_t k = 0; k < data.K; ++k) {
            data.centroidsT[d * data.K + k] = data.centroids[k * data.D + d];
        }
    }
}
#endif

void assign_labels(Data& data) {
    // For each point, find the nearest centroid using squared L2 distance
    for (size_t i = 0; i < data.N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Compute squared distance to each centroid
        for (size_t k = 0; k < data.K; ++k) {
            double d2 = 0.0;
            
            // Sum squared differences over all dimensions
            for (size_t d = 0; d < data.D; ++d) {
#ifdef TRANSPOSED_C
                // Use transposed centroids for cache-friendly access
                float diff = data.points[i * data.D + d] - data.centroidsT[d * data.K + k];
#else
                // Original row-major access
                float diff = data.points[i * data.D + d] - data.centroids[k * data.D + d];
#endif
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
            // Update best if this centroid is closer (branch-minimal pattern)
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = static_cast<int>(k);
            }
        }
        
        data.labels[i] = best_k;
    }
}

void update_centroids(Data& data) {
    // Initialize K×D double accumulators and K counts
    std::vector<double> accumulators(data.K * data.D, 0.0);
    std::vector<int> counts(data.K, 0);
    
    // Accumulate point coordinates into their centroid's accumulator
    for (size_t i = 0; i < data.N; ++i) {
        int k = data.labels[i];
        counts[static_cast<size_t>(k)]++;
        
        for (size_t d = 0; d < data.D; ++d) {
            accumulators[static_cast<size_t>(k) * data.D + d] += static_cast<double>(data.points[i * data.D + d]);
        }
    }
    
    // Update centroids by dividing accumulators by counts
    static bool warned_empty = false;
    for (size_t k = 0; k < data.K; ++k) {
        if (counts[k] > 0) {
            // Divide accumulator by count and store back to float centroids
            for (size_t d = 0; d < data.D; ++d) {
                data.centroids[k * data.D + d] = static_cast<float>(accumulators[k * data.D + d] / static_cast<double>(counts[k]));
            }
        } else {
            // Leave centroid unchanged if no points assigned
            if (!warned_empty) {
                std::cerr << "Warning: centroid " << k << " has no assigned points" << std::endl;
                warned_empty = true;
            }
        }
    }
    
#ifdef TRANSPOSED_C
    // Rebuild transposed centroids after update (cost belongs to update timer)
    transpose_centroids(data);
#endif
}

double inertia(const Data& data) {
    double total_inertia = 0.0;
    
    // For each point, compute squared distance to its assigned centroid
    for (size_t i = 0; i < data.N; ++i) {
        int k = data.labels[i];
        double d2 = 0.0;
        
        // Compute squared distance to assigned centroid
        for (size_t d = 0; d < data.D; ++d) {
            float diff = data.points[i * data.D + d] - data.centroids[static_cast<size_t>(k) * data.D + d];
            d2 += static_cast<double>(diff) * static_cast<double>(diff);
        }
        
        total_inertia += d2;
    }
    
    return total_inertia;
}