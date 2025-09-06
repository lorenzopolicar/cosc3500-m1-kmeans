#pragma once

#include <vector>

// Data structure for K-means clustering
struct Data {
    std::vector<float> points;      // N points of dimension D (row-major: [i*D + d])
    std::vector<float> centroids;   // K centroids of dimension D (row-major: [k*D + d])
    std::vector<int> labels;        // N labels (0 to K-1)
    size_t N, D, K;                 // Number of points, dimensions, clusters
    
#ifdef TRANSPOSED_C
    std::vector<float> centroidsT;  // Transposed centroids [D×K] for cache-friendly access
#endif
};

// Kernel function declarations
void assign_labels(Data& data);     // fills data.labels using squared L2; single-threaded; no SIMD
void update_centroids(Data& data);  // recompute centroids using double accumulators; single-threaded; no SIMD
double inertia(const Data& data);   // sum of squared distances for current labels

#ifdef TRANSPOSED_C
// Helper function to transpose centroids from [K×D] to [D×K]
void transpose_centroids(Data& data);
#endif