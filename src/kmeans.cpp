#include "kernels.hpp"
#include <iostream>
#include <algorithm>
#include <limits>

#ifdef TRANSPOSED_C
// Helper function to transpose centroids from [K×D] to [D×K]
void transpose_centroids(Data& data) {
    // Ensure centroidsT is properly sized
    data.centroidsT.resize(data.D * data.K);
    
    for (size_t d = 0; d < data.D; ++d) {
        for (size_t k = 0; k < data.K; ++k) {
            data.centroidsT[d * data.K + k] = data.centroids[k * data.D + d];
        }
    }
}
#endif

void assign_labels(Data& data) {
    // Combined optimization approach - all flags work together
    // Hoist invariants (from E2 HOIST)
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();

    // Choose data source based on TRANSPOSED_C
#ifdef TRANSPOSED_C
    const float* __restrict__ centroidsT = data.centroidsT.data();
    
    // K-register blocking path (E5) - only when TRANSPOSED_C is enabled
#ifdef TK
    // Specialized TK=4 with scalar accumulators for best performance
#if TK == 4
    for (size_t i = 0; i < N; ++i) {
        const float* __restrict__ px = &points[i * D];
        double best = std::numeric_limits<double>::max();
        int bestk = 0;
        
        // Process centroids in tiles of 4
        for (size_t k0 = 0; k0 < K; k0 += 4) {
            if (k0 + 3 < K) {
                // Full tile of 4 centroids with scalar accumulators
                double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
                
                for (size_t d = 0; d < D; ++d) {
                    const float x = px[d];
                    const float* base = &centroidsT[d * K + k0];  // Base pointer per dimension
                    
                    float t = x - base[0]; s0 += static_cast<double>(t) * static_cast<double>(t);
                    t = x - base[1]; s1 += static_cast<double>(t) * static_cast<double>(t);
                    t = x - base[2]; s2 += static_cast<double>(t) * static_cast<double>(t);
                    t = x - base[3]; s3 += static_cast<double>(t) * static_cast<double>(t);
                }
                
                // Branchless argmin across 4 accumulators
                int kcand = static_cast<int>(k0);
                double v = s0;
                bool b = (s1 < v); v = b ? s1 : v; kcand = b ? static_cast<int>(k0 + 1) : kcand;
                b = (s2 < v); v = b ? s2 : v; kcand = b ? static_cast<int>(k0 + 2) : kcand;
                b = (s3 < v); v = b ? s3 : v; kcand = b ? static_cast<int>(k0 + 3) : kcand;
                
                bool better = (v < best);
                best = better ? v : best;
                bestk = better ? kcand : bestk;
                continue;
            }
            
            // Remainder: scalar path for partial tile
            for (size_t k = k0; k < K; ++k) {
                double s = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    const float x = px[d];
                    const float c = centroidsT[d * K + k];
                    float t = x - c;
                    s += static_cast<double>(t) * static_cast<double>(t);
                }
                bool better = (s < best);
                best = better ? s : best;
                bestk = better ? static_cast<int>(k) : bestk;
            }
        }
        
        data.labels[i] = bestk;
    }
#else
    // Generic TK implementation with array accumulators
    for (size_t i = 0; i < N; ++i) {
        const float* __restrict__ px = &points[i * D];
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        for (size_t k0 = 0; k0 < K; k0 += TK) {
            size_t k_end = std::min(k0 + TK, K);
            size_t tile_size = k_end - k0;
            
            // Initialize accumulators for this tile
            double s[TK] = {0.0};
            
            for (size_t d = 0; d < D; ++d) {
                const float x = px[d];
                const float* base = &centroidsT[d * K + k0];  // Base pointer per dimension
                
                for (size_t j = 0; j < tile_size; ++j) {
                    float t = x - base[j];
                    s[j] += static_cast<double>(t) * static_cast<double>(t);
                }
            }
            
            // Branchless argmin across accumulators in this tile
            for (size_t j = 0; j < tile_size; ++j) {
                bool better = (s[j] < best_d2);
                best_d2 = better ? s[j] : best_d2;
                best_k = better ? static_cast<int>(k0 + j) : best_k;
            }
        }
        
        data.labels[i] = best_k;
    }
#endif
#else
    // Transposed centroids without K-blocking - use strided pointers
#ifdef STRIDE_PTR
    for (size_t i = 0; i < N; ++i) {
        const float* __restrict__ px = &points[i * D];
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            const float* __restrict__ ck = &centroidsT[k];  // Strided pointer
            
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - ck[d * K];  // Strided access
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
#ifdef BRANCHLESS
            bool better = (d2 < best_d2);
            best_k = better ? static_cast<int>(k) : best_k;
            best_d2 = better ? d2 : best_d2;
#else
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = static_cast<int>(k);
            }
#endif
        }
        
        data.labels[i] = best_k;
    }
#else
    // Transposed centroids without strided pointers
    for (size_t i = 0; i < N; ++i) {
        const float* __restrict__ px = &points[i * D];
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - centroidsT[d * K + k];
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
#ifdef BRANCHLESS
            bool better = (d2 < best_d2);
            best_k = better ? static_cast<int>(k) : best_k;
            best_d2 = better ? d2 : best_d2;
#else
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = static_cast<int>(k);
            }
#endif
        }
        
        data.labels[i] = best_k;
    }
#endif
#endif
#else
    // Non-transposed centroids (row-major) - fallback
    const float* __restrict__ centroids = data.centroids.data();
    
    for (size_t i = 0; i < N; ++i) {
        const float* __restrict__ px = &points[i * D];
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            const float* __restrict__ ck = &centroids[k * D];
            
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - ck[d];
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
#ifdef BRANCHLESS
            bool better = (d2 < best_d2);
            best_k = better ? static_cast<int>(k) : best_k;
            best_d2 = better ? d2 : best_d2;
#else
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = static_cast<int>(k);
            }
#endif
        }
        
        data.labels[i] = best_k;
    }
#endif
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