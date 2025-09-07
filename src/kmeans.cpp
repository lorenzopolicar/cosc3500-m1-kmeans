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
    // Route to E3 optimized versions if flags are defined
#ifdef TILE_K
    assign_labels_tiled(data);
    return;
#endif

#ifdef STRIDE_PTR
    assign_labels_strided(data);
    return;
#endif

#ifdef BRANCHLESS
    assign_labels_branchless(data);
    return;
#endif

#ifdef HOIST
    assign_labels_hoisted(data);
    return;
#endif

    // Original E0/E1 implementation (fallback)
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

// ============================================================================
// E2 MICRO-OPTIMIZATIONS
// ============================================================================

#ifdef HOIST
// E2: Invariant hoisting optimization
// Hoists N, D, K and data pointers outside loops to reduce repeated access
void assign_labels_hoisted(Data& data) {
    // Hoist invariants outside loops
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    
#ifdef TRANSPOSED_C
    const float* __restrict__ centroidsT = data.centroidsT.data();
    
    // For each point, find the nearest centroid using squared L2 distance
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Cache point pointer for this iteration
        const float* __restrict__ px = &points[i * D];
        
        // Compute squared distance to each centroid
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            
            // Sum squared differences over all dimensions
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - centroidsT[d * K + k];
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
#else
    const float* __restrict__ centroids = data.centroids.data();
    
    // For each point, find the nearest centroid using squared L2 distance
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Cache point pointer for this iteration
        const float* __restrict__ px = &points[i * D];
        
        // Compute squared distance to each centroid
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            
            // Sum squared differences over all dimensions
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - centroids[k * D + d];
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
#endif
}
#endif

#ifdef BRANCHLESS
// E2: Branchless argmin optimization
// Uses ternary operators instead of if-statements for better CPU branch prediction
void assign_labels_branchless(Data& data) {
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    
#ifdef TRANSPOSED_C
    const float* __restrict__ centroidsT = data.centroidsT.data();
    
    // For each point, find the nearest centroid using squared L2 distance
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Cache point pointer for this iteration
        const float* __restrict__ px = &points[i * D];
        
        // Compute squared distance to each centroid
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            
            // Sum squared differences over all dimensions
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - centroidsT[d * K + k];
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
            // Branchless update: use ternary operator instead of if-statement
            best_d2 = (d2 < best_d2) ? d2 : best_d2;
            best_k = (d2 < best_d2) ? static_cast<int>(k) : best_k;
        }
        
        data.labels[i] = best_k;
    }
#else
    const float* __restrict__ centroids = data.centroids.data();
    
    // For each point, find the nearest centroid using squared L2 distance
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Cache point pointer for this iteration
        const float* __restrict__ px = &points[i * D];
        
        // Compute squared distance to each centroid
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            
            // Sum squared differences over all dimensions
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - centroids[k * D + d];
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
            // Branchless update: use ternary operator instead of if-statement
            best_d2 = (d2 < best_d2) ? d2 : best_d2;
            best_k = (d2 < best_d2) ? static_cast<int>(k) : best_k;
        }
        
        data.labels[i] = best_k;
    }
#endif
}
#endif

#ifdef STRIDE_PTR
// E2: Strided pointer optimization
// Uses strided centroid pointers to eliminate d*K multiplies in transposed access
void assign_labels_strided(Data& data) {
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    
#ifdef TRANSPOSED_C
    const float* __restrict__ centroidsT = data.centroidsT.data();
    
    // For each point, find the nearest centroid using squared L2 distance
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Cache point pointer for this iteration
        const float* __restrict__ px = &points[i * D];
        
        // Compute squared distance to each centroid
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            
            // Use strided pointer to eliminate d*K multiplies
            const float* __restrict__ ck = &centroidsT[k];  // Start at dimension 0, centroid k
            
            // Sum squared differences over all dimensions
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - *ck;
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
                ck += K;  // Move to next dimension, same centroid (stride by K)
            }
            
            // Update best if this centroid is closer (branch-minimal pattern)
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = static_cast<int>(k);
            }
        }
        
        data.labels[i] = best_k;
    }
#else
    // Fallback to regular implementation for non-transposed case
    const float* __restrict__ centroids = data.centroids.data();
    
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        const float* __restrict__ px = &points[i * D];
        
        for (size_t k = 0; k < K; ++k) {
            double d2 = 0.0;
            const float* __restrict__ ck = &centroids[k * D];
            
            for (size_t d = 0; d < D; ++d) {
                float diff = px[d] - ck[d];
                d2 += static_cast<double>(diff) * static_cast<double>(diff);
            }
            
            if (d2 < best_d2) {
                best_d2 = d2;
                best_k = static_cast<int>(k);
            }
        }
        
        data.labels[i] = best_k;
    }
#endif
}
#endif


// ============================================================================
// E3 CACHE OPTIMIZATION: K-TILING (CENTROID BLOCKING)
// ============================================================================

#ifdef TILE_K
// E3: K-tiling (centroid blocking) optimization
// Improves cache locality by processing centroids in tiles of TK
// This is a simplified approach that maintains correctness
void assign_labels_tiled(Data& data) {
    // Hoist invariants outside loops (from E2)
    const size_t N = data.N;
    const size_t D = data.D;
    const size_t K = data.K;
    const float* __restrict__ points = data.points.data();
    
#ifdef TRANSPOSED_C
    const float* __restrict__ centroidsT = data.centroidsT.data();
    
    // For each point, find the nearest centroid using K-tiling
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        // Cache point pointer for this iteration (from E2)
        const float* __restrict__ px = &points[i * D];
        
        // Process centroids in tiles of TK for better cache locality
        for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
            size_t k_end = std::min(k0 + TILE_K, K);
            
            // Process all centroids in current tile
            for (size_t k = k0; k < k_end; ++k) {
                double d2 = 0.0;
                
                // Use strided pointer to eliminate d*K multiplies (from E2)
                const float* __restrict__ ck = &centroidsT[k];
                
                // Sum squared differences over all dimensions
                for (size_t d = 0; d < D; ++d) {
                    float diff = px[d] - *ck;
                    d2 += static_cast<double>(diff) * static_cast<double>(diff);
                    ck += K;  // Move to next dimension, same centroid (stride by K)
                }
                
                // Update best if this centroid is closer
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_k = static_cast<int>(k);
                }
            }
        }
        
        data.labels[i] = best_k;
    }
#else
    // Fallback to regular implementation for non-transposed case
    const float* __restrict__ centroids = data.centroids.data();
    
    for (size_t i = 0; i < N; ++i) {
        double best_d2 = std::numeric_limits<double>::max();
        int best_k = 0;
        
        const float* __restrict__ px = &points[i * D];
        
        // Process centroids in tiles of TK for better cache locality
        for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
            size_t k_end = std::min(k0 + TILE_K, K);
            
            for (size_t k = k0; k < k_end; ++k) {
                double d2 = 0.0;
                const float* __restrict__ ck = &centroids[k * D];
                
                for (size_t d = 0; d < D; ++d) {
                    float diff = px[d] - ck[d];
                    d2 += static_cast<double>(diff) * static_cast<double>(diff);
                }
                
                // Update best if this centroid is closer
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_k = static_cast<int>(k);
                }
            }
        }
        
        data.labels[i] = best_k;
    }
#endif
}
#endif
