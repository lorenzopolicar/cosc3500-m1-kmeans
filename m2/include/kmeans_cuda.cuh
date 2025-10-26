#pragma once

#include "kmeans_common.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA configuration parameters
constexpr int WARP_SIZE = 32;
constexpr int MAX_SHARED_MEMORY = 49152;  // 48KB shared memory per block
constexpr int DEFAULT_BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;  // For shared memory tiling

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// cuBLAS error checking macro
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", \
                    __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

// =============================================================================
// CUDA KERNELS
// =============================================================================

// Basic assign labels kernel - one thread per point
__global__ void assign_labels_kernel_basic(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    int* __restrict__ labels,              // [N]
    int N, int D, int K
);

// Optimized assign labels kernel with shared memory for centroids
__global__ void assign_labels_kernel_shared(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    int* __restrict__ labels,              // [N]
    int N, int D, int K
);

// Assign labels with transposed centroids for better memory access
__global__ void assign_labels_kernel_transposed(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroidsT,  // [D×K] (transposed)
    int* __restrict__ labels,              // [N]
    int N, int D, int K
);

// Tiled assign labels kernel with register blocking
__global__ void assign_labels_kernel_tiled(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    int* __restrict__ labels,              // [N]
    int N, int D, int K
);

// Update centroids kernel - basic reduction
__global__ void update_centroids_kernel_basic(
    const float* __restrict__ points,      // [N×D]
    const int* __restrict__ labels,        // [N]
    float* __restrict__ centroids,         // [K×D]
    int* __restrict__ counts,              // [K]
    int N, int D, int K
);

// Update centroids with atomic operations
__global__ void update_centroids_kernel_atomic(
    const float* __restrict__ points,      // [N×D]
    const int* __restrict__ labels,        // [N]
    float* __restrict__ centroid_sums,     // [K×D]
    int* __restrict__ counts,              // [K]
    int N, int D, int K
);

// Finalize centroids (divide sums by counts)
__global__ void finalize_centroids_kernel(
    const float* __restrict__ centroid_sums,  // [K×D]
    const int* __restrict__ counts,           // [K]
    float* __restrict__ centroids,            // [K×D]
    int K, int D
);

// Compute inertia (sum of squared distances)
__global__ void compute_inertia_kernel(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    const int* __restrict__ labels,        // [N]
    float* __restrict__ partial_inertias,  // [gridDim.x]
    int N, int D
);

// Transpose kernel for centroids
__global__ void transpose_kernel(
    const float* __restrict__ input,   // [rows×cols]
    float* __restrict__ output,        // [cols×rows]
    int rows, int cols
);

// =============================================================================
// CUDA IMPLEMENTATION CLASS
// =============================================================================

class KMeansCUDA : public KMeansImplementation {
private:
    // CUDA configuration
    int device_id;
    cudaDeviceProp device_props;
    dim3 block_size;
    dim3 grid_size;

    // cuBLAS handle (optional, for matrix operations)
    cublasHandle_t cublas_handle = nullptr;

    // Device memory pointers (managed in KMeansData)
    float* d_centroid_sums = nullptr;  // [K×D] for reduction
    int* d_counts = nullptr;           // [K] cluster counts
    float* d_partial_inertias = nullptr;  // For inertia reduction

    // Kernel selection
    enum KernelType {
        KERNEL_BASIC,
        KERNEL_SHARED,
        KERNEL_TRANSPOSED,
        KERNEL_TILED
    };
    KernelType assign_kernel_type = KERNEL_BASIC;
    KernelType update_kernel_type = KERNEL_BASIC;

    // Helper methods
    void allocate_device_memory(KMeansData& data);
    void free_device_memory(KMeansData& data);
    void copy_to_device(KMeansData& data);
    void copy_from_device(KMeansData& data);
    void configure_kernel_launch(size_t N, size_t K, size_t D);

public:
    // Constructor/Destructor
    KMeansCUDA(int device = 0);
    ~KMeansCUDA();

    // Implementation interface
    void initialize(KMeansData& data, const KMeansConfig& config) override;
    void assign_labels(KMeansData& data) override;
    void update_centroids(KMeansData& data) override;
    double compute_inertia(const KMeansData& data) override;
    std::string name() const override { return "CUDA"; }

    // CUDA-specific methods
    void set_kernel_type(KernelType assign_type, KernelType update_type);
    void print_device_info() const;
    size_t get_device_memory_usage() const;
};

// =============================================================================
// CUDA UTILITY FUNCTIONS
// =============================================================================

namespace CUDAUtils {
    // Device selection and query
    int get_device_count();
    void set_device(int device_id);
    cudaDeviceProp get_device_properties(int device_id);
    void print_cuda_devices();

    // Memory utilities
    size_t get_free_device_memory();
    size_t get_total_device_memory();

    // Kernel configuration helpers
    dim3 calculate_grid_size(size_t N, size_t block_size);
    int calculate_shared_memory_size(size_t K, size_t D);
    bool can_use_shared_memory(size_t K, size_t D);

    // Performance profiling
    void start_cuda_timer(cudaEvent_t& start, cudaEvent_t& stop);
    float stop_cuda_timer(cudaEvent_t& start, cudaEvent_t& stop);
}