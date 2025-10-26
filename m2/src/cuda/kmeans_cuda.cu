#include "../../include/kmeans_cuda.cuh"
#include <cstdio>
#include <cfloat>
#include <cmath>

// =============================================================================
// CUDA KERNELS IMPLEMENTATION
// =============================================================================

// Basic assign labels kernel - one thread per point
__global__ void assign_labels_kernel_basic(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    int* __restrict__ labels,              // [N]
    int N, int D, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    // Load point into registers (reused for all K centroids)
    const float* point = &points[tid * D];

    float min_dist = FLT_MAX;
    int best_label = 0;

    // Check distance to each centroid
    for (int k = 0; k < K; ++k) {
        const float* centroid = &centroids[k * D];

        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = point[d] - centroid[d];
            dist += diff * diff;
        }

        // Branchless minimum update (similar to E2 optimization)
        bool is_closer = (dist < min_dist);
        min_dist = is_closer ? dist : min_dist;
        best_label = is_closer ? k : best_label;
    }

    labels[tid] = best_label;
}

// Optimized assign labels kernel with shared memory for centroids
__global__ void assign_labels_kernel_shared(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    int* __restrict__ labels,              // [N]
    int N, int D, int K
) {
    // Shared memory for centroids (if they fit)
    extern __shared__ float shared_centroids[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Cooperatively load centroids into shared memory
    int total_centroid_elements = K * D;
    int elements_per_thread = (total_centroid_elements + blockDim.x - 1) / blockDim.x;

    for (int i = 0; i < elements_per_thread; ++i) {
        int idx = local_tid * elements_per_thread + i;
        if (idx < total_centroid_elements) {
            shared_centroids[idx] = centroids[idx];
        }
    }

    __syncthreads();

    if (tid >= N) return;

    // Load point into registers
    const float* point = &points[tid * D];

    float min_dist = FLT_MAX;
    int best_label = 0;

    // Check distance to each centroid (from shared memory)
    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;

        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float diff = point[d] - shared_centroids[k * D + d];
            dist += diff * diff;
        }

        bool is_closer = (dist < min_dist);
        min_dist = is_closer ? dist : min_dist;
        best_label = is_closer ? k : best_label;
    }

    labels[tid] = best_label;
}

// Assign labels with transposed centroids for coalesced access
__global__ void assign_labels_kernel_transposed(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroidsT,  // [D×K] (transposed)
    int* __restrict__ labels,              // [N]
    int N, int D, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    const float* point = &points[tid * D];

    float min_dist = FLT_MAX;
    int best_label = 0;

    // Process each centroid
    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;

        // Access pattern: centroidsT[d * K + k]
        // This gives better memory coalescing when multiple threads access different k
        for (int d = 0; d < D; ++d) {
            float diff = point[d] - centroidsT[d * K + k];
            dist += diff * diff;
        }

        bool is_closer = (dist < min_dist);
        min_dist = is_closer ? dist : min_dist;
        best_label = is_closer ? k : best_label;
    }

    labels[tid] = best_label;
}

// Update centroids with atomic operations
__global__ void update_centroids_kernel_atomic(
    const float* __restrict__ points,      // [N×D]
    const int* __restrict__ labels,        // [N]
    float* __restrict__ centroid_sums,     // [K×D]
    int* __restrict__ counts,              // [K]
    int N, int D, int K
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    int label = labels[tid];
    const float* point = &points[tid * D];
    float* centroid_sum = &centroid_sums[label * D];

    // Atomic add for each dimension
    for (int d = 0; d < D; ++d) {
        atomicAdd(&centroid_sum[d], point[d]);
    }

    // Atomic increment for count
    atomicAdd(&counts[label], 1);
}

// Initialize sums and counts to zero
__global__ void init_centroids_kernel(
    float* __restrict__ centroid_sums,     // [K×D]
    int* __restrict__ counts,              // [K]
    int K, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = K * D;

    if (tid < total_elements) {
        centroid_sums[tid] = 0.0f;
    }

    if (tid < K) {
        counts[tid] = 0;
    }
}

// Finalize centroids (divide sums by counts)
__global__ void finalize_centroids_kernel(
    const float* __restrict__ centroid_sums,  // [K×D]
    const int* __restrict__ counts,           // [K]
    float* __restrict__ centroids,            // [K×D]
    int K, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= K) return;

    int count = counts[tid];
    if (count > 0) {
        const float* sum = &centroid_sums[tid * D];
        float* centroid = &centroids[tid * D];

        float inv_count = 1.0f / count;
        for (int d = 0; d < D; ++d) {
            centroid[d] = sum[d] * inv_count;
        }
    }
}

// Compute inertia (sum of squared distances)
__global__ void compute_inertia_kernel(
    const float* __restrict__ points,      // [N×D]
    const float* __restrict__ centroids,   // [K×D]
    const int* __restrict__ labels,        // [N]
    float* __restrict__ partial_inertias,  // [gridDim.x]
    int N, int D
) {
    // Shared memory for block-level reduction
    extern __shared__ float shared_inertia[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    float local_inertia = 0.0f;

    if (tid < N) {
        int label = labels[tid];
        const float* point = &points[tid * D];
        const float* centroid = &centroids[label * D];

        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = point[d] - centroid[d];
            dist += diff * diff;
        }

        local_inertia = dist;
    }

    shared_inertia[local_tid] = local_inertia;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared_inertia[local_tid] += shared_inertia[local_tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (local_tid == 0) {
        partial_inertias[blockIdx.x] = shared_inertia[0];
    }
}

// Transpose kernel for centroids
__global__ void transpose_kernel(
    const float* __restrict__ input,   // [rows×cols]
    float* __restrict__ output,        // [cols×rows]
    int rows, int cols
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// =============================================================================
// KMEANS CUDA CLASS IMPLEMENTATION
// =============================================================================

KMeansCUDA::KMeansCUDA(int device) : device_id(device) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&device_props, device_id));

    // Initialize cuBLAS (optional)
    // CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Default kernel configuration
    block_size = dim3(DEFAULT_BLOCK_SIZE);
    assign_kernel_type = KERNEL_BASIC;
    update_kernel_type = KERNEL_BASIC;
}

KMeansCUDA::~KMeansCUDA() {
    // Clean up any remaining device memory
    if (d_centroid_sums) CUDA_CHECK(cudaFree(d_centroid_sums));
    if (d_counts) CUDA_CHECK(cudaFree(d_counts));
    if (d_partial_inertias) CUDA_CHECK(cudaFree(d_partial_inertias));

    // if (cublas_handle) cublasDestroy(cublas_handle);
}

void KMeansCUDA::allocate_device_memory(KMeansData& data) {
    size_t points_size = data.N * data.D * sizeof(float);
    size_t centroids_size = data.K * data.D * sizeof(float);
    size_t labels_size = data.N * sizeof(int);

    // Allocate main arrays
    CUDA_CHECK(cudaMalloc(&data.d_points, points_size));
    CUDA_CHECK(cudaMalloc(&data.d_centroids, centroids_size));
    CUDA_CHECK(cudaMalloc(&data.d_labels, labels_size));

    // Allocate transposed centroids if using transposed kernel
    if (assign_kernel_type == KERNEL_TRANSPOSED) {
        CUDA_CHECK(cudaMalloc(&data.d_centroidsT, centroids_size));
    }

    // Allocate working arrays for updates
    CUDA_CHECK(cudaMalloc(&d_centroid_sums, centroids_size));
    CUDA_CHECK(cudaMalloc(&d_counts, data.K * sizeof(int)));

    // Allocate for inertia reduction
    grid_size = CUDAUtils::calculate_grid_size(data.N, block_size.x);
    CUDA_CHECK(cudaMalloc(&d_partial_inertias, grid_size.x * sizeof(float)));
}

void KMeansCUDA::free_device_memory(KMeansData& data) {
    if (data.d_points) { CUDA_CHECK(cudaFree(data.d_points)); data.d_points = nullptr; }
    if (data.d_centroids) { CUDA_CHECK(cudaFree(data.d_centroids)); data.d_centroids = nullptr; }
    if (data.d_labels) { CUDA_CHECK(cudaFree(data.d_labels)); data.d_labels = nullptr; }
    if (data.d_centroidsT) { CUDA_CHECK(cudaFree(data.d_centroidsT)); data.d_centroidsT = nullptr; }
}

void KMeansCUDA::copy_to_device(KMeansData& data) {
    size_t points_size = data.N * data.D * sizeof(float);
    size_t centroids_size = data.K * data.D * sizeof(float);

    CUDA_CHECK(cudaMemcpy(data.d_points, data.points.data(), points_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(data.d_centroids, data.centroids.data(), centroids_size, cudaMemcpyHostToDevice));

    // Transpose centroids if needed
    if (assign_kernel_type == KERNEL_TRANSPOSED) {
        KMeansUtils::transpose_centroids(data);
        CUDA_CHECK(cudaMemcpy(data.d_centroidsT, data.centroidsT.data(), centroids_size, cudaMemcpyHostToDevice));
    }
}

void KMeansCUDA::copy_from_device(KMeansData& data) {
    size_t labels_size = data.N * sizeof(int);
    size_t centroids_size = data.K * data.D * sizeof(float);

    CUDA_CHECK(cudaMemcpy(data.labels.data(), data.d_labels, labels_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(data.centroids.data(), data.d_centroids, centroids_size, cudaMemcpyDeviceToHost));
}

void KMeansCUDA::initialize(KMeansData& data, const KMeansConfig& config) {
    // Allocate device memory
    allocate_device_memory(data);

    // Initialize centroids on host (random or k-means++)
    KMeansUtils::random_init(data, config.seed);

    // Copy data to device
    copy_to_device(data);

    // Configure kernel launch parameters
    configure_kernel_launch(data.N, data.K, data.D);
}

void KMeansCUDA::configure_kernel_launch(size_t N, size_t K, size_t D) {
    block_size = dim3(DEFAULT_BLOCK_SIZE);
    grid_size = CUDAUtils::calculate_grid_size(N, block_size.x);

    // Determine if we can use shared memory for centroids
    if (CUDAUtils::can_use_shared_memory(K, D)) {
        assign_kernel_type = KERNEL_SHARED;
        printf("Using shared memory kernel (K=%zu, D=%zu)\n", K, D);
    } else {
        assign_kernel_type = KERNEL_BASIC;
        printf("Using basic kernel (centroids too large for shared memory)\n");
    }
}

void KMeansCUDA::assign_labels(KMeansData& data) {
    switch (assign_kernel_type) {
        case KERNEL_SHARED: {
            size_t shared_size = data.K * data.D * sizeof(float);
            assign_labels_kernel_shared<<<grid_size, block_size, shared_size>>>(
                data.d_points, data.d_centroids, data.d_labels,
                data.N, data.D, data.K
            );
            break;
        }
        case KERNEL_TRANSPOSED:
            assign_labels_kernel_transposed<<<grid_size, block_size>>>(
                data.d_points, data.d_centroidsT, data.d_labels,
                data.N, data.D, data.K
            );
            break;
        case KERNEL_BASIC:
        default:
            assign_labels_kernel_basic<<<grid_size, block_size>>>(
                data.d_points, data.d_centroids, data.d_labels,
                data.N, data.D, data.K
            );
            break;
    }

    CUDA_CHECK(cudaGetLastError());
}

void KMeansCUDA::update_centroids(KMeansData& data) {
    // Initialize sums and counts to zero
    dim3 init_grid = CUDAUtils::calculate_grid_size(data.K * data.D, block_size.x);
    init_centroids_kernel<<<init_grid, block_size>>>(
        d_centroid_sums, d_counts, data.K, data.D
    );

    // Accumulate points to their assigned centroids
    update_centroids_kernel_atomic<<<grid_size, block_size>>>(
        data.d_points, data.d_labels, d_centroid_sums, d_counts,
        data.N, data.D, data.K
    );

    // Finalize centroids (divide by counts)
    dim3 final_grid = CUDAUtils::calculate_grid_size(data.K, block_size.x);
    finalize_centroids_kernel<<<final_grid, block_size>>>(
        d_centroid_sums, d_counts, data.d_centroids, data.K, data.D
    );

    // Update transposed centroids if needed
    if (assign_kernel_type == KERNEL_TRANSPOSED) {
        dim3 transpose_block(16, 16);
        dim3 transpose_grid(
            (data.K + transpose_block.x - 1) / transpose_block.x,
            (data.D + transpose_block.y - 1) / transpose_block.y
        );
        transpose_kernel<<<transpose_grid, transpose_block>>>(
            data.d_centroids, data.d_centroidsT, data.K, data.D
        );
    }

    CUDA_CHECK(cudaGetLastError());
}

double KMeansCUDA::compute_inertia(const KMeansData& data) {
    size_t shared_size = block_size.x * sizeof(float);

    compute_inertia_kernel<<<grid_size, block_size, shared_size>>>(
        data.d_points, data.d_centroids, data.d_labels, d_partial_inertias,
        data.N, data.D
    );

    // Copy partial results back to host and sum
    std::vector<float> partial_results(grid_size.x);
    CUDA_CHECK(cudaMemcpy(partial_results.data(), d_partial_inertias,
                          grid_size.x * sizeof(float), cudaMemcpyDeviceToHost));

    double total_inertia = 0.0;
    for (float val : partial_results) {
        total_inertia += val;
    }

    return total_inertia;
}

void KMeansCUDA::print_device_info() const {
    printf("=== CUDA Device %d: %s ===\n", device_id, device_props.name);
    printf("Compute Capability: %d.%d\n", device_props.major, device_props.minor);
    printf("Total Global Memory: %.2f GB\n", device_props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %zu KB\n", device_props.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", device_props.maxThreadsPerBlock);
    printf("Warp Size: %d\n", device_props.warpSize);
    printf("Number of SMs: %d\n", device_props.multiProcessorCount);
}

// =============================================================================
// CUDA UTILITY FUNCTIONS
// =============================================================================

namespace CUDAUtils {

dim3 calculate_grid_size(size_t N, size_t block_size) {
    return dim3((N + block_size - 1) / block_size);
}

bool can_use_shared_memory(size_t K, size_t D) {
    size_t required_bytes = K * D * sizeof(float);
    return required_bytes <= MAX_SHARED_MEMORY;
}

int calculate_shared_memory_size(size_t K, size_t D) {
    return K * D * sizeof(float);
}

void print_cuda_devices() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    printf("Found %d CUDA devices:\n", device_count);
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, i));
        printf("  Device %d: %s (CC %d.%d, %.1f GB)\n",
               i, props.name, props.major, props.minor,
               props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
}

} // namespace CUDAUtils