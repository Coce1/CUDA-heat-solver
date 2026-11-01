#include "../include/heat_solver.cuh"
#include "../include/cuda_utils.cuh"

#include <utility>

// const + __restrict__ : permet au compilateur d'utiliser le chemin de lecture
// en cache texture/L1 (LDG) et garantit l'absence d'aliasing entre u_n et u_next.
__global__ void heatNaive(int num_points,
                          const int* __restrict__ row_ptr,
                          const int* __restrict__ col_idx,
                          const float* __restrict__ val,
                          const float* __restrict__ u_n,
                          float* __restrict__ u_next,
                          float cx) {

    // 1 thread = 1 ligne de la matrice = 1 point physique
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_points) return;

    float dot_product = 0.0f;
    const int row_start = row_ptr[row];
    const int row_end = row_ptr[row + 1];

    // Les points de bord (Dirichlet) ont une ligne vide : produit nul, température inchangée.
    for (int i = row_start; i < row_end; ++i) {
        dot_product += val[i] * u_n[col_idx[i]];
    }

    u_next[row] = u_n[row] + cx * dot_product; // U^{n+1} = U^n + cx * (L * U^n)
}

float* solveHeatGPUNaive(int num_points, const int* d_row_ptr, const int* d_col_idx,
                         const float* d_val, float* d_u, float* d_u_tmp,
                         float alpha, float dx, float dt, int steps) {

    const float cx = heatCoefficient(alpha, dx, dt);

    const int threadsPerBlock = 256;
    const int numBlocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    for (int t = 0; t < steps; ++t) {
        heatNaive<<<numBlocks, threadsPerBlock>>>(num_points, d_row_ptr, d_col_idx,
                                                  d_val, d_u, d_u_tmp, cx);
        CUDA_CHECK_KERNEL();
        std::swap(d_u, d_u_tmp);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return d_u; // buffer contenant le dernier état calculé
}
