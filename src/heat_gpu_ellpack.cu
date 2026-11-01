#include "../include/heat_solver.cuh"
#include "../include/cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <utility>

// NNZ en paramètre de template : le #pragma unroll devient effectif (avec un
// max_nnz connu seulement à l'exécution, le compilateur ne déroule rien).
template <int NNZ>
__global__ void heatELLPACKKernel(int num_points,
                                  const int* __restrict__ col_idx,
                                  const float* __restrict__ val,
                                  const float* __restrict__ u_n,
                                  float* __restrict__ u_next,
                                  float cx) {

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_points) return;

    float dot_product = 0.0f;

    // Boucle sans divergence : tous les threads font exactement NNZ itérations.
    // Stockage colonne-major : à n fixé, les 32 threads d'un warp lisent 32 adresses
    // consécutives de val[] et col_idx[] -> accès parfaitement coalescés.
    // size_t : (n * num_points + row) déborderait un int pour de grandes grilles.
    #pragma unroll
    for (int n = 0; n < NNZ; ++n) {
        const size_t idx = static_cast<size_t>(n) * num_points + row;
        dot_product += val[idx] * u_n[col_idx[idx]];
    }

    u_next[row] = u_n[row] + cx * dot_product;
}

float* solveHeatGPUELLPACK(int num_points, int max_nnz,
                           const int* d_col_idx, const float* d_val,
                           float* d_u, float* d_u_tmp,
                           float alpha, float dx, float dt, int steps) {

    // Seul le Laplacien 2D à 5 points est instancié. Pour un autre max_nnz,
    // ajouter une instanciation template ci-dessous.
    if (max_nnz != 5) {
        std::fprintf(stderr, "[ELLPACK] max_nnz = %d non supporté (attendu : 5)\n", max_nnz);
        std::exit(EXIT_FAILURE);
    }

    const float cx = heatCoefficient(alpha, dx, dt);
    const int threadsPerBlock = 256;
    const int numBlocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;

    for (int t = 0; t < steps; ++t) {
        heatELLPACKKernel<5><<<numBlocks, threadsPerBlock>>>(num_points, d_col_idx, d_val,
                                                            d_u, d_u_tmp, cx);
        CUDA_CHECK_KERNEL();
        std::swap(d_u, d_u_tmp);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return d_u;
}
